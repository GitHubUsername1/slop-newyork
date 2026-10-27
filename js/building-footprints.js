import * as THREE from 'three/webgpu';
import { WALL_KINDS } from './tsl-materials.js';

// One merged facade/roof mesh per 1 km cell. Explicit wall coordinates keep window bays metre-sized on any angle.
function createBuildingFootprints(material, records) {
  const cells = new Map();
  records.forEach((record, index) => {
    if (!(record.height > 0)) return;
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const polygon of record.polygons) for (const p of polygon.outer) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    }
    const key = Math.floor((minX + maxX) / 2000) + ',' + Math.floor((minZ + maxZ) / 2000);
    let cell = cells.get(key);
    if (!cell) cells.set(key, (cell = { positions: [], normals: [], indices: [], boxes: [], walls: [] }));
    const height = record.height, top = height + 0.22;
    const seed = Math.floor(record.seed ?? index * 29) % 1000 + 0.1 + 0.8 * (record.tone ?? 0.5);

    for (const polygon of record.polygons) {
      const rings = [polygon.outer, ...(polygon.holes || [])].map((ring, i) => {
        const points = ring.map(p => new THREE.Vector2(p.x, p.z));
        if (points.length > 1 && points[0].equals(points[points.length - 1])) points.pop();
        // Clockwise outer walls face away from the building; courtyard walls face into their holes.
        if (THREE.ShapeUtils.isClockWise(points) !== (i === 0)) points.reverse();
        return points;
      });
      if (rings[0].length < 3) continue;
      rings.forEach((ring, i) => {
        for (let j = 0; j < ring.length; j++) {
          const a = ring[j], b = ring[(j + 1) % ring.length];
          const dx = b.x - a.x, dz = b.y - a.y, width = Math.hypot(dx, dz);
          if (width === 0) continue;
          const offset = cell.positions.length / 3;
          const kind = i ? WALL_KINDS.rear : (record.kind ?? WALL_KINDS.shop);
          // Match the positive-world-axis tangent used by the shared interior-map view ray.
          const reverse = (Math.abs(dz) >= Math.abs(dx) ? dz : dx) < 0;
          cell.positions.push(a.x, 0.22, a.y, b.x, 0.22, b.y, a.x, top, a.y, b.x, top, b.y);
          for (let k = 0; k < 4; k++) {
            cell.normals.push(-dz / width, 0, dx / width);
            cell.boxes.push(width, height, 1, seed);
            cell.walls.push(reverse ? 1 - k % 2 : k % 2, k < 2 ? height : 0, kind);
          }
          cell.indices.push(offset, offset + 1, offset + 2, offset + 2, offset + 1, offset + 3);
        }
      });
      const triangles = THREE.ShapeUtils.triangulateShape(rings[0], rings.slice(1));
      const points = rings.flat(), offset = cell.positions.length / 3;
      for (const p of points) {
        cell.positions.push(p.x, top, p.y);
        cell.normals.push(0, 1, 0);
        cell.boxes.push(1, height, 1, seed);
        cell.walls.push(0, 0, WALL_KINDS.shop);
      }
      for (const [a, b, c] of triangles) cell.indices.push(offset + a, offset + c, offset + b);
    }
  });

  const group = new THREE.Group();
  group.name = 'SurveyedBuildings';
  let first;
  for (const cell of cells.values()) {
    if (cell.indices.length === 0) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(cell.positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(cell.normals, 3));
    geometry.setAttribute('box', new THREE.Float32BufferAttribute(cell.boxes, 4));
    geometry.setAttribute('facadeWall', new THREE.Float32BufferAttribute(cell.walls, 3));
    geometry.setIndex(cell.indices);
    geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'SurveyedBuildingChunk';
    mesh.castShadow = mesh.receiveShadow = true;
    if (first) mesh.userData.sharesPipelineWith = first;
    else first = mesh;
    group.add(mesh);
  }
  return group;
}

export { createBuildingFootprints };
