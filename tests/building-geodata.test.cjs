const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const { loadModule } = require('./helpers/module.cjs');

const { REAL_BUILDINGS } = loadModule(readFileSync(join(__dirname, '../js/building-geodata.js'), 'utf8'), {
  globals: { atob }
});

function area(ring) {
  let sum = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    sum += ring[j].x * ring[i].z - ring[i].x * ring[j].z;
  }
  return sum / 2;
}

test('the building survey retains 1,000 footprints, their courtyard holes and quantized dimensions', () => {
  assert.equal(REAL_BUILDINGS.length, 1000);
  let holes = 0;
  for (const building of REAL_BUILDINGS) {
    assert.ok(Number.isInteger(building.height) && building.height >= 90 && building.height <= 472);
    assert.ok(building.year === 0 || (building.year >= 1600 && building.year <= 2100 && building.year % 10 === 0));
    const points = [];
    assert.ok(building.polygons.length > 0);
    for (const polygon of building.polygons) {
      holes += polygon.holes.length;
      assert.ok(area(polygon.outer) > 0, 'Exterior winds counterclockwise in x/z.');
      for (const hole of polygon.holes) assert.ok(area(hole) < 0, 'Courtyards wind clockwise.');
      for (const ring of [polygon.outer, ...polygon.holes]) {
        assert.ok(ring.length >= 3);
        for (let i = 0; i < ring.length; i++) {
          const point = ring[i], next = ring[(i + 1) % ring.length];
          assert.ok(Number.isInteger(point.x * 2) && Number.isInteger(point.z * 2));
          assert.ok(point.x !== next.x || point.z !== next.z, 'No duplicate closing or adjacent vertices.');
          points.push(point);
        }
      }
    }
    assert.equal(building.minX, Math.min(...points.map(p => p.x)));
    assert.equal(building.maxX, Math.max(...points.map(p => p.x)));
    assert.equal(building.minZ, Math.min(...points.map(p => p.z)));
    assert.equal(building.maxZ, Math.max(...points.map(p => p.z)));
    assert.equal(building.cx, (building.minX + building.maxX) / 2);
    assert.equal(building.cz, (building.minZ + building.maxZ) / 2);
  }
  assert.equal(holes, 8);
});

test('survey buildings align with the existing city coordinate frame', () => {
  for (const [name, x, z, height, year] of [
    ['Empire State Building', -90.5, 685.5, 378, 1930],
    ['Chrysler Building', 510, -45.75, 289, 1930],
    ['One World Trade Center', -203.25, 5253.25, 429, 2010]
  ]) {
    const building = REAL_BUILDINGS.find(b => b.cx === x && b.cz === z);
    assert.ok(building, `${name} remains in the measured position.`);
    assert.equal(building.height, height);
    assert.equal(building.year, year);
  }
});
