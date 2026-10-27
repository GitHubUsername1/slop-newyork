#!/usr/bin/env python3
"""Generate js/building-geodata.js from NYC building footprints (requires Shapely 2.1+).

python3 -m pip install 'shapely>=2.1'
python3 tools/build-buildings.py /tmp/nyc-buildings.json --fetch
python3 tools/build-buildings.py /tmp/nyc-buildings.json

The downloaded snapshot stays outside the repository. --years can supplement an older
snapshot with a JSON array containing bin and construction_year columns from the same source.
"""

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import urllib.parse
import urllib.request

from shapely import transform
from shapely.geometry import Polygon, shape
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parent.parent
SOURCE = 'https://data.cityofnewyork.us/resource/5zhs-2jue.json'
COUNT = 1000
PRECISION = 0.5
LIMIT = 50000


def load_frame():
    # Use the geodata generator's projection rather than maintaining a second one.
    script = """
console.log = () => {};
const { toWorld } = await import('./tools/geo-base.mjs');
const { GEO } = await import('./js/nyc-geodata.js');
const lon = -73.98185, lat = 40.75318;
process.stdout.write(JSON.stringify({ lon, lat, origin: toWorld(lon, lat),
  east: toWorld(lon + 1, lat), north: toWorld(lon, lat + 1), island: GEO.manhattan }));
"""
    frame = json.loads(subprocess.check_output(
        ['node', '--no-warnings', '--input-type=module', '-e', script], cwd=ROOT))
    ox, oz = frame['origin']
    ex, ez = frame['east'][0] - ox, frame['east'][1] - oz
    nx, nz = frame['north'][0] - ox, frame['north'][1] - oz

    def project(lon, lat, elevation=None):
        lon, lat = lon - frame['lon'], lat - frame['lat']
        return ox + lon * ex + lat * nx, oz + lon * ez + lat * nz

    island = frame['island']
    return project, Polygon(zip(island[::2], island[1::2]))


def canonical_ring(ring):
    points = []
    for x, z in list(ring.coords)[:-1]:
        point = (round(x / PRECISION), round(z / PRECISION))
        if not points or points[-1] != point:
            points.append(point)
    if points[-1] == points[0]:
        points.pop()
    if len(points) < 3:
        raise ValueError('A building ring collapsed during quantization.')
    start = min(range(len(points)), key=points.__getitem__)
    return points[start:] + points[:start]


def prepare(rows, supplemental_years):
    project, island = load_frame()
    candidates = []
    for row in rows:
        # Exclude parking, canopies, tanks, placeholders, temporary structures and skybridges.
        if int(row.get('feature_code', 2100)) in (1000, 1001, 1002, 1003, 1005, 2110):
            continue
        height = float(row.get('height_roof') or 0) * 0.3048
        if not math.isfinite(height) or height <= 0:
            continue
        geometry = transform(shape(row['the_geom']), project, interleaved=False)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or not island.covers(geometry.centroid):
            continue
        year = int(row.get('construction_year') or supplemental_years.get(row.get('bin'), 0))
        year = round(year / 10) * 10 if 1600 <= year <= 2100 else 0
        candidates.append((height, row.get('bin', ''), geometry, year))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].wkb))
    if len(candidates) < COUNT:
        raise ValueError(f'Only {len(candidates)} eligible buildings; expected at least {COUNT}.')

    buildings = []
    for height, _, geometry, year in candidates[:COUNT]:
        geometry = geometry.simplify(PRECISION, preserve_topology=True)
        parts = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        polygons = []
        for part in parts:
            part = orient(part, sign=1)
            rings = [canonical_ring(part.exterior)] + sorted(canonical_ring(hole) for hole in part.interiors)
            if not Polygon(rings[0], rings[1:]).is_valid:
                raise ValueError('Quantization produced an invalid building footprint.')
            polygons.append(rings)
        polygons.sort()
        # Morton order puts nearby buildings together for better delta compression.
        center = geometry.centroid
        x, z = int(center.x + 4096), int(center.y + 16384)
        order = sum(((x >> i) & 1) << (2 * i) | ((z >> i) & 1) << (2 * i + 1) for i in range(16))
        buildings.append((order, {'height': round(height), 'year': year, 'polygons': polygons}))
    return [building for _, building in sorted(buildings, key=lambda item: item[0])]


def pack(buildings):
    data = bytearray()

    def uint(value):
        while value > 127:
            data.append((value & 127) | 128)
            value >>= 7
        data.append(value)

    def sint(value):
        uint(value * 2 if value >= 0 else -value * 2 - 1)

    uint(len(buildings))
    x = z = 0
    for building in buildings:
        uint(building['height'])
        uint(building['year'] // 10 - 159 if building['year'] else 0)
        uint(len(building['polygons']))
        for polygon in building['polygons']:
            uint(len(polygon))
            for ring in polygon:
                uint(len(ring))
                for px, pz in ring:
                    sint(px - x)
                    sint(pz - z)
                    x, z = px, pz
    return data


def module(data, digest):
    return f'''// Generated by tools/build-buildings.py; do not edit.
// NYC OTI Building Footprints: {SOURCE}
// Snapshot SHA-256: {digest}
// Tallest {COUNT} buildings in the detailed map. Coordinates: 0.5 m; roof heights: 1 m; construction years: decades (0 unknown).
const data = atob('{base64.b64encode(data).decode()}');
let cursor = 0, x = 0, z = 0;
function uint() {{
  let value = 0, shift = 0, byte;
  do {{
    byte = data.charCodeAt(cursor++);
    value |= (byte & 127) << shift;
    shift += 7;
  }} while (byte & 128);
  return value;
}}
const delta = () => {{ const value = uint(); return (value >>> 1) ^ -(value & 1); }};

export const REAL_BUILDINGS = Array.from({{ length: uint() }}, () => {{
  const height = uint(), decade = uint(), polygons = [];
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let n = uint(); n > 0; n--) {{
    const rings = [];
    for (let n = uint(); n > 0; n--) {{
      const ring = [];
      for (let n = uint(); n > 0; n--) {{
        x += delta(); z += delta();
        const point = {{ x: x * 0.5, z: z * 0.5 }};
        minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
        minZ = Math.min(minZ, point.z); maxZ = Math.max(maxZ, point.z);
        ring.push(point);
      }}
      rings.push(ring);
    }}
    polygons.push({{ outer: rings[0], holes: rings.slice(1) }});
  }}
  return {{ height, year: decade ? (decade + 159) * 10 : 0, polygons,
    minX, maxX, minZ, maxZ, cx: (minX + maxX) / 2, cz: (minZ + maxZ) / 2 }};
}});
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='?', type=Path, default=Path(tempfile.gettempdir()) / 'nyc-buildings.json')
    parser.add_argument('--fetch', action='store_true', help='download a fresh Manhattan snapshot to source')
    parser.add_argument('--years', type=Path, help='optional construction-year supplement for older snapshots')
    parser.add_argument('--output', type=Path, default=ROOT / 'js/building-geodata.js')
    args = parser.parse_args()
    if args.fetch:
        query = urllib.parse.urlencode({
            '$select': 'the_geom,height_roof,bin,feature_code,construction_year',
            '$where': 'bin >= 1000000 AND bin < 2000000',
            '$order': 'bin,doitt_id', '$limit': LIMIT,
        })
        with urllib.request.urlopen(f'{SOURCE}?{query}', timeout=120) as response:
            payload = response.read()
        rows = json.loads(payload)
        if not isinstance(rows, list) or not 0 < len(rows) < LIMIT:
            raise ValueError('Expected a complete, nonempty Manhattan building snapshot.')
        args.source.write_bytes(payload)
    source = args.source.read_bytes()
    rows = json.loads(source)
    if not isinstance(rows, list) or not 0 < len(rows) < LIMIT:
        raise ValueError('Expected a complete, nonempty Manhattan building snapshot.')
    years = {}
    if args.years:
        supplement = args.years.read_bytes()
        source += supplement
        for row in json.loads(supplement):
            year, key = int(row.get('construction_year') or 0), row['bin']
            if key != '1000000' and 1600 <= year <= 2100:
                years[key] = min(year, years.get(key, year))
    buildings = prepare(rows, years)
    output = module(pack(buildings), hashlib.sha256(source).hexdigest())
    args.output.write_text(output)
    polygons = [polygon for building in buildings for polygon in building['polygons']]
    holes = sum(len(polygon) - 1 for polygon in polygons)
    vertices = sum(len(ring) for polygon in polygons for ring in polygon)
    print(f'[buildings] {len(buildings)} buildings, {len(polygons)} polygons, {holes} holes, {vertices} vertices')
    print(f'[buildings] {min(b["height"] for b in buildings)}–{max(b["height"] for b in buildings)} m; '
          f'{sum(b["year"] > 0 for b in buildings)} construction decades')
    print(f'[buildings] {len(output.encode())} bytes; {len(gzip.compress(output.encode(), compresslevel=6, mtime=0))} bytes gzip')


if __name__ == '__main__':
    main()
