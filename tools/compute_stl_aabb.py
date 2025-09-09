#!/usr/bin/env python3
import os
import struct
import numpy as np


def load_stl_vertices(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        header = f.read(80)
        if len(header) < 80:
            raise RuntimeError('Not a valid STL file (header too short)')

        # Peek next 4 bytes for triangle count to guess binary vs ascii
        tri_count_bytes = f.read(4)
        if len(tri_count_bytes) < 4:
            raise RuntimeError('Not a valid STL file (missing triangle count)')

        tri_count = struct.unpack('<I', tri_count_bytes)[0]

        # Heuristic: if resulting file size matches binary layout, assume binary
        file_size = os.fstat(f.fileno()).st_size
        expected_size = 80 + 4 + tri_count * (12 + 36 + 2)
        is_binary = (file_size == expected_size)

        if not is_binary:
            # Fallback simple ASCII parser (slow but fine for one file)
            f.seek(0)
            text = f.read().decode('utf-8', errors='ignore')
            verts = []
            for line in text.splitlines():
                line = line.strip()
                if line.startswith('vertex'):
                    _, x, y, z = line.split()
                    verts.append([float(x), float(y), float(z)])
            if not verts:
                raise RuntimeError('Failed to parse ASCII STL vertices')
            return np.array(verts, dtype=np.float64)

        # Binary STL parsing
        verts = np.empty((tri_count * 3, 3), dtype=np.float64)
        for i in range(tri_count):
            # normal (3 floats), 3 vertices (9 floats), attribute (2 bytes)
            rec = f.read(12 + 36 + 2)
            if len(rec) < 50:
                raise RuntimeError('Unexpected EOF parsing binary STL')
            # Skip normal (12 bytes)
            # Unpack 9 floats for vertices
            v = struct.unpack('<9f', rec[12:12 + 36])
            verts[i * 3 + 0] = (v[0], v[1], v[2])
            verts[i * 3 + 1] = (v[3], v[4], v[5])
            verts[i * 3 + 2] = (v[6], v[7], v[8])
        return verts


def compute_aabb(verts: np.ndarray):
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    half_extents = (vmax - vmin) / 2.0
    return vmin, vmax, center, half_extents


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True, help='Path to STL file')
    parser.add_argument('--offset', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='x y z offset to add to center (e.g., geom pos offset in MJCF)')
    args = parser.parse_args()

    verts = load_stl_vertices(args.mesh)
    vmin, vmax, center, half = compute_aabb(verts)

    center = center + np.array(args.offset)

    print('AABB min:', vmin)
    print('AABB max:', vmax)
    print('Center (local + offset):', center)
    print('Half extents (size):', half)
    print('MJCF geom box suggestion:')
    print(f'<geom class="collision" type="box" pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}"/>')


