#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mujoco


# ----------------------------- BVH parsing (minimal) -----------------------------

@dataclass
class BVHNode:
    name: str
    offset: np.ndarray
    children: List["BVHNode"] = field(default_factory=list)


def _tokenize_bvh(text: str) -> List[str]:
    # Split braces as their own tokens, keep other tokens by whitespace
    toks: List[str] = []
    acc: List[str] = []
    for ch in text:
        if ch in "{}":
            if acc:
                toks.append("".join(acc))
                acc.clear()
            toks.append(ch)
        elif ch.isspace():
            if acc:
                toks.append("".join(acc))
                acc.clear()
        else:
            acc.append(ch)
    if acc:
        toks.append("".join(acc))
    return toks


class BVHParser:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> str:
        return self.toks[self.i]

    def _next(self) -> str:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def _expect(self, s: str) -> None:
        tok = self._next()
        if tok != s:
            raise ValueError(f"Expected '{s}', got '{tok}' at token {self.i}")

    def parse(self) -> BVHNode:
        # HIERARCHY ROOT ... { ... }
        if self._next().upper() != "HIERARCHY":
            # Some files may omit the literal, try continuing
            self.i -= 1
        root = self._parse_joint(expect_root=True)
        # Optionally skip MOTION; we only need hierarchy/offsets for rest pose
        return root

    def _parse_joint(self, expect_root: bool = False) -> BVHNode:
        head = self._next()
        if expect_root:
            if head.upper() != "ROOT":
                raise ValueError("Expected ROOT at hierarchy start")
        else:
            if head.upper() not in ("JOINT",):
                raise ValueError(f"Expected JOINT, got {head}")
        name = self._next()
        self._expect("{")

        # OFFSET
        if self._next().upper() != "OFFSET":
            raise ValueError("Expected OFFSET")
        ox = float(self._next()); oy = float(self._next()); oz = float(self._next())
        offset = np.array([ox, oy, oz], dtype=np.float64)

        # CHANNELS (ignore, we only need hierarchy for size/shape)
        tok = self._next()
        if tok.upper() == "CHANNELS":
            n = int(self._next())
            # Skip channel names
            for _ in range(n):
                _ = self._next()
            tok = self._next()

        # Children until '}'
        children: List[BVHNode] = []
        while tok != "}":
            if tok.upper() == "JOINT":
                # Step back one token to reuse joint parser
                self.i -= 1
                child = self._parse_joint(expect_root=False)
                children.append(child)
            elif tok.upper() == "END":
                # END Site { OFFSET x y z }
                # Treat as leaf with given offset and name 'EndSite_of_<parent>'
                end_label = self._next()  # 'Site'
                self._expect("{")
                if self._next().upper() != "OFFSET":
                    raise ValueError("Expected OFFSET in End Site")
                ex = float(self._next()); ey = float(self._next()); ez = float(self._next())
                self._expect("}")
                end_node = BVHNode(name=f"EndSite_of_{name}", offset=np.array([ex, ey, ez], dtype=np.float64))
                children.append(end_node)
            elif tok == "{":
                # Should not happen here (already consumed '{')
                pass
            else:
                # Unexpected token
                raise ValueError(f"Unexpected token while parsing children: {tok}")
            tok = self._next()

        return BVHNode(name=name, offset=offset, children=children)


def load_bvh_hierarchy(path: str) -> BVHNode:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    toks = _tokenize_bvh(text)
    parser = BVHParser(toks)
    return parser.parse()


def compute_bvh_rest_world_positions(root: BVHNode) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    positions: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []
    names: List[str] = []

    def dfs(node: BVHNode, parent_index: Optional[int], parent_pos: np.ndarray) -> int:
        idx = len(positions)
        pos = parent_pos + node.offset
        positions.append(pos)
        names.append(node.name)
        if parent_index is not None:
            edges.append((parent_index, idx))
        for child in node.children:
            dfs(child, idx, pos)
        return idx

    dfs(root, None, np.zeros(3, dtype=np.float64))
    return np.vstack(positions), edges, names


# ----------------------------- MuJoCo G1 skeleton -----------------------------

def load_g1_body_positions(xml_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # World body is 0, skip it
    nbody = int(model.nbody)
    positions = []
    names: List[str] = []
    # Create edges from each body to its parent (exclude parent==0 if you want to skip world)
    edges: List[Tuple[int, int]] = []

    # Collect positions and names; map from body index in model to index in positions
    body_to_plot_index = {}
    for b in range(1, nbody):
        pos = np.array(data.xpos[b], dtype=np.float64)
        positions.append(pos)
        names.append(model.names[mujoco.mj_namePtr(model, mujoco.mjtObj.mjOBJ_BODY, b)].decode("utf-8", errors="ignore") if hasattr(mujoco, "mj_namePtr") else "")
        body_to_plot_index[b] = len(positions) - 1

    for b in range(1, nbody):
        parent_b = int(model.body_parentid[b])
        if parent_b <= 0:
            continue
        if b in body_to_plot_index and parent_b in body_to_plot_index:
            edges.append((body_to_plot_index[parent_b], body_to_plot_index[b]))

    return np.vstack(positions), edges, names


# ----------------------------- Plotting -----------------------------

def _set_axes_equal(ax: plt.Axes, xyz_min: np.ndarray, xyz_max: np.ndarray) -> None:
    extents = xyz_max - xyz_min
    max_size = float(np.max(extents))
    if max_size <= 0:
        max_size = 1.0
    centers = (xyz_max + xyz_min) * 0.5
    for center, axis in zip(centers, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        axis(center - 0.5 * max_size, center + 0.5 * max_size)


def plot_skeleton(ax: plt.Axes, points: np.ndarray, edges: List[Tuple[int, int]], color: str, title: str, label: Optional[str] = None) -> None:
    # Draw edges
    for i, j in edges:
        xs = [points[i, 0], points[j, 0]]
        ys = [points[i, 1], points[j, 1]]
        zs = [points[i, 2], points[j, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, alpha=0.9, label=(label if (label and i == edges[0][0] and j == edges[0][1]) else None))
    # Draw joints
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=10)
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


def main() -> None:
    # ----------------------------- Static configuration -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BVH_PATH = os.path.join(BASE_DIR, "mocap.bvh")
    XML_PATH = os.path.join(BASE_DIR, "g1_description", "g1.xml")
    OUT_PATH = os.path.join(BASE_DIR, "output", "compare_bvh_g1.png")

    OVERLAY = True            # If False, renders side-by-side
    AUTO_SCALE = True         # Scale BVH to match G1 median bone length
    BVH_SCALE: Optional[float] = None  # Manual override scale (takes precedence if set)

    # Map BVH axes (X right, Y up, Z forward) -> G1 axes (X forward, Y left, Z up)
    # R columns are BVH basis expressed in G1: [r_x | r_y | r_z]
    #   r_x (BVH +X) -> +Y (G1 left)
    #   r_y (BVH +Y) -> +Z (G1 up)
    #   r_z (BVH +Z) -> +X (G1 forward)
    # This preserves right-handedness: r_x x r_y = r_z
    ROT_BVH_TO_G1 = np.array([[0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)

    bvh_path = os.path.abspath(BVH_PATH)
    xml_path = os.path.abspath(XML_PATH)
    out_path = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- BVH ---
    bvh_root = load_bvh_hierarchy(bvh_path)
    bvh_pts, bvh_edges, _ = compute_bvh_rest_world_positions(bvh_root)
    # Apply single-step axis rotation
    bvh_pts = (ROT_BVH_TO_G1 @ bvh_pts.T).T

    # Normalize BVH root to origin (it already is if offsets are absolute from root)
    bvh_pts = bvh_pts - bvh_pts[0]

    # --- G1 ---
    g1_pts, g1_edges, _ = load_g1_body_positions(xml_path)
    # Normalize G1 pelvis/root to ground reference (subtract pelvis)
    # Pelvis is usually the first moving body (index 1). Find pelvis by name fallback.
    pelvis_index = None
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        pelvis_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    except Exception:
        pelvis_index = None

    if pelvis_index is not None and pelvis_index > 0:
        # Map model body index to our plot point index again to find pelvis point index
        # Easiest: recompute map similarly
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body_to_plot_index = {}
        positions = []
        for b in range(1, int(model.nbody)):
            positions.append(np.array(data.xpos[b], dtype=np.float64))
            body_to_plot_index[b] = len(positions) - 1
        if pelvis_index in body_to_plot_index:
            pelvis_pos = positions[body_to_plot_index[pelvis_index]]
            g1_pts = g1_pts - pelvis_pos
    else:
        # Fallback: subtract first point
        g1_pts = g1_pts - g1_pts[0]

    # Optional scaling to make sizes comparable
    scale_used: Optional[float] = None
    if BVH_SCALE is not None:
        scale_used = float(BVH_SCALE)
    elif AUTO_SCALE:
        def median_edge_length(pts: np.ndarray, e: List[Tuple[int, int]]) -> float:
            if not e:
                return 1.0
            lens = [float(np.linalg.norm(pts[i] - pts[j])) for (i, j) in e]
            lens_sorted = sorted(lens)
            m = lens_sorted[len(lens_sorted)//2] if lens_sorted else 1.0
            return max(m, 1e-6)
        m_bvh = median_edge_length(bvh_pts, bvh_edges)
        m_g1 = median_edge_length(g1_pts, g1_edges)
        scale_used = float(m_g1 / m_bvh)
    if scale_used is not None:
        bvh_pts = bvh_pts * float(scale_used)

    if OVERLAY:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        title = "BVH (blue) overlaid with G1 (orange)"
        if scale_used is not None:
            title += f"  [bvh-scale={scale_used:.3f}]"
        plot_skeleton(ax, bvh_pts, bvh_edges, color="#1f77b4", title=title, label="BVH")
        plot_skeleton(ax, g1_pts, g1_edges, color="#ff7f0e", title="", label="G1")
        ax.legend(loc="upper right")
        all_pts = np.vstack([bvh_pts, g1_pts])
        _set_axes_equal(ax, all_pts.min(axis=0), all_pts.max(axis=0))
    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        plot_skeleton(ax1, bvh_pts, bvh_edges, color="#1f77b4", title="BVH skeleton (rest pose)")
        plot_skeleton(ax2, g1_pts, g1_edges, color="#ff7f0e", title="G1 skeleton")
        _set_axes_equal(ax1, bvh_pts.min(axis=0), bvh_pts.max(axis=0))
        _set_axes_equal(ax2, g1_pts.min(axis=0), g1_pts.max(axis=0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison figure to: {out_path}")


if __name__ == "__main__":
    main()


