#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import mujoco
import mujoco.viewer


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

def _apply_t_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set joint angles to approximate a human T-pose.

    Assumptions:
    - Arms straight out to the sides (abducted ~90 deg), elbows straight, wrists neutral
    - Legs straight, ankles neutral, waist neutral
    """
    # Angles in radians
    PI = math.pi
    joint_targets: Dict[str, float] = {
        # Waist
        "waist_yaw_joint": 0.0,

        # Left arm
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": +0.5 * PI,  # abduct left arm out to the side
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,

        # Right arm (mirror)
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.5 * PI,  # abduct right arm out to the side
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,

        # Left leg
        "left_hip_pitch_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "left_ankle_roll_joint": 0.0,

        # Right leg
        "right_hip_pitch_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
    }

    for jname, angle in joint_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {jname}")
        qpos_adr = int(model.jnt_qposadr[jid])
        # Skip freejoint (has 7 DoF) by ensuring hinge type (1 DoF)
        jnt_type = int(model.jnt_type[jid])
        if jnt_type != mujoco.mjtJoint.mjJNT_HINGE:
            # Ignore non-hinge targets silently if present
            continue
        data.qpos[qpos_adr] = float(angle)

    mujoco.mj_forward(model, data)


def _auto_straighten_elbows(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Numerically choose elbow angles that make forearms colinear with upper arms.

    This avoids guessing the model's elbow zero and works across XML variants.
    """

    def _straighten_one(elbow_joint: str, elbow_body: str, wrist_site: str, shoulder_body: str) -> None:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, elbow_joint)
            if jid < 0:
                return
            if int(model.jnt_type[jid]) != mujoco.mjtJoint.mjJNT_HINGE:
                return
            qadr = int(model.jnt_qposadr[jid])

            # Search range: use joint limit if available, otherwise a safe fallback
            has_lim = bool(int(model.jnt_limited[jid])) if hasattr(model, "jnt_limited") else True
            if has_lim:
                a_min = float(model.jnt_range[jid][0])
                a_max = float(model.jnt_range[jid][1])
            else:
                a_min, a_max = -math.pi, math.pi

            elbow_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, elbow_body)
            shoulder_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_body)
            wrist_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wrist_site)
            if elbow_bid < 0 or shoulder_bid < 0 or wrist_sid < 0:
                return

            best_angle = float(data.qpos[qadr])
            best_cost = 1e9

            # Coarse grid search is robust and cheap for a single DoF
            for a in np.linspace(a_min, a_max, 121, dtype=np.float64):
                data.qpos[qadr] = float(a)
                mujoco.mj_forward(model, data)
                u = np.array(data.xpos[elbow_bid] - data.xpos[shoulder_bid], dtype=np.float64)
                v = np.array(data.site_xpos[wrist_sid] - data.xpos[elbow_bid], dtype=np.float64)
                nu = float(np.linalg.norm(u))
                nv = float(np.linalg.norm(v))
                if nu <= 1e-9 or nv <= 1e-9:
                    continue
                dot = float(np.dot(u / nu, v / nv))
                cost = 1.0 - dot  # prefer same direction (dot -> 1)
                if cost < best_cost:
                    best_cost = cost
                    best_angle = float(a)

            data.qpos[qadr] = float(best_angle)
            mujoco.mj_forward(model, data)
        except Exception:
            # Fail loudly would interrupt usage; keep best-effort here
            pass

    _straighten_one("left_elbow_joint", "left_elbow_link", "left_palm", "left_shoulder_yaw_link")
    _straighten_one("right_elbow_joint", "right_elbow_link", "right_palm", "right_shoulder_yaw_link")


def _extract_g1_body_positions(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str], Dict[int, int]]:
    """Extract body world positions and edges from current MuJoCo state.

    Returns (positions, edges, names, bodyIndex->plotIndex map).
    """
    nbody = int(model.nbody)
    positions: List[np.ndarray] = []
    names: List[str] = []
    edges: List[Tuple[int, int]] = []
    body_to_plot_index: Dict[int, int] = {}

    # World body is index 0; skip
    for b in range(1, nbody):
        pos = np.array(data.xpos[b], dtype=np.float64)
        positions.append(pos)
        if hasattr(mujoco, "mj_namePtr"):
            names.append(model.names[mujoco.mj_namePtr(model, mujoco.mjtObj.mjOBJ_BODY, b)].decode("utf-8", errors="ignore"))
        else:
            names.append("")
        body_to_plot_index[b] = len(positions) - 1

    for b in range(1, nbody):
        parent_b = int(model.body_parentid[b])
        if parent_b <= 0:
            continue
        if b in body_to_plot_index and parent_b in body_to_plot_index:
            edges.append((body_to_plot_index[parent_b], body_to_plot_index[b]))

    return np.vstack(positions), edges, names, body_to_plot_index


def _get_site_positions(model: mujoco.MjModel, data: mujoco.MjData, site_names: List[str]) -> Dict[str, np.ndarray]:
    positions: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for sname in site_names:
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                missing.append(sname)
                continue
            positions[sname] = np.array(data.site_xpos[sid], dtype=np.float64)
        except Exception:
            missing.append(sname)

    # Fallback for head if not present: use imu_in_torso as an approximation
    if "head" in missing:
        try:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
            if sid >= 0:
                positions["head"] = np.array(data.site_xpos[sid], dtype=np.float64)
                print("Note: 'head' site not found. Using 'imu_in_torso' as a head approximation.")
        except Exception:
            pass

    actually_missing = [n for n in missing if n not in positions]
    if actually_missing:
        print(f"Missing sites (skipped): {', '.join(actually_missing)}")
    return positions


# ----------------------------- Plotting -----------------------------

def _compute_wingspan_from_points(points: np.ndarray) -> float:
    """Compute lateral span (Y-axis extent) as a proxy for wingspan.

    Assumes points are expressed in G1 coordinates where +Y is left/right.
    """
    if points.size == 0:
        return 0.0
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    return max(0.0, y_max - y_min)


def _compute_g1_wingspan(g1_points: np.ndarray, extremity_sites: Dict[str, np.ndarray]) -> float:
    """Prefer palm site distance; fallback to lateral extent of body points."""
    try:
        lp = extremity_sites.get("left_palm")
        rp = extremity_sites.get("right_palm")
        if lp is not None and rp is not None:
            return float(np.linalg.norm(lp - rp))
    except Exception:
        pass
    return _compute_wingspan_from_points(g1_points)


def _estimate_bvh_arm_height(points: np.ndarray, k: int = 3) -> Optional[float]:
    """Estimate arm height in T-pose from lateral extremes in Y.

    Heuristic: take mean Z of the k-most left and k-most right points by Y.
    """
    if points.size == 0:
        return None
    k = max(1, int(k))
    order = np.argsort(points[:, 1])  # ascending by Y
    left_idxs = order[:k]
    right_idxs = order[-k:]
    zs = np.concatenate([points[left_idxs, 2], points[right_idxs, 2]])
    if zs.size == 0:
        return None
    return float(np.mean(zs))


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


def plot_site_markers(ax: plt.Axes, site_positions: Dict[str, np.ndarray], color: str = "#d62728") -> None:
    # Plot extremity sites with larger markers and labels
    first = True
    for name, pos in site_positions.items():
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=60, marker="o", label=("Extremity sites" if first else None))
        ax.text(pos[0], pos[1], pos[2], f" {name}", color=color, fontsize=8)
        first = False


def main() -> None:
    # ----------------------------- Static configuration -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BVH_PATH = os.path.join(BASE_DIR, "mocap.bvh")
    XML_PATH = os.path.join(BASE_DIR, "g1_description", "g1.xml")
    OUT_PATH = os.path.join(BASE_DIR, "output", "compare_bvh_g1.png")

    OVERLAY = True            # If False, renders side-by-side
    AUTO_SCALE = True         # Enable automatic scaling
    BVH_SCALE: Optional[float] = None  # Manual override scale (takes precedence if set)
    SHOULDER_SCALE = True     # If True and AUTO_SCALE, scale by shoulder height parity
    WINGSPAN_SCALE = False    # Keep available but disabled by default
    INTERACTIVE_VIEWER = True  # If True, open MuJoCo viewer (no physics) and draw BVH overlay
    LINE_RADIUS = 0.006        # Line radius for BVH connectors in viewer

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
    bvh_pts, bvh_edges, bvh_names = compute_bvh_rest_world_positions(bvh_root)
    # Apply single-step axis rotation
    bvh_pts = (ROT_BVH_TO_G1 @ bvh_pts.T).T

    # Normalize BVH root to origin (it already is if offsets are absolute from root)
    bvh_pts = bvh_pts - bvh_pts[0]

    # --- G1 ---
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # Set an approximate T-pose before computing positions
    _apply_t_pose(model, data)
    # Ensure elbows are fully extended for a clean T-pose
    _auto_straighten_elbows(model, data)
    g1_pts, g1_edges, g1_names, _ = _extract_g1_body_positions(model, data)
    # Pelvis information and normalized arrays for matplotlib path
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        pelvis_pos = np.array(data.xpos[pelvis_id], dtype=np.float64)
        g1_pts = g1_pts - pelvis_pos
    else:
        pelvis_pos = np.zeros(3, dtype=np.float64)
        g1_pts = g1_pts - g1_pts[0]

    # Extremity site positions (head, hands, feet)
    site_names = ["head", "left_palm", "right_palm", "left_foot", "right_foot"]
    extremity_sites = _get_site_positions(model, data, site_names)
    # Center sites by pelvis offset to match g1_pts normalization for matplotlib
    if pelvis_id >= 0:
        extremity_sites = {k: (v - pelvis_pos) for k, v in extremity_sites.items()}

    # Optional scaling to make sizes comparable (no silent fallbacks)
    scale_used: Optional[float] = None
    scale_method: Optional[str] = None
    if BVH_SCALE is not None:
        scale_used = float(BVH_SCALE)
        scale_method = "manual"
    elif AUTO_SCALE:
        if SHOULDER_SCALE:
            # G1 shoulder height from body positions (fail loudly if missing)
            l_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_yaw_link")
            r_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_yaw_link")
            if l_bid < 0 or r_bid < 0:
                raise RuntimeError("Required shoulder bodies not found in G1 model.")
            z_left = float(data.xpos[l_bid][2])
            z_right = float(data.xpos[r_bid][2])
            z_g1_shoulder = 0.5 * (z_left + z_right)
            # BVH shoulder height estimated from lateral extremes relative to BVH ground
            z_bvh_shoulder = _estimate_bvh_arm_height(bvh_pts, k=3)
            if z_bvh_shoulder is None:
                raise RuntimeError("Failed to estimate BVH shoulder height from points.")
            z_bvh_ground = float(bvh_pts[:, 2].min())
            rel_bvh_shoulder = float(z_bvh_shoulder - z_bvh_ground)
            if rel_bvh_shoulder <= 1e-9:
                raise RuntimeError("Non-positive BVH shoulder-ground height; cannot scale.")
            scale_used = float(z_g1_shoulder / rel_bvh_shoulder)
            scale_method = "shoulder-height"
        elif WINGSPAN_SCALE:
            wingspan_bvh = _compute_wingspan_from_points(bvh_pts)
            wingspan_g1 = _compute_g1_wingspan(g1_pts, extremity_sites)
            if wingspan_bvh <= 1e-9 or wingspan_g1 <= 1e-9:
                raise RuntimeError("Invalid wingspan(s); cannot scale.")
            scale_used = float(wingspan_g1 / wingspan_bvh)
            scale_method = "wingspan"
    if scale_used is not None:
        bvh_pts = bvh_pts * float(scale_used)

    if INTERACTIVE_VIEWER:
        # In viewer: draw BVH overlay in WORLD coordinates
        # - Align BVH ground to z=0 by lifting so its lowest point is on ground
        # - Anchor XY near the robot pelvis so they share the same ground plane and neighborhood
        bvh_ground = bvh_pts.copy()
        bvh_ground[:, 2] -= float(bvh_ground[:, 2].min())
        bvh_world = bvh_ground + np.array([float(pelvis_pos[0]), float(pelvis_pos[1]), 0.0], dtype=np.float64)

        def _draw_bvh_overlay(scene: mujoco.MJVSCENE, pts: np.ndarray, edges: List[Tuple[int, int]], rgba: np.ndarray) -> None:
            # Reset user scene geoms
            scene.ngeom = 0
            # Draw edges as connectors
            for i, j in edges:
                a = pts[i]
                b = pts[j]
                if int(scene.ngeom) >= int(scene.maxgeom):
                    break
                g = scene.geoms[int(scene.ngeom)]
                a3 = np.asarray(a, dtype=np.float64)
                b3 = np.asarray(b, dtype=np.float64)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, float(LINE_RADIUS), a3, b3)
                try:
                    # Set color
                    g.rgba[0] = float(rgba[0]); g.rgba[1] = float(rgba[1]); g.rgba[2] = float(rgba[2]); g.rgba[3] = float(rgba[3])
                except Exception:
                    pass
                scene.ngeom += 1

        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
            rgba = np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32)
            while viewer.is_running():
                _draw_bvh_overlay(viewer.user_scn, bvh_world, bvh_edges, rgba)
                mujoco.mj_camlight(model, data)  # keep lighting sane
                viewer.sync()
        return

    if OVERLAY:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        title = "BVH (blue) overlaid with G1 (orange)"
        if scale_used is not None:
            extra = f"bvh-scale={scale_used:.3f}"
            if scale_method:
                extra += f" by {scale_method}"
            title += f"  [{extra}]"
        plot_skeleton(ax, bvh_pts, bvh_edges, color="#1f77b4", title=title, label="BVH")
        plot_skeleton(ax, g1_pts, g1_edges, color="#ff7f0e", title="", label="G1")
        if extremity_sites:
            plot_site_markers(ax, extremity_sites, color="#d62728")
        ax.legend(loc="upper right")
        all_pts = np.vstack([bvh_pts, g1_pts])
        _set_axes_equal(ax, all_pts.min(axis=0), all_pts.max(axis=0))
    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        plot_skeleton(ax1, bvh_pts, bvh_edges, color="#1f77b4", title="BVH skeleton (rest pose)")
        plot_skeleton(ax2, g1_pts, g1_edges, color="#ff7f0e", title="G1 skeleton")
        if extremity_sites:
            plot_site_markers(ax2, extremity_sites, color="#d62728")
        _set_axes_equal(ax1, bvh_pts.min(axis=0), bvh_pts.max(axis=0))
        _set_axes_equal(ax2, g1_pts.min(axis=0), g1_pts.max(axis=0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison figure to: {out_path}")


if __name__ == "__main__":
    main()


