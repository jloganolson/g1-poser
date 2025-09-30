from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
import threading
import json
from datetime import datetime
import shutil
import subprocess


# ----------------------------- Static configuration -----------------------------
# Follow workspace rule: no CLI args; keep tunables here.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BVH_PATH = os.path.join(BASE_DIR, "mocap.bvh")
XML_PATH = os.path.join(BASE_DIR, "g1_description", "scene_g1_targets.xml")

# Mink/IK
SOLVER = "daqp"
FPS = 200.0
START_AT_FRAME: int = 0  # Initial BVH frame index to start playback from

# BVH -> G1 axis map (BVH: X right, Y up, Z fwd) to (G1: X fwd, Y left, Z up)
# Columns are BVH basis expressed in G1: [r_x | r_y | r_z]
ROT_BVH_TO_G1 = np.array(
    [
        [0.0, 0.0, 1.0],  # BVH +Z -> +X
        [1.0, 0.0, 0.0],  # BVH +X -> +Y
        [0.0, 1.0, 0.0],  # BVH +Y -> +Z
    ],
    dtype=np.float64,
)

# Scaling options
# Default to a fixed manual scale; disable auto-scaling heuristics.
AUTO_SCALE = False
BVH_SCALE: Optional[float] = 0.009  # Initial default scale
SHOULDER_SCALE = False
WINGSPAN_SCALE = False

# Viewer overlay
LINE_RADIUS = 0.006

# CoM visualization
COM_SPHERE_RADIUS = 0.05
COM_RGBA = [1.0, 0.3, 0.3, 1.0]

# IK safety clamps
FLOOR_Z = 0.0
CLAMP_HANDS_AND_FEET_TO_FLOOR = True


# BVH node names (exact, case-insensitive). Static per provided BVH.
BVH_LEFT_HAND_NAME = "left_wrist"
BVH_RIGHT_HAND_NAME = "right_wrist"
BVH_LEFT_FOOT_NAME = "left_ankle"
BVH_RIGHT_FOOT_NAME = "right_ankle"
BVH_PELVIS_NAME = "pelvis"
BVH_LEFT_SHOULDER_NAME = "left_shoulder"
BVH_RIGHT_SHOULDER_NAME = "right_shoulder"

# New BVH joint names for elbows and knees
BVH_LEFT_ELBOW_NAME = "left_elbow"
BVH_RIGHT_ELBOW_NAME = "right_elbow"
BVH_LEFT_KNEE_NAME = "left_knee"
BVH_RIGHT_KNEE_NAME = "right_knee"


# ----------------------------- Minimal BVH utilities -----------------------------
@dataclass
class BVHNode:
    name: str
    offset: np.ndarray
    children: List["BVHNode"] = field(default_factory=list)
    channels: List[str] = field(default_factory=list)
    channel_offset: int = -1


@dataclass
class BVHMotion:
    frames: np.ndarray  # (num_frames, total_channels)
    frame_time: float
    total_channels: int


def _tokenize_bvh(text: str) -> List[str]:
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
        self.total_channels = 0
        self.nodes_in_order: List[BVHNode] = []

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

    def parse_with_motion(self) -> Tuple[BVHNode, BVHMotion]:
        if self._next().upper() != "HIERARCHY":
            self.i -= 1
        root = self._parse_joint(expect_root=True)

        tok = self._next()
        if tok.upper() != "MOTION":
            raise ValueError(f"Expected MOTION section, got '{tok}' at token {self.i}")

        t = self._next()
        if not t.upper().startswith("FRAMES"):
            raise ValueError("Expected 'Frames:' in MOTION section")
        num_frames = int(self._next())

        t = self._next()
        if not t.upper().startswith("FRAME"):
            raise ValueError("Expected 'Frame Time:' in MOTION section")
        t = self._next()
        if not t.upper().startswith("TIME"):
            raise ValueError("Expected 'Frame Time:' in MOTION section")
        frame_time = float(self._next())

        if self.total_channels <= 0:
            raise ValueError("No channels described in hierarchy; cannot read motion frames")

        values: List[float] = []
        needed = int(num_frames * self.total_channels)
        for _ in range(needed):
            values.append(float(self._next()))
        frames = np.asarray(values, dtype=np.float64).reshape((num_frames, self.total_channels))
        motion = BVHMotion(frames=frames, frame_time=float(frame_time), total_channels=int(self.total_channels))
        return root, motion

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

        if self._next().upper() != "OFFSET":
            raise ValueError("Expected OFFSET")
        ox = float(self._next()); oy = float(self._next()); oz = float(self._next())
        offset = np.array([ox, oy, oz], dtype=np.float64)

        tok = self._next()
        node_channels: List[str] = []
        if tok.upper() == "CHANNELS":
            n = int(self._next())
            for _ in range(n):
                node_channels.append(self._next())
            tok = self._next()
        node = BVHNode(name=name, offset=offset, children=[], channels=node_channels, channel_offset=-1)
        if node_channels:
            node.channel_offset = int(self.total_channels)
            self.total_channels += len(node_channels)
            self.nodes_in_order.append(node)

        children: List[BVHNode] = []
        while tok != "}":
            if tok.upper() == "JOINT":
                self.i -= 1
                child = self._parse_joint(expect_root=False)
                children.append(child)
            elif tok.upper() == "END":
                end_label = self._next()  # 'Site'
                self._expect("{")
                if self._next().upper() != "OFFSET":
                    raise ValueError("Expected OFFSET in End Site")
                ex = float(self._next()); ey = float(self._next()); ez = float(self._next())
                self._expect("}")
                end_node = BVHNode(name=f"EndSite_of_{name}", offset=np.array([ex, ey, ez], dtype=np.float64))
                children.append(end_node)
            elif tok == "{":
                pass
            else:
                raise ValueError(f"Unexpected token while parsing children: {tok}")
            tok = self._next()

        node.children = children
        return node


def load_bvh_with_motion(path: str) -> Tuple[BVHNode, BVHMotion]:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    toks = _tokenize_bvh(text)
    parser = BVHParser(toks)
    return parser.parse_with_motion()


def _rot_x(a: float) -> np.ndarray:
    c = math.cos(a); s = math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c = math.cos(a); s = math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c = math.cos(a); s = math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def compute_bvh_frame_world_positions(root: BVHNode, motion: BVHMotion, frame_idx: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    values = motion.frames[int(frame_idx)]
    positions: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []
    names: List[str] = []

    def local_rotation(node: BVHNode) -> np.ndarray:
        if not node.channels:
            return np.eye(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        off = node.channel_offset
        for j, ch in enumerate(node.channels):
            v = float(values[off + j])
            if ch.lower().endswith("rotation"):
                a = math.radians(v)
                axis = ch[0].upper()
                if axis == "X":
                    R = R @ _rot_x(a)
                elif axis == "Y":
                    R = R @ _rot_y(a)
                elif axis == "Z":
                    R = R @ _rot_z(a)
        return R

    def root_translation(node: BVHNode) -> np.ndarray:
        if not node.channels:
            return np.zeros(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        off = node.channel_offset
        for j, ch in enumerate(node.channels):
            if ch.lower().endswith("position"):
                axis = ch[0].upper()
                if axis == "X":
                    t[0] = float(values[off + j])
                elif axis == "Y":
                    t[1] = float(values[off + j])
                elif axis == "Z":
                    t[2] = float(values[off + j])
        return t

    def dfs(node: BVHNode, parent_index: int, parent_pos: np.ndarray, parent_rot: np.ndarray, is_root: bool) -> int:
        idx = len(positions)
        R_local = local_rotation(node)
        R_world = parent_rot @ R_local
        pos = parent_pos + parent_rot @ node.offset
        if is_root:
            pos = pos + root_translation(node)
        positions.append(pos)
        names.append(node.name)
        if parent_index is not None and parent_index >= 0:
            edges.append((parent_index, idx))
        for child in node.children:
            dfs(child, idx, pos, R_world, False)
        return idx

    dfs(root, -1, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64), True)
    return np.vstack(positions), edges, names


def compute_bvh_frame_world_poses(
    root: BVHNode, motion: BVHMotion, frame_idx: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[str]]:
    """Compute world-space positions and rotations (3x3) for all BVH nodes.

    Returns (positions Nx3, rotations Nx3x3, edges, names).
    """
    values = motion.frames[int(frame_idx)]
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    edges: List[Tuple[int, int]] = []
    names: List[str] = []

    def local_rotation(node: BVHNode) -> np.ndarray:
        if not node.channels:
            return np.eye(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        off = node.channel_offset
        for j, ch in enumerate(node.channels):
            v = float(values[off + j])
            if ch.lower().endswith("rotation"):
                a = math.radians(v)
                axis = ch[0].upper()
                if axis == "X":
                    R = R @ _rot_x(a)
                elif axis == "Y":
                    R = R @ _rot_y(a)
                elif axis == "Z":
                    R = R @ _rot_z(a)
        return R

    def root_translation(node: BVHNode) -> np.ndarray:
        if not node.channels:
            return np.zeros(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        off = node.channel_offset
        for j, ch in enumerate(node.channels):
            if ch.lower().endswith("position"):
                axis = ch[0].upper()
                if axis == "X":
                    t[0] = float(values[off + j])
                elif axis == "Y":
                    t[1] = float(values[off + j])
                elif axis == "Z":
                    t[2] = float(values[off + j])
        return t

    def dfs(node: BVHNode, parent_index: int, parent_pos: np.ndarray, parent_rot: np.ndarray, is_root: bool) -> int:
        idx = len(positions)
        R_local = local_rotation(node)
        R_world = parent_rot @ R_local
        pos = parent_pos + parent_rot @ node.offset
        if is_root:
            pos = pos + root_translation(node)
        positions.append(pos)
        rotations.append(R_world)
        names.append(node.name)
        if parent_index is not None and parent_index >= 0:
            edges.append((parent_index, idx))
        for child in node.children:
            dfs(child, idx, pos, R_world, False)
        return idx

    dfs(root, -1, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64), True)
    return np.vstack(positions), np.stack(rotations, axis=0), edges, names


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in MuJoCo (w, x, y, z)."""
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        S = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= max(1e-12, float(np.linalg.norm(q)))
    return q


def _quat_wxyz_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw (XYZ intrinsic) in radians."""
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def _estimate_bvh_arm_height(points: np.ndarray, k: int = 3) -> Optional[float]:
    if points.size == 0:
        return None
    k = max(1, int(k))
    order = np.argsort(points[:, 1])
    left_idxs = order[:k]
    right_idxs = order[-k:]
    zs = np.concatenate([points[left_idxs, 2], points[right_idxs, 2]])
    if zs.size == 0:
        return None
    return float(np.mean(zs))


def _compute_wingspan_from_points(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    return max(0.0, y_max - y_min)


def _resolve_mocap_id_or_neg1(model: mujoco.MjModel, body_name: str) -> int:
    try:
        return int(model.body(body_name).mocapid[0])
    except Exception:
        return -1


def _resolve_required_mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    mid = _resolve_mocap_id_or_neg1(model, body_name)
    if mid == -1:
        raise RuntimeError(f"Required mocap body '{body_name}' not found in model.")
    return mid


def _clamp_floor_z(p: np.ndarray, floor_z: float) -> np.ndarray:
    """Return a copy of p with z clamped to at least floor_z."""
    v = np.asarray(p, dtype=np.float64).copy()
    if v.shape[0] >= 3 and float(v[2]) < float(floor_z):
        v[2] = float(floor_z)
    return v


def _shift_base_z_to_ground(model: mujoco.MjModel, data: mujoco.MjData, left_site: str, right_site: str) -> None:
    try:
        free_qpos_addr = None
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:
                free_qpos_addr = int(model.jnt_qposadr[j])
                break
        if free_qpos_addr is None:
            return

        mujoco.mj_forward(model, data)
        l_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_site)
        r_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, right_site)
        if l_sid == -1 or r_sid == -1:
            return
        z_l = float(data.site_xpos[l_sid][2])
        z_r = float(data.site_xpos[r_sid][2])
        z_avg = 0.5 * (z_l + z_r)
        data.qpos[free_qpos_addr + 2] -= z_avg
        mujoco.mj_forward(model, data)
    except Exception:
        # Convenience helper; not fatal
        pass


def _find_bvh_index_by_name(names: List[str], wanted_name: str) -> int:
    lname_to_idx = {n.lower(): i for i, n in enumerate(names)}
    key = wanted_name.lower()
    if key not in lname_to_idx:
        raise RuntimeError(f"Required BVH node '{wanted_name}' not found. Available: {list(lname_to_idx.keys())[:10]} ...")
    return lname_to_idx[key]


def main() -> None:
    bvh_path = os.path.abspath(BVH_PATH)
    xml_path = os.path.abspath(XML_PATH)

    # Load BVH once
    bvh_root, bvh_motion = load_bvh_with_motion(bvh_path)
    num_frames = int(bvh_motion.frames.shape[0])
    frame_time = float(bvh_motion.frame_time)

    # Create MuJoCo model/data and Mink configuration
    model = mujoco.MjModel.from_xml_path(xml_path)
    configuration = mink.Configuration(model)
    data = configuration.data

    # Stabilization and end-effector tasks
    tasks = [
        (pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        (pelvis_position_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )),
        (torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        (posture_task := mink.PostureTask(model, cost=1e-1)),
    ]

    # Shoulder tracking tasks (follow mocap targets when present)
    left_shoulder_task = mink.FrameTask(
        frame_name="left_shoulder_yaw_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    right_shoulder_task = mink.FrameTask(
        frame_name="right_shoulder_yaw_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    tasks.extend([left_shoulder_task, right_shoulder_task])

    # Elbow and knee tracking tasks (body frames)
    left_elbow_task = mink.FrameTask(
        frame_name="left_elbow_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    right_elbow_task = mink.FrameTask(
        frame_name="right_elbow_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    left_knee_task = mink.FrameTask(
        frame_name="left_knee_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    right_knee_task = mink.FrameTask(
        frame_name="right_knee_link",
        frame_type="body",
        position_cost=8.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    tasks.extend([left_elbow_task, right_elbow_task, left_knee_task, right_knee_task])

    HAND_POSITION_COST = 25.0
    FOOT_POSITION_COST = 30.0
    right_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=HAND_POSITION_COST,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    left_hand_task = mink.FrameTask(
        frame_name="left_palm",
        frame_type="site",
        position_cost=HAND_POSITION_COST,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    left_foot_task = mink.FrameTask(
        frame_name="left_ankle",
        frame_type="site",
        position_cost=FOOT_POSITION_COST,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    right_foot_task = mink.FrameTask(
        frame_name="right_ankle",
        frame_type="site",
        position_cost=FOOT_POSITION_COST,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    tasks.extend([right_hand_task, left_hand_task, left_foot_task, right_foot_task])

    # Optional foot orientation stabilization (keeps feet level-ish)
    left_foot_orientation_task = mink.FrameTask(
        frame_name="left_ankle_roll_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    right_foot_orientation_task = mink.FrameTask(
        frame_name="right_ankle_roll_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=2.0,
        lm_damping=1.0,
    )
    tasks.extend([left_foot_orientation_task, right_foot_orientation_task])

    # Collision avoidance using fromto sensors (arms vs legs)
    # Geom names defined in g1_description/g1.xml
    collision_pairs = [
        (["left_forearm", "right_forearm"], ["left_thigh", "right_thigh", "left_shin", "right_shin"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore[arg-type]
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.12,
    )
    limits = [mink.ConfigurationLimit(model), collision_avoidance_limit]

    mujoco.mj_forward(configuration.model, configuration.data)

    # Resolve mocap bodies for driving targets
    right_palm_mid = _resolve_required_mocap_id(model, "right_palm_target")
    left_palm_mid = _resolve_required_mocap_id(model, "left_palm_target")
    left_foot_mid = _resolve_required_mocap_id(model, "left_foot_target")
    right_foot_mid = _resolve_required_mocap_id(model, "right_foot_target")
    pelvis_mid = _resolve_required_mocap_id(model, "pelvis_target")
    left_shoulder_mid = _resolve_required_mocap_id(model, "left_shoulder_target")
    right_shoulder_mid = _resolve_required_mocap_id(model, "right_shoulder_target")
    left_elbow_mid = _resolve_required_mocap_id(model, "left_elbow_target")
    right_elbow_mid = _resolve_required_mocap_id(model, "right_elbow_target")
    left_knee_mid = _resolve_required_mocap_id(model, "left_knee_target")
    right_knee_mid = _resolve_required_mocap_id(model, "right_knee_target")

    # All targets are required; fail fast if any are missing
    assert min(
        right_palm_mid,
        left_palm_mid,
        left_foot_mid,
        right_foot_mid,
        pelvis_mid,
        left_shoulder_mid,
        right_shoulder_mid,
        left_elbow_mid,
        right_elbow_mid,
        left_knee_mid,
        right_knee_mid,
    ) >= 0, "Required mocap targets not found in model"

    # Initial base alignment to ground
    _shift_base_z_to_ground(model, data, left_site="left_foot", right_site="right_foot")

    # Re-initialize task targets from current configuration
    mujoco.mj_forward(configuration.model, configuration.data)
    posture_task.set_target_from_configuration(configuration)
    pelvis_orientation_task.set_target_from_configuration(configuration)
    pelvis_position_task.set_target_from_configuration(configuration)
    torso_orientation_task.set_target_from_configuration(configuration)
    left_foot_orientation_task.set_target_from_configuration(configuration)
    right_foot_orientation_task.set_target_from_configuration(configuration)

    # Precompute rest points (unscaled) for anchor computation and name mapping
    frame0_pts_unscaled, bvh_edges, frame0_names = compute_bvh_frame_world_positions(bvh_root, bvh_motion, 0)
    frame0_pts_unscaled = (ROT_BVH_TO_G1 @ frame0_pts_unscaled.T).T

    scale_used: Optional[float] = None
    if BVH_SCALE is not None:
        scale_used = float(BVH_SCALE)
    elif AUTO_SCALE:
        if SHOULDER_SCALE:
            l_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_yaw_link")
            r_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_yaw_link")
            if l_bid < 0 or r_bid < 0:
                raise RuntimeError("Required shoulder bodies not found in G1 model.")
            z_left = float(data.xpos[l_bid][2])
            z_right = float(data.xpos[r_bid][2])
            z_g1_shoulder = 0.5 * (z_left + z_right)
            # Estimate BVH shoulder height from lateral extremes
            z_bvh_shoulder = _estimate_bvh_arm_height(frame0_pts_unscaled, k=3)
            if z_bvh_shoulder is None:
                raise RuntimeError("Failed to estimate BVH shoulder height from points.")
            z_bvh_ground = float(frame0_pts_unscaled[:, 2].min())
            rel_bvh_shoulder = float(z_bvh_shoulder - z_bvh_ground)
            if rel_bvh_shoulder <= 1e-9:
                raise RuntimeError("Non-positive BVH shoulder-ground height; cannot scale.")
            scale_used = float(z_g1_shoulder / rel_bvh_shoulder)
        elif WINGSPAN_SCALE:
            wingspan_bvh = _compute_wingspan_from_points(frame0_pts_unscaled)
            # Approximate G1 wingspan as palm distance if sites exist
            lp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
            rp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
            if lp_sid < 0 or rp_sid < 0:
                raise RuntimeError("Palm sites missing; cannot wingspan-scale.")
            wingspan_g1 = float(np.linalg.norm(np.array(data.site_xpos[lp_sid]) - np.array(data.site_xpos[rp_sid])))
            if wingspan_bvh <= 1e-9 or wingspan_g1 <= 1e-9:
                raise RuntimeError("Invalid wingspan(s); cannot scale.")
            scale_used = float(wingspan_g1 / wingspan_bvh)

    # Live control UI using ttk widgets (no Entry widgets to avoid XCB issues)
    class _ControlUI:
        def __init__(self, initial_value: float, min_value: float, max_value: float, total_frames: int) -> None:
            self._value = float(initial_value)
            self._lock = threading.Lock()
            self._ready = threading.Event()
            self._min = float(min_value)
            self._max = float(max_value)
            self._paused = False
            self._reset_event = threading.Event()
            self._total_frames = int(total_frames)
            self._frame_idx_shared = 0
            self._root = None
            self._frame_label_var = None
            self._export_event = threading.Event()
            self._recording = False
            self._rec_label_var = None
            self._prev_event = threading.Event()
            self._next_event = threading.Event()
            self._export_frame_event = threading.Event()

        def start(self) -> None:
            def _run() -> None:
                import tkinter as tk
                from tkinter import ttk
                root = tk.Tk()
                root.title("BVH Controls")
                try:
                    root.geometry("520x220")
                except Exception:
                    pass
                # Frame display
                self._frame_label_var = tk.StringVar(value=f"Frame: 0 / {max(0, self._total_frames - 1)}")
                frame_label = ttk.Label(root, textvariable=self._frame_label_var, anchor="w")
                frame_label.pack(fill="x", padx=8, pady=(0, 6))

                # Recording status display
                self._rec_label_var = tk.StringVar(value="")
                rec_label = ttk.Label(root, textvariable=self._rec_label_var, anchor="w")
                rec_label.pack(fill="x", padx=8, pady=(0, 6))

                def _tick_update_frame_label() -> None:
                    with self._lock:
                        idx = int(self._frame_idx_shared)
                        total_minus_1 = max(0, self._total_frames - 1)
                        text = f"Frame: {idx} / {total_minus_1}"
                    try:
                        self._frame_label_var.set(text)
                    except Exception:
                        pass
                    try:
                        root.after(100, _tick_update_frame_label)
                    except Exception:
                        pass

                def _tick_update_recording_label() -> None:
                    with self._lock:
                        rec = bool(self._recording)
                    text = "Recording in progress..." if rec else ""
                    try:
                        self._rec_label_var.set(text)
                    except Exception:
                        pass
                    try:
                        root.after(100, _tick_update_recording_label)
                    except Exception:
                        pass


                # Scale controls
                var = tk.DoubleVar(value=float(self._value))
                label = ttk.Label(root, text=f"Scale: {float(var.get()):.3f}", anchor="e")
                label.pack(fill="x", padx=8, pady=6)

                def _on_change(*_args: object) -> None:
                    val = float(var.get())
                    with self._lock:
                        self._value = val
                    label.configure(text=f"Scale: {val:.3f}")

                var.trace_add("write", _on_change)
                scale = ttk.Scale(root, from_=self._min, to=self._max, orient="horizontal", variable=var)
                scale.pack(fill="x", padx=8, pady=6)

                # Playback controls
                controls = ttk.Frame(root)
                controls.pack(fill="x", padx=8, pady=6)

                btn_text = tk.StringVar(value="Pause")

                def _toggle_play_pause() -> None:
                    with self._lock:
                        self._paused = not self._paused
                        btn_text.set("Play" if self._paused else "Pause")

                def _request_reset() -> None:
                    self._reset_event.set()

                def _request_prev() -> None:
                    try:
                        self._prev_event.set()
                    except Exception:
                        pass

                def _request_next() -> None:
                    try:
                        self._next_event.set()
                    except Exception:
                        pass

                play_pause_btn = ttk.Button(controls, textvariable=btn_text, command=_toggle_play_pause)
                play_pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
                reset_btn = ttk.Button(controls, text="Reset", command=_request_reset)
                reset_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))
                prev_btn = ttk.Button(controls, text="Prev", command=_request_prev)
                prev_btn.pack(side="left", expand=True, fill="x", padx=(8, 0))
                next_btn = ttk.Button(controls, text="Next", command=_request_next)
                next_btn.pack(side="left", expand=True, fill="x", padx=(8, 0))
                # Export button (signals main loop to bake/export animation)
                def _request_export() -> None:
                    try:
                        self._export_event.set()
                    except Exception:
                        pass
                export_btn = ttk.Button(controls, text="Export Animation", command=_request_export)
                export_btn.pack(side="left", expand=True, fill="x", padx=(8, 0))

                # Export current frame pose (single-frame JSON like crawl-pose.json)
                def _request_export_frame() -> None:
                    try:
                        self._export_frame_event.set()
                    except Exception:
                        pass
                export_frame_btn = ttk.Button(controls, text="Export Frame", command=_request_export_frame)
                export_frame_btn.pack(side="left", expand=True, fill="x", padx=(8, 0))

                # Start periodic UI updates
                _tick_update_frame_label()
                _tick_update_recording_label()

                self._ready.set()
                root.mainloop()
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._ready.wait(timeout=2.0)

        def get(self) -> float:
            with self._lock:
                return float(self._value)

        def is_paused(self) -> bool:
            with self._lock:
                return bool(self._paused)

        def consume_reset(self) -> bool:
            if self._reset_event.is_set():
                self._reset_event.clear()
                return True
            return False

        def consume_prev(self) -> bool:
            if self._prev_event.is_set():
                self._prev_event.clear()
                return True
            return False

        def consume_next(self) -> bool:
            if self._next_event.is_set():
                self._next_event.clear()
                return True
            return False

        def set_frame_index(self, idx: int) -> None:
            with self._lock:
                self._frame_idx_shared = int(max(0, idx))

        # Export trigger flag
        _export_event: threading.Event = field(default_factory=threading.Event)  # type: ignore[assignment]

        def consume_export(self) -> bool:
            try:
                if getattr(self, "_export_event", None) is None:
                    self._export_event = threading.Event()
                if self._export_event.is_set():
                    self._export_event.clear()
                    return True
            except Exception:
                pass
            return False

        def consume_export_frame(self) -> bool:
            try:
                if getattr(self, "_export_frame_event", None) is None:
                    self._export_frame_event = threading.Event()
                if self._export_frame_event.is_set():
                    self._export_frame_event.clear()
                    return True
            except Exception:
                pass
            return False

        def set_recording(self, flag: bool) -> None:
            with self._lock:
                self._recording = bool(flag)

        def set_paused(self, paused: bool) -> None:
            with self._lock:
                self._paused = bool(paused)

    # Base scale from computed scale or manual override; slider controls absolute value
    base_scale = float(BVH_SCALE) if BVH_SCALE is not None else (float(scale_used) if scale_used is not None else 1.0)
    # Slider range: narrow +/- 0.002 around base scale (e.g., default ~0.008)
    min_scale = max(1e-4, base_scale - 0.002)
    max_scale = base_scale + 0.002
    ui = _ControlUI(initial_value=base_scale, min_value=min_scale, max_value=max_scale, total_frames=num_frames)
    ui.start()

    # Compute initial anchor targets based on frame 0 and pelvis position
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_xy0 = np.array(data.xpos[pelvis_id][0:2], dtype=np.float64) if pelvis_id >= 0 else np.zeros(2)
    target_ground_z = 0.0

    def _compute_anchor_for_targets(scale: float, ref_pts_unscaled: np.ndarray, target_xy: np.ndarray, target_zmin: float) -> np.ndarray:
        scaled = ref_pts_unscaled * float(scale)
        a_xy = target_xy - scaled[0, 0:2]
        a_z = float(target_zmin) - float(scaled[:, 2].min())
        return np.array([a_xy[0], a_xy[1], a_z], dtype=np.float64)

    anchor_offset = _compute_anchor_for_targets(base_scale, frame0_pts_unscaled, pelvis_xy0, target_ground_z)
    prev_scale = float(base_scale)

    # Build BVH name -> index using exact names
    left_hand_idx = _find_bvh_index_by_name(frame0_names, BVH_LEFT_HAND_NAME)
    right_hand_idx = _find_bvh_index_by_name(frame0_names, BVH_RIGHT_HAND_NAME)
    left_foot_idx = _find_bvh_index_by_name(frame0_names, BVH_LEFT_FOOT_NAME)
    right_foot_idx = _find_bvh_index_by_name(frame0_names, BVH_RIGHT_FOOT_NAME)
    pelvis_idx = _find_bvh_index_by_name(frame0_names, BVH_PELVIS_NAME)
    left_shoulder_idx = _find_bvh_index_by_name(frame0_names, BVH_LEFT_SHOULDER_NAME)
    right_shoulder_idx = _find_bvh_index_by_name(frame0_names, BVH_RIGHT_SHOULDER_NAME)
    left_elbow_idx = _find_bvh_index_by_name(frame0_names, BVH_LEFT_ELBOW_NAME)
    right_elbow_idx = _find_bvh_index_by_name(frame0_names, BVH_RIGHT_ELBOW_NAME)
    left_knee_idx = _find_bvh_index_by_name(frame0_names, BVH_LEFT_KNEE_NAME)
    right_knee_idx = _find_bvh_index_by_name(frame0_names, BVH_RIGHT_KNEE_NAME)



    # IK viewer loop
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        try:
            # Ensure pelvis body id variable exists for camera tracking and later resets
            pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if pelvis_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(pelvis_bid)
                viewer.cam.fixedcamid = -1
        except Exception:
            pass

        # Initialize posture targets after any last forward
        mujoco.mj_forward(configuration.model, configuration.data)
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        pelvis_position_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        left_foot_orientation_task.set_target_from_configuration(configuration)
        right_foot_orientation_task.set_target_from_configuration(configuration)
        left_shoulder_task.set_target_from_configuration(configuration)
        right_shoulder_task.set_target_from_configuration(configuration)
        left_elbow_task.set_target_from_configuration(configuration)
        right_elbow_task.set_target_from_configuration(configuration)
        left_knee_task.set_target_from_configuration(configuration)
        right_knee_task.set_target_from_configuration(configuration)

        rate = RateLimiter(frequency=float(FPS), warn=False)
        # Initialize playback time from START_AT_FRAME
        start_frame_clamped = int(min(max(0, START_AT_FRAME), max(0, num_frames - 1)))
        anim_t = float(start_frame_clamped) * float(frame_time)

        def _export_animation(live_scale: float) -> str:
            """Bake the full BVH sequence into JSON using main-proc-anim schema.

            Returns the output file path.
            """
            # Build an offline configuration starting from current configuration/data
            offline_cfg = mink.Configuration(model)
            offline_cfg.data.qpos[:] = configuration.data.qpos[:]
            mujoco.mj_forward(offline_cfg.model, offline_cfg.data)

            # Initialize task targets from offline configuration
            posture_task.set_target_from_configuration(offline_cfg)
            pelvis_orientation_task.set_target_from_configuration(offline_cfg)
            pelvis_position_task.set_target_from_configuration(offline_cfg)
            torso_orientation_task.set_target_from_configuration(offline_cfg)
            left_foot_orientation_task.set_target_from_configuration(offline_cfg)
            right_foot_orientation_task.set_target_from_configuration(offline_cfg)
            left_shoulder_task.set_target_from_configuration(offline_cfg)
            right_shoulder_task.set_target_from_configuration(offline_cfg)
            left_elbow_task.set_target_from_configuration(offline_cfg)
            right_elbow_task.set_target_from_configuration(offline_cfg)
            left_knee_task.set_target_from_configuration(offline_cfg)
            right_knee_task.set_target_from_configuration(offline_cfg)

            d_off = offline_cfg.data

            # Compute anchor using current pelvis XY and frame0 points
            pelvis_bid_local = mujoco.mj_name2id(offline_cfg.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            pelvis_xy_now = np.array(d_off.xpos[pelvis_bid_local][0:2], dtype=np.float64) if pelvis_bid_local >= 0 else np.zeros(2)
            f0_pts_us, _, _, _ = compute_bvh_frame_world_poses(bvh_root, bvh_motion, 0)
            f0_pts_us = (ROT_BVH_TO_G1 @ f0_pts_us.T).T
            local_anchor = _compute_anchor_for_targets(live_scale, f0_pts_us, pelvis_xy_now, 0.0)

            # Site metadata
            site_count = int(offline_cfg.model.nsite)
            site_names: List[str] = []
            for sid in range(site_count):
                nm = mujoco.mj_id2name(offline_cfg.model, mujoco.mjtObj.mjOBJ_SITE, sid)
                site_names.append(str(nm) if nm is not None else f"site_{sid}")

            # Contact detection site ids
            def _site_id(m: mujoco.MjModel, name: str) -> int:
                return int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name))

            site_id_FL = _site_id(offline_cfg.model, "left_palm")
            site_id_FR = _site_id(offline_cfg.model, "right_palm")
            site_id_RL = _site_id(offline_cfg.model, "left_foot")
            site_id_RR = _site_id(offline_cfg.model, "right_foot")
            contact_threshold_m = 0.01

            frames: List[List[float]] = []
            vel_frames: List[List[float]] = []
            site_positions_per_frame: List[List[List[float]]] = []

            local_solver = SOLVER
            dt = float(frame_time)
            total_steps = int(num_frames)

            for k in range(total_steps):
                f_idx = int(k)
                frame_pts_us, frame_rots_us, _, _ = compute_bvh_frame_world_poses(bvh_root, bvh_motion, f_idx)
                frame_pts_us = (ROT_BVH_TO_G1 @ frame_pts_us.T).T
                frame_rots_g1 = ROT_BVH_TO_G1 @ frame_rots_us @ ROT_BVH_TO_G1.T
                frame_pts_scaled = frame_pts_us * float(live_scale)
                bvh_world = frame_pts_scaled + local_anchor

                # Drive mocap targets
                if CLAMP_HANDS_AND_FEET_TO_FLOOR:
                    d_off.mocap_pos[right_palm_mid][0:3] = _clamp_floor_z(bvh_world[right_hand_idx], FLOOR_Z)
                    d_off.mocap_pos[left_palm_mid][0:3] = _clamp_floor_z(bvh_world[left_hand_idx], FLOOR_Z)
                    d_off.mocap_pos[left_foot_mid][0:3] = _clamp_floor_z(bvh_world[left_foot_idx], FLOOR_Z)
                    d_off.mocap_pos[right_foot_mid][0:3] = _clamp_floor_z(bvh_world[right_foot_idx], FLOOR_Z)
                else:
                    d_off.mocap_pos[right_palm_mid][0:3] = bvh_world[right_hand_idx]
                    d_off.mocap_pos[left_palm_mid][0:3] = bvh_world[left_hand_idx]
                    d_off.mocap_pos[left_foot_mid][0:3] = bvh_world[left_foot_idx]
                    d_off.mocap_pos[right_foot_mid][0:3] = bvh_world[right_foot_idx]
                d_off.mocap_quat[left_foot_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_foot_idx])
                d_off.mocap_quat[right_foot_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_foot_idx])
                d_off.mocap_pos[pelvis_mid][0:3] = bvh_world[pelvis_idx]
                d_off.mocap_quat[pelvis_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[pelvis_idx])
                d_off.mocap_pos[left_shoulder_mid][0:3] = bvh_world[left_shoulder_idx]
                d_off.mocap_quat[left_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_shoulder_idx])
                d_off.mocap_pos[right_shoulder_mid][0:3] = bvh_world[right_shoulder_idx]
                d_off.mocap_quat[right_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_shoulder_idx])
                d_off.mocap_pos[left_elbow_mid][0:3] = bvh_world[left_elbow_idx]
                d_off.mocap_quat[left_elbow_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_elbow_idx])
                d_off.mocap_pos[right_elbow_mid][0:3] = bvh_world[right_elbow_idx]
                d_off.mocap_quat[right_elbow_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_elbow_idx])
                d_off.mocap_pos[left_knee_mid][0:3] = bvh_world[left_knee_idx]
                d_off.mocap_quat[left_knee_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_knee_idx])
                d_off.mocap_pos[right_knee_mid][0:3] = bvh_world[right_knee_idx]
                d_off.mocap_quat[right_knee_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_knee_idx])

                # Update IK targets and solve

                right_hand_task.set_target(mink.SE3.from_mocap_id(d_off, right_palm_mid))
                left_hand_task.set_target(mink.SE3.from_mocap_id(d_off, left_palm_mid))
                left_foot_task.set_target(mink.SE3.from_mocap_id(d_off, left_foot_mid))
                right_foot_task.set_target(mink.SE3.from_mocap_id(d_off, right_foot_mid))
                left_shoulder_task.set_target(mink.SE3.from_mocap_id(d_off, left_shoulder_mid))
                right_shoulder_task.set_target(mink.SE3.from_mocap_id(d_off, right_shoulder_mid))
                left_elbow_task.set_target(mink.SE3.from_mocap_id(d_off, left_elbow_mid))
                right_elbow_task.set_target(mink.SE3.from_mocap_id(d_off, right_elbow_mid))
                left_knee_task.set_target(mink.SE3.from_mocap_id(d_off, left_knee_mid))
                right_knee_task.set_target(mink.SE3.from_mocap_id(d_off, right_knee_mid))

                # Ensure sensor positions are up-to-date for collision avoidance
                mujoco.mj_fwdPosition(offline_cfg.model, d_off)
                mujoco.mj_sensorPos(offline_cfg.model, d_off)
                vel_local = mink.solve_ik(offline_cfg, tasks, dt, local_solver, 1e-1, limits=limits)
                offline_cfg.integrate_inplace(vel_local, dt)

                frames.append([float(x) for x in d_off.qpos])
                vel_frames.append([float(x) for x in vel_local])

                # Record site positions
                site_xyz = []
                for sid in range(site_count):
                    site_xyz.append([
                        float(d_off.site_xpos[sid][0]),
                        float(d_off.site_xpos[sid][1]),
                        float(d_off.site_xpos[sid][2]),
                    ])
                site_positions_per_frame.append(site_xyz)

            # Recenter XY so the initial base XY becomes 0 for all frames
            free_qpos_addr = None
            for j in range(offline_cfg.model.njnt):
                if int(offline_cfg.model.jnt_type[j]) == 0:
                    free_qpos_addr = int(offline_cfg.model.jnt_qposadr[j])
                    break
            if free_qpos_addr is not None and len(frames) > 0:
                x0 = float(frames[0][free_qpos_addr + 0])
                y0 = float(frames[0][free_qpos_addr + 1])
                if x0 != 0.0 or y0 != 0.0:
                    for frame_vals in frames:
                        frame_vals[free_qpos_addr + 0] = float(frame_vals[free_qpos_addr + 0]) - x0
                        frame_vals[free_qpos_addr + 1] = float(frame_vals[free_qpos_addr + 1]) - y0
                    for site_frame in site_positions_per_frame:
                        for s in site_frame:
                            s[0] = float(s[0]) - x0
                            s[1] = float(s[1]) - y0

            # Build metadata similar to main-proc-anim.py
            base_meta = None
            free_qpos_addr2 = None
            for j in range(model.njnt):
                if int(model.jnt_type[j]) == 0:
                    free_qpos_addr2 = int(model.jnt_qposadr[j])
                    break
            if free_qpos_addr2 is not None:
                base_meta = {
                    "pos_indices": [free_qpos_addr2 + i for i in range(3)],
                    "quat_indices": [free_qpos_addr2 + 3 + i for i in range(4)],
                }

            joints_meta = []
            qpos_labels = [f"qpos[{i}]" for i in range(model.nq)]

            def _set_label(i: int, name: str) -> None:
                if 0 <= i < len(qpos_labels):
                    qpos_labels[i] = name

            for j in range(model.njnt):
                jtype = int(model.jnt_type[j])
                qadr = int(model.jnt_qposadr[j])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
                if jtype == 0:
                    jtypestr = "free"
                    qdim = 7
                elif jtype == 1:
                    jtypestr = "ball"
                    qdim = 4
                    _set_label(qadr + 0, f"ball:{name}:w")
                    _set_label(qadr + 1, f"ball:{name}:x")
                    _set_label(qadr + 2, f"ball:{name}:y")
                    _set_label(qadr + 3, f"ball:{name}:z")
                elif jtype == 2:
                    jtypestr = "slide"
                    qdim = 1
                    _set_label(qadr + 0, f"joint:{name}")
                else:
                    jtypestr = "hinge"
                    qdim = 1
                    _set_label(qadr + 0, f"joint:{name}")
                joints_meta.append({
                    "name": str(name),
                    "type": jtypestr,
                    "qposadr": int(qadr),
                    "qposdim": int(qdim),
                })

            fps = 1.0 / float(dt)
            serial = {
                "schema": "gait_animation.v1",
                "model_xml": str(xml_path),
                "dt": float(dt),
                "fps": float(fps),
                "nsite": int(model.nsite),
                "nq": int(model.nq),
                "nv": int(model.nv),
                "frames": frames,
                "vel_frames": vel_frames,
                "site_positions": site_positions_per_frame,
                "contact_flags": [
                    [
                        1 if float(sp[site_id_FL][2]) <= contact_threshold_m else 0,
                        1 if float(sp[site_id_FR][2]) <= contact_threshold_m else 0,
                        1 if float(sp[site_id_RL][2]) <= contact_threshold_m else 0,
                        1 if float(sp[site_id_RR][2]) <= contact_threshold_m else 0,
                    ] for sp in site_positions_per_frame
                ],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "animation_type": "bvh_drive",
                    "base": base_meta,
                    "joints": joints_meta,
                    "qpos_labels": qpos_labels,
                    "sites": {
                        "names": site_names,
                        "indices": list(range(len(site_names))),
                        "by_name": {name: idx for idx, name in enumerate(site_names)},
                    },
                    "contact": {
                        "order": ["FL", "FR", "RL", "RR"],
                        "threshold_m": float(contact_threshold_m),
                    },
                    "has_root_motion": True,
                    "bvh": {
                        "path": str(bvh_path),
                        "scale": float(live_scale),
                        "frame_time": float(frame_time),
                        "num_frames": int(num_frames),
                    },
                },
            }

            out_dir = os.path.join(BASE_DIR, "output")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"animation_mocap_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serial, f)
            return out_path

        def _reinit_to_frame0(live_scale: float) -> None:
            nonlocal anim_t, anchor_offset
            # Restart animation time and re-anchor using frame 0 and current pelvis XY
            anim_t = 0.0
            pelvis_xy_now = np.array(data.xpos[pelvis_bid][0:2], dtype=np.float64) if pelvis_bid >= 0 else np.zeros(2)
            anchor_offset = _compute_anchor_for_targets(live_scale, frame0_pts_unscaled, pelvis_xy_now, 0.0)
            # Reinitialize IK task targets
            mujoco.mj_forward(configuration.model, configuration.data)
            posture_task.set_target_from_configuration(configuration)
            pelvis_orientation_task.set_target_from_configuration(configuration)
            pelvis_position_task.set_target_from_configuration(configuration)
            torso_orientation_task.set_target_from_configuration(configuration)
            left_foot_orientation_task.set_target_from_configuration(configuration)
            right_foot_orientation_task.set_target_from_configuration(configuration)
            left_shoulder_task.set_target_from_configuration(configuration)
            right_shoulder_task.set_target_from_configuration(configuration)
            left_elbow_task.set_target_from_configuration(configuration)
            right_elbow_task.set_target_from_configuration(configuration)
            left_knee_task.set_target_from_configuration(configuration)
            right_knee_task.set_target_from_configuration(configuration)
            # Reset mocap to frame 0 world poses
            f0_pts_us, f0_rots_us, _, _ = compute_bvh_frame_world_poses(bvh_root, bvh_motion, 0)
            f0_pts_us = (ROT_BVH_TO_G1 @ f0_pts_us.T).T
            f0_rots_g1 = ROT_BVH_TO_G1 @ f0_rots_us @ ROT_BVH_TO_G1.T
            f0_world = f0_pts_us * float(live_scale) + anchor_offset
            # Clamp hands/feet to floor if enabled
            if CLAMP_HANDS_AND_FEET_TO_FLOOR:
                data.mocap_pos[right_palm_mid][0:3] = _clamp_floor_z(f0_world[right_hand_idx], FLOOR_Z)
                data.mocap_pos[left_palm_mid][0:3] = _clamp_floor_z(f0_world[left_hand_idx], FLOOR_Z)
                data.mocap_pos[left_foot_mid][0:3] = _clamp_floor_z(f0_world[left_foot_idx], FLOOR_Z)
                data.mocap_pos[right_foot_mid][0:3] = _clamp_floor_z(f0_world[right_foot_idx], FLOOR_Z)
            else:
                data.mocap_pos[right_palm_mid][0:3] = f0_world[right_hand_idx]
                data.mocap_pos[left_palm_mid][0:3] = f0_world[left_hand_idx]
                data.mocap_pos[left_foot_mid][0:3] = f0_world[left_foot_idx]
                data.mocap_pos[right_foot_mid][0:3] = f0_world[right_foot_idx]
            data.mocap_quat[left_foot_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[left_foot_idx])
            data.mocap_quat[right_foot_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[right_foot_idx])
            data.mocap_pos[pelvis_mid][0:3] = f0_world[pelvis_idx]
            data.mocap_quat[pelvis_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[pelvis_idx])
            data.mocap_pos[left_shoulder_mid][0:3] = f0_world[left_shoulder_idx]
            data.mocap_quat[left_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[left_shoulder_idx])
            data.mocap_pos[right_shoulder_mid][0:3] = f0_world[right_shoulder_idx]
            data.mocap_quat[right_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[right_shoulder_idx])
            data.mocap_pos[left_elbow_mid][0:3] = f0_world[left_elbow_idx]
            data.mocap_quat[left_elbow_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[left_elbow_idx])
            data.mocap_pos[right_elbow_mid][0:3] = f0_world[right_elbow_idx]
            data.mocap_quat[right_elbow_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[right_elbow_idx])
            data.mocap_pos[left_knee_mid][0:3] = f0_world[left_knee_idx]
            data.mocap_quat[left_knee_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[left_knee_idx])
            data.mocap_pos[right_knee_mid][0:3] = f0_world[right_knee_idx]
            data.mocap_quat[right_knee_mid][0:4] = _rotmat_to_quat_wxyz(f0_rots_g1[right_knee_idx])

        def _draw_bvh_overlay(scene: mujoco.MJVSCENE, pts: np.ndarray, edges: List[Tuple[int, int]], rgba: np.ndarray) -> None:
            scene.ngeom = 0
            # Reserve one slot for CoM marker
            max_for_edges = int(scene.maxgeom) - 1 if int(scene.maxgeom) > 0 else 0
            for i, j in edges:
                if int(scene.ngeom) >= int(max_for_edges):
                    break
                a3 = np.asarray(pts[i], dtype=np.float64)
                b3 = np.asarray(pts[j], dtype=np.float64)
                g = scene.geoms[int(scene.ngeom)]
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, float(LINE_RADIUS), a3, b3)
                try:
                    g.rgba[0] = 0.2; g.rgba[1] = 0.6; g.rgba[2] = 1.0; g.rgba[3] = 1.0
                except Exception:
                    pass
                scene.ngeom += 1

        def _compute_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
            """Return whole-model Center of Mass in world coordinates using subtree_com[0]."""
            return np.array(data.subtree_com[0], dtype=np.float64)

        def _draw_com(scene: mujoco.MJVSCENE, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
            if int(scene.ngeom) >= int(scene.maxgeom):
                return
            g = scene.geoms[int(scene.ngeom)]
            size = np.array([float(radius), 0.0, 0.0], dtype=np.float64)
            p = np.asarray(pos, dtype=np.float64)
            mat = np.eye(3, dtype=np.float64).reshape(9)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, p, mat, rgba.astype(np.float32))
            scene.ngeom += 1

        # Live recording state
        recording = False
        last_recorded_frame = -1
        rec_frames: List[List[float]] = []
        rec_vel_frames: List[List[float]] = []
        rec_site_positions: List[List[List[float]]] = []
        rec_site_names: List[str] = []
        site_id_FL = -1
        site_id_FR = -1
        site_id_RL = -1
        site_id_RR = -1
        contact_threshold_m = 0.01
        # 200 Hz recording helpers
        rec_initial_f = -1
        rec_seen_wrap = False
        prev_f = -1

        def _export_current_frame_pose() -> str:
            """Export the current robot pose (single frame) to output/pose_*.json like crawl-pose.json."""
            # Base RPY from free joint quaternion if present
            free_qpos_addr = None
            for j in range(model.njnt):
                if int(model.jnt_type[j]) == 0:
                    free_qpos_addr = int(model.jnt_qposadr[j])
                    break
            if free_qpos_addr is not None:
                qw = float(data.qpos[free_qpos_addr + 3])
                qx = float(data.qpos[free_qpos_addr + 4])
                qy = float(data.qpos[free_qpos_addr + 5])
                qz = float(data.qpos[free_qpos_addr + 6])
                base_rpy = list(_quat_wxyz_to_rpy(np.array([qw, qx, qy, qz], dtype=np.float64)))
            else:
                base_rpy = [0.0, 0.0, 0.0]

            # Scalar joints map
            joints_map: Dict[str, float] = {}
            for j in range(model.njnt):
                jtype = int(model.jnt_type[j])
                if jtype == 0:
                    continue
                qadr = int(model.jnt_qposadr[j])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
                if jtype == 2 or jtype == 3 or jtype >= 2:
                    # slide (2) or hinge (3); for other 1-dim joints treat similarly
                    joints_map[str(name)] = float(data.qpos[qadr + 0])
                else:
                    # Skip non-scalar joints (e.g., ball)
                    pass

            serial = {
                "poses": [
                    {
                        "base_rpy": [float(base_rpy[0]), float(base_rpy[1]), float(base_rpy[2])],
                        "joints": joints_map,
                    }
                ]
            }
            out_dir = os.path.join(BASE_DIR, "output")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"pose_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f_out:
                json.dump(serial, f_out, indent=2)
            print(f"Exported frame pose to: {out_path}")
            try:
                z = shutil.which("zenity")
                if z:
                    threading.Thread(
                        target=lambda: subprocess.run(
                            [z, "--info", "--title=Export Frame", f"--text=Saved to {os.path.basename(out_path)}"],
                            check=False,
                        ),
                        daemon=True,
                    ).start()
            except Exception:
                pass
            return out_path

        while viewer.is_running():
            # Live controls: scale change, pause, and reset
            live_scale = float(ui.get())
            if abs(live_scale - prev_scale) > 1e-6:
                _reinit_to_frame0(live_scale)
                prev_scale = float(live_scale)
            if ui.consume_reset():
                _reinit_to_frame0(prev_scale)

            # Compute current frame index before stepping
            f_curr = int((anim_t / frame_time) % max(1, num_frames))
            # Handle prev/next frame stepping
            if ui.consume_prev():
                nf = (f_curr - 1) % max(1, num_frames)
                anim_t = float(nf) * float(frame_time)
            if ui.consume_next():
                nf = (f_curr + 1) % max(1, num_frames)
                anim_t = float(nf) * float(frame_time)

            # Handle export trigger
            if ui.consume_export():
                if not recording:
                    ui.set_recording(True)
                    _reinit_to_frame0(prev_scale)
                    recording = True
                    last_recorded_frame = -1
                    rec_frames = []
                    rec_vel_frames = []
                    rec_site_positions = []
                    # Prepare site metadata for recording
                    rec_site_names = []
                    for sid in range(model.nsite):
                        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
                        rec_site_names.append(str(nm) if nm is not None else f"site_{sid}")
                    site_id_FL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
                    site_id_FR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
                    site_id_RL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
                    site_id_RR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
                    # reset 200 Hz helpers
                    rec_initial_f = -1
                    rec_seen_wrap = False
                    prev_f = -1
                else:
                    pass

            # Handle export current frame pose trigger
            if ui.consume_export_frame():
                _export_current_frame_pose()

            if not ui.is_paused():
                anim_t += rate.dt

            f = int((anim_t / frame_time) % max(1, num_frames))
            ui.set_frame_index(f)

            # Compute BVH world poses for current frame after any reset
            frame_pts_us, frame_rots_us, _, _ = compute_bvh_frame_world_poses(bvh_root, bvh_motion, f)
            frame_pts_us = (ROT_BVH_TO_G1 @ frame_pts_us.T).T
            frame_rots_g1 = ROT_BVH_TO_G1 @ frame_rots_us @ ROT_BVH_TO_G1.T
            frame_pts_scaled = frame_pts_us * float(live_scale)
            bvh_world = frame_pts_scaled + anchor_offset

            # Draw BVH overlay in viewer
            _draw_bvh_overlay(viewer.user_scn, bvh_world, bvh_edges, np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32))

            # Drive mocap targets (all required and asserted present)
            if CLAMP_HANDS_AND_FEET_TO_FLOOR:
                data.mocap_pos[right_palm_mid][0:3] = _clamp_floor_z(bvh_world[right_hand_idx], FLOOR_Z)
                data.mocap_pos[left_palm_mid][0:3] = _clamp_floor_z(bvh_world[left_hand_idx], FLOOR_Z)
                data.mocap_pos[left_foot_mid][0:3] = _clamp_floor_z(bvh_world[left_foot_idx], FLOOR_Z)
                data.mocap_pos[right_foot_mid][0:3] = _clamp_floor_z(bvh_world[right_foot_idx], FLOOR_Z)
            else:
                data.mocap_pos[right_palm_mid][0:3] = bvh_world[right_hand_idx]
                data.mocap_pos[left_palm_mid][0:3] = bvh_world[left_hand_idx]
                data.mocap_pos[left_foot_mid][0:3] = bvh_world[left_foot_idx]
                data.mocap_pos[right_foot_mid][0:3] = bvh_world[right_foot_idx]
            data.mocap_quat[left_foot_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_foot_idx])
            data.mocap_quat[right_foot_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_foot_idx])
            data.mocap_pos[pelvis_mid][0:3] = bvh_world[pelvis_idx]
            data.mocap_quat[pelvis_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[pelvis_idx])
            data.mocap_pos[left_shoulder_mid][0:3] = bvh_world[left_shoulder_idx]
            data.mocap_quat[left_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_shoulder_idx])
            data.mocap_pos[right_shoulder_mid][0:3] = bvh_world[right_shoulder_idx]
            data.mocap_quat[right_shoulder_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_shoulder_idx])
            data.mocap_pos[left_elbow_mid][0:3] = bvh_world[left_elbow_idx]
            data.mocap_quat[left_elbow_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_elbow_idx])
            data.mocap_pos[right_elbow_mid][0:3] = bvh_world[right_elbow_idx]
            data.mocap_quat[right_elbow_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_elbow_idx])
            data.mocap_pos[left_knee_mid][0:3] = bvh_world[left_knee_idx]
            data.mocap_quat[left_knee_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[left_knee_idx])
            data.mocap_pos[right_knee_mid][0:3] = bvh_world[right_knee_idx]
            data.mocap_quat[right_knee_mid][0:4] = _rotmat_to_quat_wxyz(frame_rots_g1[right_knee_idx])

            # Update task targets from required mocap bodies

            right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))
            left_hand_task.set_target(mink.SE3.from_mocap_id(data, left_palm_mid))
            left_foot_task.set_target(mink.SE3.from_mocap_id(data, left_foot_mid))
            right_foot_task.set_target(mink.SE3.from_mocap_id(data, right_foot_mid))
            left_shoulder_task.set_target(mink.SE3.from_mocap_id(data, left_shoulder_mid))
            right_shoulder_task.set_target(mink.SE3.from_mocap_id(data, right_shoulder_mid))
            left_elbow_task.set_target(mink.SE3.from_mocap_id(data, left_elbow_mid))
            right_elbow_task.set_target(mink.SE3.from_mocap_id(data, right_elbow_mid))
            left_knee_task.set_target(mink.SE3.from_mocap_id(data, left_knee_mid))
            right_knee_task.set_target(mink.SE3.from_mocap_id(data, right_knee_mid))
            pelvis_orientation_task.set_target(mink.SE3.from_mocap_id(data, pelvis_mid))
            pelvis_position_task.set_target(mink.SE3.from_mocap_id(data, pelvis_mid))

            # Solve IK step and integrate (update sensors first for collision avoidance)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            vel = mink.solve_ik(configuration, tasks, rate.dt, SOLVER, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            # Refresh derived positions and append CoM marker to the user scene
            mujoco.mj_fwdPosition(model, data)
            # Use global CoM from subtree_com[0] (whole model)
            com_w = _compute_com(model, data)
            _draw_com(
                viewer.user_scn,
                com_w,
                float(COM_SPHERE_RADIUS),
                np.array(COM_RGBA, dtype=np.float32),
            )

            # Live recording capture at 200 Hz for exactly one BVH cycle
            if recording:
                # Initialize initial frame on first tick after start
                if rec_initial_f == -1:
                    rec_initial_f = int(f)
                # Detect wrap (e.g., N-1 -> 0)
                if prev_f != -1 and int(f) < int(prev_f):
                    rec_seen_wrap = True

                # Append current integrated state every tick (200 Hz)
                rec_frames.append([float(x) for x in data.qpos])
                rec_vel_frames.append([float(x) for x in vel])
                site_xyz = []
                for sid in range(model.nsite):
                    site_xyz.append([
                        float(data.site_xpos[sid][0]),
                        float(data.site_xpos[sid][1]),
                        float(data.site_xpos[sid][2]),
                    ])
                rec_site_positions.append(site_xyz)

                # If we've wrapped and returned to initial frame index, stop and write
                if rec_seen_wrap and int(f) == int(rec_initial_f):
                    # Recenter XY so initial base XY becomes 0
                    free_qpos_addr3 = None
                    for j in range(model.njnt):
                        if int(model.jnt_type[j]) == 0:
                            free_qpos_addr3 = int(model.jnt_qposadr[j])
                            break
                    if free_qpos_addr3 is not None and len(rec_frames) > 0:
                        x0 = float(rec_frames[0][free_qpos_addr3 + 0])
                        y0 = float(rec_frames[0][free_qpos_addr3 + 1])
                        if x0 != 0.0 or y0 != 0.0:
                            for frame_vals in rec_frames:
                                frame_vals[free_qpos_addr3 + 0] = float(frame_vals[free_qpos_addr3 + 0]) - x0
                                frame_vals[free_qpos_addr3 + 1] = float(frame_vals[free_qpos_addr3 + 1]) - y0
                            for site_frame in rec_site_positions:
                                for s in site_frame:
                                    s[0] = float(s[0]) - x0
                                    s[1] = float(s[1]) - y0

                    # Build labels and joint metadata
                    base_meta2 = None
                    free_qpos_addr4 = None
                    for j in range(model.njnt):
                        if int(model.jnt_type[j]) == 0:
                            free_qpos_addr4 = int(model.jnt_qposadr[j])
                            break
                    if free_qpos_addr4 is not None:
                        base_meta2 = {
                            "pos_indices": [free_qpos_addr4 + i for i in range(3)],
                            "quat_indices": [free_qpos_addr4 + 3 + i for i in range(4)],
                        }

                    qpos_labels2 = [f"qpos[{i}]" for i in range(model.nq)]
                    def _set_label2(i: int, name: str) -> None:
                        if 0 <= i < len(qpos_labels2):
                            qpos_labels2[i] = name
                    joints_meta2 = []
                    for j in range(model.njnt):
                        jtype = int(model.jnt_type[j])
                        qadr = int(model.jnt_qposadr[j])
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
                        if jtype == 0:
                            jtypestr = "free"; qdim = 7
                        elif jtype == 1:
                            jtypestr = "ball"; qdim = 4
                            _set_label2(qadr + 0, f"ball:{name}:w")
                            _set_label2(qadr + 1, f"ball:{name}:x")
                            _set_label2(qadr + 2, f"ball:{name}:y")
                            _set_label2(qadr + 3, f"ball:{name}:z")
                        elif jtype == 2:
                            jtypestr = "slide"; qdim = 1
                            _set_label2(qadr + 0, f"joint:{name}")
                        else:
                            jtypestr = "hinge"; qdim = 1
                            _set_label2(qadr + 0, f"joint:{name}")
                        joints_meta2.append({
                            "name": str(name),
                            "type": jtypestr,
                            "qposadr": int(qadr),
                            "qposdim": int(qdim),
                        })

                    # Build serial and save (dt = rate.dt for 200 Hz capture)
                    fps2 = 1.0 / float(rate.dt)
                    serial2 = {
                        "schema": "gait_animation.v1",
                        "model_xml": str(xml_path),
                        "dt": float(rate.dt),
                        "fps": float(fps2),
                        "nsite": int(model.nsite),
                        "nq": int(model.nq),
                        "nv": int(model.nv),
                        "frames": rec_frames,
                        "vel_frames": rec_vel_frames,
                        "site_positions": rec_site_positions,
                        "contact_flags": [
                            [
                                1 if float(sp[site_id_FL][2]) <= contact_threshold_m else 0,
                                1 if float(sp[site_id_FR][2]) <= contact_threshold_m else 0,
                                1 if float(sp[site_id_RL][2]) <= contact_threshold_m else 0,
                                1 if float(sp[site_id_RR][2]) <= contact_threshold_m else 0,
                            ] for sp in rec_site_positions
                        ],
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "animation_type": "bvh_drive",
                            "base": base_meta2,
                            "joints": joints_meta2,
                            "qpos_labels": qpos_labels2,
                            "sites": {
                                "names": rec_site_names,
                                "indices": list(range(len(rec_site_names))),
                                "by_name": {name: idx for idx, name in enumerate(rec_site_names)},
                            },
                            "contact": {
                                "order": ["FL", "FR", "RL", "RR"],
                                "threshold_m": float(contact_threshold_m),
                            },
                            "has_root_motion": True,
                            "capture_hz": float(fps2),
                            "bvh": {
                                "path": str(bvh_path),
                                "scale": float(prev_scale),
                                "frame_time": float(frame_time),
                                "num_frames": int(num_frames),
                            },
                        },
                    }

                    out_dir = os.path.join(BASE_DIR, "output")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(out_dir, f"animation_mocap_{ts}.json")
                    with open(out_path, "w", encoding="utf-8") as f_out:
                        json.dump(serial2, f_out)

                    print(f"Exported animation to: {out_path}")
                    try:
                        z = shutil.which("zenity")
                        if z:
                            # Launch confirmation non-blocking so main loop continues
                            threading.Thread(
                                target=lambda: subprocess.run(
                                    [z, "--info", "--title=Export Animation", f"--text=Saved to {os.path.basename(out_path)}"],
                                    check=False,
                                ),
                                daemon=True,
                            ).start()
                    except Exception:
                        pass

                    ui.set_recording(False)
                    ui.set_paused(False)
                    recording = False

            # Track previous BVH frame each tick
            prev_f = int(f)

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()


