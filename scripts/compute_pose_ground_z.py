from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import mujoco
import numpy as np


# ----------------------------- Static configuration -----------------------------
# Follow workspace rule: no CLI args; keep tunables here.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POSE_PATH = os.path.join(BASE_DIR, "crawl-pose.json")
XML_PATH = os.path.join(BASE_DIR, "g1_description", "scene_g1_targets.xml")

# Which sites define ground contact/clearance for this pose
CONTACT_SITE_NAMES = ["left_palm", "right_palm", "left_foot", "right_foot"]

# Clearance margin added to lift the lowest contact above z=0
GROUND_CLEARANCE_MARGIN = 0.0  # meters


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw (radians) to quaternion (w,x,y,z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= max(1e-12, float(np.linalg.norm(q)))
    return q


def _resolve_free_joint_qpos_addr(model: mujoco.MjModel) -> Optional[int]:
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            return int(model.jnt_qposadr[j])
    return None


def _site_ids(model: mujoco.MjModel, names: List[str]) -> List[int]:
    out: List[int] = []
    for nm in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid == -1:
            raise RuntimeError(f"Required site '{nm}' not found in model")
        out.append(int(sid))
    return out


def _apply_pose_to_qpos(model: mujoco.MjModel, data: mujoco.MjData, pose: Dict[str, object]) -> None:
    # Base orientation
    base_rpy = pose.get("base_rpy")
    if not isinstance(base_rpy, list) or len(base_rpy) != 3:
        raise ValueError("Pose missing 'base_rpy' [roll, pitch, yaw]")
    roll = float(base_rpy[0])
    pitch = float(base_rpy[1])
    yaw = float(base_rpy[2])
    q = _rpy_to_quat_wxyz(roll, pitch, yaw)

    # Free joint position and orientation (x,y,z, qw,qx,qy,qz)
    free_adr = _resolve_free_joint_qpos_addr(model)
    if free_adr is None:
        raise RuntimeError("Model has no free joint; cannot set base pose")

    data.qpos[free_adr + 0] = 0.0
    data.qpos[free_adr + 1] = 0.0
    data.qpos[free_adr + 2] = 0.0  # z will be solved
    data.qpos[free_adr + 3] = float(q[0])
    data.qpos[free_adr + 4] = float(q[1])
    data.qpos[free_adr + 5] = float(q[2])
    data.qpos[free_adr + 6] = float(q[3])

    # Scalar joints
    joints = pose.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("Pose missing 'joints' map")

    # Build name -> (qposadr, qposdim)
    name_to_q = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        qadr = int(model.jnt_qposadr[j])
        jtype = int(model.jnt_type[j])
        if jtype == 0:
            continue  # free handled above
        qdim = 4 if jtype == 1 else 1
        name_to_q[name] = (qadr, qdim)

    for nm, val in joints.items():
        if nm not in name_to_q:
            raise RuntimeError(f"Joint '{nm}' not found in model")
        qadr, qdim = name_to_q[nm]
        if qdim != 1:
            # Expect only scalar joint values in the pose json
            continue
        data.qpos[qadr] = float(val)  # type: ignore[arg-type]


def compute_min_base_z_for_contacts(model: mujoco.MjModel, data: mujoco.MjData, site_ids: List[int], margin: float) -> float:
    mujoco.mj_forward(model, data)
    min_site_z = float(min(data.site_xpos[sid][2] for sid in site_ids))
    required_delta = -min_site_z + float(margin)
    free_adr = _resolve_free_joint_qpos_addr(model)
    if free_adr is None:
        raise RuntimeError("Model has no free joint; cannot adjust base z")
    return float(data.qpos[free_adr + 2] + required_delta)


def main() -> None:
    pose_doc = json.load(open(POSE_PATH, "r", encoding="utf-8"))
    poses = pose_doc.get("poses")
    if not isinstance(poses, list) or len(poses) == 0 or not isinstance(poses[0], dict):
        raise ValueError("Pose file must contain poses[0]")
    pose0 = poses[0]

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    _apply_pose_to_qpos(model, data, pose0)
    sids = _site_ids(model, CONTACT_SITE_NAMES)
    z = compute_min_base_z_for_contacts(model, data, sids, GROUND_CLEARANCE_MARGIN)

    print("Pose file:", POSE_PATH)
    print("Model xml:", XML_PATH)
    print("Contact sites:", CONTACT_SITE_NAMES)
    print("Recommended base_pos.z (clearance margin=%.3f m): %.6f m" % (GROUND_CLEARANCE_MARGIN, z))


if __name__ == "__main__":
    main()








