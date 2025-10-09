#!/usr/bin/env python3
"""Debug script to visualize lowest point calculation for a specific pose."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

# Configuration
POSES_FILE = "output/animation_mocap_rc0_poses_sorted.json"
POSE_INDEX = 0  # 0 for first pose (lowest Z), -1 for last pose (highest Z)
XML_PATH = "g1_description/scene_g1_targets.xml"


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict) -> None:
    """Apply pose to MuJoCo data."""
    # Set base position and orientation
    base_pos = pose["base_pos"]
    base_rpy = pose["base_rpy"]
    quat = rpy_to_quat_wxyz(base_rpy[0], base_rpy[1], base_rpy[2])
    
    # Find free joint
    free_joint_idx = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:  # free joint
            free_joint_idx = j
            break
    
    if free_joint_idx is not None:
        qpos_adr = int(model.jnt_qposadr[free_joint_idx])
        data.qpos[qpos_adr:qpos_adr+3] = base_pos
        data.qpos[qpos_adr+3:qpos_adr+7] = quat
    
    # Set joint positions
    for joint_name, value in pose["joints"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_adr = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_adr] = value


def analyze_lowest_points(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Analyze and print all low points in the robot."""
    mujoco.mj_forward(model, data)
    
    print("\n=== Analyzing lowest points ===\n")
    
    # Track all z values
    z_points = []
    
    # Check sites (hands, feet, etc.)
    print("Site positions:")
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or f"site_{site_id}"
        site_z = data.site_xpos[site_id][2]
        z_points.append(('site', site_name, site_z))
        print(f"  {site_name:30s}: z = {site_z:.4f}")
    
    # Check geometries
    print("\nGeometry positions (with size estimation):")
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        geom_pos = data.geom_xpos[geom_id]
        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id]
        
        # Estimate lowest point based on geom type
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            lowest = geom_pos[2] - geom_size[0]
            type_str = "sphere"
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            lowest = geom_pos[2] - geom_size[1]
            type_str = "capsule"
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            lowest = geom_pos[2] - geom_size[2]
            type_str = "box"
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            lowest = geom_pos[2]  # Mesh: position only (no size info)
            type_str = "mesh"
        else:
            lowest = geom_pos[2]
            type_str = f"type_{geom_type}"
        
        z_points.append(('geom', geom_name, lowest))
        if lowest < 0.2:  # Only print low geometries
            print(f"  {geom_name:30s} ({type_str:8s}): z = {geom_pos[2]:.4f}, lowest = {lowest:.4f}")
    
    # Find absolute minimum
    z_points.sort(key=lambda x: x[2])
    
    print(f"\n=== Bottom 10 lowest points ===")
    for i, (typ, name, z) in enumerate(z_points[:10]):
        print(f"{i+1:2d}. {typ:5s} {name:30s}: z = {z:.4f}")
    
    min_z = z_points[0][2]
    print(f"\nAbsolute minimum Z: {min_z:.4f}")
    print(f"Base Z in pose: {data.qpos[2]:.4f}")
    print(f"Difference (base - min): {data.qpos[2] - min_z:.4f}")


def main() -> None:
    base_dir = Path(__file__).parent.parent
    poses_path = base_dir / POSES_FILE
    xml_path = base_dir / XML_PATH
    
    # Load poses
    with open(poses_path, 'r') as f:
        poses_data = json.load(f)
    
    pose = poses_data["poses"][POSE_INDEX]
    
    print(f"Analyzing pose index {POSE_INDEX} from {POSES_FILE}")
    print(f"Base position: {pose['base_pos']}")
    print(f"Base RPY: {pose['base_rpy']}")
    
    # Load model and apply pose
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    apply_pose(model, data, pose)
    analyze_lowest_points(model, data)


if __name__ == "__main__":
    main()

