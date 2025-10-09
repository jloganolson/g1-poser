#!/usr/bin/env python3
"""Convert animation_mocap JSON to pose array with ground-adjusted z.

Reads a gait_animation.v1 JSON file and converts each frame to a pose
with base_pos x/y=0 and z adjusted so the lowest point touches the ground.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

# ----------------------------- Static configuration -----------------------------
# Change these paths to convert different animation files
INPUT_JSON = "output/animation_mocap_rc0.json"
OUTPUT_JSON = "output/animation_mocap_rc0_poses.json"
XML_PATH = "g1_description/scene_g1_targets.xml"

# Ground clearance margin (meters) - added to the lowest point
GROUND_CLEARANCE_MARGIN = 0.0

# Set to True to also output a sorted version (sorted by Z height)
OUTPUT_SORTED = True
SORTED_OUTPUT_SUFFIX = "_sorted"  # Will add this before .json


def quat_wxyz_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to roll, pitch, yaw (radians)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def compute_lowest_point_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Compute the lowest z-coordinate of the robot's geometry (excluding floor/world)."""
    mujoco.mj_forward(model, data)
    
    min_z = float('inf')
    
    # Get the world/floor body ID (typically body 0)
    world_body_id = 0
    
    # Check all body positions and geometries
    for body_id in range(model.nbody):
        # Skip world/floor body
        if body_id == world_body_id:
            continue
            
        # Get body position
        body_pos = data.xpos[body_id]
        
        # Check all geoms attached to this body
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                
                # Skip floor geometry by name
                if geom_name.lower() in ['floor', 'ground', 'plane']:
                    continue
                
                geom_pos = data.geom_xpos[geom_id]
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id]
                
                # Estimate lowest point based on geom type
                if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    lowest = geom_pos[2] - geom_size[0]
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    lowest = geom_pos[2] - geom_size[1]  # half-length
                elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    lowest = geom_pos[2] - geom_size[2]
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    # For meshes, use the geom position as approximation
                    lowest = geom_pos[2]
                else:
                    lowest = geom_pos[2]
                
                min_z = min(min_z, lowest)
    
    # Also check site positions (hands, feet, etc.)
    for site_id in range(model.nsite):
        site_z = data.site_xpos[site_id][2]
        min_z = min(min_z, site_z)
    
    return min_z


def convert_frame_to_pose(
    qpos: List[float],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_info: Dict,
    joints_info: List[Dict]
) -> Dict:
    """Convert a single qpos frame to a pose dict with ground-adjusted z."""
    
    # Extract base position and orientation
    pos_indices = base_info["pos_indices"]
    quat_indices = base_info["quat_indices"]
    
    base_x = qpos[pos_indices[0]]
    base_y = qpos[pos_indices[1]]
    base_z = qpos[pos_indices[2]]
    
    qw = qpos[quat_indices[0]]
    qx = qpos[quat_indices[1]]
    qy = qpos[quat_indices[2]]
    qz = qpos[quat_indices[3]]
    
    # Convert quaternion to RPY
    roll, pitch, yaw = quat_wxyz_to_rpy(qw, qx, qy, qz)
    
    # Set qpos in data to compute geometry positions
    data.qpos[:] = qpos
    
    # Compute lowest point
    min_z = compute_lowest_point_z(model, data)
    
    # Calculate adjustment needed to put lowest point at ground level
    z_adjustment = -min_z + GROUND_CLEARANCE_MARGIN
    adjusted_z = base_z + z_adjustment
    
    # Extract joint values (skip free joint)
    joints = {}
    for joint_info in joints_info:
        if joint_info["type"] == "free":
            continue
        if joint_info["qposdim"] == 1:  # Only scalar joints
            joints[joint_info["name"]] = qpos[joint_info["qposadr"]]
    
    return {
        "base_pos": [0.0, 0.0, adjusted_z],
        "base_rpy": [roll, pitch, yaw],
        "joints": joints
    }


def main() -> None:
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / INPUT_JSON
    output_path = base_dir / OUTPUT_JSON
    xml_path = base_dir / XML_PATH
    
    print(f"Loading animation from: {input_path}")
    
    # Load animation JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        anim_data = json.load(f)
    
    # Validate schema
    if anim_data.get("schema") != "gait_animation.v1":
        raise ValueError(f"Unexpected schema: {anim_data.get('schema')}")
    
    frames = anim_data["frames"]
    metadata = anim_data["metadata"]
    
    if "base" not in metadata:
        raise ValueError("Animation has no free base joint metadata")
    
    base_info = metadata["base"]
    joints_info = metadata["joints"]
    
    print(f"Animation has {len(frames)} frames")
    print(f"Loading model from: {xml_path}")
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Convert each frame to a pose
    poses = []
    for i, frame_qpos in enumerate(frames):
        if i % 100 == 0:
            print(f"Processing frame {i}/{len(frames)}...")
        
        pose = convert_frame_to_pose(
            frame_qpos,
            model,
            data,
            base_info,
            joints_info
        )
        poses.append(pose)
    
    # Create output document
    output_doc = {
        "poses": poses
    }
    
    # Print summary statistics
    z_values = [pose["base_pos"][2] for pose in poses]
    print(f"\nSummary:")
    print(f"  Total poses: {len(poses)}")
    print(f"  Base Z range: {min(z_values):.4f} to {max(z_values):.4f} meters")
    print(f"  Ground clearance margin: {GROUND_CLEARANCE_MARGIN:.3f} meters")
    
    # Write original output (in temporal order)
    print(f"\nWriting {len(poses)} poses (temporal order) to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_doc, f, indent=2)
    
    # Optionally write sorted output
    if OUTPUT_SORTED:
        # Sort poses by Z height (ascending)
        sorted_poses = sorted(poses, key=lambda p: p["base_pos"][2])
        sorted_output_doc = {
            "poses": sorted_poses
        }
        
        # Create sorted output path
        sorted_path = Path(str(output_path).replace('.json', f'{SORTED_OUTPUT_SUFFIX}.json'))
        
        print(f"Writing {len(sorted_poses)} poses (sorted by Z) to: {sorted_path}")
        with open(sorted_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_output_doc, f, indent=2)
        
        print(f"  Sorted Z range: {sorted_poses[0]['base_pos'][2]:.4f} (lowest) to {sorted_poses[-1]['base_pos'][2]:.4f} (highest)")
    
    print("Done!")


if __name__ == "__main__":
    main()

