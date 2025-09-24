from __future__ import annotations

from pathlib import Path
import argparse
import math

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink


_DEFAULT_XML = "./g1_description/scene_g1_targets.xml"


def _shift_base_z_to_ground(model: mujoco.MjModel, data: mujoco.MjData, left_site: str, right_site: str) -> None:
    """Translate the free base along z so average foot site height becomes 0.

    Safe to call with or without a free joint.
    """
    try:
        # Find free joint qpos base address (xyz + quat)
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
        # Non-fatal convenience helper
        pass


def _resolve_mocap_id_or_neg1(model: mujoco.MjModel, body_name: str) -> int:
    try:
        return int(model.body(body_name).mocapid[0])
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Mink IK reference with Unitree G1")
    parser.add_argument(
        "--xml",
        type=str,
        default=str(_DEFAULT_XML),
        help="Path to MuJoCo XML. Prefer a scene with mocap targets (e.g. scene_g1_targets.xml).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="daqp",
        help="QP solver to use (e.g. 'daqp'). Falls back to mink default if unavailable.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=200.0,
        help="IK loop frequency (Hz)",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())

    # Build a MINK configuration tied to this model
    configuration = mink.Configuration(model)
    data = configuration.data

    # Define core stabilization tasks
    tasks = [
        # Keep pelvis upright (no position tracking here)
        (pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        # Lightly anchor pelvis position so reaching with hands/feet doesn't drift the base
        (pelvis_position_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )),
        # Keep torso comfortable/upright
        (torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )),
        # Stay near the current configuration
        (posture_task := mink.PostureTask(model, cost=1e-1)),
    ]

    # Stabilize intermediate links to discourage awkward pitching
    left_knee_orientation_task = mink.FrameTask(
        frame_name="left_knee_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=0.8,
        lm_damping=1.0,
    )
    right_knee_orientation_task = mink.FrameTask(
        frame_name="right_knee_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=0.8,
        lm_damping=1.0,
    )
    left_elbow_orientation_task = mink.FrameTask(
        frame_name="left_elbow_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=1.2,
        lm_damping=1.0,
    )
    right_elbow_orientation_task = mink.FrameTask(
        frame_name="right_elbow_link",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=1.2,
        lm_damping=1.0,
    )
    tasks.extend([
        left_knee_orientation_task,
        right_knee_orientation_task,
        left_elbow_orientation_task,
        right_elbow_orientation_task,
    ])

    # End-effector tasks (position only). Costs are tuned for crawling-style use.
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
        frame_name="left_foot",
        frame_type="site",
        position_cost=FOOT_POSITION_COST,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    right_foot_task = mink.FrameTask(
        frame_name="right_foot",
        frame_type="site",
        position_cost=FOOT_POSITION_COST,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    tasks.extend([right_hand_task, left_hand_task, left_foot_task, right_foot_task])

    # Keep feet more level via ankle link orientations
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

    limits = [mink.ConfigurationLimit(model)]

    # Initialize state
    mujoco.mj_forward(configuration.model, configuration.data)

    # Resolve mocap bodies if present (scene_g1_targets.xml provides these)
    right_palm_mid = _resolve_mocap_id_or_neg1(model, "right_palm_target")
    left_palm_mid = _resolve_mocap_id_or_neg1(model, "left_palm_target")
    left_foot_mid = _resolve_mocap_id_or_neg1(model, "left_foot_target")
    right_foot_mid = _resolve_mocap_id_or_neg1(model, "right_foot_target")

    # Sync mocap to current frames and shift base so average feet height is ground
    if right_palm_mid != -1:
        mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
    if left_palm_mid != -1:
        mink.move_mocap_to_frame(model, data, "left_palm_target", "left_palm", "site")
    if left_foot_mid != -1:
        mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    if right_foot_mid != -1:
        mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")

    _shift_base_z_to_ground(model, data, left_site="left_foot", right_site="right_foot")
    if left_foot_mid != -1:
        mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    if right_foot_mid != -1:
        mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")

    # Re-initialize task targets AFTER base shift
    mujoco.mj_forward(configuration.model, configuration.data)
    posture_task.set_target_from_configuration(configuration)
    pelvis_orientation_task.set_target_from_configuration(configuration)
    pelvis_position_task.set_target_from_configuration(configuration)
    torso_orientation_task.set_target_from_configuration(configuration)
    left_foot_orientation_task.set_target_from_configuration(configuration)
    right_foot_orientation_task.set_target_from_configuration(configuration)
    left_knee_orientation_task.set_target_from_configuration(configuration)
    right_knee_orientation_task.set_target_from_configuration(configuration)
    left_elbow_orientation_task.set_target_from_configuration(configuration)
    right_elbow_orientation_task.set_target_from_configuration(configuration)

    solver = str(args.solver)

    # Launch passive viewer and run a simple IK loop
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Camera: track pelvis if available
        try:
            pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except Exception:
            pelvis_bid = -1
        try:
            if pelvis_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(pelvis_bid)
                viewer.cam.fixedcamid = -1
        except Exception:
            pass

        rate = RateLimiter(frequency=float(args.fps), warn=False)

        t = 0.0
        # Small, safe hand demo in front-right of pelvis at a fixed height
        hand_forward_offset = 0.35
        hand_side_offset = -0.25  # negative Y is right side in this model
        hand_radius_x = 0.06
        hand_radius_y = 0.04

        while viewer.is_running():
            # Demonstration: if right-hand mocap exists, move it in a small circle at the current height
            if right_palm_mid != -1:
                # Center the motion in front-right of pelvis to avoid intersecting the torso
                px = float(data.xpos[pelvis_bid][0]) if pelvis_bid != -1 else 0.0
                py = float(data.xpos[pelvis_bid][1]) if pelvis_bid != -1 else 0.0
                cx = px + hand_forward_offset
                cy = py + hand_side_offset
                # Keep height fixed near the current hand height, minimum 5 cm above ground
                base_z = float(data.mocap_pos[right_palm_mid][2])
                z_target = base_z if base_z > 0.05 else 0.05
                # Gentle planar Lissajous to show responsiveness without crossing the body
                data.mocap_pos[right_palm_mid][0] = cx + hand_radius_x * math.cos(2.0 * math.pi * 0.2 * t)
                data.mocap_pos[right_palm_mid][1] = cy + hand_radius_y * math.sin(2.0 * math.pi * 0.2 * t)
                data.mocap_pos[right_palm_mid][2] = z_target

            # Update end-effector targets from mocap if available; otherwise hold current configuration
            if right_palm_mid != -1:
                right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))
            else:
                right_hand_task.set_target_from_configuration(configuration)
            if left_palm_mid != -1:
                left_hand_task.set_target(mink.SE3.from_mocap_id(data, left_palm_mid))
            else:
                left_hand_task.set_target_from_configuration(configuration)
            if left_foot_mid != -1:
                left_foot_task.set_target(mink.SE3.from_mocap_id(data, left_foot_mid))
            else:
                left_foot_task.set_target_from_configuration(configuration)
            if right_foot_mid != -1:
                right_foot_task.set_target(mink.SE3.from_mocap_id(data, right_foot_mid))
            else:
                right_foot_task.set_target_from_configuration(configuration)

            # Solve a small IK step and integrate
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
            t += rate.dt


if __name__ == "__main__":
    main()


