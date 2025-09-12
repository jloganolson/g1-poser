from __future__ import annotations

from pathlib import Path
import math
import json

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink


_HERE = Path(__file__).parent
_XML = _HERE / "g1_description" / "scene_g1_targets.xml"
_POSES_JSON = _HERE / "crawl-pose.json"


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def _shift_base_z_to_ground(model: mujoco.MjModel, data: mujoco.MjData, left_site: str, right_site: str) -> None:
    """Translate the free base along z so average foot site height becomes 0.

    Keeps base orientation unchanged. Safe to call with or without a free joint.
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
        # Shift base down by average height
        data.qpos[free_qpos_addr + 2] -= z_avg
        mujoco.mj_forward(model, data)
    except Exception:
        pass


if __name__ == "__main__":
    # Load scene and build initial state
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Apply crawling pose (first pose) if available
    if _POSES_JSON.exists():
        try:
            js = json.loads(_POSES_JSON.read_text())
            first_pose = None
            poses = js.get("poses")
            if isinstance(poses, list) and poses:
                first_pose = poses[0]
            if isinstance(first_pose, dict):
                # Set base orientation from RPY if a free joint exists
                try:
                    free_qpos_addr = None
                    for j in range(model.njnt):
                        if int(model.jnt_type[j]) == 0:
                            free_qpos_addr = int(model.jnt_qposadr[j])
                            break
                    if free_qpos_addr is not None:
                        base_rpy = first_pose.get("base_rpy")
                        if base_rpy is not None and len(base_rpy) == 3:
                            qw, qx, qy, qz = _rpy_to_quat(float(base_rpy[0]), float(base_rpy[1]), float(base_rpy[2]))
                            data.qpos[free_qpos_addr + 3] = float(qw)
                            data.qpos[free_qpos_addr + 4] = float(qx)
                            data.qpos[free_qpos_addr + 5] = float(qy)
                            data.qpos[free_qpos_addr + 6] = float(qz)
                except Exception:
                    pass

                # Set named hinge/slide joints
                try:
                    joints = first_pose.get("joints", {}) or {}
                    for name, val in joints.items():
                        try:
                            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                        except Exception:
                            jid = -1
                        if jid == -1:
                            continue
                        jpos = int(model.jnt_qposadr[jid])
                        data.qpos[jpos] = float(val)
                except Exception:
                    pass
                mujoco.mj_forward(model, data)
        except Exception:
            pass

    # Build MINK configuration from the (possibly) updated state
    configuration = mink.Configuration(model)
    configuration.data.qpos[:] = data.qpos
    mujoco.mj_forward(configuration.model, configuration.data)

    # IK tasks mirroring mink_g1_pose_ik.py
    tasks = [
        # Pelvis stability
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        pelvis_position_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=10.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        ),
        # Torso comfort
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        # Stay near current configuration
        posture_task := mink.PostureTask(model, cost=1e-1),
    ]

    # Knee/elbow orientation stabilizers
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

    # End-effector tasks
    right_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    left_hand_task = mink.FrameTask(
        frame_name="left_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    left_foot_task = mink.FrameTask(
        frame_name="left_foot",
        frame_type="site",
        position_cost=10.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    right_foot_task = mink.FrameTask(
        frame_name="right_foot",
        frame_type="site",
        position_cost=10.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    tasks.extend([right_hand_task, left_hand_task, left_foot_task, right_foot_task])

    # Keep feet flat via ankle link orientations
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

    # Resolve mocap IDs
    right_palm_mid = model.body("right_palm_target").mocapid[0]
    left_palm_mid = model.body("left_palm_target").mocapid[0]
    left_foot_mid = model.body("left_foot_target").mocapid[0]
    right_foot_mid = model.body("right_foot_target").mocapid[0]

    model = configuration.model
    data = configuration.data

    # Initialize mocap bodies to match current frames
    mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
    mink.move_mocap_to_frame(model, data, "left_palm_target", "left_palm", "site")
    mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")

    # Bring base down so feet are near the ground plane
    _shift_base_z_to_ground(model, data, left_site="left_foot", right_site="right_foot")
    # Re-sync mocap after base move
    mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")

    # Plant all four end-effectors on ground plane (z = 0) and freeze their positions
    # Capture baseline world positions
    base_left_hand = tuple(float(x) for x in data.mocap_pos[left_palm_mid])
    base_right_hand = tuple(float(x) for x in data.mocap_pos[right_palm_mid])
    base_left_foot = tuple(float(x) for x in data.mocap_pos[left_foot_mid])
    base_right_foot = tuple(float(x) for x in data.mocap_pos[right_foot_mid])
    # Force z to ground for all
    base_left_hand = (base_left_hand[0], base_left_hand[1], 0.0)
    base_right_hand = (base_right_hand[0], base_right_hand[1], 0.0)
    base_left_foot = (base_left_foot[0], base_left_foot[1], 0.0)
    base_right_foot = (base_right_foot[0], base_right_foot[1], 0.0)
    # Initialize mocap to these planted positions
    data.mocap_pos[left_palm_mid][:] = base_left_hand
    data.mocap_pos[right_palm_mid][:] = base_right_hand
    data.mocap_pos[left_foot_mid][:] = base_left_foot
    data.mocap_pos[right_foot_mid][:] = base_right_foot

    # Viewer + IK loop
    solver = "daqp"
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Set targets from the current configuration
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

        rate = RateLimiter(frequency=200.0, warn=False)
        t = 0.0

        # Free joint base address and initial base position for pelvis wiggle
        free_qpos_addr = None
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:
                free_qpos_addr = int(model.jnt_qposadr[j])
                break
        base_xyz0 = None
        if free_qpos_addr is not None:
            base_xyz0 = (
                float(configuration.data.qpos[free_qpos_addr + 0]),
                float(configuration.data.qpos[free_qpos_addr + 1]),
                float(configuration.data.qpos[free_qpos_addr + 2]),
            )

        while viewer.is_running():
            t += rate.dt

            # Keep all four targets fixed at planted positions
            data.mocap_pos[left_palm_mid][:] = base_left_hand
            data.mocap_pos[right_palm_mid][:] = base_right_hand
            data.mocap_pos[left_foot_mid][:] = base_left_foot
            data.mocap_pos[right_foot_mid][:] = base_right_foot

            # Wiggle pelvis/torso by translating the free base slightly
            if base_xyz0 is not None and free_qpos_addr is not None:
                amp_xy = 0.03
                freq = 0.3
                dx = amp_xy * math.sin(2.0 * math.pi * freq * t)
                dy = amp_xy * math.cos(2.0 * math.pi * freq * t)
                configuration.data.qpos[free_qpos_addr + 0] = base_xyz0[0] + dx
                configuration.data.qpos[free_qpos_addr + 1] = base_xyz0[1] + dy
                configuration.data.qpos[free_qpos_addr + 2] = base_xyz0[2]
                mujoco.mj_forward(configuration.model, configuration.data)
                # Update pelvis/torso task targets to the new base pose
                pelvis_orientation_task.set_target_from_configuration(configuration)
                pelvis_position_task.set_target_from_configuration(configuration)
                torso_orientation_task.set_target_from_configuration(configuration)

            # Update IK targets from mocap
            right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))
            left_hand_task.set_target(mink.SE3.from_mocap_id(data, left_palm_mid))
            left_foot_task.set_target(mink.SE3.from_mocap_id(data, left_foot_mid))
            right_foot_task.set_target(mink.SE3.from_mocap_id(data, right_foot_mid))

            # Solve and integrate
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


