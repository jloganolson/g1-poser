from __future__ import annotations

from pathlib import Path
import math
import json

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
import tkinter as tk
from tkinter import ttk


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


def _swing_interp(p0: tuple[float, float, float], p1: tuple[float, float, float], phase: float, height: float) -> tuple[float, float, float]:
    """Parabolic swing interpolation between p0 and p1 on z=0 with apex height.

    phase in [0, 1]. x/y linearly interpolate; z follows a parabola peaking at `height`.
    """
    if phase <= 0.0:
        return (p0[0], p0[1], 0.0)
    if phase >= 1.0:
        return (p1[0], p1[1], 0.0)
    x = p0[0] + (p1[0] - p0[0]) * phase
    y = p0[1] + (p1[1] - p0[1]) * phase
    z = 4.0 * phase * (1.0 - phase) * height
    return (x, y, z)


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

    # ---- Lightweight Tk UI for gait tuning (no Entry widgets) ----
    root = tk.Tk()
    root.title("Crawl Gait Tuning")

    def add_scale(parent, label: str, var: tk.DoubleVar, lo: float, hi: float, resolution: float = 0.001) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=22).pack(side="left")
        val_label = ttk.Label(row, width=8, anchor="e")
        val_label.pack(side="right")
        def _update_label(*_args):
            try:
                val_label.configure(text=f"{float(var.get()):.3f}")
            except Exception:
                pass
        var.trace_add("write", _update_label)
        _update_label()
        s = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", length=260)
        s.pack(side="left", fill="x", expand=True, padx=8)

    panel = ttk.LabelFrame(root, text="Parameters", padding=8)
    panel.pack(fill="x")

    tv_speed = tk.DoubleVar(value=0.10)     # torso_fwd_speed (m/s)
    tv_bias = tk.DoubleVar(value=0.00)      # forward_bias (m)
    tv_period = tk.DoubleVar(value=1.20)    # step_period (s)
    tv_swing_frac = tk.DoubleVar(value=0.45) # swing_fraction
    tv_height = tk.DoubleVar(value=0.06)    # swing_height (m)
    tv_wiggle_amp = tk.DoubleVar(value=0.03) # wiggle amplitude (m)
    tv_wiggle_freq = tk.DoubleVar(value=0.30) # wiggle frequency (Hz)
    tv_cam_az = tk.DoubleVar(value=137.368)   # camera azimuth in degrees
    tv_cam_el = tk.DoubleVar(value=-16.395)   # camera elevation in degrees (negative looks down)
    tv_cam_dist = tk.DoubleVar(value=2.355)   # camera distance (m)

    add_scale(panel, "Torso speed (m/s)", tv_speed, 0.0, 0.40)
    add_scale(panel, "Forward bias (m)", tv_bias, -0.05, 0.10)
    add_scale(panel, "Step period (s)", tv_period, 0.40, 2.00)
    add_scale(panel, "Swing fraction", tv_swing_frac, 0.20, 0.80)
    add_scale(panel, "Swing height (m)", tv_height, 0.00, 0.12)
    add_scale(panel, "Wiggle amplitude (m)", tv_wiggle_amp, 0.00, 0.08)
    add_scale(panel, "Wiggle frequency (Hz)", tv_wiggle_freq, 0.00, 1.00)
    add_scale(panel, "Camera azimuth (deg)", tv_cam_az, -180.0, 180.0)
    add_scale(panel, "Camera elevation (deg)", tv_cam_el, -89.0, 89.0)
    add_scale(panel, "Camera distance (m)", tv_cam_dist, 0.5, 5.0)

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

    # Plant all four end-effectors on ground plane (z = 0)
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
        # Camera follow: track the pelvis body
        try:
            pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except Exception:
            pelvis_bid = -1
        try:
            if pelvis_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(pelvis_bid)
                viewer.cam.fixedcamid = -1
                # Initialize camera parameters
                viewer.cam.lookat[:] = data.xpos[pelvis_bid]
                viewer.cam.azimuth = float(tv_cam_az.get())
                viewer.cam.elevation = float(tv_cam_el.get())
                viewer.cam.distance = float(tv_cam_dist.get())
        except Exception:
            pass

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

        # Procedural crawl gait parameters (initial; will be updated live from UI)
        torso_fwd_speed = float(tv_speed.get())
        forward_bias = float(tv_bias.get())
        step_period = float(tv_period.get())
        swing_fraction = float(tv_swing_frac.get())
        swing_height = float(tv_height.get())

        # Gait state
        s_forward = 0.0
        gait_time = 0.0
        last_step_index = 0
        pair = 0  # 0: swing (right hand + left foot), 1: swing (left hand + right foot)
        swing_duration = swing_fraction * step_period

        # Resolve pelvis body id after viewer starts and compute initial offsets
        pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_pos0 = (
            float(data.xpos[pelvis_bid][0]),
            float(data.xpos[pelvis_bid][1]),
            float(data.xpos[pelvis_bid][2]),
        )

        # Desired offsets in pelvis frame (x,y) for each limb
        des_off_LH = [base_left_hand[0] - pelvis_pos0[0], base_left_hand[1] - pelvis_pos0[1]]
        des_off_RH = [base_right_hand[0] - pelvis_pos0[0], base_right_hand[1] - pelvis_pos0[1]]
        des_off_LF = [base_left_foot[0] - pelvis_pos0[0], base_left_foot[1] - pelvis_pos0[1]]
        des_off_RF = [base_right_foot[0] - pelvis_pos0[0], base_right_foot[1] - pelvis_pos0[1]]

        # Initialize first swing starts/targets using pelvis-relative offsets
        swing_start_RH = base_right_hand
        swing_start_LF = base_left_foot
        swing_start_LH = base_left_hand
        swing_start_RF = base_right_foot
        swing_target_RH = (
            pelvis_pos0[0] + des_off_RH[0] + forward_bias,
            pelvis_pos0[1] + des_off_RH[1],
            0.0,
        )
        swing_target_LF = (
            pelvis_pos0[0] + des_off_LF[0] + forward_bias,
            pelvis_pos0[1] + des_off_LF[1],
            0.0,
        )
        swing_target_LH = (
            pelvis_pos0[0] + des_off_LH[0] + forward_bias,
            pelvis_pos0[1] + des_off_LH[1],
            0.0,
        )
        swing_target_RF = (
            pelvis_pos0[0] + des_off_RF[0] + forward_bias,
            pelvis_pos0[1] + des_off_RF[1],
            0.0,
        )

        while viewer.is_running():
            # Pump Tk UI
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                pass

            # Refresh parameters from UI each frame
            torso_fwd_speed = float(tv_speed.get())
            forward_bias = float(tv_bias.get())
            step_period = max(1e-3, float(tv_period.get()))
            swing_fraction = min(0.95, max(0.05, float(tv_swing_frac.get())))
            swing_height = max(0.0, float(tv_height.get()))

            t += rate.dt
            gait_time += rate.dt

            # Advance torso forward with gentle lateral wiggle
            if base_xyz0 is not None and free_qpos_addr is not None:
                s_forward += torso_fwd_speed * rate.dt
                amp_xy = float(tv_wiggle_amp.get())
                freq = float(tv_wiggle_freq.get())
                dx = amp_xy * math.sin(2.0 * math.pi * freq * t)
                dy = amp_xy * math.cos(2.0 * math.pi * freq * t)
                configuration.data.qpos[free_qpos_addr + 0] = base_xyz0[0] + s_forward + dx
                configuration.data.qpos[free_qpos_addr + 1] = base_xyz0[1] + dy
                configuration.data.qpos[free_qpos_addr + 2] = base_xyz0[2]
                mujoco.mj_forward(configuration.model, configuration.data)
                pelvis_orientation_task.set_target_from_configuration(configuration)
                pelvis_position_task.set_target_from_configuration(configuration)
                torso_orientation_task.set_target_from_configuration(configuration)

            # Gait phase updates
            step_index = int(gait_time // step_period)
            if step_index != last_step_index:
                # Commit last swing: replant to targets
                if pair == 0:
                    base_right_hand = swing_target_RH
                    base_left_foot = swing_target_LF
                else:
                    base_left_hand = swing_target_LH
                    base_right_foot = swing_target_RF

                # Alternate swing pair
                pair = 1 - pair

                # Prepare next swing start/targets using constant pelvis-relative offsets (no accumulation)
                pelvis_pos_start = (
                    float(data.xpos[pelvis_bid][0]),
                    float(data.xpos[pelvis_bid][1]),
                    float(data.xpos[pelvis_bid][2]),
                )
                if pair == 0:
                    # Next swing: right hand + left foot; return to neutral pelvis-relative offsets
                    swing_start_RH = base_right_hand
                    swing_start_LF = base_left_foot
                    swing_target_RH = (pelvis_pos_start[0] + des_off_RH[0] + forward_bias, pelvis_pos_start[1] + des_off_RH[1], 0.0)
                    swing_target_LF = (pelvis_pos_start[0] + des_off_LF[0] + forward_bias, pelvis_pos_start[1] + des_off_LF[1], 0.0)
                else:
                    # Next swing: left hand + right foot; return to neutral pelvis-relative offsets
                    swing_start_LH = base_left_hand
                    swing_start_RF = base_right_foot
                    swing_target_LH = (pelvis_pos_start[0] + des_off_LH[0] + forward_bias, pelvis_pos_start[1] + des_off_LH[1], 0.0)
                    swing_target_RF = (pelvis_pos_start[0] + des_off_RF[0] + forward_bias, pelvis_pos_start[1] + des_off_RF[1], 0.0)
                last_step_index = step_index

            swing_duration = swing_fraction * step_period
            phase_time = gait_time - (step_index * step_period)
            phi = min(1.0, max(0.0, phase_time / max(1e-6, swing_duration)))

            # Set mocap targets for stance limbs (remain planted)
            if pair == 0:
                # Stance: left hand, right foot
                data.mocap_pos[left_palm_mid][:] = base_left_hand
                data.mocap_pos[right_foot_mid][:] = base_right_foot
                # Swing: right hand, left foot
                if phase_time < swing_duration:
                    data.mocap_pos[right_palm_mid][:] = _swing_interp(swing_start_RH, swing_target_RH, phi, swing_height)
                    data.mocap_pos[left_foot_mid][:] = _swing_interp(swing_start_LF, swing_target_LF, phi, swing_height)
                else:
                    data.mocap_pos[right_palm_mid][:] = swing_target_RH
                    data.mocap_pos[left_foot_mid][:] = swing_target_LF
            else:
                # Stance: right hand, left foot
                data.mocap_pos[right_palm_mid][:] = base_right_hand
                data.mocap_pos[left_foot_mid][:] = base_left_foot
                # Swing: left hand, right foot
                if phase_time < swing_duration:
                    data.mocap_pos[left_palm_mid][:] = _swing_interp(swing_start_LH, swing_target_LH, phi, swing_height)
                    data.mocap_pos[right_foot_mid][:] = _swing_interp(swing_start_RF, swing_target_RF, phi, swing_height)
                else:
                    data.mocap_pos[left_palm_mid][:] = swing_target_LH
                    data.mocap_pos[right_foot_mid][:] = swing_target_RF

            # Update IK targets from mocap
            right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))
            left_hand_task.set_target(mink.SE3.from_mocap_id(data, left_palm_mid))
            left_foot_task.set_target(mink.SE3.from_mocap_id(data, left_foot_mid))
            right_foot_task.set_target(mink.SE3.from_mocap_id(data, right_foot_mid))

            # Solve and integrate
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            # Keep camera centered on pelvis and apply orbit parameters
            try:
                if pelvis_bid != -1:
                    viewer.cam.lookat[0] = float(data.xpos[pelvis_bid][0])
                    viewer.cam.lookat[1] = float(data.xpos[pelvis_bid][1])
                    viewer.cam.lookat[2] = float(data.xpos[pelvis_bid][2])
                    viewer.cam.azimuth = float(tv_cam_az.get())
                    viewer.cam.elevation = float(tv_cam_el.get())
                    viewer.cam.distance = float(tv_cam_dist.get())
            except Exception:
                pass

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


