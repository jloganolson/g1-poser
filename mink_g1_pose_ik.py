from __future__ import annotations

from pathlib import Path
import json
import math

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# Tk UI for target controls
import tkinter as tk
from tkinter import ttk

_HERE = Path(__file__).parent
# Scene with G1 and mocap targets for right hand and both feet
_XML = _HERE / "g1_description" / "scene_g1_targets.xml"

# Hardcoded pose file exported from pose-tool.py
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


def _apply_pose_to_state(model: mujoco.MjModel, data: mujoco.MjData, pose: dict) -> None:
    """Apply a pose-tool pose dict to qpos (free-base orientation + named joints)."""
    try:
        # Set base orientation from RPY if a free joint is present
        free_qpos_addr = None
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:
                free_qpos_addr = int(model.jnt_qposadr[j])
                break
        if free_qpos_addr is not None:
            base_rpy = pose.get("base_rpy")
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
        joints = pose.get("joints", {}) or {}
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


if __name__ == "__main__":
    # Load scene
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Load first pose from JSON (if present) and apply
    if _POSES_JSON.exists():
        try:
            js = json.loads(_POSES_JSON.read_text())
            first_pose = None
            poses = js.get("poses")
            if isinstance(poses, list) and poses:
                first_pose = poses[0]
            if isinstance(first_pose, dict):
                _apply_pose_to_state(model, data, first_pose)
                mujoco.mj_forward(model, data)
        except Exception:
            # Ignore malformed files and continue with default state
            pass

    # Build MINK configuration
    configuration = mink.Configuration(model)
    # Copy the possibly-updated state into the configuration
    configuration.data.qpos[:] = data.qpos
    mujoco.mj_forward(configuration.model, configuration.data)

    tasks = [
        # Keep pelvis pose fixed to avoid drifting while moving hands/feet
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
        # Keep torso orientation stable for comfort
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

    # Encourage knees and elbows to stay "up" by stabilizing their link orientations.
    # These orientation tasks bias the solver away from pitching the segments downward
    # when hands/feet are moved around.
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

    # Task: right hand follows mocap target's position (ignore orientation)
    right_hand_task = mink.FrameTask(
        frame_name="right_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    tasks.append(right_hand_task)

    # Task: left hand follows mocap target's position (ignore orientation)
    left_hand_task = mink.FrameTask(
        frame_name="left_palm",
        frame_type="site",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1.0,
    )
    tasks.append(left_hand_task)

    # Tasks: feet follow mocap targets' positions (ignore orientation)
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
    tasks.extend([left_foot_task, right_foot_task])

    # Keep feet flat relative to the ground by stabilizing ankle link orientations
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

    # Resolve mocap IDs and initialize mocap bodies to current sites
    right_palm_mid = model.body("right_palm_target").mocapid[0]
    left_palm_mid = model.body("left_palm_target").mocapid[0]
    left_foot_mid = model.body("left_foot_target").mocapid[0]
    right_foot_mid = model.body("right_foot_target").mocapid[0]

    model = configuration.model
    data = configuration.data

    # Place mocap bodies at current frames
    mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")
    mink.move_mocap_to_frame(model, data, "left_palm_target", "left_palm", "site")
    mink.move_mocap_to_frame(model, data, "left_foot_target", "left_foot", "site")
    mink.move_mocap_to_frame(model, data, "right_foot_target", "right_foot", "site")

    # --- Tk UI: XY pads and Z sliders for mocap targets ---
    class XYPad(ttk.Frame):
        def __init__(self, parent: tk.Misc, x_var: tk.DoubleVar, y_var: tk.DoubleVar, x_center: float, y_center: float, x_half_range: float, y_half_range: float, width: int = 200, height: int = 200, title: str | None = None) -> None:
            super().__init__(parent, padding=4)
            self.x_var = x_var
            self.y_var = y_var
            self.x_center = float(x_center)
            self.y_center = float(y_center)
            self.x_half = float(x_half_range)
            self.y_half = float(y_half_range)
            self.w = int(width)
            self.h = int(height)

            if title:
                ttk.Label(self, text=title, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

            self.canvas = tk.Canvas(self, width=self.w, height=self.h, bg="white", highlightthickness=1, highlightbackground="#ccc")
            self.canvas.pack(fill="both", expand=True)

            # Draw bounds and crosshair
            self.canvas.create_rectangle(2, 2, self.w - 2, self.h - 2, outline="#aaa")
            self.canvas.create_line(self.w // 2, 0, self.w // 2, self.h, fill="#eee")
            self.canvas.create_line(0, self.h // 2, self.w, self.h // 2, fill="#eee")

            self.handle = self.canvas.create_oval(0, 0, 0, 0, fill="#3a7", outline="", tags=("handle",))
            self._dragging = False

            def _to_canvas_coords(x_world: float, y_world: float) -> tuple[float, float]:
                # Canvas X controls world Y (left/right relative to body center)
                # Canvas Y controls world X (forward/back relative to body center)
                u_y = (y_world - self.y_center) / self.y_half
                v_x = (x_world - self.x_center) / self.x_half
                # Left on canvas → larger +Y (left of body): negate u_y
                cx = (-u_y * 0.5 + 0.5) * self.w
                # Up on canvas → larger +X (forward): negate v_x
                cy = (-v_x * 0.5 + 0.5) * self.h
                return cx, cy

            def _to_world_coords(cx: float, cy: float) -> tuple[float, float]:
                # u: + to the right on canvas; left is negative
                # v: + downward on canvas; up is negative
                u = (cx / self.w - 0.5) * 2.0
                v = (cy / self.h - 0.5) * 2.0
                # Left on canvas (u<0) → +Y (left), Up on canvas (v<0) → +X (forward)
                xw = self.x_center - max(-1.0, min(1.0, v)) * self.x_half
                yw = self.y_center - max(-1.0, min(1.0, u)) * self.y_half
                return xw, yw

            self._to_canvas_coords = _to_canvas_coords  # type: ignore[attr-defined]
            self._to_world_coords = _to_world_coords    # type: ignore[attr-defined]

            def _redraw_handle(*_args) -> None:
                cx, cy = self._to_canvas_coords(float(self.x_var.get()), float(self.y_var.get()))
                r = 6
                self.canvas.coords(self.handle, cx - r, cy - r, cx + r, cy + r)

            def _on_press(evt) -> None:  # type: ignore[no-redef]
                self._dragging = True
                _on_drag(evt)

            def _on_drag(evt) -> None:  # type: ignore[no-redef]
                if not self._dragging:
                    return
                cx = max(0, min(self.w, evt.x))
                cy = max(0, min(self.h, evt.y))
                xw, yw = self._to_world_coords(cx, cy)
                self.x_var.set(xw)
                self.y_var.set(yw)

            def _on_release(_evt) -> None:
                self._dragging = False

            self.canvas.bind("<Button-1>", _on_press)
            self.canvas.bind("<B1-Motion>", _on_drag)
            self.canvas.bind("<ButtonRelease-1>", _on_release)

            self.x_var.trace_add("write", _redraw_handle)
            self.y_var.trace_add("write", _redraw_handle)
            _redraw_handle()

    class TargetControl(ttk.LabelFrame):
        def __init__(self, parent: tk.Misc, label: str, init_xyz: tuple[float, float, float], vrange_xy: tuple[float, float] = (0.6, 0.6), z_span: float = 0.7) -> None:
            super().__init__(parent, text=label, padding=6)
            ix, iy, iz = init_xyz
            self.x = tk.DoubleVar(value=float(ix))
            self.y = tk.DoubleVar(value=float(iy))
            self.z = tk.DoubleVar(value=float(iz))
            self.z_min = float(iz)
            self.z_max = float(iz + z_span)

            pad = XYPad(self, self.x, self.y, x_center=float(ix), y_center=float(iy), x_half_range=float(vrange_xy[0]), y_half_range=float(vrange_xy[1]), width=200, height=200)
            pad.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 8))

            ttk.Label(self, text="Z (m)").grid(row=0, column=1, sticky="w")
            # Invert so top is max height
            z_scale = ttk.Scale(self, from_=self.z_max, to=self.z_min, variable=self.z, orient="vertical", length=160)
            z_scale.grid(row=1, column=1, sticky="nsw")
            self.z_readout = ttk.Label(self, text=f"{float(self.z.get()):.3f}")
            self.z_readout.grid(row=2, column=1, sticky="w")

            def _update_readout(*_a):
                self.z_readout.configure(text=f"{float(self.z.get()):.3f}")

            self.z.trace_add("write", _update_readout)

            self.grid_columnconfigure(0, weight=1)

        def get_xyz(self) -> tuple[float, float, float]:
            return float(self.x.get()), float(self.y.get()), float(self.z.get())

    # Build the window
    root = tk.Tk()
    root.title("Mocap Targets")

    # Capture initial mocap world positions (after move_mocap_to_frame)
    init_right_palm = tuple(map(float, data.mocap_pos[right_palm_mid]))
    init_left_palm = tuple(map(float, data.mocap_pos[left_palm_mid]))
    init_left_foot = tuple(map(float, data.mocap_pos[left_foot_mid]))
    init_right_foot = tuple(map(float, data.mocap_pos[right_foot_mid]))

    controls = ttk.Frame(root, padding=6)
    controls.pack(fill="both", expand=True)

    # Place panels intuitively: Left Hand (top-left), Right Hand (top-right),
    # Left Foot (bottom-left), Right Foot (bottom-right)
    lp = TargetControl(controls, "Left Hand", init_left_palm)
    rp = TargetControl(controls, "Right Hand", init_right_palm)
    lf = TargetControl(controls, "Left Foot", init_left_foot)
    rf = TargetControl(controls, "Right Foot", init_right_foot)

    lp.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
    rp.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
    lf.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    rf.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=1)
    controls.grid_rowconfigure(0, weight=1)
    controls.grid_rowconfigure(1, weight=1)

    solver = "daqp"

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize task targets from current configuration
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
        while viewer.is_running():
            # Poll Tk UI (non-blocking)
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                pass

            # Apply UI -> mocap positions (world space)
            rpx, rpy, rpz = rp.get_xyz()
            lpx, lpy, lpz = lp.get_xyz()
            lfx, lfy, lfz = lf.get_xyz()
            rfx, rfy, rfz = rf.get_xyz()

            data.mocap_pos[right_palm_mid][0] = float(rpx)
            data.mocap_pos[right_palm_mid][1] = float(rpy)
            data.mocap_pos[right_palm_mid][2] = float(rpz)

            data.mocap_pos[left_palm_mid][0] = float(lpx)
            data.mocap_pos[left_palm_mid][1] = float(lpy)
            data.mocap_pos[left_palm_mid][2] = float(lpz)

            data.mocap_pos[left_foot_mid][0] = float(lfx)
            data.mocap_pos[left_foot_mid][1] = float(lfy)
            data.mocap_pos[left_foot_mid][2] = float(lfz)

            data.mocap_pos[right_foot_mid][0] = float(rfx)
            data.mocap_pos[right_foot_mid][1] = float(rfy)
            data.mocap_pos[right_foot_mid][2] = float(rfz)

            # Update targets from mocap bodies
            right_hand_task.set_target(mink.SE3.from_mocap_id(data, right_palm_mid))
            left_hand_task.set_target(mink.SE3.from_mocap_id(data, left_palm_mid))
            left_foot_task.set_target(mink.SE3.from_mocap_id(data, left_foot_mid))
            right_foot_task.set_target(mink.SE3.from_mocap_id(data, right_foot_mid))

            # Solve and integrate IK step
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()

    # Close Tk when viewer exits
    try:
        root.destroy()
    except Exception:
        pass


