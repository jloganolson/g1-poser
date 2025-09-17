from __future__ import annotations

from pathlib import Path
import json
import math
from datetime import datetime
import shutil
import subprocess

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

# Tk UI for target controls
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, simpledialog

_HERE = Path(__file__).parent
# Scene with G1 and mocap targets for right hand and both feet
_XML = _HERE / "g1_description" / "scene_g1_targets.xml"

# Hardcoded pose file exported from pose-tool.py
_POSES_JSON = _HERE / "crawl-pose.json"
_OUT_DIR = _HERE / "output"


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

    # --- Tk UI: Dual XY pads per leg and gait controls ---
    class DualXYPad(ttk.Frame):
        def __init__(
            self,
            parent: tk.Misc,
            plant_x: tk.DoubleVar,
            plant_y: tk.DoubleVar,
            lift_x: tk.DoubleVar,
            lift_y: tk.DoubleVar,
            x_center: float,
            y_center: float,
            x_half_range: float,
            y_half_range: float,
            width: int = 200,
            height: int = 200,
            title: str | None = None,
        ) -> None:
            super().__init__(parent, padding=4)
            self.plant_x = plant_x
            self.plant_y = plant_y
            self.lift_x = lift_x
            self.lift_y = lift_y
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

            # Handles: plant (green), lift (orange)
            self.handle_plant = self.canvas.create_oval(0, 0, 0, 0, fill="#3a7", outline="", tags=("plant",))
            self.handle_lift = self.canvas.create_oval(0, 0, 0, 0, fill="#e38", outline="", tags=("lift",))
            self._active: str | None = None

            def _to_canvas_coords(x_world: float, y_world: float) -> tuple[float, float]:
                u_y = (y_world - self.y_center) / self.y_half
                v_x = (x_world - self.x_center) / self.x_half
                cx = (-u_y * 0.5 + 0.5) * self.w
                cy = (-v_x * 0.5 + 0.5) * self.h
                return cx, cy

            def _to_world_coords(cx: float, cy: float) -> tuple[float, float]:
                u = (cx / self.w - 0.5) * 2.0
                v = (cy / self.h - 0.5) * 2.0
                xw = self.x_center - max(-1.0, min(1.0, v)) * self.x_half
                yw = self.y_center - max(-1.0, min(1.0, u)) * self.y_half
                return xw, yw

            self._to_canvas_coords = _to_canvas_coords  # type: ignore[attr-defined]
            self._to_world_coords = _to_world_coords    # type: ignore[attr-defined]

            def _redraw_handles(*_args) -> None:
                r = 6
                cx, cy = self._to_canvas_coords(float(self.plant_x.get()), float(self.plant_y.get()))
                self.canvas.coords(self.handle_plant, cx - r, cy - r, cx + r, cy + r)
                cx2, cy2 = self._to_canvas_coords(float(self.lift_x.get()), float(self.lift_y.get()))
                self.canvas.coords(self.handle_lift, cx2 - r, cy2 - r, cx2 + r, cy2 + r)
                # Optional: draw line between them
                self.canvas.delete("line")
                self.canvas.create_line(cx, cy, cx2, cy2, fill="#bbb", dash=(2, 2), tags=("line",))

            def _which_handle(cx: float, cy: float) -> str:
                p_cx, p_cy = self._to_canvas_coords(float(self.plant_x.get()), float(self.plant_y.get()))
                l_cx, l_cy = self._to_canvas_coords(float(self.lift_x.get()), float(self.lift_y.get()))
                dp = (p_cx - cx) * (p_cx - cx) + (p_cy - cy) * (p_cy - cy)
                dl = (l_cx - cx) * (l_cx - cx) + (l_cy - cy) * (l_cy - cy)
                return "plant" if dp <= dl else "lift"

            def _on_press(evt) -> None:  # type: ignore[no-redef]
                cx = max(0, min(self.w, evt.x))
                cy = max(0, min(self.h, evt.y))
                # Choose active handle based on proximity unless clicking inside a tagged handle
                items = self.canvas.find_withtag("current")
                if items and items[0] == self.handle_plant:
                    self._active = "plant"
                elif items and items[0] == self.handle_lift:
                    self._active = "lift"
                else:
                    self._active = _which_handle(cx, cy)
                _on_drag(evt)

            def _on_drag(evt) -> None:  # type: ignore[no-redef]
                if not self._active:
                    return
                cx = max(0, min(self.w, evt.x))
                cy = max(0, min(self.h, evt.y))
                xw, yw = self._to_world_coords(cx, cy)
                if self._active == "plant":
                    self.plant_x.set(xw)
                    self.plant_y.set(yw)
                else:
                    self.lift_x.set(xw)
                    self.lift_y.set(yw)

            def _on_release(_evt) -> None:
                self._active = None

            self.canvas.bind("<Button-1>", _on_press)
            self.canvas.bind("<B1-Motion>", _on_drag)
            self.canvas.bind("<ButtonRelease-1>", _on_release)

            self.plant_x.trace_add("write", _redraw_handles)
            self.plant_y.trace_add("write", _redraw_handles)
            self.lift_x.trace_add("write", _redraw_handles)
            self.lift_y.trace_add("write", _redraw_handles)
            _redraw_handles()

    class LegControl(ttk.LabelFrame):
        def __init__(self, parent: tk.Misc, label: str, init_xyz: tuple[float, float, float], vrange_xy: tuple[float, float] = (0.6, 0.6), default_step_forward: float = 0.15) -> None:
            super().__init__(parent, text=label, padding=6)
            ix, iy, iz = init_xyz
            self.base_z = float(iz)
            # Store initial values for reset
            self._init_plant_x = float(ix)
            self._init_plant_y = float(iy)
            self._init_lift_x = float(ix + default_step_forward)
            self._init_lift_y = float(iy)
            self._init_lift_h = 0.08
            # Two endpoints on ground (plant and lift)
            self.plant_x = tk.DoubleVar(value=float(ix))
            self.plant_y = tk.DoubleVar(value=float(iy))
            self.lift_x = tk.DoubleVar(value=float(ix + default_step_forward))
            self.lift_y = tk.DoubleVar(value=float(iy))
            # Lift height slider
            self.lift_h = tk.DoubleVar(value=0.08)

            pad = DualXYPad(
                self,
                self.plant_x,
                self.plant_y,
                self.lift_x,
                self.lift_y,
                x_center=float(ix),
                y_center=float(iy),
                x_half_range=float(vrange_xy[0]),
                y_half_range=float(vrange_xy[1]),
                width=200,
                height=200,
            )
            pad.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 8))

            ttk.Label(self, text="Lift height (m)").grid(row=0, column=1, sticky="w")
            lift_scale = ttk.Scale(self, from_=0.3, to=0.0, variable=self.lift_h, orient="vertical", length=160)
            lift_scale.grid(row=1, column=1, sticky="nsw")
            self.lift_readout = ttk.Label(self, text=f"{float(self.lift_h.get()):.3f}")
            self.lift_readout.grid(row=2, column=1, sticky="w")

            def _update_readout(*_a):
                self.lift_readout.configure(text=f"{float(self.lift_h.get()):.3f}")

            self.lift_h.trace_add("write", _update_readout)
            self.grid_columnconfigure(0, weight=1)

        def get_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
            return (
                (float(self.plant_x.get()), float(self.plant_y.get())),
                (float(self.lift_x.get()), float(self.lift_y.get())),
            )

        def get_lift_height(self) -> float:
            return float(self.lift_h.get())

        def reset(self) -> None:
            """Reset UI variables to initial values."""
            self.plant_x.set(self._init_plant_x)
            self.plant_y.set(self._init_plant_y)
            self.lift_x.set(self._init_lift_x)
            self.lift_y.set(self._init_lift_y)
            self.lift_h.set(self._init_lift_h)

        def set_base_z(self, new_base_z: float) -> None:
            self.base_z = float(new_base_z)

    class GlobalGaitControls(ttk.LabelFrame):
        def __init__(self, parent: tk.Misc) -> None:
            super().__init__(parent, text="Gait", padding=6)
            self.cycle_T = tk.DoubleVar(value=1.2)
            self.phase_gap = tk.DoubleVar(value=0.25)  # fraction of cycle between legs
            self.duty = tk.DoubleVar(value=0.7)        # stance fraction
            self.running = tk.BooleanVar(value=True)
            self.front_sym = tk.BooleanVar(value=False)
            self.rear_sym = tk.BooleanVar(value=False)

            # Layout sliders and readouts
            row = 0
            ttk.Label(self, text="Cycle (s)").grid(row=row, column=0, sticky="w")
            ttk.Scale(self, from_=0.2, to=5.0, variable=self.cycle_T, orient="horizontal", length=160).grid(row=row, column=1, sticky="ew", padx=(6, 0))
            self._cycle_lbl = ttk.Label(self, text=f"{float(self.cycle_T.get()):.2f}")
            self._cycle_lbl.grid(row=row, column=2, sticky="w")
            row += 1

            ttk.Label(self, text="Phase gap").grid(row=row, column=0, sticky="w")
            ttk.Scale(self, from_=0.0, to=0.5, variable=self.phase_gap, orient="horizontal", length=160).grid(row=row, column=1, sticky="ew", padx=(6, 0))
            self._gap_lbl = ttk.Label(self, text=f"{float(self.phase_gap.get()):.2f}")
            self._gap_lbl.grid(row=row, column=2, sticky="w")
            row += 1

            ttk.Label(self, text="Duty").grid(row=row, column=0, sticky="w")
            ttk.Scale(self, from_=0.5, to=0.95, variable=self.duty, orient="horizontal", length=160).grid(row=row, column=1, sticky="ew", padx=(6, 0))
            self._duty_lbl = ttk.Label(self, text=f"{float(self.duty.get()):.2f}")
            self._duty_lbl.grid(row=row, column=2, sticky="w")
            row += 1

            ttk.Checkbutton(self, text="Run", variable=self.running).grid(row=row, column=0, sticky="w")
            row += 1
            ttk.Checkbutton(self, text="Front symmetry (FL ↔ FR)", variable=self.front_sym).grid(row=row, column=0, sticky="w")
            row += 1
            ttk.Checkbutton(self, text="Rear symmetry (RL ↔ RR)", variable=self.rear_sym).grid(row=row, column=0, sticky="w")

            def _update_labels(*_a):
                self._cycle_lbl.configure(text=f"{float(self.cycle_T.get()):.2f}")
                self._gap_lbl.configure(text=f"{float(self.phase_gap.get()):.2f}")
                self._duty_lbl.configure(text=f"{float(self.duty.get()):.2f}")

            self.cycle_T.trace_add("write", _update_labels)
            self.phase_gap.trace_add("write", _update_labels)
            self.duty.trace_add("write", _update_labels)

        def reset(self) -> None:
            """Reset global gait parameters to defaults and disable symmetries."""
            self.cycle_T.set(1.2)
            self.phase_gap.set(0.25)
            self.duty.set(0.7)
            self.running.set(True)
            self.front_sym.set(False)
            self.rear_sym.set(False)

        def snapshot(self) -> dict:
            return {
                "cycle_T": float(self.cycle_T.get()),
                "phase_gap": float(self.phase_gap.get()),
                "duty": float(self.duty.get()),
                "running": bool(self.running.get()),
                "front_sym": bool(self.front_sym.get()),
                "rear_sym": bool(self.rear_sym.get()),
            }

    # Build the window
    root = tk.Tk()
    root.title("Quadruped Gait")

    # Capture initial mocap world positions (after move_mocap_to_frame)
    init_right_palm = tuple(map(float, data.mocap_pos[right_palm_mid]))
    init_left_palm = tuple(map(float, data.mocap_pos[left_palm_mid]))
    init_left_foot = tuple(map(float, data.mocap_pos[left_foot_mid]))
    init_right_foot = tuple(map(float, data.mocap_pos[right_foot_mid]))

    controls = ttk.Frame(root, padding=6)
    controls.pack(fill="both", expand=True)

    # Global gait controls across the top
    global_ctrl = GlobalGaitControls(controls)
    global_ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 8))

    # Per-leg panels: Front Left (Left Hand), Right Rear (Right Foot),
    # Front Right (Right Hand), Left Rear (Left Foot)
    fl = LegControl(controls, "Front Left (Left Hand)", init_left_palm)
    rr = LegControl(controls, "Rear Right (Right Foot)", init_right_foot)
    fr = LegControl(controls, "Front Right (Right Hand)", init_right_palm)
    rl = LegControl(controls, "Rear Left (Left Foot)", init_left_foot)

    fl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
    fr.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
    rl.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
    rr.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)

    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=1)
    controls.grid_rowconfigure(1, weight=1)
    controls.grid_rowconfigure(2, weight=1)

    # Symmetry bindings: mirror Y across midline (y -> -y), copy X
    def _bind_symmetry_pair(left: LegControl, right: LegControl, sym_var: tk.BooleanVar) -> None:
        in_update = {"flag": False}

        def _mirror(src: LegControl, dst: LegControl) -> None:
            dst.plant_x.set(float(src.plant_x.get()))
            dst.plant_y.set(-float(src.plant_y.get()))
            dst.lift_x.set(float(src.lift_x.get()))
            dst.lift_y.set(-float(src.lift_y.get()))
            dst.lift_h.set(float(src.lift_h.get()))

        def _on_left(*_a) -> None:
            if not bool(sym_var.get()) or in_update["flag"]:
                return
            in_update["flag"] = True
            _mirror(left, right)
            in_update["flag"] = False

        def _on_right(*_a) -> None:
            if not bool(sym_var.get()) or in_update["flag"]:
                return
            in_update["flag"] = True
            _mirror(right, left)
            in_update["flag"] = False

        left.plant_x.trace_add("write", _on_left)
        left.plant_y.trace_add("write", _on_left)
        left.lift_x.trace_add("write", _on_left)
        left.lift_y.trace_add("write", _on_left)
        left.lift_h.trace_add("write", _on_left)

        right.plant_x.trace_add("write", _on_right)
        right.plant_y.trace_add("write", _on_right)
        right.lift_x.trace_add("write", _on_right)
        right.lift_y.trace_add("write", _on_right)
        right.lift_h.trace_add("write", _on_right)

        def _on_toggle(*_a) -> None:
            if bool(sym_var.get()):
                in_update["flag"] = True
                _mirror(left, right)
                in_update["flag"] = False

        sym_var.trace_add("write", _on_toggle)

    # Bind front and rear pairs
    _bind_symmetry_pair(fl, fr, global_ctrl.front_sym)
    _bind_symmetry_pair(rl, rr, global_ctrl.rear_sym)

    # Global gait phase time (seconds)
    t_sim = 0.0

    # Reset handler to restore UI and mocap targets
    def _reset_all() -> None:
        global t_sim
        try:
            # Reset global parameters first to avoid symmetry mirroring during leg resets
            global_ctrl.reset()

            # Reset each leg UI to its initial state
            fl.reset()
            fr.reset()
            rl.reset()
            rr.reset()

            # Reset gait phase time
            t_sim = 0.0

            # Restore mocap target positions to initial captured values
            data.mocap_pos[right_palm_mid][0] = init_right_palm[0]
            data.mocap_pos[right_palm_mid][1] = init_right_palm[1]
            data.mocap_pos[right_palm_mid][2] = init_right_palm[2]

            data.mocap_pos[left_palm_mid][0] = init_left_palm[0]
            data.mocap_pos[left_palm_mid][1] = init_left_palm[1]
            data.mocap_pos[left_palm_mid][2] = init_left_palm[2]

            data.mocap_pos[left_foot_mid][0] = init_left_foot[0]
            data.mocap_pos[left_foot_mid][1] = init_left_foot[1]
            data.mocap_pos[left_foot_mid][2] = init_left_foot[2]

            data.mocap_pos[right_foot_mid][0] = init_right_foot[0]
            data.mocap_pos[right_foot_mid][1] = init_right_foot[1]
            data.mocap_pos[right_foot_mid][2] = init_right_foot[2]

            # Nudge UI redraw
            try:
                root.update_idletasks()
            except Exception:
                pass
        except Exception:
            pass

    # Add Reset button to global controls
    ttk.Button(global_ctrl, text="Reset", command=_reset_all).grid(row=7, column=0, sticky="w", pady=(6, 0))

    # --- JSON import/export helpers (reuse strategy from pose-tool.py) ---
    def _safe_open_json_path(parent, initialdir: Path) -> Path | None:
        try:
            zenity = shutil.which("zenity")
            if zenity:
                cmd = [
                    zenity,
                    "--file-selection",
                    "--title=Import Configuration",
                    f"--filename={str(initialdir)}/",
                    "--file-filter=JSON files | *.json *.JSON",
                ]
                proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0:
                    selected = proc.stdout.strip()
                    if selected:
                        p = Path(selected)
                        if p.exists():
                            return p
                return None
        except Exception:
            print("Zenity path failed for any reason, falling through to simple prompt")
            pass

        try:
            path_str = simpledialog.askstring(
                title="Import Configuration",
                prompt="Path to JSON file:",
                parent=parent,
            )
            if path_str:
                p = Path(path_str).expanduser()
                if p.exists():
                    return p
        except Exception:
            pass
        return None

    def _snapshot_config() -> dict:
        def leg_snapshot(ctrl: LegControl) -> dict:
            (px, py), (lx, ly) = ctrl.get_endpoints()
            return {
                "plant": [float(px), float(py)],
                "lift": [float(lx), float(ly)],
                "lift_h": float(ctrl.get_lift_height()),
                "base_z": float(ctrl.base_z),
            }

        return {
            "global": global_ctrl.snapshot(),
            "legs": {
                "FL": leg_snapshot(fl),
                "FR": leg_snapshot(fr),
                "RL": leg_snapshot(rl),
                "RR": leg_snapshot(rr),
            },
            "timestamp": datetime.now().isoformat(),
            "schema": "gait_config.v1",
        }

    def _apply_config(cfg: dict) -> None:
        global t_sim
        try:
            # Temporarily disable symmetry to avoid feedback during set
            old_front = bool(global_ctrl.front_sym.get())
            old_rear = bool(global_ctrl.rear_sym.get())
            global_ctrl.front_sym.set(False)
            global_ctrl.rear_sym.set(False)

            g = cfg.get("global", {}) or {}
            try:
                if "cycle_T" in g:
                    global_ctrl.cycle_T.set(float(g.get("cycle_T", float(global_ctrl.cycle_T.get()))))
                if "phase_gap" in g:
                    global_ctrl.phase_gap.set(float(g.get("phase_gap", float(global_ctrl.phase_gap.get()))))
                if "duty" in g:
                    global_ctrl.duty.set(float(g.get("duty", float(global_ctrl.duty.get()))))
                if "running" in g:
                    global_ctrl.running.set(bool(g.get("running", bool(global_ctrl.running.get()))))
            except Exception:
                pass

            legs = cfg.get("legs", {}) or {}

            def set_leg(name: str, ctrl: LegControl) -> None:
                s = legs.get(name) or {}
                try:
                    plant = s.get("plant")
                    if isinstance(plant, (list, tuple)) and len(plant) == 2:
                        ctrl.plant_x.set(float(plant[0]))
                        ctrl.plant_y.set(float(plant[1]))
                    lift = s.get("lift")
                    if isinstance(lift, (list, tuple)) and len(lift) == 2:
                        ctrl.lift_x.set(float(lift[0]))
                        ctrl.lift_y.set(float(lift[1]))
                    if "lift_h" in s:
                        ctrl.lift_h.set(float(s.get("lift_h", ctrl.get_lift_height())))
                    if "base_z" in s:
                        ctrl.set_base_z(float(s.get("base_z", ctrl.base_z)))
                except Exception:
                    pass

            set_leg("FL", fl)
            set_leg("FR", fr)
            set_leg("RL", rl)
            set_leg("RR", rr)

            # Restore symmetry flags from config
            try:
                if "front_sym" in g:
                    global_ctrl.front_sym.set(bool(g.get("front_sym", old_front)))
                else:
                    global_ctrl.front_sym.set(old_front)
                if "rear_sym" in g:
                    global_ctrl.rear_sym.set(bool(g.get("rear_sym", old_rear)))
                else:
                    global_ctrl.rear_sym.set(old_rear)
            except Exception:
                pass

            # Reset phase so gait starts consistently
            t_sim = 0.0
        except Exception:
            pass

    def _on_import() -> None:
        p = _safe_open_json_path(root, _HERE)
        if p is None:
            return
        try:
            cfg = json.loads(p.read_text())
            if not isinstance(cfg, dict):
                raise ValueError("Invalid configuration file format")
            _apply_config(cfg)
            try:
                messagebox.showinfo("Import", f"Loaded configuration from {p.name}")
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Import failed", str(e))
            except Exception:
                pass

    def _on_export() -> None:
        try:
            serial = _snapshot_config()
            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = _OUT_DIR / f"gait_{ts}.json"
            path.write_text(json.dumps(serial, indent=2))
            try:
                messagebox.showinfo("Export", f"Saved to {path.name}")
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Export failed", str(e))
            except Exception:
                pass

    # Import/Export buttons
    ttk.Button(global_ctrl, text="Import…", command=_on_import).grid(row=7, column=1, sticky="w", pady=(6, 0), padx=(8, 0))
    ttk.Button(global_ctrl, text="Export", command=_on_export).grid(row=7, column=2, sticky="w", pady=(6, 0), padx=(8, 0))

    def _on_export_animation() -> None:
        try:
            # Snapshot current UI state (do not modify live state during export)
            T = float(global_ctrl.cycle_T.get())
            gap = float(global_ctrl.phase_gap.get())
            duty = float(global_ctrl.duty.get())

            # Offline sampling setup: choose N so that dt = T/N for seamless looping
            target_fps = 200.0
            if T <= 1e-6:
                T = 1e-6
            num_steps = max(2, int(round(T * target_fps)))
            dt = T / float(num_steps)
            total_steps = 3 * num_steps  # export three full cycles

            # Snapshot leg parameters
            ui_snap = {}
            for leg_name, ui_ctrl in {
                "FL": fl,
                "FR": fr,
                "RL": rl,
                "RR": rr,
            }.items():
                (plant_x, plant_y), (lift_x, lift_y) = ui_ctrl.get_endpoints()
                ui_snap[leg_name] = {
                    "plant_x": float(plant_x),
                    "plant_y": float(plant_y),
                    "lift_x": float(lift_x),
                    "lift_y": float(lift_y),
                    "base_z": float(ui_ctrl.base_z),
                    "lift_h": float(ui_ctrl.get_lift_height()),
                }

            # Create an offline configuration to avoid perturbing the live viewer
            offline_cfg = mink.Configuration(model)
            offline_cfg.data.qpos[:] = configuration.data.qpos[:]
            mujoco.mj_forward(offline_cfg.model, offline_cfg.data)

            # Initialize task targets from this offline configuration
            posture_task.set_target_from_configuration(offline_cfg)
            pelvis_orientation_task.set_target_from_configuration(offline_cfg)
            pelvis_position_task.set_target_from_configuration(offline_cfg)
            torso_orientation_task.set_target_from_configuration(offline_cfg)
            left_foot_orientation_task.set_target_from_configuration(offline_cfg)
            right_foot_orientation_task.set_target_from_configuration(offline_cfg)
            left_knee_orientation_task.set_target_from_configuration(offline_cfg)
            right_knee_orientation_task.set_target_from_configuration(offline_cfg)
            left_elbow_orientation_task.set_target_from_configuration(offline_cfg)
            right_elbow_orientation_task.set_target_from_configuration(offline_cfg)

            d_off = offline_cfg.data

            # Helpers
            def _smoothstep(u: float) -> float:
                u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
                return u * u * (3.0 - 2.0 * u)

            phase_offsets = {
                "FL": 0.0,
                "RR": gap,
                "FR": 2.0 * gap,
                "RL": 3.0 * gap,
            }

            frames: list[list[float]] = []
            local_solver = "daqp"

            # Offline rollout for three cycles (continuous integration across cycles)
            for k in range(total_steps):
                t_local = float(k) * dt
                for leg in ("FL", "RR", "FR", "RL"):
                    s = ui_snap[leg]
                    plant_x = s["plant_x"]
                    plant_y = s["plant_y"]
                    lift_x = s["lift_x"]
                    lift_y = s["lift_y"]
                    base_z = s["base_z"]
                    lift_h = s["lift_h"]

                    # Local phase in [0,1)
                    phi_total = t_local / T + float(phase_offsets[leg])
                    phi = phi_total - math.floor(phi_total)

                    if phi < duty:
                        # Stance: move on ground from lift -> plant
                        ss = phi / max(1e-6, duty)
                        ssm = _smoothstep(ss)
                        tx = (1.0 - ssm) * lift_x + ssm * plant_x
                        ty = (1.0 - ssm) * lift_y + ssm * plant_y
                        tz = base_z
                    else:
                        # Swing: arc in air from plant -> lift
                        ss = (phi - duty) / max(1e-6, (1.0 - duty))
                        ssm = _smoothstep(ss)
                        tx = (1.0 - ssm) * plant_x + ssm * lift_x
                        ty = (1.0 - ssm) * plant_y + ssm * lift_y
                        tz = base_z + lift_h * math.sin(math.pi * ss)

                    if leg == "FL":
                        d_off.mocap_pos[left_palm_mid][0] = float(tx)
                        d_off.mocap_pos[left_palm_mid][1] = float(ty)
                        d_off.mocap_pos[left_palm_mid][2] = float(tz)
                    elif leg == "FR":
                        d_off.mocap_pos[right_palm_mid][0] = float(tx)
                        d_off.mocap_pos[right_palm_mid][1] = float(ty)
                        d_off.mocap_pos[right_palm_mid][2] = float(tz)
                    elif leg == "RL":
                        d_off.mocap_pos[left_foot_mid][0] = float(tx)
                        d_off.mocap_pos[left_foot_mid][1] = float(ty)
                        d_off.mocap_pos[left_foot_mid][2] = float(tz)
                    else:  # RR
                        d_off.mocap_pos[right_foot_mid][0] = float(tx)
                        d_off.mocap_pos[right_foot_mid][1] = float(ty)
                        d_off.mocap_pos[right_foot_mid][2] = float(tz)

                # Update task targets from mocap
                right_hand_task.set_target(mink.SE3.from_mocap_id(d_off, right_palm_mid))
                left_hand_task.set_target(mink.SE3.from_mocap_id(d_off, left_palm_mid))
                left_foot_task.set_target(mink.SE3.from_mocap_id(d_off, left_foot_mid))
                right_foot_task.set_target(mink.SE3.from_mocap_id(d_off, right_foot_mid))

                # IK solve and integrate on the offline configuration
                vel = mink.solve_ik(offline_cfg, tasks, dt, local_solver, 1e-1, limits=limits)
                offline_cfg.integrate_inplace(vel, dt)

                # Record qpos snapshot
                frames.append([float(x) for x in d_off.qpos])

            # Build metadata for external tools
            # Base indices (if free joint exists)
            base_meta = None
            try:
                free_qpos_addr = None
                for j in range(model.njnt):
                    if int(model.jnt_type[j]) == 0:
                        free_qpos_addr = int(model.jnt_qposadr[j])
                        break
                if free_qpos_addr is not None:
                    base_meta = {
                        "pos_indices": [free_qpos_addr + i for i in range(3)],
                        "quat_indices": [free_qpos_addr + 3 + i for i in range(4)],
                    }
            except Exception:
                pass

            # Per-joint mapping and qpos labels
            joints_meta = []
            qpos_labels = [f"qpos[{i}]" for i in range(model.nq)]

            def _set_label(i: int, name: str) -> None:
                if 0 <= i < len(qpos_labels):
                    qpos_labels[i] = name

            for j in range(model.njnt):
                try:
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
                except Exception:
                    pass

            fps = 1.0 / float(dt)

            # Serialize animation with embedded metadata
            serial = {
                "schema": "gait_animation.v1",
                "model_xml": str(_XML),
                "dt": float(dt),
                "fps": float(fps),
                "nq": int(model.nq),
                "frames": frames,
                "cycles": 3,
                "cycle_T": float(T),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "base": base_meta,
                    "joints": joints_meta,
                    "qpos_labels": qpos_labels,
                },
            }

            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = _OUT_DIR / f"animation_{ts}.json"
            path.write_text(json.dumps(serial))

            # No XML sidecar; all metadata embedded in JSON

            try:
                messagebox.showinfo("Export Animation", f"Saved animation to {path.name} ({len(frames)} frames, 3 cycles)")
            except Exception:
                pass
        except Exception as e:
            try:
                messagebox.showerror("Export Animation failed", str(e))
            except Exception:
                pass

    ttk.Button(global_ctrl, text="Export Animation", command=_on_export_animation).grid(row=7, column=3, sticky="w", pady=(6, 0), padx=(8, 0))

    solver = "daqp"

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
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
        # Leg order and phase offsets per user: FL, RR, FR, RL
        LEG_ORDER = ("FL", "RR", "FR", "RL")
        # Map legs to UI panels and mocap ids
        LEG_TO_UI = {
            "FL": fl,
            "FR": fr,
            "RL": rl,
            "RR": rr,
        }
        LEG_TO_MID = {
            "FL": left_palm_mid,
            "FR": right_palm_mid,
            "RL": left_foot_mid,
            "RR": right_foot_mid,
        }

        def _smoothstep(u: float) -> float:
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
            return u * u * (3.0 - 2.0 * u)

        while viewer.is_running():
            # Poll Tk UI (non-blocking)
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                pass

            # Advance simulation time if running
            if bool(global_ctrl.running.get()):
                t_sim += rate.dt

            T = float(global_ctrl.cycle_T.get())
            gap = float(global_ctrl.phase_gap.get())
            duty = float(global_ctrl.duty.get())
            if T <= 1e-6:
                T = 1e-6
            # Compute phase offsets per leg
            phase_offsets = {
                "FL": 0.0,
                "RR": gap,
                "FR": 2.0 * gap,
                "RL": 3.0 * gap,
            }

            # For each leg compute target
            for leg in LEG_ORDER:
                ui = LEG_TO_UI[leg]
                mid = LEG_TO_MID[leg]
                (plant_x, plant_y), (lift_x, lift_y) = ui.get_endpoints()
                base_z = ui.base_z
                lift_h = ui.get_lift_height()

                # Local phase for this leg in [0,1)
                phi_total = t_sim / T + float(phase_offsets[leg])
                phi = phi_total - math.floor(phi_total)

                if phi < duty:
                    # Stance: move on ground from lift (back) -> plant (front)
                    s = phi / max(1e-6, duty)
                    s_smooth = _smoothstep(s)
                    tx = (1.0 - s_smooth) * lift_x + s_smooth * plant_x
                    ty = (1.0 - s_smooth) * lift_y + s_smooth * plant_y
                    tz = base_z
                else:
                    # Swing: arc in air from plant (front) -> lift (back)
                    s = (phi - duty) / max(1e-6, (1.0 - duty))
                    s_smooth = _smoothstep(s)
                    tx = (1.0 - s_smooth) * plant_x + s_smooth * lift_x
                    ty = (1.0 - s_smooth) * plant_y + s_smooth * lift_y
                    tz = base_z + lift_h * math.sin(math.pi * s)

                data.mocap_pos[mid][0] = float(tx)
                data.mocap_pos[mid][1] = float(ty)
                data.mocap_pos[mid][2] = float(tz)

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


