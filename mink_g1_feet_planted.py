from __future__ import annotations

from pathlib import Path
import math
import json
import shutil
import subprocess

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime


_HERE = Path(__file__).parent
_XML = _HERE / "g1_description" / "scene_g1_targets.xml"
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
    tv_cam_az = tk.DoubleVar(value=137.368)   # camera azimuth in degrees
    tv_cam_el = tk.DoubleVar(value=-16.395)   # camera elevation in degrees (negative looks down)
    tv_cam_dist = tk.DoubleVar(value=2.355)   # camera distance (m)

    add_scale(panel, "Torso speed (m/s)", tv_speed, 0.0, 0.40)
    add_scale(panel, "Camera azimuth (deg)", tv_cam_az, -180.0, 180.0)
    add_scale(panel, "Camera elevation (deg)", tv_cam_el, -89.0, 89.0)
    add_scale(panel, "Camera distance (m)", tv_cam_dist, 0.5, 5.0)

    # ---- Minimal 2D Foot Placement UI (front/rear with symmetry) ----
    class XYPad(ttk.Frame):
        def __init__(
            self,
            parent: tk.Misc,
            var_x: tk.DoubleVar,
            var_y: tk.DoubleVar,
            x_center: float,
            y_center: float,
            x_half_range: float,
            y_half_range: float,
            width: int = 180,
            height: int = 180,
            title: str | None = None,
        ) -> None:
            super().__init__(parent, padding=4)
            self.var_x = var_x
            self.var_y = var_y
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

            self.handle = self.canvas.create_oval(0, 0, 0, 0, fill="#3a7", outline="")
            self._active = False

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

            def _redraw(*_a) -> None:
                r = 6
                cx, cy = self._to_canvas_coords(float(self.var_x.get()), float(self.var_y.get()))
                self.canvas.coords(self.handle, cx - r, cy - r, cx + r, cy + r)

            def _on_press(evt) -> None:  # type: ignore[no-redef]
                self._active = True
                _on_drag(evt)

            def _on_drag(evt) -> None:  # type: ignore[no-redef]
                if not self._active:
                    return
                cx = max(0, min(self.w, evt.x))
                cy = max(0, min(self.h, evt.y))
                xw, yw = self._to_world_coords(cx, cy)
                self.var_x.set(xw)
                self.var_y.set(yw)

            def _on_release(_evt) -> None:
                self._active = False

            self.canvas.bind("<Button-1>", _on_press)
            self.canvas.bind("<B1-Motion>", _on_drag)
            self.canvas.bind("<ButtonRelease-1>", _on_release)

            self.var_x.trace_add("write", _redraw)
            self.var_y.trace_add("write", _redraw)
            _redraw()

    # --- Per-leg controls similar to mink_g1_pose_ik.py ---
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

            self.canvas.create_rectangle(2, 2, self.w - 2, self.h - 2, outline="#aaa")
            self.canvas.create_line(self.w // 2, 0, self.w // 2, self.h, fill="#eee")
            self.canvas.create_line(0, self.h // 2, self.w, self.h // 2, fill="#eee")

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
            self._init_plant_x = float(ix)
            self._init_plant_y = float(iy)
            self._init_lift_x = float(ix + default_step_forward)
            self._init_lift_y = float(iy)
            self._init_lift_h = 0.08
            self.plant_x = tk.DoubleVar(value=float(ix))
            self.plant_y = tk.DoubleVar(value=float(iy))
            self.lift_x = tk.DoubleVar(value=float(ix + default_step_forward))
            self.lift_y = tk.DoubleVar(value=float(iy))
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
            self.phase_gap = tk.DoubleVar(value=0.25)
            self.duty = tk.DoubleVar(value=0.7)
            self.running = tk.BooleanVar(value=True)
            self.front_sym = tk.BooleanVar(value=False)
            self.rear_sym = tk.BooleanVar(value=False)

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
            try:
                self.cycle_T.set(1.2)
                self.phase_gap.set(0.25)
                self.duty.set(0.7)
                self.running.set(True)
                self.front_sym.set(False)
                self.rear_sym.set(False)
            except Exception:
                pass

        def snapshot(self) -> dict:
            return {
                "cycle_T": float(self.cycle_T.get()),
                "phase_gap": float(self.phase_gap.get()),
                "duty": float(self.duty.get()),
                "running": bool(self.running.get()),
                "front_sym": bool(self.front_sym.get()),
                "rear_sym": bool(self.rear_sym.get()),
            }

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

    HAND_POSITION_COST = 25.0
    FOOT_POSITION_COST = 30.0

    # End-effector tasks
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

    # Snapshot initial generalized coordinates (after base shift and mocap planting)
    qpos_init = [float(x) for x in data.qpos]

    # Viewer + IK loop
    solver = "daqp"
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
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
        # Free joint base address and initial base position for pelvis motion
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

        # Gait state
        s_forward = 0.0
        gait_time = 0.0
        last_step_index = 0
        pair = 0  # 0: swing (right hand + left foot), 1: swing (left hand + right foot)

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

        # Baseline offsets captured at startup; UI deltas apply on top of these
        base_off_LH = [des_off_LH[0], des_off_LH[1]]
        base_off_RH = [des_off_RH[0], des_off_RH[1]]
        base_off_LF = [des_off_LF[0], des_off_LF[1]]
        base_off_RF = [des_off_RF[0], des_off_RF[1]]

        # Determine correct mirror sign for Y based on initial offsets
        # If left/right Y have opposite signs (usual), mirror factor is -1; otherwise +1
        try:
            mirror_front_sign = -1.0 if (des_off_LH[1] * des_off_RH[1]) < 0.0 else 1.0
        except Exception:
            mirror_front_sign = -1.0
        try:
            mirror_rear_sign = -1.0 if (des_off_LF[1] * des_off_RF[1]) < 0.0 else 1.0
        except Exception:
            mirror_rear_sign = -1.0

        # Build per-leg control UI
        controls = ttk.Frame(root, padding=6)
        controls.pack(fill="both", expand=True)
        global_ctrl = GlobalGaitControls(controls)
        global_ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 8))

        # Initialize per-leg controls using pelvis-relative offsets (x,y), with ground z
        fl = LegControl(controls, "Front Left (Left Hand)", (des_off_LH[0], des_off_LH[1], base_left_hand[2]))
        rr = LegControl(controls, "Rear Right (Right Foot)", (des_off_RF[0], des_off_RF[1], base_right_foot[2]))
        fr = LegControl(controls, "Front Right (Right Hand)", (des_off_RH[0], des_off_RH[1], base_right_hand[2]))
        rl = LegControl(controls, "Rear Left (Left Foot)", (des_off_LF[0], des_off_LF[1], base_left_foot[2]))

        fl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        fr.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        rl.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        rr.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_rowconfigure(1, weight=1)
        controls.grid_rowconfigure(2, weight=1)

        # Symmetry bindings
        def _bind_symmetry_pair(left_ctrl: LegControl, right_ctrl: LegControl, sym_var: tk.BooleanVar) -> None:
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
                _mirror(left_ctrl, right_ctrl)
                in_update["flag"] = False

            def _on_right(*_a) -> None:
                if not bool(sym_var.get()) or in_update["flag"]:
                    return
                in_update["flag"] = True
                _mirror(right_ctrl, left_ctrl)
                in_update["flag"] = False

            left_ctrl.plant_x.trace_add("write", _on_left)
            left_ctrl.plant_y.trace_add("write", _on_left)
            left_ctrl.lift_x.trace_add("write", _on_left)
            left_ctrl.lift_y.trace_add("write", _on_left)
            left_ctrl.lift_h.trace_add("write", _on_left)

            right_ctrl.plant_x.trace_add("write", _on_right)
            right_ctrl.plant_y.trace_add("write", _on_right)
            right_ctrl.lift_x.trace_add("write", _on_right)
            right_ctrl.lift_y.trace_add("write", _on_right)
            right_ctrl.lift_h.trace_add("write", _on_right)

            def _on_toggle(*_a) -> None:
                if bool(sym_var.get()):
                    in_update["flag"] = True
                    _mirror(left_ctrl, right_ctrl)
                    in_update["flag"] = False

            sym_var.trace_add("write", _on_toggle)

        _bind_symmetry_pair(fl, fr, global_ctrl.front_sym)
        _bind_symmetry_pair(rl, rr, global_ctrl.rear_sym)

        # Per-leg gait state
        t_sim = 0.0
        LEG_ORDER = ("FL", "RR", "FR", "RL")
        LEG_TO_UI = {"FL": fl, "FR": fr, "RL": rl, "RR": rr}
        LEG_TO_MID = {"FL": left_palm_mid, "FR": right_palm_mid, "RL": left_foot_mid, "RR": right_foot_mid}

        def _smoothstep(u: float) -> float:
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
            return u * u * (3.0 - 2.0 * u)

        # Initialize per-leg world-anchored state
        pelvis_x0, pelvis_y0 = pelvis_pos0[0], pelvis_pos0[1]
        LEG_STATE: dict[str, dict[str, object]] = {}
        # Initial duty parameters for stance determination at t=0
        T0 = float(global_ctrl.cycle_T.get())
        gap0 = float(global_ctrl.phase_gap.get())
        duty0 = float(global_ctrl.duty.get())
        if T0 <= 1e-6:
            T0 = 1e-6
        phase_offsets0 = {"FL": 0.0, "RR": gap0, "FR": 2.0 * gap0, "RL": 3.0 * gap0}

        base_xy = {
            "FL": (base_left_hand[0], base_left_hand[1]),
            "FR": (base_right_hand[0], base_right_hand[1]),
            "RL": (base_left_foot[0], base_left_foot[1]),
            "RR": (base_right_foot[0], base_right_foot[1]),
        }

        for leg in LEG_ORDER:
            ui = LEG_TO_UI[leg]
            (_, _), (lift_x, lift_y) = ui.get_endpoints()
            target_w0 = (pelvis_x0 + float(lift_x), pelvis_y0 + float(lift_y))
            phi0 = phase_offsets0[leg] - math.floor(phase_offsets0[leg])
            in_stance0 = bool(phi0 < duty0)
            LEG_STATE[leg] = {
                "contact_w": base_xy[leg],
                "target_w": target_w0,
                "in_stance": in_stance0,
            }

        # Reset button handler and wiring
        def _reset_all() -> None:
            global t_sim, s_forward
            try:
                # Reset UI controls
                tv_speed.set(0.10)
                tv_cam_az.set(137.368)
                tv_cam_el.set(-16.395)
                tv_cam_dist.set(2.355)
                global_ctrl.reset()
                fl.reset()
                fr.reset()
                rl.reset()
                rr.reset()

                # Reset timers and base forward displacement
                t_sim = 0.0
                s_forward = 0.0

                # Restore full generalized coordinates to startup snapshot
                for i, v in enumerate(qpos_init):
                    configuration.data.qpos[i] = float(v)
                mujoco.mj_forward(configuration.model, configuration.data)
                # Refresh all task targets to current configuration
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

                # Recompute initial stance state using current parameters
                T0 = float(global_ctrl.cycle_T.get())
                gap0 = float(global_ctrl.phase_gap.get())
                duty0 = float(global_ctrl.duty.get())
                if T0 <= 1e-6:
                    T0 = 1e-6
                phase_offsets0 = {"FL": 0.0, "RR": gap0, "FR": 2.0 * gap0, "RL": 3.0 * gap0}

                pelvis_x = float(data.xpos[pelvis_bid][0]) if pelvis_bid != -1 else 0.0
                pelvis_y = float(data.xpos[pelvis_bid][1]) if pelvis_bid != -1 else 0.0

                LEG_STATE.clear()
                for leg in LEG_ORDER:
                    ui = LEG_TO_UI[leg]
                    (plant_x, plant_y), (lift_x, lift_y) = ui.get_endpoints()
                    # On reset: set contact to plant position relative to pelvis at reset
                    contact_w0 = (pelvis_x + float(plant_x), pelvis_y + float(plant_y))
                    target_w0 = (pelvis_x + float(lift_x), pelvis_y + float(lift_y))
                    phi0 = phase_offsets0[leg] - math.floor(phase_offsets0[leg])
                    in_stance0 = bool(phi0 < duty0)
                    LEG_STATE[leg] = {
                        "contact_w": contact_w0,
                        "target_w": target_w0,
                        "in_stance": in_stance0,
                    }

                    # Reset mocap to planted contact position at ground/base z
                    mid = LEG_TO_MID[leg]
                    cx, cy = contact_w0
                    tz = ui.base_z
                    data.mocap_pos[mid][0] = float(cx)
                    data.mocap_pos[mid][1] = float(cy)
                    data.mocap_pos[mid][2] = float(tz)

                try:
                    root.update_idletasks()
                except Exception:
                    pass
            except Exception:
                pass

        ttk.Button(global_ctrl, text="Reset", command=_reset_all).grid(row=6, column=0, sticky="w", pady=(6, 0))

        # --- JSON import/export helpers (no Entry widgets; prefer Zenity) ---
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
                pass

            try:
                messagebox.showerror("Import", "Zenity not available. Please install 'zenity' or provide a JSON path via CLI.")
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
                "schema": "gait_config.v1",
                "timestamp": datetime.now().isoformat(),
                "speed": float(tv_speed.get()),
                "camera": {
                    "azimuth": float(tv_cam_az.get()),
                    "elevation": float(tv_cam_el.get()),
                    "distance": float(tv_cam_dist.get()),
                },
                "global": global_ctrl.snapshot(),
                "legs": {
                    "FL": leg_snapshot(fl),
                    "FR": leg_snapshot(fr),
                    "RL": leg_snapshot(rl),
                    "RR": leg_snapshot(rr),
                },
            }

        def _apply_config(cfg: dict) -> None:
            try:
                # Temporarily disable symmetry to avoid feedback during set
                old_front = bool(global_ctrl.front_sym.get())
                old_rear = bool(global_ctrl.rear_sym.get())
                global_ctrl.front_sym.set(False)
                global_ctrl.rear_sym.set(False)

                # Optional: speed and camera
                try:
                    if "speed" in cfg:
                        tv_speed.set(float(cfg.get("speed", float(tv_speed.get()))))
                    cam = cfg.get("camera", {}) or {}
                    if isinstance(cam, dict):
                        if "azimuth" in cam:
                            tv_cam_az.set(float(cam.get("azimuth", float(tv_cam_az.get()))))
                        if "elevation" in cam:
                            tv_cam_el.set(float(cam.get("elevation", float(tv_cam_el.get()))))
                        if "distance" in cam:
                            tv_cam_dist.set(float(cam.get("distance", float(tv_cam_dist.get()))))
                except Exception:
                    pass

                # Global gait
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

                # Legs
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

        # Import/Export buttons next to Reset
        ttk.Button(global_ctrl, text="Import…", command=_on_import).grid(row=6, column=1, sticky="w", pady=(6, 0), padx=(8, 0))
        ttk.Button(global_ctrl, text="Export", command=_on_export).grid(row=6, column=2, sticky="w", pady=(6, 0), padx=(8, 0))

        def _on_export_animation() -> None:
            try:
                # Snapshot current UI parameters
                T = float(global_ctrl.cycle_T.get())
                gap = float(global_ctrl.phase_gap.get())
                duty = float(global_ctrl.duty.get())
                torso_fwd_speed = float(tv_speed.get())

                target_fps = 200.0
                if T <= 1e-6:
                    T = 1e-6
                num_steps = max(2, int(round(T * target_fps)))
                dt = T / float(num_steps)
                total_steps = 3 * num_steps  # export three full cycles

                # Snapshot leg UI
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

                # Offline configuration to avoid disturbing the live viewer
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

                phase_offsets = {"FL": 0.0, "RR": gap, "FR": 2.0 * gap, "RL": 3.0 * gap}

                # Pelvis id and free-joint base
                try:
                    pelvis_bid = mujoco.mj_name2id(offline_cfg.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                except Exception:
                    pelvis_bid = -1

                free_qpos_addr = None
                for j in range(offline_cfg.model.njnt):
                    if int(offline_cfg.model.jnt_type[j]) == 0:
                        free_qpos_addr = int(offline_cfg.model.jnt_qposadr[j])
                        break

                base_xyz0 = None
                if free_qpos_addr is not None:
                    base_xyz0 = (
                        float(d_off.qpos[free_qpos_addr + 0]),
                        float(d_off.qpos[free_qpos_addr + 1]),
                        float(d_off.qpos[free_qpos_addr + 2]),
                    )

                # Initialize per-leg world-anchored state at t=0 similar to runtime reset
                pelvis_x0 = float(d_off.xpos[pelvis_bid][0]) if pelvis_bid != -1 else 0.0
                pelvis_y0 = float(d_off.xpos[pelvis_bid][1]) if pelvis_bid != -1 else 0.0
                LEG_STATE = {}
                for leg in ("FL", "RR", "FR", "RL"):
                    s = ui_snap[leg]
                    plant_x = s["plant_x"]
                    plant_y = s["plant_y"]
                    lift_x = s["lift_x"]
                    lift_y = s["lift_y"]
                    phi0 = phase_offsets[leg] - math.floor(phase_offsets[leg])
                    in_stance0 = bool(phi0 < duty)
                    contact_w0 = (pelvis_x0 + float(plant_x), pelvis_y0 + float(plant_y))
                    target_w0 = (pelvis_x0 + float(lift_x), pelvis_y0 + float(lift_y))
                    LEG_STATE[leg] = {
                        "contact_w": contact_w0,
                        "target_w": target_w0,
                        "in_stance": in_stance0,
                    }

                frames = []
                local_solver = "daqp"
                s_forward = 0.0

                for k in range(total_steps):
                    t_local = float(k) * dt

                    # Advance pelvis base forward if a free joint exists
                    if base_xyz0 is not None and free_qpos_addr is not None:
                        s_forward += torso_fwd_speed * dt
                        d_off.qpos[free_qpos_addr + 0] = base_xyz0[0] + s_forward
                        d_off.qpos[free_qpos_addr + 1] = base_xyz0[1]
                        d_off.qpos[free_qpos_addr + 2] = base_xyz0[2]
                        mujoco.mj_forward(offline_cfg.model, d_off)
                        # Keep pelvis/torso orientation/position tasks following the moved base
                        pelvis_orientation_task.set_target_from_configuration(offline_cfg)
                        pelvis_position_task.set_target_from_configuration(offline_cfg)
                        torso_orientation_task.set_target_from_configuration(offline_cfg)

                    # Current pelvis world position
                    pelvis_x = float(d_off.xpos[pelvis_bid][0]) if pelvis_bid != -1 else 0.0
                    pelvis_y = float(d_off.xpos[pelvis_bid][1]) if pelvis_bid != -1 else 0.0

                    for leg, mid in ("FL", left_palm_mid), ("RR", right_foot_mid), ("FR", right_palm_mid), ("RL", left_foot_mid):
                        s = ui_snap[leg]
                        plant_x = s["plant_x"]
                        plant_y = s["plant_y"]
                        lift_x = s["lift_x"]
                        lift_y = s["lift_y"]
                        base_z = s["base_z"]
                        lift_h = s["lift_h"]

                        phi_total = t_local / T + float(phase_offsets[leg])
                        phi = phi_total - math.floor(phi_total)
                        stance_now = bool(phi < duty)

                        state = LEG_STATE[leg]
                        # Transitions
                        if bool(state["in_stance"]) and not stance_now:
                            state["target_w"] = (pelvis_x + float(lift_x), pelvis_y + float(lift_y))
                        elif (not bool(state["in_stance"])) and stance_now:
                            state["contact_w"] = tuple(state["target_w"])  # type: ignore[arg-type]
                        state["in_stance"] = stance_now

                        contact_w = tuple(state["contact_w"])  # type: ignore[arg-type]
                        target_w = tuple(state["target_w"])    # type: ignore[arg-type]

                        if stance_now:
                            tx = float(contact_w[0])
                            ty = float(contact_w[1])
                            tz = float(base_z)
                        else:
                            sphi = (phi - duty) / max(1e-6, (1.0 - duty))
                            ssm = _smoothstep(sphi)
                            tx = (1.0 - ssm) * float(contact_w[0]) + ssm * float(target_w[0])
                            ty = (1.0 - ssm) * float(contact_w[1]) + ssm * float(target_w[1])
                            tz = float(base_z) + float(lift_h) * math.sin(math.pi * sphi)

                        d_off.mocap_pos[mid][0] = float(tx)
                        d_off.mocap_pos[mid][1] = float(ty)
                        d_off.mocap_pos[mid][2] = float(tz)

                    # Update IK targets from mocap and solve
                    right_hand_task.set_target(mink.SE3.from_mocap_id(d_off, right_palm_mid))
                    left_hand_task.set_target(mink.SE3.from_mocap_id(d_off, left_palm_mid))
                    left_foot_task.set_target(mink.SE3.from_mocap_id(d_off, left_foot_mid))
                    right_foot_task.set_target(mink.SE3.from_mocap_id(d_off, right_foot_mid))

                    vel = mink.solve_ik(offline_cfg, tasks, dt, local_solver, 1e-1, limits=limits)
                    offline_cfg.integrate_inplace(vel, dt)

                    frames.append([float(x) for x in d_off.qpos])

                # Recenter XY plane so the initial base X/Y become 0 across all frames
                if free_qpos_addr is not None and len(frames) > 0:
                    try:
                        x0 = float(frames[0][free_qpos_addr + 0])
                        y0 = float(frames[0][free_qpos_addr + 1])
                        if x0 != 0.0 or y0 != 0.0:
                            for frame_vals in frames:
                                frame_vals[free_qpos_addr + 0] = float(frame_vals[free_qpos_addr + 0]) - x0
                                frame_vals[free_qpos_addr + 1] = float(frame_vals[free_qpos_addr + 1]) - y0
                    except Exception:
                        pass

                # Build metadata similar to mink_g1_pose_ik.py
                base_meta = None
                try:
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
                except Exception:
                    pass

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

                try:
                    messagebox.showinfo("Export Animation", f"Saved animation to {path.name} ({len(frames)} frames, 3 cycles)")
                except Exception:
                    pass
            except Exception as e:
                try:
                    messagebox.showerror("Export Animation failed", str(e))
                except Exception:
                    pass

        ttk.Button(global_ctrl, text="Export Animation", command=_on_export_animation).grid(row=6, column=3, sticky="w", pady=(6, 0), padx=(8, 0))

        while viewer.is_running():
            # Pump Tk UI
            try:
                root.update_idletasks()
                root.update()
            except Exception:
                pass

            # Refresh parameters from UI each frame
            torso_fwd_speed = float(tv_speed.get())

            gait_time += rate.dt

            # Advance torso forward
            if base_xyz0 is not None and free_qpos_addr is not None:
                s_forward += torso_fwd_speed * rate.dt
                configuration.data.qpos[free_qpos_addr + 0] = base_xyz0[0] + s_forward
                configuration.data.qpos[free_qpos_addr + 1] = base_xyz0[1]
                configuration.data.qpos[free_qpos_addr + 2] = base_xyz0[2]
                mujoco.mj_forward(configuration.model, configuration.data)
                pelvis_orientation_task.set_target_from_configuration(configuration)
                pelvis_position_task.set_target_from_configuration(configuration)
                torso_orientation_task.set_target_from_configuration(configuration)

            # Per-leg sequential gait with symmetry options
            if bool(global_ctrl.running.get()):
                t_sim += rate.dt

            T = float(global_ctrl.cycle_T.get())
            gap = float(global_ctrl.phase_gap.get())
            duty = float(global_ctrl.duty.get())
            if T <= 1e-6:
                T = 1e-6
            phase_offsets = {"FL": 0.0, "RR": gap, "FR": 2.0 * gap, "RL": 3.0 * gap}

            # Current pelvis world position
            pelvis_x = float(data.xpos[pelvis_bid][0]) if pelvis_bid != -1 else 0.0
            pelvis_y = float(data.xpos[pelvis_bid][1]) if pelvis_bid != -1 else 0.0

            for leg in LEG_ORDER:
                ui = LEG_TO_UI[leg]
                mid = LEG_TO_MID[leg]
                state = LEG_STATE[leg]
                (plant_x, plant_y), (lift_x, lift_y) = ui.get_endpoints()
                base_z = ui.base_z
                lift_h = ui.get_lift_height()

                phi_total = t_sim / T + float(phase_offsets[leg])
                phi = phi_total - math.floor(phi_total)
                stance_now = bool(phi < duty)

                # Transitions
                if bool(state["in_stance"]) and not stance_now:
                    # Entering swing: freeze a world target ahead based on current pelvis and UI lift
                    state["target_w"] = (pelvis_x + float(lift_x), pelvis_y + float(lift_y))
                elif (not bool(state["in_stance"])) and stance_now:
                    # Entering stance: lock contact at the previously set target
                    state["contact_w"] = tuple(state["target_w"])  # type: ignore[arg-type]

                state["in_stance"] = stance_now

                contact_w = tuple(state["contact_w"])  # type: ignore[arg-type]
                target_w = tuple(state["target_w"])    # type: ignore[arg-type]

                if stance_now:
                    tx = float(contact_w[0])
                    ty = float(contact_w[1])
                    tz = base_z
                else:
                    s = (phi - duty) / max(1e-6, (1.0 - duty))
                    s_smooth = _smoothstep(s)
                    tx = (1.0 - s_smooth) * float(contact_w[0]) + s_smooth * float(target_w[0])
                    ty = (1.0 - s_smooth) * float(contact_w[1]) + s_smooth * float(target_w[1])
                    tz = base_z + lift_h * math.sin(math.pi * s)

                data.mocap_pos[mid][0] = float(tx)
                data.mocap_pos[mid][1] = float(ty)
                data.mocap_pos[mid][2] = float(tz)

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


