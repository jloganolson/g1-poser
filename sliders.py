import math
import time
from pathlib import Path
from loop_rate_limiters import RateLimiter

import mujoco
import mujoco.viewer

# --- NEW: tiny Tk UI for joint sliders ---
import tkinter as tk
from tkinter import ttk

_HERE = Path(__file__).parent
_XML = _HERE / "g1_description" / "g1.xml"


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll-pitch-yaw (radians) to MuJoCo quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # ZYX rotation order
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def build_joint_ui(root, model):
    """
    Create sliders for base orientation and all hinge/slide joints.
    Returns: (joint_vars_by_id, base_vars or None)
    """
    vars_by_joint = {}
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)

    row = 0

    # Base orientation: only if a FREE joint exists
    base_vars = None
    has_free = any(model.jnt_type[j_id] == 0 for j_id in range(model.njnt))
    if has_free:
        base_vars = {"roll": tk.DoubleVar(value=0.0), "pitch": tk.DoubleVar(value=0.0), "yaw": tk.DoubleVar(value=0.0)}
        ttk.Label(frame, text="Base orientation (roll, pitch, yaw)", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
        row += 1
        for label, key in (("Roll (x)", "roll"), ("Pitch (y)", "pitch"), ("Yaw (z)", "yaw")):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Scale(frame, from_=-math.pi, to=math.pi, variable=base_vars[key], orient="horizontal", length=280).grid(row=row, column=1, sticky="ew", padx=6, pady=2)
            row += 1
        def reset_base():
            base_vars["roll"].set(0.0)
            base_vars["pitch"].set(0.0)
            base_vars["yaw"].set(0.0)
        ttk.Button(frame, text="Reset base", command=reset_base).grid(row=row, column=0, pady=(6, 10), sticky="w")
        row += 1

    ttk.Label(frame, text="Joint Sliders (hinge/slide)", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
    row += 1
    for j_id in range(model.njnt):
        jtype = model.jnt_type[j_id]
        # 0=FREE, 1=BALL, 2=SLIDE, 3=HINGE
        if jtype not in (2, 3):
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or f"joint_{j_id}"
        # Range: if not defined, fall back to a reasonable default
        lo, hi = model.jnt_range[j_id]
        if lo == 0.0 and hi == 0.0:
            # No explicit limits in XML -> pick a symmetric range
            hi = math.radians(180.0) if jtype == 3 else 0.5  # hinge vs slide
            lo = -hi

        v = tk.DoubleVar(value=0.0)
        vars_by_joint[j_id] = v

        ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(frame, from_=lo, to=hi, variable=v, orient="horizontal", length=280)
        scale.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        row += 1

    # Reset button
    def reset_all():
        for v in vars_by_joint.values():
            v.set(0.0)
    ttk.Button(frame, text="Reset pose", command=reset_all).grid(row=row, column=0, pady=(8,0), sticky="w")

    # Make the second column expand
    frame.grid_columnconfigure(1, weight=1)
    return vars_by_joint, base_vars


def main() -> None:
    """Manual posing with a Tkinter slider UI + MuJoCo viewer (no physics)."""
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Optional: disable gravity if your model drops when not actuated
    # model.opt.gravity = (0.0, 0.0, 0.0)

    # Build the Tk UI before opening the viewer.
    root = tk.Tk()
    root.title("G1 Joint Posing")
    joint_vars, base_vars = build_joint_ui(root, model)

    # Precompute qpos addresses for each joint we control.
    # For hinge/slide joints, nq = 1, so it's just a single address.
    jpos_addr = {j_id: model.jnt_qposadr[j_id] for j_id in joint_vars.keys()}

    # Locate the free joint qpos address if available
    free_qpos_addr = None
    for j_id in range(model.njnt):
        if model.jnt_type[j_id] == 0:
            free_qpos_addr = model.jnt_qposadr[j_id]
            break

    rate = RateLimiter(frequency=60.0, warn=False)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            # 1) Poll the Tk UI without blocking
            root.update_idletasks()
            root.update()

            # 2) Update base orientation quaternion from RPY sliders
            if base_vars is not None and free_qpos_addr is not None:
                roll = float(base_vars["roll"].get())
                pitch = float(base_vars["pitch"].get())
                yaw = float(base_vars["yaw"].get())
                qw, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)
                # Layout: [x, y, z, qw, qx, qy, qz]
                data.qpos[free_qpos_addr + 3] = qw
                data.qpos[free_qpos_addr + 4] = qx
                data.qpos[free_qpos_addr + 5] = qy
                data.qpos[free_qpos_addr + 6] = qz

            # 3) Read hinge/slide slider values into qpos
            for j_id, var in joint_vars.items():
                adr = jpos_addr[j_id]
                data.qpos[adr] = var.get()

            # 4) Recompute kinematics and redraw
            mujoco.mj_forward(model, data)
            viewer.sync()

            rate.sleep()

    # If the MuJoCo window closes, also close the Tk window
    try:
        root.destroy()
    except:
        pass


if __name__ == "__main__":
    main()
