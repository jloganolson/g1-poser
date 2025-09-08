import math
import time
import json
from datetime import datetime
from pathlib import Path
from loop_rate_limiters import RateLimiter
import shutil
import subprocess

import mujoco
import mujoco.viewer

# --- NEW: tiny Tk UI for joint sliders ---
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog

_HERE = Path(__file__).parent
_XML = _HERE / "g1_description" / "g1.xml"
_AUTOSAVE = _HERE / ".poses_autosave.json"


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


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_slerp(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float], t: float) -> tuple[float, float, float, float]:
    """Spherical linear interpolation for quaternions in (w, x, y, z)."""
    w0, x0, y0, z0 = quat_normalize(q0)
    w1, x1, y1, z1 = quat_normalize(q1)

    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    if dot < 0.0:
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Very close - use linear interpolation and normalize
        w = w0 + t * (w1 - w0)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        return quat_normalize((w, x, y, z))

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    w = s0 * w0 + s1 * w1
    x = s0 * x0 + s1 * x1
    y = s0 * y0 + s1 * y1
    z = s0 * z0 + s1 * z1
    return w, x, y, z


def quat_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw (radians), ZYX order."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def build_joint_ui(root, model):
    """
    Create sliders for base orientation and all hinge/slide joints.
    Returns: (joint_vars_by_id, base_vars or None, mirror_var)
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
            ttk.Button(frame, text="↺", width=2, command=lambda k=key: base_vars[k].set(0.0)).grid(row=row, column=2, sticky="w")
            row += 1
        def reset_base():
            base_vars["roll"].set(0.0)
            base_vars["pitch"].set(0.0)
            base_vars["yaw"].set(0.0)
        ttk.Button(frame, text="Reset base", command=reset_base).grid(row=row, column=0, pady=(6, 10), sticky="w")
        row += 1

    ttk.Label(frame, text="Joint Sliders (hinge/slide)", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
    row += 1

    # Mirror checkbox
    mirror_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="Mirror left/right", variable=mirror_var).grid(row=row, column=0, sticky="w", pady=(0, 4))
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
        ttk.Button(frame, text="↺", width=2, command=lambda var=v: var.set(0.0)).grid(row=row, column=2, sticky="w")
        row += 1

    # Reset button
    def reset_all():
        for v in vars_by_joint.values():
            v.set(0.0)
    ttk.Button(frame, text="Reset pose", command=reset_all).grid(row=row, column=0, pady=(8,0), sticky="w")

    # Make the second column expand
    frame.grid_columnconfigure(1, weight=1)
    return vars_by_joint, base_vars, mirror_var


def _safe_open_json_path(parent, initialdir: Path) -> Path | None:
    """Prefer an external file picker (zenity) to avoid X11/GLFW/Tk conflicts.

    Falls back to a simple text prompt for a path if zenity is unavailable.
    Returns a Path or None if cancelled.
    """
    try:
        zenity = shutil.which("zenity")
        if zenity:
            # Use zenity to select a .json file, starting in the initialdir
            cmd = [
                zenity,
                "--file-selection",
                "--title=Import Poses",
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
        # If zenity path failed for any reason, fall through to simple prompt
        pass

    # Fallback: simple text input to avoid Tk filedialog
    try:
        path_str = simpledialog.askstring(
            title="Import Poses",
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


class PosesController:
    def __init__(self, root, model, joint_vars: dict[int, tk.DoubleVar], base_vars: dict[str, tk.DoubleVar] | None, on_activate):
        self.model = model
        self.joint_vars = joint_vars
        self.base_vars = base_vars
        self.on_activate = on_activate
        self.parent = root

        self.poses: list[dict] = []

        frame = ttk.LabelFrame(root, text="Poses", padding=8)
        frame.pack(fill="x", padx=8, pady=8)

        inner = ttk.Frame(frame)
        inner.pack(fill="x")

        self.listbox = tk.Listbox(inner, height=6, exportselection=False)
        self.listbox.grid(row=0, column=0, rowspan=6, sticky="nsew")

        btns = ttk.Frame(inner)
        btns.grid(row=0, column=1, sticky="nw", padx=(8, 0))

        ttk.Button(btns, text="Add", command=self.add_pose).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Update", command=self.update_pose).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Delete", command=self.delete_pose).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Activate", command=self.activate_selected).grid(row=3, column=0, sticky="ew", pady=2)

        ttk.Separator(btns, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=(6, 4))
        ttk.Button(btns, text="Import…", command=self.import_poses).grid(row=5, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Export", command=self.export_poses).grid(row=6, column=0, sticky="ew", pady=2)

        inner.grid_columnconfigure(0, weight=1)

    def snapshot_current(self) -> dict:
        base_rpy = None
        if self.base_vars is not None:
            base_rpy = (
                float(self.base_vars["roll"].get()),
                float(self.base_vars["pitch"].get()),
                float(self.base_vars["yaw"].get()),
            )
        joints = {int(j_id): float(var.get()) for j_id, var in self.joint_vars.items()}
        return {"base_rpy": base_rpy, "joints": joints}

    def add_pose(self) -> None:
        pose = self.snapshot_current()
        self.poses.append(pose)
        self.refresh()
        self.autosave()

    def update_pose(self) -> None:
        sel = self._selected_index()
        if sel is None:
            return
        self.poses[sel] = self.snapshot_current()
        self.refresh(select=sel)
        self.autosave()

    def delete_pose(self) -> None:
        sel = self._selected_index()
        if sel is None:
            return
        if not messagebox.askyesno("Delete pose", f"Delete Pose {sel + 1}?"):
            return
        del self.poses[sel]
        self.refresh(select=min(sel, len(self.poses) - 1))
        self.autosave()

    def activate_selected(self) -> None:
        sel = self._selected_index()
        if sel is None:
            return
        self.on_activate(self.poses[sel])

    def import_poses(self) -> None:
        # Use a safer, external dialog if possible to avoid X11 threading issues
        p = _safe_open_json_path(self.parent, _HERE)
        if p is None:
            return  # cancelled
        try:
            self.load_from_file(p)
            self.autosave()
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def export_poses(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _HERE / f"poses_{ts}.json"
        try:
            self.save_to_file(path)
            messagebox.showinfo("Exported", f"Saved to {path.name}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _selected_index(self) -> int | None:
        try:
            sel = self.listbox.curselection()
            if not sel:
                return None
            return int(sel[0])
        except Exception:
            return None

    def refresh(self, select: int | None = None) -> None:
        self.listbox.delete(0, tk.END)
        for i, _ in enumerate(self.poses, start=1):
            self.listbox.insert(tk.END, f"Pose {i}")
        if select is not None and 0 <= select < len(self.poses):
            self.listbox.selection_set(select)
            self.listbox.see(select)

    def autosave(self) -> None:
        try:
            self.save_to_file(_AUTOSAVE)
        except Exception:
            pass

    def save_to_file(self, path: Path) -> None:
        serial = {
            "poses": [
                {
                    "base_rpy": pose["base_rpy"],
                    "joints": {str(k): float(v) for k, v in pose["joints"].items()},
                }
                for pose in self.poses
            ]
        }
        path.write_text(json.dumps(serial, indent=2))

    def load_from_file(self, path: Path) -> None:
        data = json.loads(path.read_text())
        poses = []
        for item in data.get("poses", []):
            base_rpy = tuple(item.get("base_rpy")) if item.get("base_rpy") is not None else None
            joints = {int(k): float(v) for k, v in item.get("joints", {}).items()}
            poses.append({"base_rpy": base_rpy, "joints": joints})
        self.poses = poses
        self.refresh(select=0 if self.poses else None)


def main() -> None:
    """Manual posing with a Tkinter slider UI + MuJoCo viewer (no physics)."""
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Optional: disable gravity if your model drops when not actuated
    # model.opt.gravity = (0.0, 0.0, 0.0)

    # Build the Tk UI before opening the viewer.
    root = tk.Tk()
    root.title("G1 Joint Posing")
    joint_vars, base_vars, mirror_var = build_joint_ui(root, model)

    # Precompute qpos addresses for each joint we control.
    # For hinge/slide joints, nq = 1, so it's just a single address.
    jpos_addr = {j_id: model.jnt_qposadr[j_id] for j_id in joint_vars.keys()}

    # Locate the free joint qpos address if available
    free_qpos_addr = None
    for j_id in range(model.njnt):
        if model.jnt_type[j_id] == 0:
            free_qpos_addr = model.jnt_qposadr[j_id]
            break

    # State for activation lerp
    lerp_active = {"running": False, "t0": 0.0, "duration": 0.3, "start": None, "target": None}

    # --- Mirror wiring ---
    # Map joint names to ids and build left/right mirror pairs.
    joint_name_to_id: dict[str, int] = {}
    for j_id in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if nm is not None:
            joint_name_to_id[nm] = j_id

    # Bases that should invert sign when mirroring
    invert_bases = {
        "hip_roll",
        "hip_yaw",
        "shoulder_roll",
        "ankle_roll",
        "shoulder_yaw",
        "wrist_roll",
    }

    # Build id->(peer_id, invert) mapping in both directions for all left/right pairs
    mirror_map: dict[int, tuple[int, bool]] = {}
    for name, a_id in joint_name_to_id.items():
        if not name.endswith("_joint"):
            continue
        if name.startswith("left_"):
            base = name[len("left_"):]
            right_name = "right_" + base
            b_id = joint_name_to_id.get(right_name)
            if b_id is None:
                continue
            if a_id not in joint_vars or b_id not in joint_vars:
                continue
            base_no_joint = base[:-len("_joint")] if base.endswith("_joint") else base
            invert = base_no_joint in invert_bases
            mirror_map[a_id] = (b_id, invert)
            mirror_map[b_id] = (a_id, invert)

    # Recursion guard to prevent feedback when setting mirrored values
    _mirroring = {"active": False}

    def _on_joint_var_changed(joint_id: int, *args) -> None:
        # Skip during lerp or if mirroring disabled or if already mirroring
        if not mirror_var.get() or lerp_active["running"] or _mirroring["active"]:
            return
        peer = mirror_map.get(joint_id)
        if not peer:
            return
        peer_id, invert = peer
        src_val = float(joint_vars[joint_id].get())
        dst_val = -src_val if invert else src_val
        try:
            _mirroring["active"] = True
            joint_vars[peer_id].set(dst_val)
        finally:
            _mirroring["active"] = False

    # Attach traces
    for j_id, var in joint_vars.items():
        var.trace_add("write", lambda *_args, jid=j_id: _on_joint_var_changed(jid))

    def capture_current_pose_dict() -> dict:
        base_rpy_now = None
        if base_vars is not None:
            base_rpy_now = (
                float(base_vars["roll"].get()),
                float(base_vars["pitch"].get()),
                float(base_vars["yaw"].get()),
            )
        joints_now = {int(j_id): float(var.get()) for j_id, var in joint_vars.items()}
        return {"base_rpy": base_rpy_now, "joints": joints_now}

    # Poses controller callback
    poses_controller = None  # will be assigned after class definition

    def on_activate(pose: dict) -> None:
        # Start a lerp from current to target pose over fixed duration
        start = capture_current_pose_dict()
        lerp_active["running"] = True
        lerp_active["t0"] = time.time()
        lerp_active["start"] = start
        lerp_active["target"] = pose

    poses_controller = PosesController(root, model, joint_vars, base_vars, on_activate)

    rate = RateLimiter(frequency=60.0, warn=False)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            # 1) Poll the Tk UI without blocking
            root.update_idletasks()
            root.update()

            # 2) If a lerp is active, drive sliders toward target
            if lerp_active["running"] and lerp_active["start"] is not None and lerp_active["target"] is not None:
                t = (time.time() - lerp_active["t0"]) / lerp_active["duration"]
                if t >= 1.0:
                    t = 1.0
                start_pose = lerp_active["start"]
                target_pose = lerp_active["target"]

                # Base orientation via quaternion slerp
                if base_vars is not None and free_qpos_addr is not None:
                    sr, sp, sy = start_pose["base_rpy"] if start_pose["base_rpy"] is not None else (0.0, 0.0, 0.0)
                    tr, tp, ty = target_pose["base_rpy"] if target_pose["base_rpy"] is not None else (0.0, 0.0, 0.0)
                    q0 = rpy_to_quat(sr, sp, sy)
                    q1 = rpy_to_quat(tr, tp, ty)
                    qw, qx, qy, qz = quat_slerp(q0, q1, t)
                    # Reflect RPY sliders for user visibility
                    rr, pp, yy = quat_to_rpy(qw, qx, qy, qz)
                    base_vars["roll"].set(rr)
                    base_vars["pitch"].set(pp)
                    base_vars["yaw"].set(yy)

                # Joints linear
                for j_id, var in joint_vars.items():
                    s = float(start_pose["joints"].get(int(j_id), 0.0))
                    e = float(target_pose["joints"].get(int(j_id), 0.0))
                    var.set(s + (e - s) * t)

                if t >= 1.0:
                    lerp_active["running"] = False

            # 3) Update base orientation quaternion from RPY sliders
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

            # 4) Read hinge/slide slider values into qpos
            for j_id, var in joint_vars.items():
                adr = jpos_addr[j_id]
                data.qpos[adr] = var.get()

            # 5) Recompute kinematics and redraw
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
