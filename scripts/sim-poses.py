import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import tkinter as tk
from tkinter import ttk


# Hardcoded poses JSON path (no CLI argument)
POSES_JSON_PATH: Path = Path("output-example.json")

# Optional: start from a specific pose in the JSON (1-based index). Set to None to disable.
START_POSE_INDEX: int | None = 2  # e.g., set to 1 to start at "Pose 1"

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
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
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0, 0.0
    return w / n, x / n, y / n, z / n


def quat_slerp(q0: tuple[float, float, float, float], q1: tuple[float, float, float, float], t: float) -> tuple[float, float, float, float]:
    w0, x0, y0, z0 = quat_normalize(q0)
    w1, x1, y1, z1 = quat_normalize(q1)
    dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    if dot < 0.0:
        w1, x1, y1, z1 = -w1, -x1, -y1, -z1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
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


def load_poses(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    poses: list[dict] = []
    for item in data.get("poses", []):
        base_rpy = item.get("base_rpy")
        joints_raw = item.get("joints", {})
        # Support new format with joint-name keys; also accept legacy index keys
        joints: dict[int | str, float] = {}
        for k, v in joints_raw.items():
            try:
                # If key is numeric (e.g., "12"), keep as int for backward compatibility
                k_int = int(k)  # type: ignore[arg-type]
                joints[k_int] = float(v)
            except (ValueError, TypeError):
                joints[str(k)] = float(v)
        poses.append({"base_rpy": base_rpy, "joints": joints})
    return poses


def find_latest_json(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("poses_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict, free_qpos_addr: int | None, face_down_fallback: bool = True) -> None:
    # Orientation is not adjusted when applying poses (simulate real-test behavior).
    # Hinge/slide joints from pose (ids are MuJoCo joint ids)
    for j_key, val in pose.get("joints", {}).items():
        # Resolve joint id from either int id or joint name
        if isinstance(j_key, int):
            j_id = j_key
        else:
            try:
                j_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j_key)))
            except Exception:
                continue
        # Skip FREE/BALL joints; only SLIDE/HINGE have nq=1
        if model.jnt_type[int(j_id)] not in (2, 3):
            continue
        adr = model.jnt_qposadr[int(j_id)]
        data.qpos[adr] = float(val)


class PosesUI:
    def __init__(self, root: tk.Tk, poses: list[dict], on_apply):
        self.root = root
        self.poses = poses
        self.on_apply = on_apply

        frame = ttk.LabelFrame(root, text="Poses", padding=8)
        frame.pack(fill="x", padx=8, pady=8)

        inner = ttk.Frame(frame)
        inner.pack(fill="x")

        self.listbox = tk.Listbox(inner, height=8, exportselection=False)
        self.listbox.grid(row=0, column=0, rowspan=4, sticky="nsew")

        btns = ttk.Frame(inner)
        btns.grid(row=0, column=1, sticky="nw", padx=(8, 0))

        ttk.Button(btns, text="Apply", command=self.apply_selected).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Quit", command=root.quit).grid(row=1, column=0, sticky="ew", pady=2)

        inner.grid_columnconfigure(0, weight=1)

        self.refresh(select=0)

    def refresh(self, select: int | None = None) -> None:
        self.listbox.delete(0, tk.END)
        for i, _ in enumerate(self.poses, start=1):
            self.listbox.insert(tk.END, f"Pose {i}")
        if select is not None and 0 <= select < len(self.poses):
            self.listbox.selection_set(select)
            self.listbox.see(select)

    def selected_index(self) -> int | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        return int(sel[0])

    def apply_selected(self) -> None:
        idx = self.selected_index()
        if idx is None:
            return
        self.on_apply(self.poses[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview JSON poses in MuJoCo viewer (with physics)")
    parser.add_argument("--scene", type=Path, default=Path("./g1_description/scene_torso_collision_test.xml"), help="MuJoCo XML scene path")
    parser.add_argument("--lerp-duration", type=float, default=0.5, help="Lerp duration seconds when applying a pose")
    # Start configuration
    parser.add_argument("--start-face-down", action=argparse.BooleanOptionalAction, default=True, help="If a FREE joint exists, start the base face-down (roll=pi)")
    parser.add_argument("--drop-height", type=float, default=0.5, help="Initial z height to drop from when starting face-down (meters)")
    args = parser.parse_args()

    poses_path = POSES_JSON_PATH

    poses = load_poses(poses_path)
    if not poses:
        raise SystemExit(f"No poses found in {poses_path}")

    model = mujoco.MjModel.from_xml_path(args.scene.as_posix())
    data = mujoco.MjData(model)

    # Find a FREE joint if present
    free_qpos_addr: int | None = None
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:
            free_qpos_addr = int(model.jnt_qposadr[j])
            break

    # Optionally start face-down with a small drop if a FREE joint is present
    if args.start_face_down and free_qpos_addr is not None:
        # Layout for FREE joint qpos: [x, y, z, qw, qx, qy, qz]
        data.qpos[free_qpos_addr + 0] = 0.0
        data.qpos[free_qpos_addr + 1] = 0.0
        data.qpos[free_qpos_addr + 2] = float(max(0.0, args.drop_height))
        qw, qx, qy, qz = rpy_to_quat( 0.0,math.pi/2, 0.0)
        data.qpos[free_qpos_addr + 3] = qw
        data.qpos[free_qpos_addr + 4] = qx
        data.qpos[free_qpos_addr + 5] = qy
        data.qpos[free_qpos_addr + 6] = qz

    # Optionally set joint configuration from a pose in the JSON (applied before first forward)
    initial_select_idx: int | None = None
    if START_POSE_INDEX is not None and len(poses) > 0:
        idx0 = max(0, min(len(poses) - 1, int(START_POSE_INDEX) - 1))
        apply_pose(model, data, poses[idx0], free_qpos_addr, face_down_fallback=False)
        initial_select_idx = idx0
    else:
        initial_select_idx = 0

    # Compute kinematics for the initial configuration
    mujoco.mj_forward(model, data)

    print(f"Loaded {len(poses)} poses from {poses_path.name}")

    # Build a minimal Tk UI for selecting and applying poses
    root = tk.Tk()
    root.title("G1 Pose Tester (Physics)")

    # Lerp state
    lerp = {"running": False, "t0": 0.0, "duration": float(args.lerp_duration), "start": None, "target": None}

    def _capture_current() -> dict:
        # Capture all hinge/slide joints (no orientation changes)
        joints = {}
        for j in range(model.njnt):
            if model.jnt_type[j] in (2, 3):
                adr = model.jnt_qposadr[j]
                joints[int(j)] = float(data.qpos[adr])
        return {"joints": joints}

    def on_apply(pose: dict) -> None:
        start = _capture_current()
        # Build target: keep current by default, override with provided joints
        target_joints = dict(start["joints"])  # type: ignore[index]
        for j_key, val in pose.get("joints", {}).items():
            # Resolve joint id from either int id or joint name
            if isinstance(j_key, int):
                j_id = int(j_key)
            else:
                try:
                    j_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(j_key)))
                except Exception:
                    continue
            if model.jnt_type[j_id] in (2, 3):
                target_joints[j_id] = float(val)
        lerp["running"] = True
        lerp["t0"] = time.time()
        lerp["start"] = start
        lerp["target"] = {"joints": target_joints}

    ui = PosesUI(root, poses, on_apply)
    if initial_select_idx is not None:
        ui.refresh(select=initial_select_idx)

    # Build mapping from joint id -> actuator id (names are aligned in XML)
    joint_to_actuator: dict[int, int] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] not in (2, 3):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname is None:
            continue
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        except Exception:
            aid = -1
        if aid != -1:
            joint_to_actuator[int(j)] = int(aid)

    controlled_joint_ids = sorted(joint_to_actuator.keys())

    # Hold targets initialized at current qpos so nothing moves
    hold_targets: dict[int, float] = {}
    for j in controlled_joint_ids:
        adr = model.jnt_qposadr[j]
        hold_targets[j] = float(data.qpos[adr])

    # Timer for periodic joint value printing
    last_print_time = time.time()

    with viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as v:
        # Physics stepping loop with Tk UI polling
        while v.is_running():
            # Process UI events
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                break

            if lerp["running"] and lerp["start"] is not None and lerp["target"] is not None:
                t = (time.time() - float(lerp["t0"])) / float(lerp["duration"])
                if t >= 1.0:
                    t = 1.0
                start_pose = lerp["start"]  # type: ignore[assignment]
                target_pose = lerp["target"]  # type: ignore[assignment]

                # Update hold targets via linear interpolation
                for j_id, start_val in start_pose["joints"].items():  # type: ignore[index]
                    if model.jnt_type[int(j_id)] not in (2, 3):
                        continue
                    end_val = float(target_pose["joints"].get(int(j_id), start_val))  # type: ignore[index]
                    hold_targets[int(j_id)] = float(start_val) + (end_val - float(start_val)) * float(t)

                if t >= 1.0:
                    # Lock targets exactly to the final target and keep commanding them
                    for j_id, end_val in target_pose["joints"].items():  # type: ignore[index]
                        if model.jnt_type[int(j_id)] in (2, 3):
                            hold_targets[int(j_id)] = float(end_val)
                    lerp["running"] = False

            # Apply PD position targets to actuators every tick to hold pose
            for j in controlled_joint_ids:
                aid = joint_to_actuator[j]
                data.ctrl[aid] = hold_targets[j]

            # Physics step and render
            mujoco.mj_step(model, data)
            v.sync()

            time.sleep(max(0.0, model.opt.timestep * 0.5))

            # Print joint qpos every 1 second
            now = time.time()
            if now - last_print_time >= 1.0:
                print("Joint qpos (actual -> target):")
                for j in range(model.njnt):
                    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    name = jname if jname is not None else f"joint_{j}"
                    adr = model.jnt_qposadr[j]
                    jtype = int(model.jnt_type[j])
                    # FREE=0 -> 7, BALL=1 -> 4, SLIDE/HINGE -> 1
                    count = 7 if jtype == 0 else (4 if jtype == 1 else 1)
                    if count == 1:
                        actual_val = float(data.qpos[adr])
                        target_val = hold_targets.get(int(j)) if int(j) in hold_targets else None
                        tstr = f"{target_val:.6f}" if isinstance(target_val, float) else "-"
                        print(f"  {name}: {actual_val:.6f} -> {tstr}")
                    else:
                        actual_list = [float(x) for x in data.qpos[adr: adr + count]]
                        # No scalar target for multi-DoF joints
                        print(f"  {name}: {actual_list} -> -")
                last_print_time = now

    try:
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()


