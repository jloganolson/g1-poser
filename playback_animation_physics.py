from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time
import math

import mujoco
import mujoco.viewer


def _normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _slerp(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float], t: float) -> tuple[float, float, float, float]:
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    if dot < 0.0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        dot = -dot
    if dot > 0.9995:
        res = (
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1]),
            q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3]),
        )
        return _normalize_quat(res)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return (
        s0 * q1[0] + s1 * q2[0],
        s0 * q1[1] + s1 * q2[1],
        s0 * q1[2] + s1 * q2[2],
        s0 * q1[3] + s1 * q2[3],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Play a MuJoCo qpos animation with physics (actuator-driven joints)")
    ap.add_argument("animation", nargs="?", type=Path, help="Path to animation_*.json. If omitted, uses newest in output/")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    ap.add_argument("--fps", type=float, default=60.0, help="Viewer present frame rate (default 60)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1.0)")
    args = ap.parse_args()

    root = Path(__file__).parent
    anim_path: Path
    if args.animation is not None:
        anim_path = args.animation.expanduser().resolve()
    else:
        out_dir = (root / "output").resolve()
        if not out_dir.exists():
            print("No output directory and no animation path provided.")
            return 2
        candidates = sorted(out_dir.glob("animation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No animation_*.json files found in output/.")
            return 2
        anim_path = candidates[0]

    if not anim_path.exists():
        print(f"Animation file not found: {anim_path}")
        return 2

    anim = json.loads(anim_path.read_text())
    if not isinstance(anim, dict):
        print("Invalid animation file")
        return 2

    model_xml = anim.get("model_xml")
    dt_capture = float(anim.get("dt", 1.0 / 200.0))
    frames = anim.get("frames")
    nq = int(anim.get("nq", 0))

    if not model_xml or not isinstance(frames, list) or nq <= 0:
        print("Animation missing required fields (model_xml, frames, nq)")
        return 2

    model_xml_path = Path(model_xml)
    if not model_xml_path.exists():
        alt = (root / model_xml).resolve()
        if alt.exists():
            model_xml_path = alt
        else:
            print(f"Model XML not found: {model_xml}")
            return 2

    model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
    data = mujoco.MjData(model)

    # Build mapping from SLIDE/HINGE joints -> actuator ids (by aligned names in XML)
    joint_to_actuator: dict[int, int] = {}
    joint_qposadr: dict[int, int] = {}
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        if jtype not in (2, 3):
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
            joint_qposadr[int(j)] = int(model.jnt_qposadr[j])

    if not joint_to_actuator:
        print("Warning: No position actuators aligned to joints were found. Nothing to drive.")

    # Apply first frame fully to set initial configuration (including base)
    f0 = frames[0]
    if len(f0) < model.nq:
        print("Invalid first frame length")
        return 2
    for j in range(model.nq):
        data.qpos[j] = float(f0[j])
    mujoco.mj_forward(model, data)

    # Optional free joint quaternion address for nicer interpolation if needed in future
    free_qpos_adr = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            free_qpos_adr = int(model.jnt_qposadr[j])
            break

    present_dt = 1.0 / max(1e-6, float(args.fps))
    speed = max(1e-6, float(args.speed))

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        try:
            pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except Exception:
            pelvis_bid = -1
        try:
            if pelvis_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(pelvis_bid)
                viewer.cam.fixedcamid = -1
                viewer.cam.lookat[:] = data.xpos[pelvis_bid]
        except Exception:
            pass

        total = len(frames)
        t_capture = 0.0
        last_wall = time.perf_counter()
        last_present = last_wall
        phys_accum = 0.0

        # Print one-time info
        print(f"Loaded animation: {anim_path.name} ({total} frames @ {1.0/dt_capture:.1f} fps capture) | Model nq={model.nq}")
        print(f"Driving {len(joint_to_actuator)} joints via actuators; base and other DoFs evolve by physics.")

        while viewer.is_running():
            now = time.perf_counter()
            dt_wall = now - last_wall
            last_wall = now
            phys_accum += dt_wall

            # Step physics at model timestep resolution
            while phys_accum >= model.opt.timestep:
                # Advance capture time based on real time and playback speed
                t_capture += model.opt.timestep * speed

                # Compute frame indices and interpolation factor
                u = t_capture / dt_capture
                if args.loop:
                    i0 = int(math.floor(u)) % total
                    alpha = u - math.floor(u)
                    i1 = (i0 + 1) % total
                else:
                    if u >= total - 1:
                        i0 = total - 1
                        i1 = i0
                        alpha = 0.0
                    else:
                        i0 = int(math.floor(u))
                        alpha = u - math.floor(u)
                        i1 = i0 + 1

                f0 = frames[i0]
                f1 = frames[i1]

                # Update actuator targets for SLIDE/HINGE joints
                for j_id, aid in joint_to_actuator.items():
                    adr = joint_qposadr[j_id]
                    a = float(f0[adr])
                    b = float(f1[adr])
                    target = (1.0 - alpha) * a + alpha * b
                    data.ctrl[aid] = float(target)

                # Step physics
                mujoco.mj_step(model, data)
                phys_accum -= model.opt.timestep

            # Keep camera centered on pelvis if available
            try:
                if pelvis_bid != -1:
                    viewer.cam.lookat[0] = float(data.xpos[pelvis_bid][0])
                    viewer.cam.lookat[1] = float(data.xpos[pelvis_bid][1])
                    viewer.cam.lookat[2] = float(data.xpos[pelvis_bid][2])
            except Exception:
                pass

            viewer.sync()

            # Pace to presentation framerate
            if present_dt > 0.0:
                target_present = last_present + present_dt
                while True:
                    now2 = time.perf_counter()
                    if now2 >= target_present:
                        break
                    time.sleep(0.001)
                last_present = target_present

    return 0


if __name__ == "__main__":
    sys.exit(main())


