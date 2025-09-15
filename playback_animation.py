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
    # Ensure shortest path
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    if dot < 0.0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        dot = -dot
    # If close, fall back to lerp
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
    ap = argparse.ArgumentParser(description="Play back a MuJoCo qpos animation exported by mink_g1_pose_ik.py")
    ap.add_argument("animation", nargs="?", type=Path, help="Path to animation_*.json. If omitted, uses newest in output/")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    ap.add_argument("--fps", type=float, default=60.0, help="Presentation frame rate (default 60)")
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
    dt = float(anim.get("dt", 1.0 / 200.0))
    frames = anim.get("frames")
    nq = int(anim.get("nq", 0))

    if not model_xml or not isinstance(frames, list) or nq <= 0:
        print("Animation missing required fields (model_xml, frames, nq)")
        return 2

    model_xml_path = Path(model_xml)
    if not model_xml_path.exists():
        # Try resolve relative to project root
        alt = (root / model_xml).resolve()
        if alt.exists():
            model_xml_path = alt
        else:
            print(f"Model XML not found: {model_xml}")
            return 2

    model = mujoco.MjModel.from_xml_path(model_xml_path.as_posix())
    data = mujoco.MjData(model)

    # Determine free joint quaternion range (if exists) for better interpolation
    free_qpos_adr = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            free_qpos_adr = int(model.jnt_qposadr[j])
            break

    present_dt = 1.0 / max(1e-6, float(args.fps))
    speed = max(1e-6, float(args.speed))

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # Try to track the pelvis body if present for forward-motion animations
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
        total = len(frames)
        t_capture = 0.0
        t0 = time.perf_counter()
        last_present = t0

        while viewer.is_running():
            now = time.perf_counter()
            # Advance capture time based on wall clock and speed
            t_capture += (now - last_present) * speed
            last_present = now

            # Compute fractional frame index in capture timeline
            u = t_capture / dt
            if args.loop:
                # Wrap for looping
                i0 = int(math.floor(u)) % total
                alpha = u - math.floor(u)
                i1 = (i0 + 1) % total
            else:
                # Clamp at the last frame
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

            # Interpolate qpos
            # Start with linear interpolation
            qpos_interp = [0.0] * model.nq
            for j in range(model.nq):
                a = float(f0[j])
                b = float(f1[j])
                # If this dof is a hinge joint angle, wrap for shortest path
                # MuJoCo stores hinges as single qpos in radians; detect by mapping joint adr
                # Build a reverse map from qpos index to joint id range (once)
                qpos_interp[j] = (1.0 - alpha) * a + alpha * b

            # If there is a free joint, slerp its quaternion portion
            if free_qpos_adr is not None and free_qpos_adr + 7 <= model.nq:
                # positions 0..2 linear (already), quaternion 3..6 slerp
                idx = free_qpos_adr + 3
                q0 = (
                    float(f0[idx + 0]),
                    float(f0[idx + 1]),
                    float(f0[idx + 2]),
                    float(f0[idx + 3]),
                )
                q1 = (
                    float(f1[idx + 0]),
                    float(f1[idx + 1]),
                    float(f1[idx + 2]),
                    float(f1[idx + 3]),
                )
                qs = _slerp(_normalize_quat(q0), _normalize_quat(q1), alpha)
                qpos_interp[idx + 0] = qs[0]
                qpos_interp[idx + 1] = qs[1]
                qpos_interp[idx + 2] = qs[2]
                qpos_interp[idx + 3] = qs[3]

            # Apply and render
            for j in range(model.nq):
                data.qpos[j] = float(qpos_interp[j])
            mujoco.mj_forward(model, data)

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
            # Busy-wait minimally to align to present_dt without oversleeping too much
            if present_dt > 0.0:
                target = last_present + present_dt
                while True:
                    now2 = time.perf_counter()
                    if now2 >= target:
                        break
                    # short sleep to yield CPU
                    time.sleep(0.001)

    return 0


if __name__ == "__main__":
    sys.exit(main())


