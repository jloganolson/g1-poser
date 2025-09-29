from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Optional

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter


# ----------------------------- Static configuration -----------------------------
# No CLI args; tune here.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# If None, the script will auto-pick the newest animation file from OUTPUT_DIR.
# Otherwise set an absolute path to a specific JSON file.
ANIMATION_PATH: Optional[str] = None


def _find_latest_animation_path() -> Optional[str]:
    out = Path(OUTPUT_DIR)
    if not out.exists():
        return None
    # Prefer BVH-driven exports first
    cand = sorted(out.glob("animation_mocap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        # Fallback to any animation exports
        cand = sorted(out.glob("animation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        return None
    return cand[0].as_posix()


def _load_animation_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_xml_path(model_xml: str) -> str:
    p = Path(model_xml)
    if p.exists():
        return p.as_posix()
    # Try resolving relative to repo root if not absolute/exists
    q = Path(BASE_DIR) / model_xml
    return q.as_posix()


def main() -> None:
    # Determine animation path
    anim_path = ANIMATION_PATH or _find_latest_animation_path()
    if not anim_path or not Path(anim_path).exists():
        raise FileNotFoundError(
            f"No animation JSON found. Set ANIMATION_PATH or export to {OUTPUT_DIR} first."
        )
    print(f"Loading animation: {anim_path}")

    payload = _load_animation_json(anim_path)
    schema = payload.get("schema", "")
    if schema != "gait_animation.v1":
        raise ValueError(f"Unsupported schema '{schema}'. Expected 'gait_animation.v1'.")

    model_xml = _resolve_model_xml_path(str(payload.get("model_xml", "")))
    dt = float(payload.get("dt", 0.0) or 0.0)
    if dt <= 0.0:
        fps = float(payload.get("fps", 0.0) or 0.0)
        if fps <= 0.0:
            raise ValueError("Missing dt/fps in animation payload.")
        dt = 1.0 / fps

    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Animation payload has no frames.")
    n_frames = len(frames)

    # Build model/data
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    # Initialize to first frame
    q0 = frames[0]
    if len(q0) != int(model.nq):
        raise ValueError(f"Frame qpos length {len(q0)} does not match model.nq {model.nq}.")
    for i in range(model.nq):
        data.qpos[i] = float(q0[i])
    for i in range(model.nv):
        data.qvel[i] = 0.0
    mujoco.mj_forward(model, data)

    # Viewer playback
    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=True, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # Camera follow pelvis if present
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

        rate = RateLimiter(frequency=max(1.0, 1.0 / max(1e-6, dt)), warn=False)
        t_accum = 0.0
        t0 = time.perf_counter()

        while viewer.is_running():
            now = time.perf_counter()
            t_accum += (now - t0)
            t0 = now

            # Frame index based on dt
            f = int((t_accum / dt) % n_frames)

            q = frames[f]
            if len(q) != int(model.nq):
                raise ValueError(f"Frame {f} qpos length {len(q)} != model.nq {model.nq}")

            for i in range(model.nq):
                data.qpos[i] = float(q[i])
            for i in range(model.nv):
                data.qvel[i] = 0.0
            mujoco.mj_forward(model, data)

            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()


