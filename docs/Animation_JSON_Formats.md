## Animation and Pose JSON formats (BVH drive vs Pose IK)

This document describes the JSON outputs produced by `scripts/bvh_drive_g1_mink.py` and `archive/mink_g1_pose_ik.py`, so an importer can support both. Both scripts emit animations with schema `gait_animation.v1`; additionally, `bvh_drive_g1_mink.py` can export a single-frame pose JSON compatible with `crawl-pose.json`.

### Terminology
- **Animation JSON**: time series of `qpos` frames (and optionally velocities/sites), schema `gait_animation.v1`.
- **Pose JSON**: single-frame pose list with `base_rpy` and named 1-DoF `joints`.

---

## Common fields in animation JSON (both scripts)
These fields appear in all animation JSONs produced by both scripts:

- `schema`: always `"gait_animation.v1"`
- `model_xml`: string path to the MuJoCo model used (absolute or relative)
- `dt`: seconds per frame (float)
- `fps`: frames per second; typically `1.0 / dt`
- `nq`: number of generalized positions (`qpos` length)
- `frames`: list of length `T`, each an array of length `nq` (qpos per frame)
- `timestamp`: ISO-8601 string
- `metadata.base` (present when a free base joint exists):
  - `pos_indices`: three indices into `qpos` for base position
  - `quat_indices`: four indices into `qpos` for base orientation quaternion `(w, x, y, z)`
- `metadata.joints`: list of per-joint descriptors
  - `{ "name": str, "type": "free"|"ball"|"slide"|"hinge", "qposadr": int, "qposdim": int }`
- `metadata.qpos_labels`: list of length `nq` with human-readable labels for `qpos` entries

Importer requirement: If `schema == "gait_animation.v1"` and `frames` exists, treat as an animation.

---

## BVH-driven animation (scripts/bvh_drive_g1_mink.py)

In addition to the common fields above, BVH exports add rich kinematic and contact context.

### Additional top-level fields
- `nv`: number of velocity DoFs
- `nsite`: number of sites in the model
- `vel_frames`: `[T][nv]` generalized velocities per frame
- `site_positions`: `[T][nsite][3]` world-space site positions per frame
- `contact_flags`: `[T][4]` binary contact indicators for four end-effectors (see ordering below)

### Additional metadata
- `animation_type`: always `"bvh_drive"`
- `sites`:
  - `names`: list of site names (length `nsite`)
  - `indices`: list `[0..nsite-1]`
  - `by_name`: mapping `name -> index`
- `contact`:
  - `order`: `["FL", "FR", "RL", "RR"]`
  - `threshold_m`: z-height threshold (meters) used to mark contact in `contact_flags`
- `has_root_motion`: boolean; true when the free base trajectory is present
- `bvh`: description of the BVH sampling used
  - `path`: path to the BVH file used
  - `scale`: BVH to model scale factor used during drive
  - `frame_time`: BVH frame time (seconds)
  - `num_frames`: number of BVH frames sampled
- `capture_hz`: only in the "live recording" export; equals the capture rate (e.g., 200 Hz)

### Behavior notes
- XY recentering: exports subtract the initial base `(x, y)` of frame 0 from all `frames` and `site_positions` so the sequence starts at `(0, 0)`.
- Two export modes:
  1) Offline bake ("Export Animation" outside the viewer loop): `dt` equals BVH `frame_time`, `T == num_frames`.
  2) Live recording (button inside the viewer loop): fixed `dt ≈ 1/200`, records exactly one BVH cycle, adds `capture_hz`.

### Minimal BVH animation example
```json
{
  "schema": "gait_animation.v1",
  "model_xml": "/abs/path/to/g1_description/scene_g1_targets.xml",
  "dt": 0.005,
  "fps": 200.0,
  "nsite": 42,
  "nq": 56,
  "nv": 55,
  "frames": [[/* nq */], /* ... */],
  "vel_frames": [[/* nv */], /* ... */],
  "site_positions": [[[0.0, 0.0, 0.0] /* nsite */], /* ... */],
  "contact_flags": [[0, 1, 0, 1], /* ... */],
  "timestamp": "2025-09-29T09:50:56",
  "metadata": {
    "animation_type": "bvh_drive",
    "base": { "pos_indices": [0,1,2], "quat_indices": [3,4,5,6] },
    "joints": [{ "name": "left_hip_yaw", "type": "hinge", "qposadr": 7, "qposdim": 1 } /* ... */],
    "qpos_labels": ["qpos[0]", "qpos[1]" /* ... */],
    "sites": {
      "names": ["left_palm", "right_palm", "left_foot", "right_foot" /* ... */],
      "indices": [0, 1, 2, 3 /* ... */],
      "by_name": { "left_palm": 0, "right_palm": 1, "left_foot": 2, "right_foot": 3 }
    },
    "contact": { "order": ["FL", "FR", "RL", "RR"], "threshold_m": 0.01 },
    "has_root_motion": true,
    "bvh": { "path": "/abs/path/to/mocap.bvh", "scale": 0.009, "frame_time": 0.005, "num_frames": 1200 }
  }
}
```

Importer guidance for BVH:
- Treat missing or empty `vel_frames`, `site_positions`, or `contact_flags` as optional (but current exports include them).
- Use `metadata.sites.by_name` or `names`/`indices` for site lookup.
- Interpret `contact_flags[t]` using `metadata.contact.order`.

---

## Pose-IK animation (archive/mink_g1_pose_ik.py)

This animation is a gait synthesized by a UI, not driven by BVH. It contains the common fields plus:

### Additional top-level fields
- `cycles`: integer number of cycles baked (typically 3)
- `cycle_T`: seconds per cycle used in synthesis

### Not present (compared to BVH export)
- No `nv`, `nsite`, `vel_frames`, `site_positions`, `contact_flags`
- No `metadata.sites`, `metadata.contact`, `metadata.animation_type`, `metadata.has_root_motion`, `metadata.bvh`

### Behavior notes
- Sampling chooses `num_steps ≈ round(T * 200 Hz)` so `dt = T / num_steps` for seamless looping.
- XY recentering is not performed in these exports.

### Minimal Pose-IK animation example
```json
{
  "schema": "gait_animation.v1",
  "model_xml": "/abs/path/to/g1_description/scene_g1_targets.xml",
  "dt": 0.006,
  "fps": 166.6667,
  "nq": 56,
  "frames": [[/* nq */], /* ... */],
  "cycles": 3,
  "cycle_T": 1.2,
  "timestamp": "2025-09-15T13:40:53",
  "metadata": {
    "base": { "pos_indices": [0,1,2], "quat_indices": [3,4,5,6] },
    "joints": [{ "name": "left_hip_yaw", "type": "hinge", "qposadr": 7, "qposdim": 1 } /* ... */],
    "qpos_labels": ["qpos[0]", "qpos[1]" /* ... */]
  }
}
```

Importer guidance for Pose-IK:
- `frames`, `dt`, `nq` are sufficient to play; ignore extra BVH-only fields if absent.
- `cycles` and `cycle_T` are informative; playback does not require them.

---

## Single-frame pose JSON (from bvh_drive_g1_mink.py)

The "Export Frame" button writes a pose file compatible with `crawl-pose.json`. It is not an animation and has no `schema` field.

### Structure
- `poses`: list of pose dicts (typically length 1)
  - `base_rpy`: `[roll, pitch, yaw]` in radians; applied only if a free base exists
  - `joints`: mapping from 1-DoF joint name to scalar joint position

### Example
```json
{
  "poses": [
    {
      "base_rpy": [0.0, 0.05, 0.0],
      "joints": {
        "left_hip_yaw": -0.1,
        "left_knee": 0.8,
        "right_knee": 0.8
      }
    }
  ]
}
```

Importer detection: If `poses` is present (and no `schema`), treat as a pose file. Apply to a MuJoCo state by converting `base_rpy` to quaternion for the free joint (if any) and assigning named scalar joints.

---

## Importer checklist (robust handling)

- Detect type:
  - Animation: `schema == "gait_animation.v1" && frames` present
  - Pose: `poses` present
- Animation minimal fields to consume: `frames`, `dt`, `nq`
- Optional animation fields:
  - `nv`, `vel_frames`
  - `nsite`, `site_positions`, `metadata.sites`
  - `contact_flags`, `metadata.contact`
  - `metadata.base`, `metadata.joints`, `metadata.qpos_labels`
  - `metadata.bvh`, `animation_type`, `has_root_motion`, `capture_hz`, `cycles`, `cycle_T`
- Paths: `model_xml` may be absolute or relative; importer should not assume existence unless needed.
- Root motion: if `metadata.base` exists, treat as free-base animation; otherwise, rigid-base.
- Contact flags ordering: use `metadata.contact.order` if present; default to `["FL","FR","RL","RR"]` if missing.


