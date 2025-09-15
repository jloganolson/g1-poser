## G1 Poser (2-hour prototype)

Manual slider UI to pose the MuJoCo G1 model and save/load poses 

Make sure to install cyclonedds per the [unitree_sdk2_python repo](https://github.com/unitreerobotics/unitree_sdk2_python) and export before uv sync
```bash 
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
```

![App screenshot](screenshot.png)

### Quick start
```bash
pip3 install uv #if you dont have uv?
uv sync
uv run main.py
```

### Scripts
- `pose-tool.py` - Manual posing with joint sliders and pose management
- `mink_g1_pose_ik.py` - Quadruped gait generator with IK (working baseline)
- `temp.py` - Enhanced gait generator with additional features (linear stance, coplanar constraints)

### Usage
- **Adjust joints**: Use List or Body Map UI (tabs at top). Base R/P/Y appears if the model has a FREE joint.
- **Mirror**: Toggle to sync left/right joints.
- **Poses panel**: Add/Update/Delete/Activate poses (will lerp b/w them)
- **Export/Import**: Exports `poses_YYYYMMDD_HHMMSS.json`; autosaves to `.poses_autosave.json`; import a JSON to load.
- **Model**: uses `g1_description/g1.xml`.

## 🚨 Critical Threading Issue

**Important for developers**: This project has a critical threading compatibility issue when combining Tkinter GUI with MuJoCo viewer.

### Problem
Using `ttk.Entry` or `tk.Entry` widgets alongside MuJoCo's viewer causes XCB threading crashes:
```
[xcb] Unknown sequence number while appending request
[xcb] Aborting, sorry about that.
python3: ../../src/xcb_io.c:157: append_pending_request: Assertion `!xcb_xlib_unknown_seq_number' failed
```

### Solution
- **Never use Entry widgets** - use `ttk.Label` with auto-updating display instead
- **Defer mujoco operations** until after viewer initialization
- See `.cursorrules` for detailed safe coding patterns

### Example Safe Pattern
```python
# ❌ Causes crashes
entry = ttk.Entry(parent, textvariable=my_var)

# ✅ Safe alternative  
display = ttk.Label(parent, text="0.000", relief="sunken")
def update_display(*args):
    display.configure(text=f"{float(my_var.get()):.3f}")
my_var.trace_add("write", update_display)
```

This issue was discovered and resolved in December 2024. The `temp.py` file demonstrates the working solutions.

## Animation export (for Isaac Sim)

`mink_g1_feet_planted.py` can export a time-series of MuJoCo `qpos` vectors as an animation JSON. In the UI, click "Export Animation" to write `output/animation_YYYYMMDD_HHMMSS.json`.

### File format: `gait_animation.v1`

- **schema**: Always `"gait_animation.v1"`.
- **model_xml**: Path to the XML used to generate the animation. Use it to verify joint order.
- **dt**: Seconds per frame.
- **fps**: Convenience frames-per-second (`1/dt`).
- **nq**: Length of each `qpos` vector.
- **frames**: List of length-`nq` float arrays. Each array is the full MuJoCo `qpos` at that frame in MJCF order.
- **cycles**: Number of gait cycles exported (typically 3). Informational.
- **cycle_T**: Gait cycle duration used when generating, in seconds. Informational.
- **timestamp**: ISO8601 string.
- **metadata**: Auxiliary mapping info for loaders.
  - **base**: Present if the model has a free root joint. Fields:
    - **pos_indices**: `[ix, iy, iz]` indices of the root position within `qpos`.
    - **quat_indices**: `[iw, ix, iy, iz]` indices of the root quaternion within `qpos` (MuJoCo order `w,x,y,z`).
  - **joints**: Array of objects describing each joint:
    - **name**: Joint name (matches MJCF).
    - **type**: One of `free`, `ball`, `slide`, `hinge`.
    - **qposadr**: Start index of this joint's state in `qpos`.
    - **qposdim**: Number of `qpos` elements this joint occupies (free=7, ball=4, slide|hinge=1).
  - **qpos_labels**: Length-`nq` human-friendly labels for many entries (hinge/slide and ball joints are labeled; free-root entries may remain generic like `qpos[i]`).

### Minimal example

```json
{
  "schema": "gait_animation.v1",
  "model_xml": "g1_description/scene_g1_targets.xml",
  "dt": 0.005,
  "fps": 200.0,
  "nq": 78,
  "frames": [
    [0.000, 0.000, 0.280, 1.0, 0.0, 0.0, 0.0, 0.12, 0.03, /* ... nq values ... */],
    [0.001, 0.000, 0.280, 1.0, 0.0, 0.0, 0.0, 0.12, 0.03, /* ... */]
  ],
  "cycles": 3,
  "cycle_T": 1.2,
  "timestamp": "2025-09-15T13:49:44.123456",
  "metadata": {
    "base": { "pos_indices": [0, 1, 2], "quat_indices": [3, 4, 5, 6] },
    "joints": [
      { "name": "pelvis", "type": "free",  "qposadr": 0, "qposdim": 7 },
      { "name": "left_hip_yaw", "type": "hinge", "qposadr": 7, "qposdim": 1 }
      /* ... more joints ... */
    ],
    "qpos_labels": [
      "qpos[0]", "qpos[1]", "qpos[2]", "qpos[3]", "qpos[4]", "qpos[5]", "qpos[6]",
      "joint:left_hip_yaw", /* ... length=nq */
    ]
  }
}
```

### Isaac Sim loading tips

- **Root pose**: If `metadata.base` is present, set the articulation root prim pose from `qpos[pos_indices]` and the quaternion at `quat_indices`.
  - MuJoCo quaternion order is `w,x,y,z`; Isaac Sim quaternions are typically `x,y,z,w`. Convert accordingly.
- **Joint mapping**: Use `metadata.joints` to map from MJCF joint names to `qpos` indices (`qposadr`) when setting Isaac Sim DOF states. Hinge/slide joints consume one `qpos` each.
- **Timing**: Advance one frame per `dt` (or resample to your sim step).
- **Validation**: Optionally verify your articulation has `nq` total DoF state elements in the same order as MJCF; otherwise, build an explicit index map using names and `qpos_labels`.
