## Mink IK with Unitree G1 — Reference Guide

This guide distills practical IK techniques used in `main-proc-anim.py` into a compact reference you can reuse for further Mink work with the G1 model (`g1_description/g1.xml` and scenes).

### TL;DR — Start here

- Use the reference script:
```bash
uv run scripts/mink_g1_ik_reference.py
```
- It demonstrates a stable task set, mocap target usage, and a small animated right-hand trajectory that drives IK at 200 Hz.

### Model and scene

- Prefer a scene that includes mocap bodies for hands/feet, e.g. `g1_description/scene_g1_targets.xml`.
- If your XML lacks mocap targets, you can still set task targets from the current configuration via `set_target_from_configuration`.

### Core tasks pattern

Recommended baseline (position/orientation costs are tuned for crawling/placing):

- Pelvis orientation (keep upright): `FrameTask("pelvis", body, orientation_cost=1.0)`
- Pelvis position (limit drift): `FrameTask("pelvis", body, position_cost=10.0)`
- Torso orientation (comfort): `FrameTask("torso_link", body, orientation_cost=1.0)`
- Posture (stay near current): `PostureTask(model, cost=1e-1)`
- End-effectors (position only):
  - Hands: `FrameTask("left_palm"|"right_palm", site, position_cost=~25.0)`
  - Feet: `FrameTask("left_foot"|"right_foot", site, position_cost=~30.0)`
- Segment stabilizers (discourage awkward pitching):
  - Knees: `FrameTask("*_knee_link", body, orientation_cost=0.8)`
  - Elbows: `FrameTask("*_elbow_link", body, orientation_cost=1.2)`
- Keep feet level: `FrameTask("*_ankle_roll_link", body, orientation_cost=2.0)`

Initialize once from the current configuration:
```python
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
```

### Targets from mocap vs configuration

- With mocap bodies (recommended):
```python
mid = model.body("right_palm_target").mocapid[0]
right_hand_task.set_target(mink.SE3.from_mocap_id(data, mid))
```
- Without mocap, hold the current frame:
```python
right_hand_task.set_target_from_configuration(configuration)
```

Use `mink.move_mocap_to_frame(model, data, "right_palm_target", "right_palm", "site")` to initialize mocap bodies at their corresponding frames.

### Grounding the base

If you have a free joint, shift base-z so average foot height is at z=0:
```python
# see scripts/mink_g1_ik_reference.py:_shift_base_z_to_ground
```
This keeps early experiments visually grounded and reduces solver surprises.

### IK loop skeleton

```python
rate = RateLimiter(frequency=200.0, warn=False)
limits = [mink.ConfigurationLimit(model)]
solver = "daqp"  # falls back if not available
while viewer.is_running():
    # update targets (from mocap or configuration)
    vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
    configuration.integrate_inplace(vel, rate.dt)
    viewer.sync(); rate.sleep()
```

### Camera and viewer

- Launch with `mujoco.viewer.launch_passive`.
- Track the pelvis when possible for easier debugging.

### Stability tips

- Start with conservative costs; increase end-effector costs gradually.
- Keep posture task enabled (small cost) to avoid joint drift.
- For locomotion-like motions, update mocap targets in world coordinates and let IK follow them.

### Tkinter + MuJoCo warning (critical)

- Avoid `ttk.Entry` / `tk.Entry` widgets with the MuJoCo viewer (can trigger XCB threading crashes). Use Labels + traces or external prompts. See workspace rules for details.

### Reference implementation

- The file `scripts/mink_g1_ik_reference.py` is a compact, documented example. Use it as a template when building new Mink-driven behaviors.


