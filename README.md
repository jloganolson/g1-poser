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
