Manual slider UI to pose the MuJoCo G1 model and save/load poses 

![App screenshot](screenshot.png)

### Quick start
```bash
pip3 install uv #if you dont have uv?
uv sync
uv run main.py
```

### Usage
- **Adjust joints**: Use List or Body Map UI (tabs at top). Base R/P/Y appears if the model has a FREE joint.
- **Mirror**: Toggle to sync left/right joints.
- **Poses panel**: Add/Update/Delete/Activate poses (will lerp b/w them)
- **Export/Import**: Exports `poses_YYYYMMDD_HHMMSS.json`; autosaves to `.poses_autosave.json`; import a JSON to load.
- **Model**: uses `g1_description/g1.xml`.
