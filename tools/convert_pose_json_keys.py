import argparse
import json
from pathlib import Path
from typing import Any

import mujoco


def _load_model(xml_path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(xml_path.as_posix())


def _convert_joints_to_names(model: mujoco.MjModel, joints_obj: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (joints_obj or {}).items():
        key_str = str(k)
        name: str | None = None
        try:
            jid = int(key_str)
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            name = nm if nm is not None else None
        except Exception:
            # Not an int key: assume it's already a joint name
            name = key_str
        out[name if name is not None else key_str] = float(v)
    return out


def convert_file(model: mujoco.MjModel, src: Path, dst: Path, inplace: bool = False) -> None:
    data = json.loads(src.read_text())
    new_poses: list[dict[str, Any]] = []
    for item in data.get("poses", []):
        base_rpy = item.get("base_rpy")
        joints_named = _convert_joints_to_names(model, item.get("joints", {}) or {})
        new_poses.append({"base_rpy": base_rpy, "joints": joints_named})
    serial = {"poses": new_poses}
    dst_path = src if inplace else dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(serial, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pose JSONs to use joint NAMES as keys")
    parser.add_argument("--xml", type=Path, default=Path("./g1_description/g1.xml"), help="MuJoCo XML path")
    parser.add_argument("--dir", type=Path, default=Path("./output"), help="Directory containing poses_*.json")
    parser.add_argument("--inplace", action="store_true", help="Overwrite files in place (default: write *_names.json)")
    parser.add_argument("paths", nargs="*", type=Path, help="Specific JSON files to convert (overrides --dir glob)")
    args = parser.parse_args()

    model = _load_model(args.xml)

    files: list[Path]
    if args.paths:
        files = [p for p in args.paths if p.exists() and p.suffix.lower() == ".json"]
    else:
        files = sorted(args.dir.glob("poses_*.json"))

    if not files:
        print("No pose JSONs found to convert.")
        return

    for src in files:
        if args.inplace:
            dst = src
        else:
            dst = src.with_name(src.stem + "_names" + src.suffix)
        convert_file(model, src, dst, inplace=args.inplace)
        print(f"Converted {src.name} -> {dst.name}")


if __name__ == "__main__":
    main()


