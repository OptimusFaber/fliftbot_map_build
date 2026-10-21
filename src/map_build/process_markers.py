#!/usr/bin/env python3
"""Build a marker map from observations and an existing mapping run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="existing mapping run; supplies all paths not explicitly overridden")
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--camera-position", type=Path)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--lidar-config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ceiling-z", type=float)
    parser.add_argument("--per-track-height", action="store_true")
    parser.add_argument("--merge-radius-m", type=float, default=0.10)
    parser.add_argument("--time-offset-s", type=float, default=0.0)
    args = parser.parse_args()
    run = args.run_dir.resolve() if args.run_dir else None
    defaults = {
        "observations": run / "observations.json" if run else None,
        "trajectory": run / "map/trajectory.json" if run else None,
        "map_dir": run / "map" if run else None,
        "camera_position": run / "input/camera_position.txt" if run else None,
        "camera_json": run / "input/camera.json" if run else None,
        "lidar_config": run / "input/lidar.markers.config" if run else None,
        "out_dir": run / "markers" if run else None,
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --run-dir")
    for name in ("observations", "trajectory", "map_dir", "camera_position", "camera_json", "lidar_config"):
        if not getattr(args, name).exists():
            parser.error(f"{name.replace('_', '-')}: path not found: {getattr(args, name)}")
    command = [
        sys.executable,
        str(ROOT / "vendor/fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_stored_ray_corrected.py"),
        "--observations", str(args.observations), "--trajectory", str(args.trajectory),
        "--map-dir", str(args.map_dir), "--camera-position", str(args.camera_position),
        "--camera-json", str(args.camera_json), "--lidar-config", str(args.lidar_config),
        "--out-dir", str(args.out_dir), "--merge-radius-m", str(args.merge_radius_m),
        "--time-offset-s", str(args.time_offset_s),
    ]
    if args.ceiling_z is not None:
        command += ["--ceiling-z", str(args.ceiling_z)]
    if args.per_track_height:
        command.append("--per-track-height")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_parameters.json").write_text(json.dumps({
        "schema": "fliftbot.marker_run_parameters.v1",
        "inputs": {name: str(getattr(args, name).resolve()) for name in (
            "observations", "trajectory", "map_dir", "camera_position", "camera_json", "lidar_config")},
        "parameters": {"ceiling_z_m": args.ceiling_z, "per_track_height": args.per_track_height,
                       "merge_radius_m": args.merge_radius_m, "time_offset_s": args.time_offset_s},
        "command": command,
    }, indent=2) + "\n", encoding="utf-8")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
