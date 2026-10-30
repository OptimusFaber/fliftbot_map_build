#!/usr/bin/env python3
"""Build a marker map from observations and an existing mapping run."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from marker_profiles import (add_marker_options,
                             resolve_marker_parameters, marker_command_options)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        help="existing mapping run; supplies all paths not explicitly overridden")
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frame-timestamps", type=Path)
    parser.add_argument("--camera-config", type=Path,
                        help="camera component config used to regenerate observation sidecars")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--camera-position", type=Path)
    parser.add_argument("--camera-json", type=Path)
    parser.add_argument("--lidar-config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--time-offset-s", type=float,
                        help="seconds added to camera frame timestamps; defaults to camera config/run metadata")
    parser.add_argument("--group-by-code", action="store_true",
                        help="diagnostic mode for detector-only observations without persistent track IDs")
    add_marker_options(parser)
    args = parser.parse_args()
    try:
        parameters, profile = resolve_marker_parameters(args)
    except ValueError as exc:
        parser.error(str(exc))
    run = args.run_dir.resolve() if args.run_dir else None
    run_parameters = {}
    if run and (run / "run_parameters.json").is_file():
        run_parameters = json.loads((run / "run_parameters.json").read_text(encoding="utf-8"))
    observations_supplied = args.observations is not None
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

    if run and not observations_supplied and not args.observations.exists():
        stored = run_parameters.get("inputs", {}).get("observations")
        generated = run / "input/video/observations.json"
        if stored and Path(stored).is_file():
            args.observations = Path(stored)
        elif generated.is_file():
            args.observations = generated

    if args.camera_config:
        config_path = args.camera_config.resolve()
        camera = json.loads(config_path.read_text(encoding="utf-8"))
        if args.time_offset_s is None:
            args.time_offset_s = float(camera.get("time_offset_s", 0.0))
        position, angles = camera["position_m"], camera["orientation_deg"]
        args.camera_position.parent.mkdir(parents=True, exist_ok=True)
        args.camera_json.parent.mkdir(parents=True, exist_ok=True)
        args.camera_position.write_text(
            f"{position['x']*100:.9g} {position['y']*100:.9g} {position['z']*100:.9g}\n"
            f"{math.radians(angles['roll']):.12g} {math.radians(angles['pitch']):.12g} "
            f"{math.radians(angles.get('yaw', 0.0)):.12g}\n",
            encoding="utf-8",
        )
        intrinsics = Path(camera["intrinsics"]).expanduser()
        if not intrinsics.is_absolute():
            intrinsics = (config_path.parent / intrinsics).resolve()
        shutil.copy2(intrinsics, args.camera_json)

    if args.time_offset_s is None and run is not None:
        args.time_offset_s = float(run_parameters.get("markers", {}).get(
            "time_offset_s", run_parameters.get("camera", {}).get("time_offset_s", 0.0)))
    if args.time_offset_s is None:
        args.time_offset_s = 0.0

    if not args.observations.exists() and args.video and args.frame_timestamps:
        video_dir = run / "input/marker_video" if run else args.out_dir / "marker_video"
        video_dir.mkdir(parents=True, exist_ok=True)
        staged_video = video_dir / args.video.name
        staged_timestamps = video_dir / "frame_timestamps.csv"
        if not staged_video.exists():
            staged_video.symlink_to(args.video.resolve())
        if not staged_timestamps.exists():
            staged_timestamps.symlink_to(args.frame_timestamps.resolve())
        observation_command = [
            sys.executable, str(ROOT / "src/auto_map/get_video_data.py"),
            "--video", str(video_dir), "--camera-json", str(args.camera_json),
            "--camera-position", str(args.camera_position),
        ]
        subprocess.run(observation_command, cwd=ROOT / "src/auto_map", check=True)
        generated = video_dir / "observations.json"
        if not generated.exists():
            raise RuntimeError(f"video processor did not create {generated}")
        shutil.copy2(generated, args.observations)

    for name in ("observations", "trajectory", "map_dir", "camera_position", "camera_json", "lidar_config"):
        if not getattr(args, name).exists():
            parser.error(f"{name.replace('_', '-')}: path not found: {getattr(args, name)}")
    command = [
        sys.executable,
        str(ROOT / "src/fliftbot_mapping/evaluation/ceiling_marker_localization_stored_ray_corrected.py"),
        "--observations", str(args.observations), "--trajectory", str(args.trajectory),
        "--map-dir", str(args.map_dir), "--camera-position", str(args.camera_position),
        "--camera-json", str(args.camera_json), "--lidar-config", str(args.lidar_config),
        "--out-dir", str(args.out_dir),
        "--time-offset-s", str(args.time_offset_s),
    ]
    command += marker_command_options(parameters)
    if args.group_by_code:
        command.append("--group-by-code")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_parameters.json").write_text(json.dumps({
        "schema": "fliftbot.marker_run_parameters.v1",
        "inputs": {name: str(getattr(args, name).resolve()) for name in (
            "observations", "trajectory", "map_dir", "camera_position", "camera_json", "lidar_config")},
        "profile": profile,
        "parameters": {**parameters, "time_offset_s": args.time_offset_s,
                       "group_by_code": args.group_by_code},
        "command": command,
    }, indent=2) + "\n", encoding="utf-8")
    returncode = subprocess.call(command)
    if returncode == 0 and run is not None and args.out_dir.resolve() == (run / "markers").resolve():
        # Keep all user-facing marker exports in sync when an existing map is
        # reprocessed without rebuilding the map itself.
        from build_from_canonical import export_marker_coordinates, refresh_artifacts

        if (run / "web_map/pbstream_to_local_matrix.npy").is_file():
            export_marker_coordinates(run)
        rendered = args.out_dir / "stored_ray_corrected_map.png"
        if rendered.is_file():
            shutil.copy2(rendered, run / "map_with_markers.png")
        refresh_artifacts(run)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
