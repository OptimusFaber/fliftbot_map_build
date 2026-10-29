#!/usr/bin/env python3
"""Build a marker map from observations and an existing mapping run."""
import argparse
import json
import math
import shutil
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
    parser.add_argument("--ceiling-z", type=float)
    parser.add_argument("--per-track-height", action="store_true")
    parser.add_argument("--max-marker-height-m", type=float)
    parser.add_argument("--merge-radius-m", type=float, default=0.10)
    parser.add_argument("--min-camera-baseline-m", type=float)
    parser.add_argument("--max-track-height-uncertainty-m", type=float)
    parser.add_argument("--height-quality-method", choices=("legacy", "block_cv"))
    parser.add_argument("--block-distance-m", type=float)
    parser.add_argument("--max-block-height-instability-m", type=float)
    parser.add_argument("--stationary-height-filter", action="store_true")
    parser.add_argument("--time-offset-s", type=float,
                        help="seconds added to camera frame timestamps; defaults to camera config/run metadata")
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

    if args.camera_config:
        config_path = args.camera_config.resolve()
        camera = json.loads(config_path.read_text(encoding="utf-8"))
        if args.time_offset_s is None:
            args.time_offset_s = float(camera.get("time_offset_s", 0.0))
        position, angles = camera["position_m"], camera["orientation_deg"]
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
        run_parameters = run / "run_parameters.json"
        if run_parameters.is_file():
            camera = json.loads(run_parameters.read_text(encoding="utf-8")).get("camera", {})
            args.time_offset_s = float(camera.get("time_offset_s", 0.0))
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
        "--out-dir", str(args.out_dir), "--merge-radius-m", str(args.merge_radius_m),
        "--time-offset-s", str(args.time_offset_s),
    ]
    if args.ceiling_z is not None:
        command += ["--ceiling-z", str(args.ceiling_z)]
    if args.per_track_height:
        command.append("--per-track-height")
    if args.max_marker_height_m is not None:
        command += ["--max-marker-height-m", str(args.max_marker_height_m)]
    for name in ("min_camera_baseline_m", "max_track_height_uncertainty_m",
                 "height_quality_method", "block_distance_m", "max_block_height_instability_m"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    if args.stationary_height_filter:
        command.append("--stationary-height-filter")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_parameters.json").write_text(json.dumps({
        "schema": "fliftbot.marker_run_parameters.v1",
        "inputs": {name: str(getattr(args, name).resolve()) for name in (
            "observations", "trajectory", "map_dir", "camera_position", "camera_json", "lidar_config")},
        "parameters": {"ceiling_z_m": args.ceiling_z, "per_track_height": args.per_track_height,
                       "max_marker_height_m": args.max_marker_height_m,
                       "merge_radius_m": args.merge_radius_m,
                       "min_camera_baseline_m": args.min_camera_baseline_m,
                       "max_track_height_uncertainty_m": args.max_track_height_uncertainty_m,
                       "height_quality_method": args.height_quality_method,
                       "block_distance_m": args.block_distance_m,
                       "max_block_height_instability_m": args.max_block_height_instability_m,
                       "stationary_height_filter": args.stationary_height_filter,
                       "time_offset_s": args.time_offset_s},
        "command": command,
    }, indent=2) + "\n", encoding="utf-8")
    returncode = subprocess.call(command)
    if returncode == 0 and run is not None and args.out_dir.resolve() == (run / "markers").resolve():
        # Keep all user-facing marker exports in sync when an existing map is
        # reprocessed without rebuilding the map itself.
        from build_from_canonical import export_marker_coordinates, refresh_artifacts

        export_marker_coordinates(run)
        rendered = args.out_dir / "stored_ray_corrected_map.png"
        if rendered.is_file():
            shutil.copy2(rendered, run / "map_with_markers.png")
        refresh_artifacts(run)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
