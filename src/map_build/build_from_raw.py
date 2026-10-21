#!/usr/bin/env python3
"""Run the complete pipeline starting with raw M360 and odometry files."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--odometry-kind", choices=("auto", "log", "csv"), default="auto")
    parser.add_argument(
        "--m360-point-time-mode", choices=("packet-host", "device-linear"),
        default="packet-host")
    parser.add_argument("--lidar-start-unix", type=float)
    parser.add_argument("--lidar-end-unix", type=float)
    parser.add_argument("--canonical-output", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--camera-position-file", type=Path)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="mapping")
    parser.add_argument("--lidar-only", action="store_true")
    parser.add_argument("--wheelbase", type=float, help="override robot wheelbase in metres")
    parser.add_argument("--fork-position", choices=("raised", "lowered"), default="raised")
    parser.add_argument("--lowered-forks-offset-m", type=float, default=0.05)
    for prefix in ("camera", "lidar"):
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            parser.add_argument(f"--{prefix}-{axis}", type=float)
    parser.add_argument("--encoder-speed-sign", type=float, choices=(-1.0, 1.0))
    parser.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0))
    parser.add_argument("--mapping-yaw", type=float)
    parser.add_argument("--odometry-time-offset", type=float, default=0.0)
    parser.add_argument("--web-resolution-m-per-px", type=float, default=0.01)
    parser.add_argument("--no-web-align", action="store_true")
    parser.add_argument("--origin-map-xy", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--origin-marker-id")
    parser.add_argument("--origin-yaw-deg", type=float, default=0.0)
    parser.add_argument("--origin-name", default="manual")
    parser.add_argument("--deploy-output", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frame-timestamps", type=Path)
    parser.add_argument("--labels2coords", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--ceiling-z", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve user-provided paths before starting child processes with
    # cwd=HERE. Otherwise repository-relative paths would be resolved from the
    # internal source directory.
    for attribute in (
        "scan", "odometry", "canonical_output", "camera_config", "camera_position_file",
        "lidar_config", "robot_config", "output", "deploy_output", "video",
        "frame_timestamps", "labels2coords", "observations",
    ):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())

    convert = [sys.executable, str(HERE / "convert_recording.py"), "--scan", str(args.scan),
               "--odometry", str(args.odometry), "--odometry-kind", args.odometry_kind,
               "--output", str(args.canonical_output), "--lidar-config", str(args.lidar_config)]
    for option, value in (("--camera-config", args.camera_config), ("--robot-config", args.robot_config)):
        if value is not None:
            convert += [option, str(value)]
    convert += ["--m360-point-time-mode", args.m360_point_time_mode]
    if args.lidar_start_unix is not None:
        convert += ["--lidar-start-unix", str(args.lidar_start_unix)]
    if args.lidar_end_unix is not None:
        convert += ["--lidar-end-unix", str(args.lidar_end_unix)]
    subprocess.run(convert, cwd=HERE, check=True)
    command = [sys.executable, str(HERE / "build_from_canonical.py"),
               "--recording", str(args.canonical_output), "--output", str(args.output),
               "--name", args.name, "--web-resolution-m-per-px", str(args.web_resolution_m_per_px)]
    for option, value in (("--camera-config", args.camera_config), ("--robot-config", args.robot_config),
                          ("--camera-position-file", args.camera_position_file)):
        if value is not None:
            command += [option, str(value)]
    for prefix in ("camera", "lidar"):
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            value = getattr(args, f"{prefix}_{axis}")
            if value is not None:
                command += [f"--{prefix}-{axis}", str(value)]
    for flag, enabled in (("--lidar-only", args.lidar_only), ("--no-web-align", args.no_web_align),
                          ("--dry-run", args.dry_run)):
        if enabled:
            command.append(flag)
    if args.encoder_speed_sign is not None:
        command += ["--encoder-speed-sign", str(args.encoder_speed_sign)]
    if args.wheelbase is not None:
        command += ["--wheelbase", str(args.wheelbase)]
    command += ["--fork-position", args.fork_position,
                "--lowered-forks-offset-m", str(args.lowered_forks_offset_m)]
    if args.steering_sign is not None:
        command += ["--steering-sign", str(args.steering_sign)]
    if args.mapping_yaw is not None:
        command += ["--mapping-yaw", str(args.mapping_yaw)]
    command += ["--odometry-time-offset", str(args.odometry_time_offset)]
    if args.origin_map_xy:
        command += ["--origin-map-xy", *map(str, args.origin_map_xy),
                    "--origin-yaw-deg", str(args.origin_yaw_deg), "--origin-name", args.origin_name]
    elif args.origin_marker_id:
        command += ["--origin-marker-id", args.origin_marker_id,
                    "--origin-yaw-deg", str(args.origin_yaw_deg), "--origin-name", args.origin_name]
    for option, value in (("--deploy-output", args.deploy_output), ("--video", args.video),
                          ("--frame-timestamps", args.frame_timestamps),
                          ("--labels2coords", args.labels2coords), ("--observations", args.observations),
                          ("--ceiling-z", args.ceiling_z)):
        if value is not None:
            command += [option, str(value)]
    return subprocess.call(command, cwd=HERE)


if __name__ == "__main__":
    raise SystemExit(main())
