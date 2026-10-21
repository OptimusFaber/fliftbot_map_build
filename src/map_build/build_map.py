#!/usr/bin/env python3
"""Run the existing standalone mapper from a canonical recording."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--camera-config", type=Path)
    p.add_argument("--camera-position-file", type=Path)
    p.add_argument("--lidar-config", type=Path, required=True)
    p.add_argument("--robot-config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="mapping")
    p.add_argument("--lidar-only", action="store_true")
    p.add_argument("--wheelbase", type=float, help="override robot wheelbase in metres")
    p.add_argument("--fork-position", choices=("raised", "lowered"), default="raised")
    p.add_argument("--lowered-forks-offset-m", type=float, default=0.05)
    for prefix in ("camera", "lidar"):
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            p.add_argument(f"--{prefix}-{axis}", type=float)
    p.add_argument("--encoder-speed-sign", type=float, choices=(-1.0, 1.0))
    p.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0))
    p.add_argument("--mapping-yaw", type=float)
    p.add_argument("--odometry-time-offset", type=float, default=0.0)
    p.add_argument("--cartographer-config", default="fliftbot_2d_offline.lua")
    p.add_argument("--with-markers", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--frame-timestamps", type=Path)
    p.add_argument("--labels2coords", type=Path)
    p.add_argument("--observations", type=Path)
    p.add_argument("--ceiling-z", type=float)
    p.add_argument("--result-path-file", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    command = [sys.executable, str(HERE / "run_pipeline.py"),
               "--lidar-config", str(args.lidar_config),
               "--lidar-data", str(args.recording / "horizon"),
               "--encoder", str(args.recording / "encoder.jsonl"),
               "--output", str(args.output), "--name", args.name]
    for option, value in (("--camera-config", args.camera_config), ("--robot-config", args.robot_config),
                          ("--camera-position-file", args.camera_position_file)):
        if value is not None: command += [option, str(value)]
    for prefix in ("camera", "lidar"):
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            value = getattr(args, f"{prefix}_{axis}")
            if value is not None: command += [f"--{prefix}-{axis}", str(value)]
    command.append("--with-markers" if args.with_markers else "--no-markers")
    for option, value in (("--video", args.video), ("--frame-timestamps", args.frame_timestamps),
                          ("--labels2coords", args.labels2coords), ("--observations", args.observations),
                          ("--ceiling-z", args.ceiling_z)):
        if value is not None:
            command += [option, str(value)]
    if args.lidar_only:
        command.append("--lidar-only")
    if args.wheelbase is not None:
        command += ["--wheelbase", str(args.wheelbase)]
    command += ["--fork-position", args.fork_position,
                "--lowered-forks-offset-m", str(args.lowered_forks_offset_m)]
    if args.encoder_speed_sign is not None:
        command += ["--encoder-speed-sign", str(args.encoder_speed_sign)]
    if args.steering_sign is not None:
        command += ["--steering-sign", str(args.steering_sign)]
    if args.mapping_yaw is not None:
        command += ["--mapping-yaw", str(args.mapping_yaw)]
    command += ["--odometry-time-offset", str(args.odometry_time_offset)]
    command += ["--cartographer-config", args.cartographer_config]
    if args.result_path_file:
        command += ["--result-path-file", str(args.result_path_file)]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command, cwd=HERE)

if __name__ == "__main__":
    raise SystemExit(main())
