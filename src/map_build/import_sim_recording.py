#!/usr/bin/env python3
"""Stage fliftbot_sim NPZ/wheel feedback as canonical data; never import GT."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from run_pipeline import hardlink_or_copy


def import_recording(recording: Path, output: Path, wheelbase: float = 1.410) -> dict:
    source = recording.resolve()
    if (source / "lidar/horizon").is_dir():
        source = source / "lidar"
    lidar = json.loads((source / "lidar.config").read_text(encoding="utf-8"))
    if lidar.get("source") != "simulation_robot_geometry":
        raise ValueError("input is not a fliftbot_sim recording (simulation_robot_geometry required)")
    packets = sorted((source / "horizon").glob("*.npz"))
    if not packets or not (source / "encoder.jsonl").is_file():
        raise ValueError("simulation input requires horizon/*.npz and encoder.jsonl")
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("canonical output must be empty")
    (output / "horizon").mkdir(parents=True)
    modes = {"copy": 0, "hardlink": 0}
    for path in packets:
        modes[hardlink_or_copy(path, output / "horizon" / path.name)] += 1
    # Explicit whitelist strips GT/legacy odometry fields from the wheel stream.
    with (source / "encoder.jsonl").open() as stream, (output / "encoder.jsonl").open("w") as target:
        for line in stream:
            row = json.loads(line)
            clean = {key: row[key] for key in ("enc_unix", "enc_v", "enc_dgr")}
            clean["enc_available"] = row.get("enc_available", True)
            target.write(json.dumps(clean) + "\n")
    lidar.update({"schema": "fliftbot.component_config.v1", "component": "lidar",
                  "name": "simulation_" + lidar["lidar_type"], "frame": "rotation_axis",
                  "points_frame": "sensor", "encoder_speed_sign": -1.0,
                  "max_range_m": lidar.get("max_range_m", 40.0),
                  "mapping_orientation_deg": dict(lidar["orientation_deg"])})
    (output / "lidar.config").write_text(json.dumps(lidar, indent=2) + "\n")
    robot = {"schema": "fliftbot.component_config.v1", "component": "robot",
             "name": "simulation", "frame": "rotation_axis", "wheelbase_m": wheelbase,
             "steering_sign": 1.0, "steering_offset_deg": 0.0}
    robot_source = source / "liftbot.config"
    if robot_source.is_file():
        robot = json.loads(robot_source.read_text(encoding="utf-8"))
    (output / "liftbot.config").write_text(json.dumps(robot, indent=2) + "\n")
    report = {"schema": "fliftbot.canonical_recording.v1", "source": str(source),
              "odometry_kind": "simulation", "clock": "simulation time (seconds)",
              "lidar": {"frames": len(packets), "staging": modes},
              "inputs": "horizon XY/beam times and measured encoder only; GT excluded"}
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True, help="sim RUN or RUN/lidar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelbase", type=float, default=1.410, help="legacy records without liftbot.config")
    args = parser.parse_args()
    print(json.dumps(import_recording(args.recording, args.output.resolve(), args.wheelbase), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
