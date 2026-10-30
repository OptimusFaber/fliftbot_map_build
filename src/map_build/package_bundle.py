#!/usr/bin/env python3
"""Create a calibrated, standalone map catalog for fliftbot_lidar_nav."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.resolve()
    required = ("map/map.pbstream", "map/map.pgm", "map/map.yaml", "map/map.png",
                "web_map/map_metadata.json", "web_map/pbstream_to_local_matrix.npy",
                "run_parameters.json")
    for relative in required:
        if not (source / relative).is_file():
            parser.error(f"incomplete mapping run: missing {relative}")
    if output.exists() and any(output.iterdir()):
        parser.error("deploy output must be empty; choose a new --output directory")
    parameters = json.loads((source / "run_parameters.json").read_text(encoding="utf-8"))
    lidar, robot = parameters["lidar"], parameters["robot"]
    orientation = lidar.get("mapping_orientation_deg", lidar["orientation_deg"])
    if any(abs(float(orientation.get(axis, 0))) > 1e-9 for axis in ("roll", "pitch")):
        parser.error("2D navigation requires zero LiDAR roll/pitch")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in ("map/map.pbstream", "map/map.pgm", "map/map.yaml", "map/map.png",
                     "web_map/map_dark.jpg", "web_map/map_light.jpg", "web_map/map_metadata.json",
                     "web_map/map_transform_matrix.npy", "web_map/pbstream_to_local_matrix.npy",
                     "web_map/pbstream_to_raster_matrix.npy"):
        src = source / relative
        if src.is_file():
            destination = output / src.name
            shutil.copy2(src, destination)
            files.append(destination)
    explicit_reference = source / "web_map/origin_reference.json"
    if explicit_reference.is_file():
        shutil.copy2(explicit_reference, output / "origin_reference.json")
    else:
        metadata = json.loads((source / "web_map/map_metadata.json").read_text(encoding="utf-8"))
        origin = metadata["origin"]
        pb_to_local = np.load(source / "web_map/pbstream_to_local_matrix.npy")
        yaw_deg = float(origin.get("origin_map_yaw_deg",
                                   -math.degrees(math.atan2(pb_to_local[1, 0], pb_to_local[0, 0]))))
        reference = {"schema": "fliftbot.map_origin_reference.v1", "frame": "cartographer_map",
                     "origin_name": origin.get("name", origin.get("type", "web_map_lower_left")),
                     "origin_map_xy_m": origin["pbstream_map_xy_m"],
                     "origin_map_yaw_rad": yaw_deg * 3.141592653589793 / 180.0,
                     "origin_map_yaw_deg": yaw_deg, "axis_contract": "+X right, +Y up",
                     "source": "map.pbstream"}
        (output / "origin_reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    files.append(output / "origin_reference.json")
    navigation_config = {
        "schema": "fliftbot.navigation_config.v1",
        "lidar_type": lidar["lidar_type"],
        "points_frame": lidar.get("points_frame", "sensor"),
        "scan_to_base": {"x_m": float(lidar["position_m"]["x"]),
                         "y_m": float(lidar["position_m"]["y"]),
                         "yaw_deg": float(orientation.get("yaw", 0))},
        "max_range_m": float(lidar["max_range_m"]),
        "odometry": {"speed_sign": float(lidar["encoder_speed_sign"]),
                     "steering_sign": float(robot.get("steering_sign", 1)),
                     "steering_zero_deg": -float(robot.get("steering_offset_deg", 0)) /
                                          float(robot.get("steering_sign", 1)),
                     "wheelbase_m": float(robot["wheelbase_m"]),
                     "model": "drive_wheel"},
    }
    config_path = output / "navigation_config.json"
    config_path.write_text(json.dumps(navigation_config, indent=2) + "\n", encoding="utf-8")
    files.append(config_path)
    manifest = {"schema": "fliftbot.deploy_catalog.v1", "source_run": str(source),
                "deployment": "pass this directory to fliftbot_lidar_nav/nav.sh --map-dir",
                "files": [{"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
                          for path in sorted(files)]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
