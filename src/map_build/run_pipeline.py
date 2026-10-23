#!/usr/bin/env python3
"""One entry point for Cartographer mapping, trajectory and ceiling markers."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_CAMERA_POSITION_M = {"x": 1.077, "y": -0.274, "z": 1.248}
DEFAULT_CAMERA_ORIENTATION_DEG = {"roll": 0.0, "pitch": 0.0, "yaw": -90.0}
DEFAULT_WHEELBASE_M = 1.39


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def component(path: Path, expected: str) -> dict[str, Any]:
    data = read_json(path)
    if data.get("schema") != "fliftbot.component_config.v1" or data.get("component") != expected:
        raise ValueError(f"{path}: expected fliftbot.component_config.v1 component={expected}")
    return data


def link(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def hardlink_or_copy(source: Path, target: Path) -> str:
    """Materialize a regular input file without duplicating it when possible."""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.copy2(source, target)
        return "copy"


def stage_mapping_inputs(horizon: Path, encoder: Path, run_input: Path) -> dict[str, Any]:
    """Create the real directory/files required by the standalone input guard.

    The guard rejects symlinks by design. Hard links satisfy that contract and
    use no additional payload space on the usual same-filesystem setup.
    """
    if not horizon.is_dir():
        raise FileNotFoundError(f"LiDAR horizon directory not found: {horizon}")
    packets = sorted(horizon.glob("*.npz"))
    if not packets:
        raise FileNotFoundError(f"No LiDAR packets found: {horizon}/*.npz")
    if not encoder.is_file():
        raise FileNotFoundError(f"Encoder file not found: {encoder}")
    staged_horizon = run_input / "horizon"
    staged_horizon.mkdir(parents=True)
    modes: dict[str, int] = {"hardlink": 0, "copy": 0}
    for source in packets:
        mode = hardlink_or_copy(source, staged_horizon / source.name)
        modes[mode] += 1
    encoder_mode = hardlink_or_copy(encoder, run_input / "encoder.jsonl")
    return {"lidar_packets": len(packets), "lidar_file_modes": modes, "encoder_file_mode": encoder_mode}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_meta(path: Path) -> tuple[float, list[float], str]:
    # map_server YAML emitted here is deliberately simple; avoid another dependency.
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    origin = json.loads(values["origin"].replace("'", '"'))
    return float(values["resolution"]), [float(v) for v in origin], values["image"].strip("'\"")


def render_trajectory(map_dir: Path, output: Path, trajectory_path: Path | None = None) -> None:
    resolution, origin, image_name = map_meta(map_dir / "map.yaml")
    image = Image.open(map_dir / image_name).convert("RGB")
    trajectory = read_json(trajectory_path or (map_dir / "trajectory.json"))
    nodes = trajectory.get("nodes") or trajectory.get("poses") or []
    points = []
    for row in nodes:
        x, y = float(row["x"]), float(row["y"])
        gx = (x - origin[0]) / resolution
        gy = image.height - 1 - (y - origin[1]) / resolution
        points.append((gx, gy))
    draw = ImageDraw.Draw(image)
    if len(points) > 1:
        draw.line(points, fill=(225, 29, 72), width=max(2, round(0.06 / resolution)))
        r = max(3, round(0.12 / resolution))
        for point, color in ((points[0], (22, 163, 74)), (points[-1], (37, 99, 235))):
            draw.ellipse((point[0]-r, point[1]-r, point[0]+r, point[1]+r), fill=color)
    image.save(output)


def lidar_to_robot_trajectory(source: Path, output: Path, lidar: dict[str, Any]) -> None:
    """Convert Cartographer LiDAR poses to the forklift rotation-axis poses."""
    document = read_json(source)
    rows = document.get("poses") or document.get("nodes")
    if rows is None:
        raise ValueError(f"No poses/nodes in {source}")
    mount = lidar["position_m"]
    mount_yaw = math.radians(float(lidar["mapping_orientation_deg"].get("yaw", 0.0)))
    converted = []
    for row in rows:
        item = dict(row)
        lidar_yaw = 2.0 * math.atan2(float(row.get("qz", 0.0)), float(row.get("qw", 1.0)))
        axis_yaw = lidar_yaw - mount_yaw
        c, s = math.cos(axis_yaw), math.sin(axis_yaw)
        item["x"] = float(row["x"]) - c * float(mount["x"]) + s * float(mount["y"])
        item["y"] = float(row["y"]) - s * float(mount["x"]) - c * float(mount["y"])
        item["z"] = float(row.get("z", 0.0)) - float(mount.get("z", 0.0))
        item["qx"], item["qy"] = 0.0, 0.0
        item["qz"], item["qw"] = math.sin(axis_yaw / 2.0), math.cos(axis_yaw / 2.0)
        converted.append(item)
    key = "poses" if "poses" in document else "nodes"
    result = dict(document)
    result[key] = converted
    result["schema"] = "fliftbot.robot_axis_trajectory.v1"
    result["tracking_frame"] = "rotation_axis"
    result["converted_from"] = str(source.resolve())
    result["lidar_mount_position_m"] = mount
    result["lidar_mapping_yaw_deg"] = math.degrees(mount_yaw)
    write_json(output, result)


def wheel_odometry_trajectory(encoder_path: Path, output: Path, robot: dict[str, Any],
                              speed_sign: float) -> None:
    rows = []
    for line in encoder_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("enc_available", True):
            rows.append((float(item["enc_unix"]), float(item["enc_v"]), float(item["enc_dgr"])))
    rows = sorted(dict((row[0], row) for row in rows).values())
    wheelbase = float(robot["wheelbase_m"])
    steering_sign = float(robot.get("steering_sign", 1.0))
    steering_offset = math.radians(float(robot.get("steering_offset_deg", 0.0)))
    x = y = yaw = 0.0
    poses = []
    for index, (timestamp, velocity, steering_deg) in enumerate(rows):
        if index:
            previous = rows[index - 1]
            dt = timestamp - previous[0]
            if 0.0 < dt <= 0.5:
                speed = speed_sign * 0.5 * (velocity + previous[1])
                steering = steering_sign * math.radians(0.5 * (steering_deg + previous[2])) + steering_offset
                angular_speed = speed * math.sin(steering) / wheelbase
                theta = angular_speed * dt
                forward_speed = speed * math.cos(steering)
                if abs(angular_speed) < 1e-9:
                    dx, dy = forward_speed * dt, 0.0
                else:
                    dx = forward_speed * math.sin(theta) / angular_speed
                    dy = forward_speed * (1.0 - math.cos(theta)) / angular_speed
                x += math.cos(yaw) * dx - math.sin(yaw) * dy
                y += math.sin(yaw) * dx + math.cos(yaw) * dy
                yaw += theta
        poses.append({"timestamp_unix": timestamp, "x": x, "y": y, "z": 0.0,
                      "qx": 0.0, "qy": 0.0, "qz": math.sin(yaw / 2.0), "qw": math.cos(yaw / 2.0)})
    write_json(output, {"schema": "fliftbot.wheel_odometry_trajectory.v1",
                        "tracking_frame": "rotation_axis", "origin": "first_encoder_sample",
                        "wheelbase_m": wheelbase, "encoder_speed_sign": speed_sign,
                        "steering_sign": steering_sign,
                        "steering_offset_deg": float(robot.get("steering_offset_deg", 0.0)),
                        "poses": poses})


def camera_sidecars(camera: dict[str, Any], run_input: Path, config_base: Path) -> tuple[Path, Path]:
    position = camera["position_m"]
    angles = camera["orientation_deg"]
    camera_position = run_input / "camera_position.txt"
    camera_position.write_text(
        f"{position['x']*100:.9g} {position['y']*100:.9g} {position['z']*100:.9g}\n"
        f"{math.radians(angles['roll']):.12g} {math.radians(angles['pitch']):.12g} {math.radians(angles['yaw']):.12g}\n",
        encoding="utf-8",
    )
    intrinsics = resolve(config_base, camera.get("intrinsics"))
    if intrinsics is None:
        raise ValueError("camera config has no intrinsics")
    camera_json = run_input / "camera.json"
    shutil.copy2(intrinsics, camera_json)
    return camera_position, camera_json


def read_camera_position_file(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2 or len(lines[0]) < 3 or len(lines[1]) < 2:
        raise ValueError(f"{path}: expected XYZ in centimetres and roll/pitch[/yaw] in radians")
    position = dict(zip(("x", "y", "z"), (float(value) / 100.0 for value in lines[0][:3])))
    angles = dict(zip(("roll", "pitch", "yaw"),
                      (math.degrees(float(value)) for value in lines[1][:3])))
    return position, angles


def make_observations(inputs: dict[str, Path | None], run_input: Path, camera_json: Path,
                      camera_position: Path, output_dir: Path) -> Path:
    existing = inputs.get("observations")
    if existing:
        return Path(existing)
    required = ("video", "frame_timestamps", "labels2coords")
    if any(not inputs.get(key) for key in required):
        raise ValueError("markers need observations or video + frame_timestamps + labels2coords")
    video_dir = run_input / "video"
    video_dir.mkdir()
    video = Path(inputs["video"])
    link(video, video_dir / f"{video_dir.name}{video.suffix}")
    link(Path(inputs["frame_timestamps"]), video_dir / "frame_timestamps.csv")
    command = [
        sys.executable, str(PROJECT_ROOT / "src/auto_map/get_video_data.py"),
        "--video", str(video_dir), "--camera-json", str(camera_json),
        "--camera-position", str(camera_position), "--labels", str(inputs["labels2coords"]),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT / "src/auto_map", check=True)
    observations = video_dir / "observations.json"
    if not observations.exists():
        raise RuntimeError("video processor did not create observations.json")
    copied = output_dir / "observations.json"
    shutil.copy2(observations, copied)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional integrated run JSON preset")
    parser.add_argument("--camera-config", type=Path, help="camera component config")
    parser.add_argument("--camera-position-file", type=Path,
                        help="legacy camera.txt: XYZ cm, then roll/pitch[/yaw] radians")
    parser.add_argument("--lidar-config", type=Path, help="LiDAR component config")
    parser.add_argument("--robot-config", type=Path, help="forklift component config")
    parser.add_argument("--lidar-data", type=Path, help="horizon directory (overrides config)")
    parser.add_argument("--encoder", type=Path, help="encoder.jsonl (overrides config)")
    parser.add_argument("--video", type=Path, help="camera video for marker localization")
    parser.add_argument("--frame-timestamps", type=Path, help="frame_timestamps.csv for video")
    parser.add_argument("--labels2coords", type=Path, help="marker labels2coords JSON/pickle")
    parser.add_argument("--observations", type=Path, help="ready observations.json; skips video processing")
    parser.add_argument("--output", type=Path, help="result root (overrides config)")
    parser.add_argument("--name", default=None, help="name included in the result directory")
    parser.add_argument("--lidar-x", type=float); parser.add_argument("--lidar-y", type=float); parser.add_argument("--lidar-z", type=float)
    parser.add_argument("--yaw", "--lidar-yaw", dest="lidar_yaw", type=float, help="override LiDAR yaw in degrees")
    parser.add_argument("--mapping-yaw", type=float, help="override effective LiDAR yaw used by Cartographer")
    parser.add_argument("--max-range", type=float, help="maximum LiDAR range used by Cartographer, metres")
    parser.add_argument("--roll", "--lidar-roll", dest="lidar_roll", type=float, help="override LiDAR roll in degrees")
    parser.add_argument("--pitch", "--lidar-pitch", dest="lidar_pitch", type=float, help="override LiDAR pitch in degrees")
    parser.add_argument("--camera-x", type=float); parser.add_argument("--camera-y", type=float); parser.add_argument("--camera-z", type=float)
    parser.add_argument("--camera-roll", type=float); parser.add_argument("--camera-pitch", type=float); parser.add_argument("--camera-yaw", type=float)
    parser.add_argument("--wheelbase", type=float, help="override wheelbase in metres")
    parser.add_argument(
        "--fork-position", choices=("raised", "lowered"), default="raised",
        help="fork position; lowered adds the configured longitudinal geometry offset (default: raised)",
    )
    parser.add_argument(
        "--lowered-forks-offset-m", type=float, default=0.05,
        help="longitudinal offset added to camera x, LiDAR x and wheelbase for lowered forks",
    )
    parser.add_argument("--encoder-speed-sign", type=float, choices=(-1.0, 1.0), help="override encoder velocity sign")
    parser.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0), help="steering angle sign (default: robot config or +1)")
    parser.add_argument("--steering-offset-deg", type=float,
                        help="constant angle added to encoder steering measurements, degrees")
    parser.add_argument("--odometry-time-offset", type=float, default=0.0,
                        help="seconds added to encoder timestamps before Cartographer")
    parser.add_argument("--cartographer-config", default="fliftbot_2d_offline.lua",
                        help="Lua configuration filename in the Cartographer config directory")
    parser.add_argument("--cartographer-config-dir", type=Path,
                        help="optional directory containing the Lua configuration")
    odometry_group = parser.add_mutually_exclusive_group()
    odometry_group.add_argument(
        "--lidar-only", action="store_true",
        help="do not feed encoder odometry to Cartographer (encoder is still saved with the run)",
    )
    odometry_group.add_argument(
        "--use-odometry", action="store_true",
        help="feed encoder-derived odometry to Cartographer (default)",
    )
    marker_group = parser.add_mutually_exclusive_group()
    marker_group.add_argument("--with-markers", action="store_true", help="build marker overlay")
    marker_group.add_argument("--no-markers", action="store_true", help="skip marker processing")
    parser.add_argument("--ceiling-z", type=float, help="measured ceiling height in metres")
    parser.add_argument("--dry-run", action="store_true", help="validate and materialize inputs only")
    parser.add_argument("--result-path-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        config_path = args.config.resolve()
        spec = read_json(config_path)
        if spec.get("schema") != "fliftbot.integrated_mapping_run.v1":
            raise ValueError("unsupported run config schema")
        base = config_path.parent
    else:
        config_path = None
        base = Path.cwd()
        spec = {"schema": "fliftbot.integrated_mapping_run.v1", "configs": {}, "inputs": {}, "markers": {}}
    config_refs = spec.get("configs", {})

    def select_config(cli_path: Path | None, key: str, required: bool = True) -> Path | None:
        selected = cli_path.resolve() if cli_path else resolve(base, config_refs.get(key))
        if selected is None and required:
            raise ValueError(f"--{key if key != 'robot' else 'robot'}-config is required without a preset")
        return selected

    camera_path = select_config(args.camera_config, "camera", required=False)
    lidar_path = select_config(args.lidar_config, "lidar")
    robot_path = select_config(args.robot_config, "robot", required=False)
    camera = component(camera_path, "camera") if camera_path else {
        "schema": "fliftbot.component_config.v1", "component": "camera", "name": "default",
        "frame": "rotation_axis", "position_m": dict(DEFAULT_CAMERA_POSITION_M),
        "orientation_deg": dict(DEFAULT_CAMERA_ORIENTATION_DEG),
        "intrinsics": str((PROJECT_ROOT / "configs/camera_intrinsics.json").resolve()),
    }
    lidar = component(lidar_path, "lidar")
    robot = component(robot_path, "robot") if robot_path else {
        "schema": "fliftbot.component_config.v1", "component": "robot", "name": "default",
        "frame": "rotation_axis", "wheelbase_m": DEFAULT_WHEELBASE_M, "steering_offset_deg": 0.0,
    }
    camera["position_m"] = dict(camera["position_m"])
    camera["orientation_deg"] = dict(camera.get("orientation_deg", DEFAULT_CAMERA_ORIENTATION_DEG))
    if args.camera_position_file:
        sidecar_position, sidecar_angles = read_camera_position_file(args.camera_position_file.resolve())
        if not camera_path:
            camera["position_m"].update(sidecar_position)
        camera["orientation_deg"].update({key: sidecar_angles[key] for key in ("roll", "pitch")})
    for key in ("roll", "pitch", "yaw"):
        value = getattr(args, f"camera_{key}")
        if value is not None: camera["orientation_deg"][key] = value
    lidar["orientation_deg"] = dict(lidar["orientation_deg"])
    lidar["position_m"] = dict(lidar["position_m"])
    for key in ("yaw", "roll", "pitch"):
        value = getattr(args, f"lidar_{key}")
        if value is not None:
            lidar["orientation_deg"][key] = value
    lidar["mapping_orientation_deg"] = dict(
        lidar.get("mapping_orientation_deg", lidar["orientation_deg"])
    )
    if args.mapping_yaw is not None:
        lidar["mapping_orientation_deg"]["yaw"] = args.mapping_yaw
    if args.max_range is not None:
        lidar["max_range_m"] = args.max_range
    if float(lidar.get("max_range_m", 0.0)) <= 0:
        raise ValueError("LiDAR config must define a positive max_range_m")
    if args.lowered_forks_offset_m < 0:
        raise ValueError("--lowered-forks-offset-m must be non-negative")
    fork_offset_m = args.lowered_forks_offset_m if args.fork_position == "lowered" else 0.0
    base_geometry = {
        "camera_x_m": float(camera["position_m"]["x"]),
        "lidar_x_m": float(lidar["position_m"]["x"]),
        "wheelbase_m": float(robot["wheelbase_m"]),
    }
    if fork_offset_m:
        camera["position_m"] = dict(camera["position_m"])
        lidar["position_m"] = dict(lidar["position_m"])
        camera["position_m"]["x"] = base_geometry["camera_x_m"] + fork_offset_m
        lidar["position_m"]["x"] = base_geometry["lidar_x_m"] + fork_offset_m
        robot["wheelbase_m"] = base_geometry["wheelbase_m"] + fork_offset_m
    # Explicit CLI geometry is always the final authority, including when a
    # fork-state correction was requested.
    for key in ("x", "y", "z"):
        camera_value = getattr(args, f"camera_{key}")
        lidar_value = getattr(args, f"lidar_{key}")
        if camera_value is not None:
            camera["position_m"][key] = camera_value
        if lidar_value is not None:
            lidar["position_m"][key] = lidar_value
    if args.wheelbase is not None:
        robot["wheelbase_m"] = args.wheelbase
    if args.encoder_speed_sign is not None:
        lidar["encoder_speed_sign"] = args.encoder_speed_sign
    if args.steering_sign is not None:
        robot["steering_sign"] = args.steering_sign
    if args.steering_offset_deg is not None:
        robot["steering_offset_deg"] = args.steering_offset_deg
    if abs(float(lidar["mapping_orientation_deg"].get("roll", 0))) > 1e-9 or abs(float(lidar["mapping_orientation_deg"].get("pitch", 0))) > 1e-9:
        raise ValueError("2D Cartographer cannot apply non-zero LiDAR roll/pitch; level the 3D cloud first")

    raw_inputs = spec.get("inputs", {})
    inputs = {key: resolve(base, value) for key, value in raw_inputs.items()}
    cli_inputs = {
        "lidar": args.lidar_data, "encoder": args.encoder, "video": args.video,
        "frame_timestamps": args.frame_timestamps, "labels2coords": args.labels2coords,
        "observations": args.observations,
    }
    for key, value in cli_inputs.items():
        if value is not None:
            inputs[key] = value.resolve()
    if not inputs.get("lidar") or not inputs.get("encoder"):
        raise ValueError("lidar and encoder inputs are required")

    root = (args.output or resolve(base, spec.get("output")) or (PROJECT_ROOT / "ros-data/results")).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or spec.get("name", "mapping")
    out = root / f"{stamp}_{run_name}"
    run_input, map_dir = out / "input", out / "map"
    run_input.mkdir(parents=True)
    map_dir.mkdir()
    lidar_data = Path(inputs["lidar"])
    horizon = lidar_data / "horizon" if (lidar_data / "horizon").is_dir() else lidar_data
    staging = stage_mapping_inputs(horizon, Path(inputs["encoder"]), run_input)
    lidar_runtime = {
        "format": "lidar.config.v1", "lidar_type": lidar.get("lidar_type"),
        "points_frame": lidar.get("points_frame", "sensor"), "position_m": lidar["position_m"],
        "orientation_deg": lidar["mapping_orientation_deg"], "ray_correction_deg": lidar.get("ray_correction_deg", {"roll": 0, "pitch": 0, "yaw": 0}),
        "encoder_speed_sign": lidar.get("encoder_speed_sign", -1.0),
    }
    write_json(run_input / "lidar.config", lidar_runtime)
    marker_lidar_runtime = dict(lidar_runtime)
    marker_lidar_runtime["orientation_deg"] = lidar["orientation_deg"]
    write_json(run_input / "lidar.markers.config", marker_lidar_runtime)
    camera_position, camera_json = camera_sidecars(
        camera, run_input, camera_path.parent if camera_path else PROJECT_ROOT)
    markers = dict(spec.get("markers", {}))
    if args.with_markers:
        markers["enabled"] = True
    if args.no_markers:
        markers["enabled"] = False
    if args.ceiling_z is not None:
        markers["ceiling_z_m"] = args.ceiling_z
    resolved = {
        "schema": "fliftbot.integrated_mapping_parameters.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(config_path) if config_path else None, "camera": camera, "lidar": lidar, "robot": robot,
        "inputs": {key: str(value) if value else None for key, value in inputs.items()},
        "input_staging": staging, "output": str(out), "markers": markers,
        "mapping": {"use_odometry": not args.lidar_only, "cartographer_config": args.cartographer_config},
        "fork_geometry": {
            "position": args.fork_position,
            "lowered_offset_m": fork_offset_m,
            "base": base_geometry,
            "effective": {
                "camera_x_m": float(camera["position_m"]["x"]),
                "lidar_x_m": float(lidar["position_m"]["x"]),
                "wheelbase_m": float(robot["wheelbase_m"]),
            },
        },
        "odometry_time_offset_s": args.odometry_time_offset,
    }
    write_json(out / "run_parameters.json", resolved)
    if args.dry_run:
        if args.result_path_file:
            args.result_path_file.resolve().write_text(str(out) + "\n", encoding="utf-8")
        print(out)
        return 0

    runner = PROJECT_ROOT / "src/map_build/run_standalone_cartographer.sh"
    config_dir = args.cartographer_config_dir.resolve() if args.cartographer_config_dir else \
        PROJECT_ROOT / "configs/cartographer"
    mapping_command = [
        str(runner), "--run_dir", str(run_input), "--output", str(map_dir),
        "--configuration_dir", str(config_dir),
        "--configuration_file", args.cartographer_config,
        "--wheelbase", str(robot["wheelbase_m"]), "--steering_offset_deg", str(robot.get("steering_offset_deg", 0.0)),
        "--steering_sign", str(robot.get("steering_sign", 1.0)),
        "--encoder_speed_sign", str(lidar_runtime["encoder_speed_sign"]),
        "--max_range", str(lidar["max_range_m"]),
        "--odometry_time_offset", str(args.odometry_time_offset),
    ]
    if args.lidar_only:
        mapping_command.append("--use_odometry=false")
    subprocess.run(mapping_command, check=True)
    shutil.copy2(map_dir / "map.png", out / "map.png")
    robot_trajectory = out / "trajectory_robot_axis.json"
    lidar_to_robot_trajectory(map_dir / "trajectory.json", robot_trajectory, lidar)
    wheel_odometry_trajectory(run_input / "encoder.jsonl", out / "trajectory_wheel_odometry.json",
                              robot, float(lidar_runtime["encoder_speed_sign"]))
    render_trajectory(map_dir, out / "map_with_lidar_trajectory.png")
    render_trajectory(map_dir, out / "map_with_trajectory.png", robot_trajectory)
    shutil.copy2(out / "map_with_trajectory.png", map_dir / "track.png")

    if markers.get("enabled", False):
        observations = make_observations(inputs, run_input, camera_json, camera_position, out)
        marker_dir = out / "markers"
        command = [
            sys.executable, str(PROJECT_ROOT / "src/fliftbot_mapping/evaluation/ceiling_marker_localization_stored_ray_corrected.py"),
            "--observations", str(observations), "--trajectory", str(map_dir / "trajectory.json"),
            "--map-dir", str(map_dir), "--camera-position", str(camera_position),
            "--camera-json", str(camera_json),
            "--lidar-config", str(run_input / "lidar.markers.config"), "--out-dir", str(marker_dir),
        ]
        if markers.get("ceiling_z_m") is not None:
            command += ["--ceiling-z", str(markers["ceiling_z_m"])]
        write_json(marker_dir / "run_parameters.json", {
            "schema": "fliftbot.marker_run_parameters.v1",
            "camera": camera, "lidar": marker_lidar_runtime,
            "inputs": {"observations": str(observations), "trajectory": str(map_dir / "trajectory.json")},
            "parameters": markers, "command": command,
        })
        subprocess.run(command, check=True)
        shutil.copy2(marker_dir / "stored_ray_corrected_map.png", out / "map_with_markers.png")

    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and "input/horizon" not in str(path):
            artifacts.append({"path": str(path.relative_to(out)), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "artifacts.json", {"schema": "fliftbot.mapping_artifacts.v1", "artifacts": artifacts})
    if args.result_path_file:
        args.result_path_file.resolve().write_text(str(out) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
