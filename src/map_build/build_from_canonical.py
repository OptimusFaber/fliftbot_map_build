#!/usr/bin/env python3
"""Build, post-process and validate a complete map bundle from canonical data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=HERE, check=True)


def canonical_config(recording: Path, supplied: Path | None, name: str, required: bool = True) -> Path | None:
    selected = supplied.resolve() if supplied else recording / name
    if not selected.is_file():
        if required:
            raise FileNotFoundError(f"configuration is required and was not found: {selected}")
        return None
    return selected


def default_encoder_speed_sign(recording: Path, lidar_config: Path) -> float:
    for name in ("conversion_report.json", "recording_metadata.json"):
        path = recording / name
        if path.is_file():
            kind = json.loads(path.read_text(encoding="utf-8")).get("odometry_kind")
            if kind == "log":
                return -1.0
            if kind == "csv":
                return 1.0
    document = json.loads(lidar_config.read_text(encoding="utf-8"))
    if document.get("encoder_speed_sign") in (-1, -1.0, 1, 1.0):
        return float(document["encoder_speed_sign"])
    raise ValueError(
        "cannot infer encoder sign from recording metadata or lidar config; "
        "pass --encoder-speed-sign explicitly")


def odometry_kind(recording: Path) -> str | None:
    for name in ("conversion_report.json", "recording_metadata.json"):
        path = recording / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")).get("odometry_kind")
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_artifacts(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if path.is_file() and not str(relative).startswith("input/horizon/") and relative.name != "artifacts.json":
            artifacts.append({"path": str(relative), "size": path.stat().st_size, "sha256": sha256(path)})
    (run_dir / "artifacts.json").write_text(json.dumps(
        {"schema": "fliftbot.mapping_artifacts.v1", "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")


def export_marker_coordinates(run_dir: Path) -> None:
    marker_dir = run_dir / "markers"
    source = marker_dir / "marker_points_stored_ray_corrected.json"
    if not source.is_file():
        return
    report = json.loads(source.read_text(encoding="utf-8"))
    pb_to_local = np.load(run_dir / "web_map/pbstream_to_local_matrix.npy")
    pb_to_raster = np.load(run_dir / "web_map/pbstream_to_raster_matrix.npy")
    markers = []
    for marker in report.get("markers", []):
        x, y, z = marker["point_xyz_m"]
        local = pb_to_local @ np.array([x, y, 1.0])
        raster = pb_to_raster @ np.array([x, y, 1.0])
        markers.append({"marker_id": marker["marker_id"], "decoder_label": marker.get("decoder_label", "?"),
                        "decoder_label_votes": marker.get("decoder_label_votes", {}),
                        "pbstream_map_xyz_m": [x, y, z], "local_map_xyz_m": [float(local[0]), float(local[1]), z],
                        "web_raster_uv_px": [float(raster[0]), float(raster[1])],
                        "track_segments": marker.get("track_segments", []),
                        "observations": marker.get("observations", 0), "spread_m": marker.get("spread_m")})
    document = {"schema": "fliftbot.markers_map.v1", "status": report.get("status"),
                "identity_contract": "marker_id follows physical tracks; decoder_label is majority vote or ?",
                "markers": markers}
    (marker_dir / "markers_map.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rendered = marker_dir / "stored_ray_corrected_map.png"
    if rendered.is_file():
        shutil.copy2(rendered, marker_dir / "markers_map.png")


def append_geometry_options(command: list[str], args: argparse.Namespace,
                            camera: Path | None, robot: Path | None) -> None:
    for option, value in (("--camera-config", camera), ("--robot-config", robot),
                          ("--camera-position-file", args.camera_position_file)):
        if value is not None:
            command += [option, str(value)]
    for prefix in ("camera", "lidar"):
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            value = getattr(args, f"{prefix}_{axis}")
            if value is not None:
                command += [f"--{prefix}-{axis}", str(value)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--camera-position-file", type=Path)
    parser.add_argument("--lidar-config", type=Path)
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
    parser.add_argument("--cartographer-config", default="fliftbot_2d_offline.lua")
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

    # Resolve paths while still in the caller's working directory. Child
    # scripts run with cwd=map_build, so forwarding relative marker paths
    # unchanged would turn auto_map/... into map_build/auto_map/....
    for attribute in ("video", "frame_timestamps", "labels2coords", "observations",
                      "deploy_output"):
        value = getattr(args, attribute)
        if value is not None:
            setattr(args, attribute, value.resolve())

    recording = args.recording.resolve()
    camera = canonical_config(recording, args.camera_config, "camera.config", required=False)
    lidar = canonical_config(recording, args.lidar_config, "lidar.config")
    robot = canonical_config(recording, args.robot_config, "liftbot.config", required=False)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation = output / f"{args.name}_recording_validation.json"
    encoder_speed_sign = (args.encoder_speed_sign if args.encoder_speed_sign is not None
                          else default_encoder_speed_sign(recording, lidar))
    kind = odometry_kind(recording)
    expected_sign = {"log": -1.0, "csv": 1.0}.get(kind)
    if args.encoder_speed_sign is not None and expected_sign is not None and args.encoder_speed_sign != expected_sign:
        warnings.warn(
            f"--encoder-speed-sign={args.encoder_speed_sign:g} is unusual for odometry kind {kind}; "
            f"the usual value is {expected_sign:g}. The explicit value has priority.", stacklevel=1)
    run([sys.executable, str(HERE / "validate_recording.py"), str(recording), "--json", str(validation)])

    if args.dry_run:
        command = [sys.executable, str(HERE / "build_map.py"), "--recording", str(recording),
                   "--lidar-config", str(lidar), "--output", str(output), "--name", args.name,
                   "--dry-run"]
        append_geometry_options(command, args, camera, robot)
        if args.lidar_only:
            command.append("--lidar-only")
        if args.wheelbase is not None:
            command += ["--wheelbase", str(args.wheelbase)]
        command += ["--fork-position", args.fork_position,
                    "--lowered-forks-offset-m", str(args.lowered_forks_offset_m)]
        command += ["--encoder-speed-sign", str(encoder_speed_sign)]
        if args.steering_sign is not None:
            command += ["--steering-sign", str(args.steering_sign)]
        if args.mapping_yaw is not None:
            command += ["--mapping-yaw", str(args.mapping_yaw)]
        command += ["--odometry-time-offset", str(args.odometry_time_offset)]
        command += ["--cartographer-config", args.cartographer_config]
        run(command)
        print(output)
        return 0

    with tempfile.NamedTemporaryFile(prefix="map-build-result-", suffix=".txt") as marker:
        command = [sys.executable, str(HERE / "build_map.py"), "--recording", str(recording),
                   "--lidar-config", str(lidar), "--output", str(output), "--name", args.name,
                   "--result-path-file", marker.name]
        append_geometry_options(command, args, camera, robot)
        if args.lidar_only:
            command.append("--lidar-only")
        if args.wheelbase is not None:
            command += ["--wheelbase", str(args.wheelbase)]
        command += ["--fork-position", args.fork_position,
                    "--lowered-forks-offset-m", str(args.lowered_forks_offset_m)]
        command += ["--encoder-speed-sign", str(encoder_speed_sign)]
        if args.steering_sign is not None:
            command += ["--steering-sign", str(args.steering_sign)]
        if args.mapping_yaw is not None:
            command += ["--mapping-yaw", str(args.mapping_yaw)]
        command += ["--odometry-time-offset", str(args.odometry_time_offset)]
        command += ["--cartographer-config", args.cartographer_config]
        marker_inputs = (("--video", args.video), ("--frame-timestamps", args.frame_timestamps),
                         ("--labels2coords", args.labels2coords), ("--observations", args.observations),
                         ("--ceiling-z", args.ceiling_z))
        if args.observations or (args.video and args.frame_timestamps and args.labels2coords):
            command.append("--with-markers")
        for option, value in marker_inputs:
            if value is not None:
                command += [option, str(value)]
        run(command)
        run_dir = Path(Path(marker.name).read_text(encoding="utf-8").strip()).resolve()
    shutil.copy2(validation, run_dir / "recording_validation_report.json")

    web_command = [sys.executable, str(HERE / "build_web_map.py"), "--map-dir", str(run_dir / "map"),
                   "--output", str(run_dir / "web_map"), "--web-resolution-m-per-px",
                   str(args.web_resolution_m_per_px)]
    if args.no_web_align:
        web_command.append("--no-align")
    run(web_command)
    export_marker_coordinates(run_dir)
    if args.origin_map_xy or args.origin_marker_id:
        origin_command = [sys.executable, str(HERE / "set_map_origin.py"), "--web-map", str(run_dir / "web_map"),
                          "--yaw-deg", str(args.origin_yaw_deg), "--origin-name", args.origin_name]
        if args.origin_marker_id:
            origin_command += ["--marker-id", args.origin_marker_id,
                               "--markers", str(run_dir / "markers/markers_map.json")]
        else:
            origin_command += ["--origin-map-xy", str(args.origin_map_xy[0]), str(args.origin_map_xy[1])]
        run(origin_command)
        export_marker_coordinates(run_dir)
    bundle_report = run_dir / "validation_report.json"
    run([sys.executable, str(HERE / "validate_bundle.py"), str(run_dir), "--json", str(bundle_report)])
    refresh_artifacts(run_dir)
    if args.deploy_output:
        run([sys.executable, str(HERE / "package_bundle.py"), "--run", str(run_dir),
             "--output", str(args.deploy_output.resolve())])
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
