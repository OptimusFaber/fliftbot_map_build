#!/usr/bin/env python3
"""Convert raw M360 + move logger/encoder CSV into a canonical recording."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "realtime_mapping_scripts"))
from m360_to_run import write_horizon as write_m360_horizon  # type: ignore  # noqa: E402
from rplidar_to_run import (  # type: ignore  # noqa: E402
    POINT_TIME_MODES as RPLIDAR_POINT_TIME_MODES,
    RAW_MAGIC as RPLIDAR_MAGIC,
    write_horizon as write_rplidar_horizon,
)

LOCAL_TZ = ZoneInfo("Europe/Moscow")
LOG_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"


def parse_log_time(value: str) -> float:
    return datetime.strptime(value.strip(), LOG_FORMAT).replace(tzinfo=LOCAL_TZ).timestamp()


def nearest_slice(rows: list[dict], start: float, end: float) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("encoder source contains no valid rows")
    begin = min(range(len(rows)), key=lambda i: abs(rows[i]["enc_unix"] - start))
    finish = min(range(len(rows)), key=lambda i: abs(rows[i]["enc_unix"] - end))
    if begin > finish:
        begin, finish = finish, begin
    selected = rows[begin:finish + 1]
    return selected, {
        "lidar_start_unix": start, "lidar_end_unix": end,
        "selected_encoder_start_unix": rows[begin]["enc_unix"],
        "selected_encoder_end_unix": rows[finish]["enc_unix"],
        "start_abs_error_s": abs(rows[begin]["enc_unix"] - start),
        "end_abs_error_s": abs(rows[finish]["enc_unix"] - end),
        "source_start_index": begin, "source_end_index": finish,
    }


def load_log(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line_no, row in enumerate(csv.DictReader(stream), 2):
            raw_time = (row.get("enc_t") or "").strip()
            if not raw_time or raw_time.startswith("1970_"):
                continue
            try:
                timestamp = parse_log_time(raw_time)
                velocity = float(row["enc_v"])
                angle = float(row["enc_dgr"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid encoder row") from exc
            if not all(math.isfinite(v) for v in (timestamp, velocity, angle)):
                continue
            rows.append({"enc_t": raw_time, "enc_unix": timestamp,
                         "enc_v": velocity, "enc_dgr": angle,
                         "enc_available": True})
    rows.sort(key=lambda row: row["enc_unix"])
    return rows


def load_encoder_csv(path: Path) -> tuple[list[dict], int, int]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_sec", "linear_velocity_m_per_s", "wheel_angle_rad"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_no, row in enumerate(reader, 2):
            try:
                timestamp = float(row["timestamp_sec"])
                velocity = float(row["linear_velocity_m_per_s"])
                angle = math.degrees(float(row["wheel_angle_rad"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid encoder row") from exc
            if all(math.isfinite(v) for v in (timestamp, velocity, angle)):
                rows.append({"enc_unix": timestamp, "enc_v": velocity,
                             "enc_dgr": angle, "enc_available": True})
    rows.sort(key=lambda row: row["enc_unix"])
    unique = []
    duplicates = 0
    for row in rows:
        if unique and row["enc_unix"] == unique[-1]["enc_unix"]:
            duplicates += 1
            continue
        unique.append(row)
    for row in unique:
        row["enc_t"] = datetime.fromtimestamp(row["enc_unix"], LOCAL_TZ).strftime(LOG_FORMAT)
    return unique, len(rows), duplicates


def detect_odometry_kind(path: Path) -> str:
    """Detect the supported odometry dialect from its CSV header."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = {value.strip() for value in next(reader, [])}
    if {"enc_t", "enc_v", "enc_dgr"} <= header:
        return "log"
    if {"timestamp_sec", "linear_velocity_m_per_s", "wheel_angle_rad"} <= header:
        return "csv"
    raise ValueError(
        f"{path}: cannot detect odometry format from content; expected move logger "
        "enc_t/enc_v/enc_dgr or encoder timestamp_sec/linear_velocity_m_per_s/wheel_angle_rad"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--odometry-kind", choices=("auto", "log", "csv"), default="auto")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument(
        "--m360-point-time-mode", choices=("packet-host", "device-linear"),
        default="packet-host",
        help="experimental device-linear uses M360 TimeStamp + x/(HZ*2300)",
    )
    parser.add_argument(
        "--rplidar-point-time-mode", choices=RPLIDAR_POINT_TIME_MODES,
        default="azimuth-device-end",
        help="reconstruct RPLIDAR beam times from azimuth by default",
    )
    parser.add_argument("--lidar-start-unix", type=float)
    parser.add_argument("--lidar-end-unix", type=float)
    args = parser.parse_args()
    detected_kind = detect_odometry_kind(args.odometry.resolve())
    extension_kind = {".log": "log", ".csv": "csv"}.get(args.odometry.suffix.lower())
    if extension_kind and extension_kind != detected_kind:
        warnings.warn(
            f"odometry extension suggests {extension_kind}, but content is {detected_kind}; "
            f"using {detected_kind if args.odometry_kind == 'auto' else args.odometry_kind}",
            stacklevel=1,
        )
    if args.odometry_kind == "auto":
        kind = detected_kind
    else:
        kind = args.odometry_kind
        if kind != detected_kind:
            warnings.warn(
                f"--odometry-kind={kind} does not match detected {detected_kind} content; "
                "the explicit type has priority and conversion may fail",
                stacklevel=1,
            )
    out = args.output.resolve()
    horizon = out / "horizon"
    if any(horizon.glob("*.npz")) or (out / "encoder.jsonl").exists():
        parser.error("canonical output already contains data; choose a new --output directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.lidar_start_unix is not None and args.lidar_end_unix is not None:
        if args.lidar_start_unix >= args.lidar_end_unix:
            parser.error("--lidar-start-unix must be less than --lidar-end-unix")
    scan_path = args.scan.resolve()
    with scan_path.open("rb") as stream:
        scan_magic = stream.read(len(RPLIDAR_MAGIC))
    if scan_magic == RPLIDAR_MAGIC:
        if args.lidar_start_unix is not None or args.lidar_end_unix is not None:
            parser.error("LiDAR interval trimming is not implemented for RPLRAW1 captures")
        print("[convert] Detected RPLRAW1; converting RPLIDAR revolutions...", flush=True)
        lidar_report = write_rplidar_horizon(
            scan_path, horizon, point_time_mode=args.rplidar_point_time_mode)
    else:
        print("[convert] Detected M360RAW1; converting M360 packets...", flush=True)
        lidar_report = write_m360_horizon(
            scan_path, horizon,
            point_time_mode=args.m360_point_time_mode,
            start_unix=args.lidar_start_unix, end_unix=args.lidar_end_unix)
    if not lidar_report["first_ts"] or not lidar_report["last_ts"]:
        raise RuntimeError("LiDAR conversion produced no scans")
    if kind == "log":
        all_rows = load_log(args.odometry.resolve())
        rows, sync = nearest_slice(all_rows, lidar_report["first_ts"], lidar_report["last_ts"])
        encoder_stats = {"source_samples": len(all_rows), "duplicates_removed": 0}
    else:
        all_rows, source_samples, duplicates = load_encoder_csv(args.odometry.resolve())
        if not all_rows:
            raise ValueError("encoder CSV contains no valid rows")
        if args.lidar_start_unix is not None or args.lidar_end_unix is not None:
            rows, sync = nearest_slice(
                all_rows, lidar_report["first_ts"], lidar_report["last_ts"])
            sync["mode"] = "interval_csv"
        else:
            rows = all_rows
            sync = {"mode": "full_csv", "lidar_start_unix": lidar_report["first_ts"],
                    "lidar_end_unix": lidar_report["last_ts"],
                    "encoder_start_unix": rows[0]["enc_unix"],
                    "encoder_end_unix": rows[-1]["enc_unix"]}
        encoder_stats = {"source_samples": source_samples, "duplicates_removed": duplicates}
    write_jsonl(out / "encoder.jsonl", rows)
    copied_configs = {}
    for source_config, target_name in ((args.camera_config, "camera.config"),
                                       (args.lidar_config, "lidar.config"),
                                       (args.robot_config, "liftbot.config")):
        if source_config is not None:
            source_config = source_config.resolve()
            shutil.copy2(source_config, out / target_name)
            copied_configs[target_name] = {"source": str(source_config), "sha256": sha256(source_config)}
    if args.camera_config is not None:
        camera_source = args.camera_config.resolve()
        camera_document = json.loads(camera_source.read_text(encoding="utf-8"))
        intrinsics_value = camera_document.get("intrinsics")
        if intrinsics_value:
            intrinsics_source = Path(intrinsics_value).expanduser()
            if not intrinsics_source.is_absolute():
                intrinsics_source = (camera_source.parent / intrinsics_source).resolve()
            if not intrinsics_source.is_file():
                raise FileNotFoundError(f"camera intrinsics not found: {intrinsics_source}")
            intrinsics_name = "camera_intrinsics" + intrinsics_source.suffix
            shutil.copy2(intrinsics_source, out / intrinsics_name)
            camera_document["intrinsics"] = intrinsics_name
            (out / "camera.config").write_text(
                json.dumps(camera_document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            copied_configs[intrinsics_name] = {"source": str(intrinsics_source),
                                               "sha256": sha256(intrinsics_source)}
    metadata = {"schema": "fliftbot.canonical_recording.v1", "lidar_source": str(args.scan.resolve()),
                "odometry_source": str(args.odometry.resolve()), "odometry_kind": kind,
                "odometry_detected_kind": detected_kind,
                "sources": {"lidar_sha256": sha256(args.scan.resolve()),
                            "odometry_sha256": sha256(args.odometry.resolve())},
                "lidar": lidar_report, "encoder_samples": len(rows), "encoder": encoder_stats,
                "synchronization": sync, "configs": copied_configs}
    report = out / "conversion_report.json"
    report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(report, out / "recording_metadata.json")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
