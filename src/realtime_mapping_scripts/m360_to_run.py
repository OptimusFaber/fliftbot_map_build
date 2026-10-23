#!/usr/bin/env python3
"""Convert an M360RAW1 capture and encoder data into a canonical run.

The resulting ``horizon/*.npz`` files are consumed by this workspace's
offline mapper. They are not a Cartographer input by themselves; Cartographer
needs a ROS 2 bag containing LaserScan/odometry/TF messages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from receive_scan import (  # noqa: E402
    HEADER,
    INVALID_DISTANCE,
    POINTS_PER_PACKET,
    POINTS_PER_SCAN,
    decode_packet,
    iter_raw_records,
)

POINT_TIME_MODES = ("packet-host", "device-linear")


def _device_to_host_clock(m360_path: Path, start_unix: float | None,
                          end_unix: float | None) -> tuple[float, float, dict]:
    """Fit host packet-receive time to the M360 device clock.

    The manual defines TimeStamp as the first firing in a packet.  A UDP
    datagram can only be received after all 115 firings, so the fit pairs host
    receive time with the nominal device time immediately after the packet.
    Two robust least-squares passes reject scheduler/network-delay outliers.
    """
    device_end, host_receive = [], []
    margin = 1.0
    with m360_path.open("rb") as handle:
        handle.seek(0)
    for host_t, raw in iter_raw_records(str(m360_path)):
        if start_unix is not None and host_t < start_unix - margin:
            continue
        if end_unix is not None and host_t > end_unix + margin:
            break
        _version, _revolution, frequency_hz, _encoder, low, high = HEADER.unpack_from(raw)
        if frequency_hz <= 0:
            continue
        first = high + low / 2**32
        device_end.append(first + POINTS_PER_PACKET / (POINTS_PER_SCAN * frequency_hz))
        host_receive.append(host_t)
    if len(device_end) < 20:
        raise ValueError("not enough M360 packets to fit device clock to host clock")
    x = np.asarray(device_end, dtype=np.float64)
    y = np.asarray(host_receive, dtype=np.float64)
    x0, y0 = float(np.median(x)), float(np.median(y))
    keep = np.ones(len(x), dtype=bool)
    slope, intercept_centered = 1.0, 0.0
    for _ in range(2):
        slope, intercept_centered = np.polyfit(x[keep] - x0, y[keep] - y0, 1)
        residual = (y - y0) - (slope * (x - x0) + intercept_centered)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(0.001, 6.0 * 1.4826 * mad)
        keep = np.abs(residual - median) <= threshold
    intercept = y0 + intercept_centered - slope * x0
    residual = y - (slope * x + intercept)
    report = {
        "mode": "device-linear",
        "manual_contract": "TimeStamp is first point; dt=1/(HZ*2300)",
        "clock_model": "host_receive = slope * device_packet_end + intercept",
        "slope": float(slope),
        "drift_ppm": float((slope - 1.0) * 1e6),
        "intercept_s": float(intercept),
        "samples": int(len(x)),
        "inliers": int(keep.sum()),
        "residual_abs_p95_ms": float(np.percentile(np.abs(residual[keep]), 95) * 1e3),
    }
    return float(slope), float(intercept), report


def stamp_name(unix_t: float) -> str:
    text = datetime.fromtimestamp(unix_t, tz=timezone.utc).isoformat()
    return text.replace(":", "-").replace("+00:00", "+00-00")


def enc_t_name(unix_t: float) -> str:
    dt = datetime.fromtimestamp(unix_t, tz=timezone.utc)
    return dt.strftime("%Y_%m_%d_%H_%M_%S_") + f"{dt.microsecond:06d}"


def write_encoder_jsonl(csv_path: Path, out_path: Path) -> dict:
    samples = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "timestamp_sec",
            "linear_velocity_m_per_s",
            "wheel_angle_rad",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{csv_path}: missing encoder columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                unix_t = float(row["timestamp_sec"])
                velocity = float(row["linear_velocity_m_per_s"])
                wheel_angle = float(row["wheel_angle_rad"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: invalid encoder sample"
                ) from exc
            if not all(math.isfinite(value) for value in (unix_t, velocity, wheel_angle)):
                raise ValueError(f"{csv_path}:{line_number}: non-finite encoder sample")
            samples.append({
                "enc_unix": unix_t,
                "enc_t": enc_t_name(unix_t),
                "enc_v": velocity,
                "enc_dgr": math.degrees(wheel_angle),
                "enc_available": True,
            })
    _write_jsonl(samples, out_path)
    return _encoder_stats(samples)


def write_encoder_jsonl_copy(jsonl_path: Path, out_path: Path) -> dict:
    """Validate/copy an already canonical encoder.jsonl input.

    This is useful when the robot logger has already emitted the mapper's JSONL
    format and there is no intermediate wheel_encoder_data.csv to convert.
    Extra fields (for example integrated odometry) are preserved.
    """
    samples = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                unix_t = float(row["enc_unix"])
                velocity = float(row["enc_v"])
                wheel_angle = float(row["enc_dgr"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{jsonl_path}:{line_number}: invalid canonical encoder sample"
                ) from exc
            if not all(math.isfinite(value) for value in (unix_t, velocity, wheel_angle)):
                raise ValueError(f"{jsonl_path}:{line_number}: non-finite encoder sample")
            if not isinstance(row, dict):
                raise ValueError(f"{jsonl_path}:{line_number}: sample is not an object")
            row["enc_unix"] = unix_t
            row["enc_t"] = str(row.get("enc_t") or enc_t_name(unix_t))
            row["enc_v"] = velocity
            row["enc_dgr"] = wheel_angle
            row["enc_available"] = bool(row.get("enc_available", True))
            samples.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(samples, out_path)
    return _encoder_stats(samples)


def _write_jsonl(samples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")


def _encoder_stats(samples: list[dict]) -> dict:
    timestamps = [float(sample["enc_unix"]) for sample in samples]
    return {
        "samples": len(samples),
        "first_ts": timestamps[0] if timestamps else None,
        "last_ts": timestamps[-1] if timestamps else None,
        "duration_s": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
    }


def save_scan(
    horizon_dir: Path, frame_index: int, scan, min_points: int = 200,
    point_time_mode: str = "packet-host", clock_model: tuple[float, float] | None = None,
    start_unix: float | None = None, end_unix: float | None = None,
) -> float | None:
    xs, ys, inten, times = [], [], [], []
    for index, measurement in enumerate(scan):
        if measurement is None:
            continue
        distance_mm, intensity, device_t, host_t, frequency_hz, packet_point = measurement
        if distance_mm == INVALID_DISTANCE:
            continue
        distance_m = distance_mm / 1000.0
        angle = index * 2.0 * math.pi / POINTS_PER_SCAN
        xs.append(distance_m * math.cos(angle))
        ys.append(distance_m * math.sin(angle))
        inten.append(intensity)
        if point_time_mode == "device-linear":
            if clock_model is None or frequency_hz <= 0:
                raise ValueError("device-linear timing needs a valid clock model and HZ")
            slope, intercept = clock_model
            device_point_t = device_t + packet_point / (POINTS_PER_SCAN * frequency_hz)
            times.append(slope * device_point_t + intercept)
        else:
            times.append(host_t)
    if len(xs) < min_points:
        return None
    xy = np.column_stack((np.asarray(xs, np.float32), np.asarray(ys, np.float32)))
    xyz = np.column_stack((xy, np.zeros(len(xs), np.float32)))
    times_a = np.asarray(times, np.float64)
    stamp = float(np.median(times_a))
    if ((start_unix is not None and stamp < start_unix) or
            (end_unix is not None and stamp > end_unix)):
        return None
    path = horizon_dir / f"{frame_index:05d}_{stamp_name(stamp)}.npz"
    np.savez(
        path,
        xy=xy,
        xyz=xyz,
        intensity=np.asarray(inten, np.float32),
        t=times_a,
        stamp=np.asarray(stamp, np.float64),
        frame=np.asarray(int(frame_index), np.int32),
        channel=np.asarray(0, np.int32),
    )
    return stamp


def write_horizon(
    m360_path: Path, horizon_dir: Path, min_points: int = 200,
    point_time_mode: str = "packet-host", start_unix: float | None = None,
    end_unix: float | None = None,
) -> dict:
    """Convert raw M360 packets into one canonical NPZ per revolution.

    Packet timestamps in a raw capture are deliberately not used as the
    recording clock: with ``time_sync=NONE`` they are device uptime.  The
    host receive timestamp stored by ``receive_scan.py`` is the Unix timeline
    shared with the wheel encoder and is therefore used for NPZ ``t``/``stamp``.
    """
    if point_time_mode not in POINT_TIME_MODES:
        raise ValueError(f"point_time_mode must be one of {POINT_TIME_MODES}")
    horizon_dir.mkdir(parents=True, exist_ok=True)
    clock_model = None
    timing_report = {"mode": point_time_mode}
    if point_time_mode == "device-linear":
        slope, intercept, timing_report = _device_to_host_clock(
            m360_path, start_unix, end_unix)
        clock_model = (slope, intercept)
    current_rev = None
    scan = [None] * POINTS_PER_SCAN
    frame_index = 0
    saved = 0
    skipped = 0
    stamps = []
    packets = 0
    invalid_packets = 0

    def flush():
        nonlocal frame_index, saved, skipped
        frame_index += 1
        stamp = save_scan(
            horizon_dir, frame_index, scan, min_points=min_points,
            point_time_mode=point_time_mode, clock_model=clock_model,
            start_unix=start_unix, end_unix=end_unix)
        if stamp is None:
            skipped += 1
        else:
            saved += 1
            stamps.append(stamp)

    for host_t, raw in iter_raw_records(str(m360_path)):
        if start_unix is not None and host_t < start_unix - 0.1:
            continue
        if end_unix is not None and host_t > end_unix + 0.1:
            break
        packet = decode_packet(raw, host_t)
        packets += 1
        if current_rev is None:
            current_rev = packet.revolution
        elif packet.revolution != current_rev:
            flush()
            scan = [None] * POINTS_PER_SCAN
            current_rev = packet.revolution
        start = packet.encoder
        if start >= POINTS_PER_SCAN:
            invalid_packets += 1
            continue
        end = min(start + POINTS_PER_PACKET, POINTS_PER_SCAN)
        if start < POINTS_PER_SCAN:
            scan[start:end] = [
                (distance, intensity, packet.timestamp, packet.host_time,
                 packet.frequency_hz, point_index)
                for point_index, (distance, intensity)
                in enumerate(packet.points[: end - start])
            ]
    if current_rev is not None:
        flush()
    report = {
        "packets": packets,
        "invalid_packets": invalid_packets,
        "frames": saved,
        "skipped_sparse": skipped,
        "duration_s": (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0,
        "first_ts": stamps[0] if stamps else None,
        "last_ts": stamps[-1] if stamps else None,
        "clock": "jetson_host_time",
        "point_timing": timing_report,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m360", type=Path, required=True)
    encoder = parser.add_mutually_exclusive_group(required=True)
    encoder.add_argument("--encoder-csv", type=Path)
    encoder.add_argument(
        "--encoder-jsonl",
        type=Path,
        help="already canonical encoder.jsonl; validate and copy it",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--min-points",
        type=int,
        default=200,
        help="minimum valid returns required to save a revolution (default: 200)",
    )
    args = parser.parse_args()
    if args.min_points < 1:
        parser.error("--min-points must be >= 1")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.encoder_csv is not None:
        encoder_stats = write_encoder_jsonl(args.encoder_csv, run_dir / "encoder.jsonl")
        encoder_source = args.encoder_csv
    else:
        encoder_stats = write_encoder_jsonl_copy(
            args.encoder_jsonl, run_dir / "encoder.jsonl"
        )
        encoder_source = args.encoder_jsonl
    lidar_stats = write_horizon(
        args.m360, run_dir / "horizon", min_points=args.min_points
    )
    meta = {
        "format": "canonical_recording",
        "lidar_source": str(args.m360),
        "encoder_source": str(encoder_source),
        "lidar": lidar_stats,
        "encoder": encoder_stats,
    }
    (run_dir / "recording_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"fork_state": "down", "lidar": "dephan_m360"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "lidar.config").write_text(json.dumps({
        "format": "lidar.config.v1",
        "lidar_type": "dephan_m360",
        "sensor": "DePHAN_M360",
        # Raw M360 xy is expressed in the sensor frame; the mount transform
        # is applied by downstream consumers once the mount is measured.
        "points_frame": "sensor",
        "coordinate_system": {
            "frame": "rotation_axis", "units": "m",
            "x": "forward", "y": "left", "z": "up",
        },
        "position_m": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "source": "recording_mount_must_be_measured",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **meta}, indent=2))


if __name__ == "__main__":
    main()
