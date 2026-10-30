#!/usr/bin/env python3
"""Convert an RPLRAW1 point capture into canonical horizon NPZ scans."""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RAW_MAGIC = b"RPLRAW1\0"
RAW_RECORD = struct.Struct("<ddddB7x")
POINT_TIME_MODES = (
    "azimuth-device-end",
    "azimuth-host-end",
    "azimuth-device-start",
    "legacy-device",
)


@dataclass
class ScanTiming:
    points: int
    finite_points: int
    device_min: float
    device_max: float
    host_min: float
    host_max: float


def iter_raw_records(path: Path):
    with path.open("rb") as stream:
        if stream.read(len(RAW_MAGIC)) != RAW_MAGIC:
            raise ValueError(f"{path} is not an RPLRAW1 capture")
        while True:
            blob = stream.read(RAW_RECORD.size)
            if not blob:
                return
            if len(blob) != RAW_RECORD.size:
                raise ValueError(
                    f"{path}: truncated record ({len(blob)}/{RAW_RECORD.size} bytes)"
                )
            yield RAW_RECORD.unpack(blob)


def _wrapped_delta(current: float, previous: float) -> float:
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _analyse_source(source: Path) -> tuple[list[ScanTiming], dict]:
    """Collect revolution metadata and fit one stable device-to-host clock."""
    scans: list[ScanTiming] = []
    previous_device = previous_angle = None
    has_device_resets = False
    direction_sum = 0.0
    count = 0
    mean_device = mean_host = covariance = device_variance = 0.0
    current_points = current_finite = 0
    device_min = host_min = math.inf
    device_max = host_max = -math.inf

    def flush() -> None:
        nonlocal current_points, current_finite
        nonlocal device_min, device_max, host_min, host_max
        if current_points:
            scans.append(ScanTiming(
                current_points, current_finite,
                device_min, device_max, host_min, host_max,
            ))
        current_points = current_finite = 0
        device_min = host_min = math.inf
        device_max = host_max = -math.inf

    # Determine one boundary signal for the complete recording. Device-clock
    # resets and angle wraps must not both split the same revolution.
    for _host_t, _x, _y, device_t, _lidar_id in iter_raw_records(source):
        if previous_device is not None and device_t + 1e-9 < previous_device:
            has_device_resets = True
        previous_device = device_t

    boundary_kind = "device_t_reset" if has_device_resets else "angle_wrap"
    previous_device = previous_angle = None
    for host_t, x, y, device_t, _lidar_id in iter_raw_records(source):
        angle = math.atan2(y, x)
        time_reset = previous_device is not None and device_t + 1e-9 < previous_device
        angle_wrap = previous_angle is not None and abs(angle - previous_angle) > math.pi
        if current_points and (time_reset if has_device_resets else angle_wrap):
            flush()

        current_points += 1
        finite = all(map(math.isfinite, (host_t, x, y, device_t)))
        if finite:
            current_finite += 1
            device_min = min(device_min, device_t)
            device_max = max(device_max, device_t)
            host_min = min(host_min, host_t)
            host_max = max(host_max, host_t)

            count += 1
            delta_device = device_t - mean_device
            mean_device += delta_device / count
            delta_host = host_t - mean_host
            mean_host += delta_host / count
            device_variance += delta_device * (device_t - mean_device)
            covariance += delta_device * (host_t - mean_host)

        if previous_angle is not None:
            delta = _wrapped_delta(angle, previous_angle)
            if abs(delta) < 0.5:
                direction_sum += delta
        previous_device, previous_angle = device_t, angle
    flush()

    slope = covariance / device_variance if device_variance > 0.0 else math.nan
    intercept = mean_host - slope * mean_device
    clock_is_valid = (
        not has_device_resets and math.isfinite(slope) and math.isfinite(intercept)
        and 0.95 <= slope <= 1.05
    )
    return scans, {
        "boundary_kind": boundary_kind,
        "has_device_resets": has_device_resets,
        "clock_slope": slope,
        "clock_intercept": intercept,
        "clock_is_valid": clock_is_valid,
        "direction": -1.0 if direction_sum < 0.0 else 1.0,
        "direction_name": "clockwise" if direction_sum < 0.0 else "counter_clockwise",
        "records": count,
    }


def _anchors(scans: list[ScanTiming], analysis: dict, mode: str) -> np.ndarray:
    if mode.startswith("azimuth-device") and analysis["clock_is_valid"]:
        slope = analysis["clock_slope"]
        intercept = analysis["clock_intercept"]
        return np.asarray([
            slope * (scan.device_min if mode.endswith("start") else scan.device_max) + intercept
            for scan in scans
        ], dtype=np.float64)
    return np.asarray([
        scan.host_min if mode.endswith("start") else scan.host_max
        for scan in scans
    ], dtype=np.float64)


def _scan_periods(anchors: np.ndarray) -> tuple[np.ndarray, dict]:
    differences = np.diff(anchors)
    broad = differences[(differences > 0.05) & (differences < 0.25)]
    if not len(broad):
        raise ValueError("cannot estimate RPLIDAR revolution period")
    median = float(np.median(broad))
    lower, upper = 0.8 * median, 1.2 * median
    valid = differences[(differences >= lower) & (differences <= upper)]
    periods = np.full(len(anchors), median, dtype=np.float64)
    for index in range(len(periods)):
        candidates = [median]
        if index > 0 and lower <= differences[index - 1] <= upper:
            candidates.append(float(differences[index - 1]))
        if index < len(differences) and lower <= differences[index] <= upper:
            candidates.append(float(differences[index]))
        # Keep adjacent intervals disjoint after conversion to 100 ns ticks.
        periods[index] = min(candidates) * 0.999
    return periods, {
        "median_period_s": median,
        "minimum_period_s": float(np.min(valid)),
        "maximum_period_s": float(np.max(valid)),
        "valid_period_intervals": int(len(valid)),
        "rejected_period_intervals": int(len(differences) - len(valid)),
    }


def _azimuth_times(
    xy: np.ndarray, anchor: float, period: float, direction: float, stamp_is_end: bool,
) -> tuple[np.ndarray, float]:
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    phase = np.zeros(len(angles), dtype=np.float64)
    progress = 0.0
    for index in range(1, len(angles)):
        delta = _wrapped_delta(float(angles[index]), float(angles[index - 1]))
        # Preserve forward gaps caused by missing returns, but do not allow
        # bearing jitter to make acquisition time run backwards.
        progress += max(0.0, direction * delta)
        phase[index] = min(1.0, progress / (2.0 * math.pi))
    start = anchor - period if stamp_is_end else anchor
    return start + phase * period, progress / (2.0 * math.pi)


def _save_scan(
    horizon: Path, index: int, records: list[tuple], min_points: int,
    mode: str, anchor: float | None, period: float | None, direction: float,
):
    finite = [row for row in records if math.isfinite(row[1]) and math.isfinite(row[2])]
    if len(finite) < min_points:
        return None
    host = np.asarray([row[0] for row in finite], dtype=np.float64)
    xy = np.asarray([[row[1], row[2]] for row in finite], dtype=np.float32)
    device_t = np.asarray([row[3] for row in finite], dtype=np.float64)
    lidar_id = np.asarray([row[4] for row in finite], dtype=np.uint8)
    if mode == "legacy-device":
        if np.ptp(device_t) > 0 and np.all(np.diff(device_t) >= -1e-9):
            point_t = np.median(host - device_t) + device_t
        else:
            point_t = host
    else:
        if anchor is None or period is None:
            raise ValueError("azimuth timing requires an anchor and revolution period")
        point_t, angular_coverage = _azimuth_times(
            xy, anchor, period, direction, mode.endswith("end"))
        # Recorder startup/shutdown can leave a point-rich but incomplete
        # fraction of a revolution. It has no valid neighbouring-period
        # anchor and must not be stretched into a complete scan.
        if angular_coverage < 0.75:
            return None
    stamp = float(point_t[-1])
    np.savez(
        horizon / f"{index:05d}_{stamp:.6f}.npz",
        xy=xy,
        xyz=np.column_stack((xy, np.zeros(len(xy), np.float32))),
        intensity=np.zeros(len(xy), np.float32),
        t=point_t,
        device_t=device_t,
        lidar_id=lidar_id,
        stamp=np.asarray(stamp, np.float64),
        frame=np.asarray(index, np.int32),
        channel=np.asarray(0, np.int32),
    )
    return stamp, float(point_t[0]), float(point_t[-1])


def write_horizon(
    source: Path, horizon: Path, min_points: int = 100,
    point_time_mode: str = "azimuth-device-end",
) -> dict:
    if point_time_mode not in POINT_TIME_MODES:
        raise ValueError(f"point_time_mode must be one of {POINT_TIME_MODES}")
    horizon.mkdir(parents=True, exist_ok=True)
    scan_meta, analysis = _analyse_source(source)
    anchors = periods = None
    period_report = {}
    if point_time_mode != "legacy-device":
        anchors = _anchors(scan_meta, analysis, point_time_mode)
        periods, period_report = _scan_periods(anchors)

    current: list[tuple] = []
    stamps: list[float] = []
    intervals: list[tuple[float, float]] = []
    skipped = total = scan_index = 0
    previous_device = previous_angle = None

    def flush() -> None:
        nonlocal current, skipped, scan_index
        if not current:
            return
        anchor = None if anchors is None else float(anchors[scan_index])
        period = None if periods is None else float(periods[scan_index])
        saved = _save_scan(
            horizon, scan_index + 1, current, min_points,
            point_time_mode, anchor, period, analysis["direction"],
        )
        if saved is None:
            skipped += 1
        else:
            stamp, first, last = saved
            stamps.append(stamp)
            intervals.append((first, last))
        scan_index += 1
        current = []

    total_records = max(0, (source.stat().st_size - len(RAW_MAGIC)) // RAW_RECORD.size)
    for host_t, x, y, device_t, lidar_id in iter_raw_records(source):
        angle = math.atan2(y, x)
        time_reset = previous_device is not None and device_t + 1e-9 < previous_device
        angle_wrap = previous_angle is not None and abs(angle - previous_angle) > math.pi
        if current and (time_reset if analysis["has_device_resets"] else angle_wrap):
            flush()
        current.append((host_t, x, y, device_t, lidar_id))
        previous_device, previous_angle = device_t, angle
        total += 1
        if total % 100000 == 0 or total == total_records:
            fraction = total / total_records if total_records else 1.0
            width = 30
            filled = int(width * fraction)
            sys.stderr.write(
                f"\r[{'#' * filled}{'-' * (width - filled)}] "
                f"{fraction * 100:5.1f}% converting RPLIDAR points "
                f"{total}/{total_records} scans={len(stamps)}"
            )
            sys.stderr.flush()
    flush()
    sys.stderr.write("\n")

    overlaps = 0
    maximum_overlap = 0.0
    for previous, following in zip(intervals, intervals[1:]):
        overlap = previous[1] - following[0]
        if overlap >= -1e-7:
            overlaps += 1
            maximum_overlap = max(maximum_overlap, overlap)
    return {
        "format": "RPLRAW1",
        "points": total,
        "frames": len(stamps),
        "skipped_sparse": skipped,
        "first_ts": stamps[0] if stamps else None,
        "last_ts": stamps[-1] if stamps else None,
        "duration_s": stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0,
        "clock": "jetson_host_time",
        "scan_boundary": analysis["boundary_kind"],
        "point_time_mode": point_time_mode,
        "device_to_host_clock": {
            "slope": analysis["clock_slope"],
            "intercept": analysis["clock_intercept"],
            "valid": analysis["clock_is_valid"],
        },
        "rotation_direction": analysis["direction_name"],
        "timing": period_report,
        "overlapping_output_scans": overlaps,
        "maximum_output_overlap_s": maximum_overlap,
    }
