#!/usr/bin/env python3
"""Downsample a canonical recording at an approximately fixed travel distance."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from bisect import bisect_left
from pathlib import Path

import numpy as np


def load_encoder(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: float(row["enc_unix"]))
    if len(rows) < 2:
        raise ValueError("encoder.jsonl must contain at least two samples")
    return rows


def cumulative_distance(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([float(row["enc_unix"]) for row in rows])
    speed = np.abs(np.asarray([float(row["enc_v"]) for row in rows]))
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("encoder timestamps must be strictly increasing")
    distance = np.concatenate(([0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * dt)))
    return times, distance


def select_by_distance(distance: np.ndarray, step: float) -> list[int]:
    selected = [0]
    target = step
    while target < distance[-1]:
        index = int(np.searchsorted(distance, target, side="left"))
        if index != selected[-1]:
            selected.append(index)
        target += step
    if selected[-1] != len(distance) - 1:
        selected.append(len(distance) - 1)
    return selected


def scan_stamp(path: Path) -> float:
    with np.load(path) as data:
        return float(data["stamp"])


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-m", type=float, default=0.045)
    args = parser.parse_args()
    source = args.recording.resolve()
    output = args.output.resolve()
    if args.step_m <= 0:
        parser.error("--step-m must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    rows = load_encoder(source / "encoder.jsonl")
    encoder_times, distance = cumulative_distance(rows)
    encoder_indices = select_by_distance(distance, args.step_m)

    scans = sorted((source / "horizon").glob("*.npz"))
    if not scans:
        raise FileNotFoundError(f"no NPZ scans in {source / 'horizon'}")
    scan_times = np.asarray([scan_stamp(path) for path in scans])
    scan_distance = np.interp(scan_times, encoder_times, distance)
    scan_indices = select_by_distance(scan_distance, args.step_m)

    (output / "horizon").mkdir(parents=True, exist_ok=True)
    with (output / "encoder.jsonl").open("w") as stream:
        for index in encoder_indices:
            stream.write(json.dumps(rows[index], separators=(",", ":")) + "\n")
    for index in scan_indices:
        hardlink_or_copy(scans[index], output / "horizon" / scans[index].name)

    for name in ("camera.config", "lidar.config", "liftbot.config", "camera_intrinsics.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)

    scan_steps = np.diff(scan_distance[scan_indices])
    encoder_steps = np.diff(distance[encoder_indices])
    report = {
        "schema": "fliftbot.distance_downsample.v1",
        "source": str(source),
        "target_step_m": args.step_m,
        "travel_distance_m": float(distance[-1]),
        "lidar": {
            "source_samples": len(scans),
            "selected_samples": len(scan_indices),
            "step_median": float(np.median(scan_steps)),
            "step_p95": float(np.percentile(scan_steps, 95)),
            "step_max": float(np.max(scan_steps)),
        },
        "encoder": {
            "source_samples": len(rows),
            "selected_samples": len(encoder_indices),
            "step_median": float(np.median(encoder_steps)),
            "step_p95": float(np.percentile(encoder_steps, 95)),
            "step_max": float(np.max(encoder_steps)),
        },
        "limitations": [
            "This simulates sparse spatial sampling, not motion distortion within a LiDAR revolution.",
            "Original timestamps and measured velocities are preserved.",
        ],
    }
    (output / "downsample_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
