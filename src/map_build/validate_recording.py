#!/usr/bin/env python3
"""Validate canonical LiDAR/encoder fields, ordering and time coverage."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def interval_stats(values: list[float]) -> dict:
    gaps = np.diff(values)
    duration = values[-1] - values[0] if len(values) > 1 else 0.0
    return {"samples": len(values), "first_unix": values[0], "last_unix": values[-1],
            "first_utc": datetime.fromtimestamp(values[0], timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(values[-1], timezone.utc).isoformat(),
            "duration_s": duration, "mean_rate_hz": ((len(values) - 1) / duration if duration > 0 else 0.0),
            "maximum_gap_s": (float(np.max(gaps)) if len(gaps) else 0.0)}


def moving_encoder_gaps(rows: list[tuple[float, float, float]], threshold: float) -> list[float]:
    result = []
    for left, right in zip(rows, rows[1:]):
        gap = right[0] - left[0]
        if gap > threshold and (abs(left[1]) > 1e-6 or abs(right[1]) > 1e-6):
            result.append(gap)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = args.recording.resolve()
    report = {"schema": "fliftbot.recording_validation.v1", "path": str(root),
              "errors": [], "warnings": []}
    files = sorted((root / "horizon").glob("*.npz")) if (root / "horizon").is_dir() else []
    if not files:
        report["errors"].append("no horizon/*.npz")
    stamps: list[float] = []
    bad_npz = 0
    for path in files:
        try:
            with np.load(path) as data:
                stamp = float(np.asarray(data["stamp"]).reshape(()))
                if not np.isfinite(stamp):
                    raise ValueError("non-finite LiDAR timestamp")
                stamps.append(stamp)
                if data["xy"].ndim != 2 or data["xy"].shape[1] != 2:
                    bad_npz += 1
        except Exception:
            bad_npz += 1
    if bad_npz:
        report["errors"].append(f"invalid NPZ files: {bad_npz}")
    encoder_path = root / "encoder.jsonl"
    encoder_rows: list[tuple[float, float, float]] = []
    if not encoder_path.is_file():
        report["errors"].append("missing encoder.jsonl")
    else:
        for number, line in enumerate(encoder_path.open(encoding="utf-8"), 1):
            try:
                row = json.loads(line)
                values = (float(row["enc_unix"]), float(row["enc_v"]), float(row["enc_dgr"]))
                if not all(np.isfinite(v) for v in values):
                    raise ValueError("non-finite value")
                encoder_rows.append(values)
            except Exception:
                report["errors"].append(f"invalid encoder row {number}")
    if encoder_path.is_file() and not encoder_rows:
        report["errors"].append("encoder.jsonl contains no valid samples")
    if any(right <= left for left, right in zip(stamps, stamps[1:])):
        report["errors"].append("LiDAR timestamps are not strictly increasing in file order")
    encoder_times = [row[0] for row in encoder_rows]
    duplicates = sum(right == left for left, right in zip(encoder_times, encoder_times[1:]))
    reversals = sum(right < left for left, right in zip(encoder_times, encoder_times[1:]))
    if duplicates:
        report["warnings"].append(f"encoder duplicate timestamps: {duplicates}")
    if reversals:
        report["errors"].append(f"encoder timestamp reversals: {reversals}")
    if stamps:
        report["lidar"] = interval_stats(stamps)
    if encoder_rows:
        report["encoder"] = interval_stats(encoder_times)
        moving = sum(abs(row[1]) > 1e-6 for row in encoder_rows)
        report["encoder"].update({"duplicate_timestamps": duplicates, "timestamp_reversals": reversals,
                                  "moving_samples": moving, "moving_fraction": moving / len(encoder_rows)})
        if stamps and (encoder_times[-1] < stamps[0] or encoder_times[0] > stamps[-1]):
            report["errors"].append("encoder and LiDAR time ranges do not overlap")
        elif stamps:
            if encoder_times[0] > stamps[0]:
                report["warnings"].append(f"encoder starts {encoder_times[0] - stamps[0]:.6f}s after LiDAR")
            if encoder_times[-1] < stamps[-1]:
                report["warnings"].append(f"encoder ends {stamps[-1] - encoder_times[-1]:.6f}s before LiDAR")
        if moving == 0:
            report["warnings"].append("encoder velocity is zero for all samples")
        moving_gaps = moving_encoder_gaps(encoder_rows, 0.2)
        if moving_gaps:
            report["encoder"]["moving_gaps_over_0_2_s"] = len(moving_gaps)
            report["warnings"].append(
                f"encoder gaps while moving: {len(moving_gaps)}, maximum {max(moving_gaps):.6f}s"
            )
        unrecoverable_gaps = [gap for gap in moving_gaps if gap > 1.5]
        if unrecoverable_gaps:
            report["errors"].append(
                "encoder gap while moving exceeds repair limit: "
                f"{max(unrecoverable_gaps):.6f}s"
            )
    if stamps:
        lidar_gaps = [right - left for left, right in zip(stamps, stamps[1:])
                      if right - left > 0.1]
        if lidar_gaps:
            report["lidar"]["gaps_over_0_1_s"] = len(lidar_gaps)
            report["warnings"].append(
                f"LiDAR gaps: {len(lidar_gaps)}, maximum {max(lidar_gaps):.6f}s"
            )
    report["status"] = "rejected" if report["errors"] else (
        "accepted_with_warnings" if report["warnings"] else "accepted")
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.json:
        args.json.resolve().write_text(text, encoding="utf-8")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
