#!/usr/bin/env python3
"""Copy a canonical recording while removing beams around depth discontinuities."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jump-m", type=float, default=2.0)
    parser.add_argument("--half-width-beams", type=int, default=2)
    parser.add_argument("--max-bearing-gap-deg", type=float, default=0.25)
    args = parser.parse_args()
    source, output = args.recording.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    (output / "horizon").mkdir(parents=True)
    removed_total = input_total = 0
    per_scan = []
    for path in sorted((source / "horizon").glob("*.npz")):
        with np.load(path) as z:
            arrays = {name: np.asarray(z[name]) for name in z.files}
        xy = arrays["xy"]
        ranges = np.linalg.norm(xy, axis=1)
        bearings = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
        order = np.argsort(bearings)
        sorted_ranges, sorted_bearings = ranges[order], bearings[order]
        adjacent = ((np.abs(np.diff(sorted_ranges)) >= args.jump_m) &
                    (np.diff(sorted_bearings) <= np.deg2rad(args.max_bearing_gap_deg)))
        remove_sorted = np.zeros(len(xy), dtype=bool)
        for boundary in np.flatnonzero(adjacent):
            lo = max(0, boundary + 1 - args.half_width_beams)
            hi = min(len(xy), boundary + 1 + args.half_width_beams)
            remove_sorted[lo:hi] = True
        remove = np.zeros(len(xy), dtype=bool)
        remove[order] = remove_sorted
        keep = ~remove
        filtered = {
            name: value[keep] if value.ndim >= 1 and value.shape[0] == len(xy) else value
            for name, value in arrays.items()
        }
        np.savez_compressed(output / "horizon" / path.name, **filtered)
        removed = int(remove.sum())
        input_total += len(xy); removed_total += removed
        per_scan.append(removed)
    for name in ("encoder.jsonl", "camera.config", "lidar.config", "liftbot.config",
                 "camera_intrinsics.json", "interval_report.json", "tail_report.json"):
        path = source / name
        if path.is_file():
            link_or_copy(path, output / name)
    report = {
        "schema": "fliftbot.depth_edge_filter.v1", "source": str(source),
        "jump_m": args.jump_m, "half_width_beams": args.half_width_beams,
        "max_bearing_gap_deg": args.max_bearing_gap_deg,
        "scans": len(per_scan), "input_points": input_total, "removed_points": removed_total,
        "removed_fraction": removed_total / input_total,
        "removed_per_scan": {
            "median": float(np.median(per_scan)), "p95": float(np.percentile(per_scan, 95)),
            "max": int(max(per_scan)),
        },
    }
    (output / "depth_edge_filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
