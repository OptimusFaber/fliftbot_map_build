#!/usr/bin/env python3
"""Render evenly spaced raw LiDAR revolutions without involving SLAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path, help="canonical recording or horizon directory")
    parser.add_argument("--last-seconds", type=float, default=180.0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-range", type=float, default=200.0)
    parser.add_argument("--display-range", type=float, default=65.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    horizon = args.recording / "horizon" if (args.recording / "horizon").is_dir() else args.recording
    paths = sorted(horizon.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No NPZ scans in {horizon}")
    stamps = np.asarray([float(np.load(path)["stamp"]) for path in paths])
    eligible = np.flatnonzero(stamps >= stamps[-1] - args.last_seconds)
    selected = eligible[np.linspace(0, len(eligible) - 1, args.count).round().astype(int)]

    fig, axes = plt.subplots(
        2, args.count, figsize=(4 * args.count, 8), constrained_layout=True, squeeze=False
    )
    report = {"source": str(horizon.resolve()), "last_seconds": args.last_seconds,
              "max_range_m": args.max_range, "scans": []}
    for column, index in enumerate(selected):
        with np.load(paths[index]) as data:
            xy = np.asarray(data["xy"], dtype=float)
            intensity = np.asarray(data["intensity"], dtype=float)
            stamp = float(data["stamp"])
            frame = int(data["frame"])
            point_time = np.asarray(data["t"], dtype=float)
        ranges = np.linalg.norm(xy, axis=1)
        valid = np.isfinite(ranges) & (ranges <= args.max_range)
        xy, ranges, intensity = xy[valid], ranges[valid], intensity[valid]
        order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
        sorted_ranges = ranges[order]
        rounded, counts = np.unique(np.round(ranges, 2), return_counts=True)
        repeated = sorted(zip(counts, rounded), reverse=True)[:5]
        row = {
            "file": paths[index].name, "frame": frame, "stamp": stamp,
            "seconds_before_end": float(stamps[-1] - stamp), "points": int(len(ranges)),
            "scan_duration_ms": float((point_time.max() - point_time.min()) * 1000),
            "range_m": {"median": float(np.median(ranges)), "p95": float(np.percentile(ranges, 95)),
                        "max": float(ranges.max())},
            "fraction_over_18m": float(np.mean(ranges > 18)),
            "fraction_over_60m": float(np.mean(ranges > 60)),
            "largest_adjacent_range_jump_m": float(np.max(np.abs(np.diff(sorted_ranges)))),
            "most_common_ranges_rounded_cm": [
                {"range_m": float(radius), "count": int(count)} for count, radius in repeated
            ],
        }
        report["scans"].append(row)

        ax = axes[0, column]
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=2, cmap="viridis", rasterized=True)
        ax.scatter([0], [0], marker="+", color="red", s=60)
        ax.set(xlim=(-args.display_range, args.display_range), ylim=(-args.display_range, args.display_range),
               aspect="equal", title=f"frame {frame}\n{row['seconds_before_end']:.1f} s before end")
        ax.grid(alpha=.2)
        if column == args.count - 1:
            fig.colorbar(scatter, ax=ax, shrink=.7, label="intensity")

        ax = axes[1, column]
        angles = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
        ax.scatter(angles, ranges, c=intensity, s=2, cmap="viridis", rasterized=True)
        ax.axhline(18, color="tab:red", lw=1, label="18 m")
        ax.axhline(60, color="tab:orange", lw=1, label="60 m")
        ax.set(xlim=(-180, 180), ylim=(0, args.display_range), xlabel="azimuth, deg", ylabel="range, m")
        ax.grid(alpha=.2)
        if column == 0:
            ax.legend()

    args.output.mkdir(parents=True, exist_ok=True)
    png = args.output / "raw_scans_last_3min.png"
    fig.suptitle(f"Raw LiDAR scans (filter <= {args.max_range:g} m); no Cartographer")
    fig.savefig(png, dpi=180)
    plt.close(fig)
    (args.output / "raw_scans_last_3min.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(png)


if __name__ == "__main__":
    main()
