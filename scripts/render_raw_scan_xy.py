#!/usr/bin/env python3
"""Render selected raw LiDAR scans as large top-down XY plots."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(ax, xy: np.ndarray, intensity: np.ndarray, title: str, extent: float):
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=9, cmap="turbo", linewidths=0)
    ax.scatter([0], [0], marker="+", color="red", s=150, linewidths=2.5, zorder=5)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="X, m", ylabel="Y, m", title=title)
    ax.grid(alpha=.25)
    return scatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("horizon", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-range", type=float, default=200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    horizon = args.horizon / "horizon" if (args.horizon / "horizon").is_dir() else args.horizon
    paths = sorted(horizon.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No scans in {horizon}")
    indices = np.linspace(0, len(paths) - 1, args.count).round().astype(int)
    scans = []
    for index in indices:
        with np.load(paths[index]) as data:
            xy = np.asarray(data["xy"], dtype=float)
            intensity = np.asarray(data["intensity"], dtype=float)
            frame, stamp = int(data["frame"]), float(data["stamp"])
        ranges = np.linalg.norm(xy, axis=1)
        valid = np.isfinite(ranges) & (ranges <= args.max_range)
        scans.append((xy[valid], intensity[valid], frame, stamp, paths[index].name))
    largest = max(float(np.linalg.norm(xy, axis=1).max()) for xy, *_ in scans)
    extent = np.ceil(largest / 5) * 5 + 2
    args.output.mkdir(parents=True, exist_ok=True)

    columns = min(3, args.count)
    rows = math.ceil(args.count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows),
                             constrained_layout=True, squeeze=False)
    for column, (xy, intensity, frame, stamp, _) in enumerate(scans):
        elapsed = stamp - scans[0][3]
        scatter = draw(axes.ravel()[column], xy, intensity,
                       f"frame {frame}\nt = +{elapsed:.1f} s, {len(xy)} points", extent)
    for ax in axes.ravel()[len(scans):]:
        ax.set_visible(False)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=.72, label="intensity")
    fig.suptitle(f"Five raw LiDAR scans; all returns ≤ {args.max_range:g} m; no SLAM", fontsize=17)
    fig.savefig(args.output / "five_raw_scans_xy.png", dpi=180)
    plt.close(fig)

    for number, (xy, intensity, frame, stamp, _) in enumerate(scans, 1):
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        scatter = draw(ax, xy, intensity,
                       f"Raw LiDAR frame {frame} ({number}/{args.count}) — {len(xy)} points", extent)
        fig.colorbar(scatter, ax=ax, shrink=.8, label="intensity")
        fig.savefig(args.output / f"frame_{number:02d}_{frame}.png", dpi=220)
        plt.close(fig)
    print(args.output / "five_raw_scans_xy.png")


if __name__ == "__main__":
    main()
