#!/usr/bin/env python3
"""Render the same trajectory segment with progressively more raw scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UNIX_EPOCH_TICKS = 62135596800.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--center-index", type=int)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 5, 20, 100, 500])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-range", type=float, default=200.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    poses = json.loads((run / "map" / "trajectory.json").read_text())["poses"]
    pt = np.array([x["timestamp_universal"] / 1e7 - UNIX_EPOCH_TICKS for x in poses])
    px = np.array([x["x"] for x in poses]); py = np.array([x["y"] for x in poses])
    pyaw = np.unwrap(np.array([2 * np.arctan2(x["qz"], x["qw"]) for x in poses]))
    paths = sorted((run / "input" / "horizon").glob("*.npz"))[::max(1, args.stride)]
    stamps = np.array([float(np.load(q)["stamp"]) for q in paths])
    if not len(paths): raise SystemExit("no horizon scans")
    center = args.center_index if args.center_index is not None else len(paths) // 2
    center = max(0, min(center, len(paths) - 1))
    chunks = []
    for path, stamp in zip(paths, stamps):
        with np.load(path) as d: xy = np.asarray(d["xy"], float)
        rr = np.linalg.norm(xy, axis=1); xy = xy[np.isfinite(rr) & (rr <= args.max_range)]
        x = np.interp(stamp, pt, px); y = np.interp(stamp, pt, py); a = np.interp(stamp, pt, pyaw)
        c, s = np.cos(a), np.sin(a)
        world = np.column_stack((x + c * xy[:, 0] - s * xy[:, 1], y + s * xy[:, 0] + c * xy[:, 1]))
        chunks.append(world)
    fig, axes = plt.subplots(1, len(args.windows), figsize=(6 * len(args.windows), 6), squeeze=False)
    for ax, window in zip(axes[0], args.windows):
        half = max(0, int(window) // 2); lo, hi = max(0, center-half), min(len(chunks), center-half+int(window))
        points = np.vstack(chunks[lo:hi]); ax.scatter(points[:, 0], points[:, 1], s=.35, c=np.arange(len(points)), cmap="turbo", linewidths=0, rasterized=True)
        center_stamp = stamps[center]
        ax.plot(px, py, color="black", lw=.5, alpha=.5)
        ax.plot(np.interp(center_stamp, pt, px), np.interp(center_stamp, pt, py), "r+")
        ax.set_aspect("equal"); ax.set_title(f"{hi-lo} scans\nframes {lo}..{hi-1}"); ax.grid(alpha=.2)
    fig.suptitle(f"Accumulation at center scan {center}; max range {args.max_range:g} m")
    fig.tight_layout(); fig.savefig(out / "accumulation_windows.png", dpi=200); plt.close(fig)
    (out / "report.json").write_text(json.dumps({"center_index": center, "windows": args.windows, "stride": args.stride, "max_range_m": args.max_range, "scan_count": len(paths)}, indent=2) + "\n")
    print(out / "accumulation_windows.png")


if __name__ == "__main__": main()
