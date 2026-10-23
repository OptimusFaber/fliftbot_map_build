#!/usr/bin/env python3
"""Accumulate raw scan endpoints with Cartographer poses, colored by scan time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UNIVERSAL_TO_UNIX_SECONDS = 62135596800.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-range", type=float, default=200.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--xlim", type=float, nargs=2)
    parser.add_argument("--ylim", type=float, nargs=2)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "diagnostics" / "time_colored_endpoints").resolve()
    output.mkdir(parents=True, exist_ok=True)

    params = json.loads((run / "run_parameters.json").read_text())
    stats = json.loads((run / "map" / "run_stats.json").read_text())
    # A trajectory whose tracking frame is the physical LiDAR already carries
    # the sensor orientation.  Apply mount yaw only for a body/rotation-axis
    # tracking trajectory, matching runner.cc's AddSensorData contract.
    mount_yaw = 0.0 if stats.get("tracking_frame") == "lidar" else np.deg2rad(
        float(params["lidar"]["mapping_orientation_deg"]["yaw"])
    )
    poses = json.loads((run / "map" / "trajectory.json").read_text())["poses"]
    pose_t = np.asarray([
        row["timestamp_universal"] / 1e7 - UNIVERSAL_TO_UNIX_SECONDS for row in poses
    ])
    pose_x = np.asarray([row["x"] for row in poses])
    pose_y = np.asarray([row["y"] for row in poses])
    pose_yaw = np.unwrap(np.asarray([2 * np.arctan2(row["qz"], row["qw"]) for row in poses]))

    paths = sorted((run / "input" / "horizon").glob("*.npz"))[::args.stride]
    chunks, colors = [], []
    first_stamp = None
    for path in paths:
        with np.load(path) as data:
            xy = np.asarray(data["xy"], dtype=float)
            stamp = float(data["stamp"])
        if first_stamp is None:
            first_stamp = stamp
        ranges = np.linalg.norm(xy, axis=1)
        xy = xy[np.isfinite(ranges) & (ranges <= args.max_range)]
        x = np.interp(stamp, pose_t, pose_x)
        y = np.interp(stamp, pose_t, pose_y)
        yaw = np.interp(stamp, pose_t, pose_yaw) + mount_yaw
        c, s = np.cos(yaw), np.sin(yaw)
        world = np.column_stack((x + c * xy[:, 0] - s * xy[:, 1],
                                 y + s * xy[:, 0] + c * xy[:, 1]))
        chunks.append(world)
        colors.append(np.full(len(world), stamp - first_stamp))
    points, elapsed = np.vstack(chunks), np.concatenate(colors)

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=elapsed, cmap="turbo", s=.35,
                         linewidths=0, rasterized=True)
    ax.plot(pose_x, pose_y, color="black", lw=1.0, label="Cartographer LiDAR trajectory")
    ax.set(aspect="equal", xlabel="map X, m", ylabel="map Y, m",
           title=f"Raw endpoints transformed by Cartographer poses; every {args.stride}th scan")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right")
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)
    fig.colorbar(scatter, ax=ax, label="seconds from first scan")
    target = output / "endpoints_by_time.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
