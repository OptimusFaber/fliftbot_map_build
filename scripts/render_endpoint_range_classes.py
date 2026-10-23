#!/usr/bin/env python3
"""Accumulate endpoints and separate near from far raw returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

UNIVERSAL_TO_UNIX_SECONDS = 62135596800.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--split-range", type=float, default=18.0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    run = a.run.resolve()
    output = (a.output or run / "diagnostics" / "endpoint_range_classes").resolve()
    output.mkdir(parents=True, exist_ok=True)
    poses = json.loads((run / "map" / "trajectory.json").read_text())["poses"]
    pt = np.asarray([q["timestamp_universal"] / 1e7 - UNIVERSAL_TO_UNIX_SECONDS for q in poses])
    px = np.asarray([q["x"] for q in poses]); py = np.asarray([q["y"] for q in poses])
    pa = np.unwrap(np.asarray([2 * np.arctan2(q["qz"], q["qw"]) for q in poses]))
    near, far = [], []
    for path in sorted((run / "input" / "horizon").glob("*.npz"))[::a.stride]:
        with np.load(path) as z:
            xy = np.asarray(z["xy"], float); stamp = float(z["stamp"])
        ranges = np.linalg.norm(xy, axis=1)
        x = np.interp(stamp, pt, px); y = np.interp(stamp, pt, py); yaw = np.interp(stamp, pt, pa)
        c, s = np.cos(yaw), np.sin(yaw)
        world = np.column_stack((x + c * xy[:, 0] - s * xy[:, 1], y + s * xy[:, 0] + c * xy[:, 1]))
        near.append(world[ranges <= a.split_range]); far.append(world[ranges > a.split_range])
    near = np.vstack(near); far = np.vstack(far)
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    ax.scatter(far[:, 0], far[:, 1], s=.45, c="#e11d48", alpha=.55, linewidths=0,
               label=f"raw range > {a.split_range:g} m")
    ax.scatter(near[:, 0], near[:, 1], s=.5, c="#0891b2", alpha=.65, linewidths=0,
               label=f"raw range ≤ {a.split_range:g} m")
    ax.plot(px, py, c="black", lw=1, label="trajectory")
    ax.set(aspect="equal", xlabel="map X, m", ylabel="map Y, m",
           title="Accumulated raw endpoints by original LiDAR range")
    ax.grid(alpha=.2); ax.legend(markerscale=8)
    target = output / "endpoints_near_vs_far.png"
    fig.savefig(target, dpi=200); plt.close(fig)
    yaml_text = (run / "map" / "map.yaml").read_text()
    resolution = float(next(line.split(":", 1)[1] for line in yaml_text.splitlines()
                            if line.startswith("resolution:")))
    origin_text = next(line.split(":", 1)[1] for line in yaml_text.splitlines()
                       if line.startswith("origin:"))
    origin = np.fromstring(origin_text.strip().strip("[]"), sep=",")
    image = np.asarray(Image.open(run / "map" / "map.pgm"))
    def pixels(points):
        return np.column_stack(((points[:, 0] - origin[0]) / resolution,
                                image.shape[0] - 1 - (points[:, 1] - origin[1]) / resolution))
    near_px, far_px = pixels(near), pixels(far)
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.scatter(far_px[:, 0], far_px[:, 1], s=.35, c="#e11d48", alpha=.35, linewidths=0,
               label=f"> {a.split_range:g} m")
    ax.scatter(near_px[:, 0], near_px[:, 1], s=.4, c="#06b6d4", alpha=.45, linewidths=0,
               label=f"≤ {a.split_range:g} m")
    ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0), aspect="equal",
           title="Raw endpoints over occupancy map: near versus far returns")
    ax.axis("off"); ax.legend(markerscale=10, loc="lower right")
    fig.savefig(output / "endpoints_over_occupancy.png", dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
