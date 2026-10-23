#!/usr/bin/env python3
"""Recalculate marker segment heights with an experimental camera calibration.

The script intentionally preserves the physical-marker/track-segment grouping
from an existing localization result.  Rays themselves are rebuilt from the
recorded pixel centres, supplied intrinsics and supplied camera roll/pitch.
The resulting Cartographer points are rigidly registered to four surveyed
control markers and rendered in that surveyed frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/fliftbot_mapping/evaluation"))
import ceiling_marker_localization_stored_ray_corrected as stored  # noqa: E402
import ceiling_marker_localization_track_candidate as base  # noqa: E402


def rotation3(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(-roll), math.sin(-roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    return rz @ ry @ rx


def load_intrinsics(path: Path, fx_px: float, fy_px: float) -> tuple[np.ndarray, np.ndarray]:
    entry = next(iter(json.loads(path.read_text()).values()))
    scale = float(max(entry["width"], entry["height"]))
    matrix = np.asarray([
        [fx_px, 0, entry["width"] / 2 + entry["c_x"] * scale],
        [0, fy_px, entry["height"] / 2 + entry["c_y"] * scale],
        [0, 0, 1],
    ], float)
    distortion = np.asarray([entry.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], float)
    return matrix, distortion


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The surveyed frame is explicitly anchored at the first control marker.
    # Fit only rotation after subtracting that anchor so marker 018 remains
    # exactly (0, 0), rather than moving it in a four-point least-squares fit.
    source_zero = source - source[0]
    target_zero = target - target[0]
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target[0] - source[0] @ rotation
    residual = source @ rotation + translation - target
    return rotation, translation, residual


def parse_map_yaml(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            result[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            result[key.strip()] = value.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-marker-dir", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--camera-position", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fx-px", type=float, required=True)
    parser.add_argument("--fy-px", type=float, required=True)
    parser.add_argument("--roll-deg", type=float, required=True)
    parser.add_argument("--pitch-deg", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows, _ = stored.rows_from_stored(args.observations, args.camera_json)
    matrix, distortion = load_intrinsics(args.camera_json, args.fx_px, args.fy_px)
    pixels = np.asarray([row["center"] for row in rows], float)
    optical = cv2.undistortPoints(pixels[:, None, :], matrix, distortion)[:, 0, :]
    optical = np.column_stack((optical, np.ones(len(optical))))
    mount = rotation3(math.radians(args.roll_deg), math.radians(args.pitch_deg), math.radians(args.yaw_deg))
    body = optical @ mount.T
    body /= np.linalg.norm(body, axis=1)[:, None]

    times, trajectory, trajectory_meta = base.load_trajectory(args.run / "map/trajectory.json")
    poses, inside = base.interpolate(np.asarray([row["timestamp_s"] for row in rows]), times, trajectory)
    lidar_doc = json.loads((args.run / "input/lidar.markers.config").read_text())
    lidar = np.asarray([lidar_doc["position_m"][k] for k in ("x", "y", "z")], float)
    camera = np.asarray(stored.sidecar(args.camera_position)["position_m"], float)
    lever = camera[:2] - lidar[:2]
    heading = poses[:, 2] - math.radians(float(lidar_doc["orientation_deg"]["yaw"]))
    origins = np.column_stack((
        poses[:, 0] + np.cos(heading) * lever[0] - np.sin(heading) * lever[1],
        poses[:, 1] + np.sin(heading) * lever[0] + np.cos(heading) * lever[1],
        np.full(len(rows), camera[2]),
    ))
    directions = np.column_stack((
        np.cos(heading) * body[:, 0] - np.sin(heading) * body[:, 1],
        np.sin(heading) * body[:, 0] + np.cos(heading) * body[:, 1],
        body[:, 2],
    ))
    valid = inside & np.isfinite(directions).all(1) & (directions[:, 2] > 1.0e-6)
    segments = base.split_segments(rows)
    old = json.loads((args.source_marker_dir / "marker_points_stored_ray_corrected.json").read_text())

    segment_fits: dict[int, dict] = {}
    for sid, values in enumerate(segments):
        indices = np.asarray([i for i in values if valid[i]], int)
        if len(indices) < 8:
            continue
        weights = np.asarray([rows[i].get("screen_weight", 1.0) for i in indices], float)
        fit = base.estimate_track_height(origins, directions, indices, weights)
        if fit is None:
            continue
        baseline = float(np.max(np.linalg.norm(origins[indices, :2] - origins[indices[0], :2], axis=1)))
        segment_fits[sid] = {
            "segment_id": sid,
            "observations": len(indices),
            "baseline_m": baseline,
            **fit,
        }

    markers = []
    for marker in old["markers"]:
        fits = [segment_fits[sid] for sid in marker["track_segments"] if sid in segment_fits]
        fits = [f for f in fits if 5.5 <= f["height_m"] <= 8.5 and f["median_residual_m"] <= 0.10]
        if not fits:
            continue
        heights = np.asarray([f["height_m"] for f in fits])
        height = float(np.median(heights))
        keep = np.abs(heights - height) <= 0.25
        if not np.any(keep):
            keep[int(np.argmin(np.abs(heights - height)))] = True
        accepted = [f for f, use in zip(fits, keep) if use]
        centers = np.asarray([f["center_xy_m"] for f in accepted])
        center0 = np.median(centers, axis=0)
        spatial_keep = np.linalg.norm(centers - center0, axis=1) <= 0.30
        accepted = [f for f, use in zip(accepted, spatial_keep) if use]
        weights = np.asarray([f["observations"] / max(f["median_residual_m"], 0.01) for f in accepted])
        xy = np.average(np.asarray([f["center_xy_m"] for f in accepted]), axis=0, weights=weights)
        height = float(np.median([f["height_m"] for f in accepted]))
        markers.append({
            "marker_id": marker["marker_id"],
            "map_xyz_m": [float(xy[0]), float(xy[1]), height],
            "track_segments": [f["segment_id"] for f in accepted],
            "observations": int(sum(f["observations"] for f in accepted)),
            "height_segment_min_m": float(min(f["height_m"] for f in accepted)),
            "height_segment_max_m": float(max(f["height_m"] for f in accepted)),
            "median_ray_spread_m": float(np.median([f["median_residual_m"] for f in accepted])),
        })

    by_id = {m["marker_id"]: m for m in markers}
    control_ids = ["marker_018", "marker_004", "marker_022", "marker_007"]
    surveyed = np.asarray([[0, 0], [0, 4.89], [3.02, 0], [3.04, 4.89]], float)
    source = np.asarray([by_id[mid]["map_xyz_m"][:2] for mid in control_ids])
    rotation, translation, control_residuals = rigid_fit(source, surveyed)
    for marker in markers:
        local = np.asarray(marker["map_xyz_m"][:2]) @ rotation + translation
        marker["surveyed_xyz_m"] = [float(local[0]), float(local[1]), marker["map_xyz_m"][2]]
        if marker["marker_id"] in control_ids:
            control_index = control_ids.index(marker["marker_id"])
            marker["triangulated_surveyed_xyz_m"] = marker["surveyed_xyz_m"]
            marker["surveyed_xyz_m"] = [
                float(surveyed[control_index, 0]),
                float(surveyed[control_index, 1]),
                float([7.175, 7.175, 7.157, 7.160][control_index]),
            ]

    report = {
        "method": "pixel rays rebuilt with experimental intrinsics/extrinsics; per-segment height fit",
        "calibration": {
            "fx_px": args.fx_px, "fy_px": args.fy_px,
            "roll_deg": args.roll_deg, "pitch_deg": args.pitch_deg, "yaw_deg": args.yaw_deg,
            "principal_point_and_distortion_source": str(args.camera_json.resolve()),
            "camera_position_source": str(args.camera_position.resolve()),
        },
        "trajectory": trajectory_meta,
        "valid_observations": int(valid.sum()),
        "total_observations": len(rows),
        "surveyed_frame": {
            "control_policy": "hard anchors in exported surveyed coordinates; triangulated values retained for diagnostics",
            "controls": dict(zip(control_ids, surveyed.tolist())),
            "map_to_surveyed_rotation_row_vector": rotation.tolist(),
            "map_to_surveyed_translation_m": translation.tolist(),
            "control_residuals_m": control_residuals.tolist(),
            "control_rmse_m": float(np.sqrt(np.mean(np.sum(control_residuals**2, axis=1)))),
        },
        "markers": markers,
    }
    (args.output / "marker_points_calibrated_camera.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "marker_heights_calibrated_camera.csv").open("w", newline="") as handle:
        fields = ["marker_id", "x_m", "y_m", "height_m", "observations", "height_segment_min_m", "height_segment_max_m", "median_ray_spread_m"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for marker in markers:
            x, y, z = marker["surveyed_xyz_m"]
            writer.writerow({"marker_id": marker["marker_id"], "x_m": x, "y_m": y, "height_m": z,
                             "observations": marker["observations"], "height_segment_min_m": marker["height_segment_min_m"],
                             "height_segment_max_m": marker["height_segment_max_m"], "median_ray_spread_m": marker["median_ray_spread_m"]})

    map_meta = parse_map_yaml(args.run / "map/map.yaml")
    image = plt.imread(args.run / "map" / map_meta["image"])
    resolution = float(map_meta["resolution"])
    ox, oy = map_meta["origin"][:2]
    extent = [ox, ox + image.shape[1] * resolution, oy, oy + image.shape[0] * resolution]
    # Row-vector p_local=p_map@R+t corresponds to this column-vector affine.
    affine = Affine2D.from_values(rotation[0, 0], rotation[0, 1], rotation[1, 0], rotation[1, 1], translation[0], translation[1])
    points = np.asarray([m["surveyed_xyz_m"] for m in markers])
    fig, ax = plt.subplots(figsize=(16, 12), dpi=180)
    ax.imshow(np.flipud(image), cmap="gray", origin="lower", extent=extent,
              transform=affine + ax.transData, interpolation="nearest", alpha=0.78)
    ax.scatter(points[:, 0], points[:, 1], s=28, c=points[:, 2], cmap="turbo", vmin=6.8, vmax=7.5,
               edgecolors="black", linewidths=0.5, zorder=3)
    for marker in markers:
        x, y, z = marker["surveyed_xyz_m"]
        label = marker["marker_id"].replace("marker_", "")
        ax.annotate(f"{label}\nH={z:.3f}", (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color="black", bbox={"facecolor": "white", "alpha": 0.72, "pad": 1, "edgecolor": "none"})
    ax.scatter([0], [0], marker="x", s=100, c="red", linewidths=2.2, zorder=5)
    ax.annotate("(0, 0) / marker 018", (0, 0), xytext=(8, -16), textcoords="offset points", color="red", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("X, m (surveyed marker frame)")
    ax.set_ylabel("Y, m (surveyed marker frame)")
    ax.set_title("Markers recalculated with experimental camera calibration")
    ax.grid(True, which="major", color="#1677ff", alpha=0.45, linewidth=0.65)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlim(float(points[:, 0].min() - 2), float(points[:, 0].max() + 2))
    ax.set_ylim(float(points[:, 1].min() - 2), float(points[:, 1].max() + 2))
    fig.colorbar(ax.collections[0], ax=ax, label="Height, m", shrink=0.72)
    fig.tight_layout()
    fig.savefig(args.output / "map_markers_calibrated_camera_grid_1m.png")
    plt.close(fig)
    print(json.dumps({"markers": len(markers), "control_rmse_m": report["surveyed_frame"]["control_rmse_m"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
