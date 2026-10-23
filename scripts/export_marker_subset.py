#!/usr/bin/env python3
"""Render selected marker distances and export one source observation per marker."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


UNIVERSAL_EPOCH_OFFSET_S = 62135596800.0


def split_segments(rows, max_frame_gap=3, max_time_gap_s=1.5):
    by_track = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[row["track_id"]].append(index)
    result = []
    for values in by_track.values():
        values.sort(key=lambda index: rows[index]["frame_index"])
        current = [values[0]]
        for index in values[1:]:
            previous = current[-1]
            if (rows[index]["frame_index"] - rows[previous]["frame_index"] > max_frame_gap
                    or rows[index]["timestamp_s"] - rows[previous]["timestamp_s"] > max_time_gap_s):
                result.append(current)
                current = [index]
            else:
                current.append(index)
        result.append(current)
    return result


def video_clock(seconds):
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--marker-ids", nargs="+", required=True)
    parser.add_argument("--edges", nargs="+", required=True, help="Pairs such as 001-002")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    observations = json.loads(args.observations.read_text())
    marker_report = json.loads((args.run / "markers/marker_points_stored_ray_corrected.json").read_text())
    wanted = {value.zfill(3) for value in args.marker_ids}
    markers = {
        marker["marker_id"].split("_")[-1]: marker
        for marker in marker_report["markers"]
        if marker["marker_id"].split("_")[-1] in wanted
    }
    missing = wanted - markers.keys()
    if missing:
        raise ValueError(f"Markers absent from report: {sorted(missing)}")

    times = observations["frame_timestamps"]
    rows = []
    for frame_index, frame in enumerate(observations["data"]):
        if not isinstance(frame, dict):
            continue
        for track_id, item in frame.items():
            if isinstance(item, dict) and item.get("ray") is not None:
                rows.append({
                    "frame_index": frame_index,
                    "timestamp_s": float(times[frame_index]),
                    "track_id": str(track_id),
                    "item": item,
                })
    segments = split_segments(rows)

    trajectory = json.loads((args.run / "map/trajectory.json").read_text())
    trajectory_times = [
        float(pose["timestamp_universal"]) / 1.0e7 - UNIVERSAL_EPOCH_OFFSET_S
        for pose in trajectory["poses"]
    ]
    start_time, end_time = min(trajectory_times), max(trajectory_times)

    selected = {}
    for marker_id, marker in markers.items():
        candidates = []
        for segment_id in marker["track_segments"]:
            for row_index in segments[int(segment_id)]:
                row = rows[row_index]
                if not start_time <= row["timestamp_s"] <= end_time:
                    continue
                box = row["item"].get("tlbr", [])
                if len(box) != 4:
                    continue
                area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
                candidates.append((area, row))
        if not candidates:
            raise ValueError(f"No in-range observation for marker {marker_id}")
        selected[marker_id] = max(candidates, key=lambda value: value[0])[1]

    # Keep the conventional observations.json frame list, but only four
    # frames are populated and every marker has exactly one (top-1) item.
    subset_data = [{} for _ in observations["data"]]
    selection_report = []
    video_start = float(times[0])
    for marker_id, row in sorted(selected.items()):
        frame_index = row["frame_index"]
        item = dict(row["item"])
        seconds = row["timestamp_s"] - video_start
        item.update({
            "code": int(marker_id),
            "marker_id": f"marker_{marker_id}",
            "selection_rank": 1,
            "selection_metric": "largest_valid_bbox_area",
            "source_track_id": row["track_id"],
            "source_frame_index_zero_based": frame_index,
            "source_frame_number_one_based": frame_index + 1,
            "source_timestamp_unix_s": row["timestamp_s"],
            "source_video_time_s": seconds,
            "source_video_time_hhmmss_ms": video_clock(seconds),
        })
        subset_data[frame_index][marker_id] = item
        selection_report.append({key: item[key] for key in (
            "marker_id", "source_track_id", "source_frame_index_zero_based",
            "source_frame_number_one_based", "source_timestamp_unix_s",
            "source_video_time_s", "source_video_time_hhmmss_ms",
            "selection_rank", "selection_metric",
        )})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset = {
        "video_path": observations.get("video_path"),
        "data": subset_data,
        "frame_timestamps": times,
        "camera_origin": observations.get("camera_origin"),
        "ray_frame": observations.get("ray_frame"),
        "camera_origin_frame": observations.get("camera_origin_frame"),
        "ray_source": observations.get("ray_source"),
        "selection": {
            "policy": "top-1 largest valid bounding box per selected physical marker",
            "source_observations": str(args.observations.resolve()),
            "source_video": observations.get("video_path"),
            "markers": selection_report,
        },
    }
    (args.out_dir / "observations_top1.json").write_text(
        json.dumps(subset, ensure_ascii=False, indent=2) + "\n"
    )

    map_image = cv2.imread(str(args.run / "map/map.png"), cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        raise FileNotFoundError(args.run / "map/map.png")
    height, width = map_image.shape
    map_frame = trajectory["map_frame"]
    origin_x, origin_y = map_frame["occupancy_grid_origin_xy_m"]
    resolution = float(map_frame["occupancy_grid_resolution_m"])

    def pixel(point):
        return ((point[0] - origin_x) / resolution,
                height - 1 - (point[1] - origin_y) / resolution)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=180)
    ax.imshow(map_image, cmap="gray", origin="upper")
    colors = {"001": "#e41a1c", "002": "#377eb8", "003": "#4daf4a", "005": "#984ea3"}
    for marker_id, marker in markers.items():
        x, y = pixel(marker["point_xyz_m"])
        ax.scatter([x], [y], s=100, color=colors.get(marker_id, "red"), edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(marker_id, (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=11, weight="bold", color=colors.get(marker_id, "red"), zorder=5)

    edge_report = []
    for value in args.edges:
        left, right = [part.zfill(3) for part in value.split("-", 1)]
        pa, pb = markers[left]["point_xyz_m"], markers[right]["point_xyz_m"]
        distance = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        xa, ya = pixel(pa); xb, yb = pixel(pb)
        ax.plot([xa, xb], [ya, yb], color="#ff8c00", linewidth=2.2, zorder=3)
        ax.annotate(f"{distance:.3f} m", ((xa + xb) / 2, (ya + yb) / 2),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10,
                    color="black", weight="bold",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#ff8c00"})
        edge_report.append({"from": left, "to": right, "distance_xy_m": distance})

    points = np.asarray([pixel(marker["point_xyz_m"]) for marker in markers.values()])
    padding = 120
    ax.set_xlim(max(0, points[:, 0].min() - padding), min(width, points[:, 0].max() + padding))
    ax.set_ylim(min(height, points[:, 1].max() + padding), max(0, points[:, 1].min() - padding))
    ax.set_title("Selected markers and XY distances")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "selected_markers_distances.png", bbox_inches="tight")
    plt.close(fig)
    (args.out_dir / "distances.json").write_text(json.dumps(edge_report, indent=2) + "\n")

    print(json.dumps({
        "map": str((args.out_dir / "selected_markers_distances.png").resolve()),
        "observations": str((args.out_dir / "observations_top1.json").resolve()),
        "selected_frames": selection_report,
        "distances": edge_report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
