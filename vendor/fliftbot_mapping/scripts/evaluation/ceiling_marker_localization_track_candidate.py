#!/usr/bin/env python3
"""Production candidate: metadata-driven, track-first ceiling localization.

Runtime inputs are observations, frame timestamps, camera sidecars, the
measured camera/LiDAR mounts, and a timestamped 2-D LiDAR trajectory.  GT,
floorplan geometry, marker anchors, V5/V6 angle constants, and fitted time
offsets are forbidden.

Legacy observations store ``tlbr=[y1,x1,y2,x2]``.  ``item.center`` and the
stored ray were exported with the axes swapped and are audit-only here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


UNIX_EPOCH_TICKS = 621355968000000000
def rotation3(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    entry = next(iter(json.loads(path.read_text(encoding="utf-8")).values()))
    width, height = int(entry["width"]), int(entry["height"])
    scale = float(max(width, height))
    matrix = np.asarray([
        [float(entry["focal_x"]) * scale, 0.0, width / 2.0 + float(entry["c_x"]) * scale],
        [0.0, float(entry["focal_y"]) * scale, height / 2.0 + float(entry["c_y"]) * scale],
        [0.0, 0.0, 1.0],
    ])
    distortion = np.asarray([float(entry.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2", "k3")])
    return matrix, distortion, width, height


def load_sidecar_mount(path: Path, mount_mode: str = "legacy_plus90_in_mount") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_position_cm = np.asarray([float(value) for value in lines[0].split()], dtype=float)
    source_angles = np.asarray([float(value) for value in lines[1].split()], dtype=float)
    if source_angles.shape not in ((2,), (3,)):
        raise ValueError(f"Expected roll pitch [yaw] in {path}")
    # Exact contract in auto_map/get_video_data.py::_build_camera_model_from_sidecars.
    # Legacy two-angle sidecars use +90 deg mount yaw.
    yaw = float(source_angles[2]) if len(source_angles) == 3 else (math.pi / 2.0 if mount_mode == "legacy_plus90_in_mount" else 0.0)
    applied = np.asarray([-source_angles[0], source_angles[1], yaw], dtype=float)
    camera_axis_m = source_position_cm / 100.0
    if camera_axis_m.shape != (3,) or not np.isfinite(camera_axis_m).all():
        raise ValueError(f"Expected finite camera position X Y Z in centimetres in {path}")
    return rotation3(*applied), camera_axis_m, {
        "source": str(path.resolve()),
        "source_position_cm": source_position_cm.tolist(),
        "applied_camera_axis_m": camera_axis_m.tolist(),
        "source_angles_rad": source_angles.tolist(),
        "applied_camera_model_tilts_rad": applied.tolist(),
        "euler_order": "Rz @ Ry @ Rx",
        "contract_source": "auto_map/get_video_data.py::_build_camera_model_from_sidecars and camera_model.py::_rotation_matrix_from_euler",
    }


def load_frame_times(payload: dict[str, Any], observations: Path) -> tuple[np.ndarray, str]:
    embedded = payload.get("frame_timestamps")
    frames = payload.get("data", [])
    if embedded and len(embedded) >= len(frames):
        values = np.asarray(embedded[:len(frames)], dtype=float)
        if np.isfinite(values).all():
            return values, "observations.json:frame_timestamps"
    path = observations.parent / "frame_timestamps.csv"
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            values[int(row.get("frame_index", ordinal))] = float(row["time_take_img_sec"])
    return np.asarray([values[index] for index in range(len(frames))], dtype=float), str(path.resolve())


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("tracking_frame") != "lidar":
        raise ValueError("Candidate requires tracking_frame=lidar")
    times, poses = [], []
    for row in payload.get("poses", []):
        stamp = (float(row["timestamp_universal"]) - UNIX_EPOCH_TICKS) / 1.0e7
        qx, qy, qz, qw = (float(row.get(key, 0.0)) for key in ("qx", "qy", "qz", "qw"))
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        times.append(stamp); poses.append([float(row["x"]), float(row["y"]), yaw])
    order = np.argsort(times)
    times_a, poses_a = np.asarray(times)[order], np.asarray(poses)[order]
    return times_a, poses_a, {"source": str(path.resolve()), "tracking_frame": "lidar", "timestamp_domain": "Unix seconds converted from universal ticks"}


def interpolate(query: np.ndarray, times: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inside = (query >= times[0]) & (query <= times[-1])
    return np.column_stack((np.interp(query, times, poses[:, 0]), np.interp(query, times, poses[:, 1]), np.interp(query, times, np.unwrap(poses[:, 2])))), inside


def read_rows(observations: Path, camera_json: Path, mount_rotation: np.ndarray) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    payload = json.loads(observations.read_text(encoding="utf-8"))
    frame_times, frame_source = load_frame_times(payload, observations)
    matrix, distortion, width, height = load_camera(camera_json)
    rows, audit_angles = [], []
    for frame_index, frame in enumerate(payload.get("data", [])):
        if not isinstance(frame, dict):
            continue
        for track_id, item in frame.items():
            if not isinstance(item, dict):
                continue
            box = np.asarray(item.get("tlbr", []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            y1, x1, y2, x2 = box
            pixel = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                continue
            normalized = cv2.undistortPoints(pixel.reshape(1, 1, 2), matrix, distortion).reshape(2)
            optical = np.asarray([normalized[0], normalized[1], 1.0], dtype=float)
            optical /= np.linalg.norm(optical)
            body = mount_rotation @ optical
            body /= np.linalg.norm(body)
            try:
                code = int(item.get("code", -1))
            except (TypeError, ValueError):
                code = -1
            stored = np.asarray(item.get("ray", []), dtype=float)
            if stored.shape == (3,) and np.isfinite(stored).all():
                stored /= np.linalg.norm(stored)
                audit_angles.append(math.degrees(math.acos(float(np.clip(stored @ body, -1.0, 1.0)))))
            rows.append({"frame_index": frame_index, "timestamp_s": float(frame_times[frame_index]), "track_id": str(track_id), "code": code, "pixel_uv": pixel.tolist(), "body_ray_unit": body.tolist()})
    audit = {"stored_ray_policy": "audit_only; known swapped-pixel export", "stored_vs_rebuilt_body_ray_median_deg": float(np.median(audit_angles)), "stored_vs_rebuilt_body_ray_p95_deg": float(np.percentile(audit_angles, 95)), "samples": len(audit_angles)}
    return rows, frame_source, audit


def split_segments(rows: list[dict[str, Any]], max_frame_gap: int = 3, max_time_gap_s: float = 1.5) -> list[list[int]]:
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): by_track[row["track_id"]].append(index)
    result = []
    for values in by_track.values():
        values.sort(key=lambda index: rows[index]["frame_index"])
        current = [values[0]]
        for index in values[1:]:
            previous = current[-1]
            if rows[index]["frame_index"] - rows[previous]["frame_index"] > max_frame_gap or rows[index]["timestamp_s"] - rows[previous]["timestamp_s"] > max_time_gap_s:
                result.append(current); current = [index]
            else:
                current.append(index)
        result.append(current)
    return result


def estimate_track_height(origins: np.ndarray, directions: np.ndarray, indices: np.ndarray) -> dict[str, Any] | None:
    o, d = origins[indices], directions[indices]
    q = d[:, :2] / d[:, 2, None]
    a = o[:, :2] - o[:, 2, None] * q
    da, dq = a - a.mean(axis=0), q - q.mean(axis=0)
    denominator = float(np.sum(dq * dq))
    if denominator < 1.0e-4:
        return None
    height = -float(np.sum(da * dq)) / denominator
    points = a + height * q
    center = np.median(points, axis=0)
    residuals = np.linalg.norm(points - center, axis=1)
    return {"height_m": height, "center_xy_m": center.tolist(), "median_residual_m": float(np.median(residuals)), "p95_residual_m": float(np.percentile(residuals, 95)), "conditioning": denominator}


def merge_markers(markers: list[dict[str, Any]], radius_m: float) -> list[dict[str, Any]]:
    pending, groups = set(range(len(markers))), []
    while pending:
        seed = min(pending); pending.remove(seed); stack, group = [seed], [seed]
        while stack:
            i = stack.pop(); point = np.asarray(markers[i]["point_xy_m"])
            for j in list(pending):
                if np.linalg.norm(point - np.asarray(markers[j]["point_xy_m"])) <= radius_m:
                    pending.remove(j); stack.append(j); group.append(j)
        groups.append(group)
    result = []
    for group in groups:
        weights = np.asarray([markers[i]["observations"] / max(markers[i]["spread_m"], 0.01) for i in group])
        point = np.average(np.asarray([markers[i]["point_xy_m"] for i in group]), axis=0, weights=weights)
        result.append({"point_xyz_m": [float(point[0]), float(point[1]), float(markers[group[0]]["ceiling_z_m"])], "track_segments": [markers[i]["segment_id"] for i in group], "candidate_codes": sorted({code for i in group for code in markers[i]["candidate_codes"] if code >= 0}), "observations": int(sum(markers[i]["observations"] for i in group)), "spread_m": float(np.median([markers[i]["spread_m"] for i in group]))})
    return result


def read_map_yaml(path: Path) -> dict[str, Any]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line: continue
        key, value = line.split(":", 1); value = value.strip()
        try: result[key] = json.loads(value)
        except Exception:
            try: result[key] = float(value)
            except ValueError: result[key] = value.strip("'\"")
    return result


def render(path: Path, map_dir: Path, trajectory: np.ndarray, origins: np.ndarray, directions: np.ndarray, markers: list[dict[str, Any]], status: str, ceiling_z: float | None) -> None:
    meta = read_map_yaml(map_dir / "map.yaml")
    image = np.asarray(Image.open(map_dir / str(meta["image"])).convert("L"))
    resolution, origin = float(meta["resolution"]), np.asarray(meta["origin"], dtype=float)
    extent = [origin[0], origin[0] + image.shape[1] * resolution, origin[1], origin[1] + image.shape[0] * resolution]
    fig, ax = plt.subplots(figsize=(14, 11), dpi=180)
    ax.imshow(np.flipud(image), cmap="gray", origin="lower", extent=extent, interpolation="nearest")
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#2563eb", lw=0.7, alpha=0.65, label="LiDAR trajectory")
    if ceiling_z is not None and len(origins):
        stride = max(1, len(origins) // 800)
        depth = (ceiling_z - origins[::stride, 2]) / directions[::stride, 2]
        ends = origins[::stride, :2] + depth[:, None] * directions[::stride, :2]
        for start, end in zip(origins[::stride, :2], ends):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#f59e0b", lw=0.25, alpha=0.08)
    if markers:
        points = np.asarray([row["point_xyz_m"][:2] for row in markers])
        ax.scatter(points[:, 0], points[:, 1], color="#dc2626", edgecolors="white", lw=0.4, s=38, label="track-first markers")
    ax.set_aspect("equal"); ax.grid(alpha=0.15); ax.legend(loc="best")
    ax.set_title(f"Track-first production candidate — {status} — ceiling={ceiling_z}")
    fig.tight_layout(); fig.savefig(path)
    # The occupancy image can contain long-range LiDAR returns far outside the
    # explored room.  Keep the full-map diagnostic above, and also write a
    # close-up where marker clusters remain readable.
    focus = trajectory[:, :2]
    if markers:
        focus = np.vstack((focus, np.asarray([row["point_xyz_m"][:2] for row in markers])))
    if len(focus):
        lo, hi = np.min(focus, axis=0), np.max(focus, axis=0)
        span = np.maximum(hi - lo, 1.0)
        pad = np.maximum(0.15 * span, 0.75)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_title(f"Track-first markers close-up — {status} — ceiling={ceiling_z}")
        closeup = path.with_name(path.stem + "_closeup" + path.suffix)
        fig.tight_layout(); fig.savefig(closeup)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--camera-position", type=Path, required=True)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract-calibration", type=Path)
    parser.add_argument("--min-segment-observations", type=int, default=8)
    parser.add_argument("--min-camera-baseline-m", type=float, default=0.35)
    parser.add_argument("--merge-radius-m", type=float, default=0.10)
    args = parser.parse_args()

    lidar_doc = json.loads(args.lidar_config.read_text(encoding="utf-8"))
    lidar = np.asarray([lidar_doc["position_m"][key] for key in ("x", "y", "z")], dtype=float)
    if lidar.shape != (3,) or not np.isfinite(lidar).all():
        raise ValueError(f"Expected finite LiDAR position x/y/z in {args.lidar_config}")
    yaw_sign, yaw_offset, time_offset = 1.0, 0.0, 0.0
    mount_mode = "legacy_plus90_in_mount"
    calibration_meta = None
    if args.contract_calibration:
        calibration_doc = json.loads(args.contract_calibration.read_text(encoding="utf-8"))
        if calibration_doc.get("uses_gt") is not False:
            raise ValueError("Contract calibration must explicitly declare uses_gt=false")
        selected = calibration_doc["selected"]
        run_name = next((part for part in args.observations.parts if part.startswith("run_")), args.observations.parent.name)
        run_fit = selected["selected_pair"][run_name]
        yaw_sign = float(selected["yaw_sign"]); yaw_offset = float(selected["yaw_offset_rad"])
        time_offset = float(run_fit["dt_s"]); mount_mode = str(selected["mount_mode"])
        calibration_meta = {"source": str(args.contract_calibration.resolve()), "run": run_name,
                            "uses_gt": False, "yaw_sign": yaw_sign, "yaw_offset_rad": yaw_offset,
                            "time_offset_s": time_offset, "mount_mode": mount_mode,
                            "shared_ceiling_z_audit_m": selected.get("shared_ceiling_z_m")}
    mount_rotation, camera_axis, mount_meta = load_sidecar_mount(args.camera_position, mount_mode)
    camera_from_lidar_xy = camera_axis[:2] - lidar[:2]
    rows, frame_source, ray_audit = read_rows(args.observations, args.camera_json, mount_rotation)
    trajectory_times, trajectory, trajectory_meta = load_trajectory(args.trajectory)
    query = np.asarray([row["timestamp_s"] for row in rows], dtype=float) + time_offset
    poses, inside = interpolate(query, trajectory_times, trajectory)
    poses[:, 2] = yaw_sign * poses[:, 2] + yaw_offset
    body = np.asarray([row["body_ray_unit"] for row in rows])
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    origins = np.column_stack((poses[:, 0] + c * camera_from_lidar_xy[0] - s * camera_from_lidar_xy[1], poses[:, 1] + s * camera_from_lidar_xy[0] + c * camera_from_lidar_xy[1], np.full(len(poses), camera_axis[2])))
    directions = np.column_stack((c * body[:, 0] - s * body[:, 1], s * body[:, 0] + c * body[:, 1], body[:, 2]))
    valid = inside & (directions[:, 2] > 1.0e-6) & np.isfinite(directions).all(axis=1)

    segment_records = []
    for segment_id, values in enumerate(split_segments(rows)):
        indices = np.asarray([index for index in values if valid[index]], dtype=int)
        if len(indices) < args.min_segment_observations: continue
        baseline = float(np.max(np.linalg.norm(origins[indices, :2] - origins[indices[0], :2], axis=1)))
        if baseline < args.min_camera_baseline_m: continue
        fit = estimate_track_height(origins, directions, indices)
        if fit is None: continue
        codes = Counter(rows[index]["code"] for index in indices if rows[index]["code"] >= 0)
        segment_records.append({"segment_id": segment_id, "indices": indices, "observations": len(indices), "baseline_m": baseline, "codes": [code for code, _count in codes.most_common()], **fit})

    plausible = [row for row in segment_records if camera_axis[2] + 0.5 <= row["height_m"] <= camera_axis[2] + 8.0 and row["median_residual_m"] <= 0.20]
    heights = np.asarray([row["height_m"] for row in plausible], dtype=float)
    all_heights = np.asarray([row["height_m"] for row in segment_records], dtype=float)
    all_residuals = np.asarray([row["median_residual_m"] for row in segment_records], dtype=float)
    height_report = {
        "candidate_segments": int(len(heights)),
        "inlier_segments": 0,
        "ceiling_z_m": None,
        "robust_sigma_m": None,
        "confidence": 0.0,
        "decision": "abstain_no_plausible_track_height_consensus",
        "all_fitted_height_quantiles_m": dict(zip(("min", "p10", "median", "p90", "max"), np.percentile(all_heights, (0, 10, 50, 90, 100)).tolist())) if len(all_heights) else None,
        "all_track_median_residual_quantiles_m": dict(zip(("min", "p10", "median", "p90", "max"), np.percentile(all_residuals, (0, 10, 50, 90, 100)).tolist())) if len(all_residuals) else None,
        "rejections": {
            "height_outside_camera_plus_0_5_to_8m": int(sum(not (camera_axis[2] + 0.5 <= row["height_m"] <= camera_axis[2] + 8.0) for row in segment_records)),
            "median_residual_over_0_20m": int(sum(row["median_residual_m"] > 0.20 for row in segment_records)),
        },
    }
    ceiling_z = None
    if len(heights) >= 8:
        center = float(np.median(heights)); mad = float(np.median(np.abs(heights - center)))
        gate = max(0.12, 3.0 * 1.4826 * mad)
        keep = np.abs(heights - center) <= gate
        sigma = float(1.4826 * np.median(np.abs(heights[keep] - np.median(heights[keep])))) if keep.any() else math.inf
        confidence = min(1.0, float(keep.sum()) / 20.0) * math.exp(-sigma / 0.10)
        height_report.update({"inlier_segments": int(keep.sum()), "ceiling_z_m": float(np.median(heights[keep])) if keep.any() else None, "robust_sigma_m": sigma, "confidence": confidence, "decision": "accepted" if int(keep.sum()) >= 8 and sigma <= 0.12 and confidence >= 0.25 else "abstain_unstable_common_plane"})
        if int(keep.sum()) >= 8 and sigma <= 0.12 and confidence >= 0.25:
            ceiling_z = float(np.median(heights[keep]))

    track_markers = []
    if ceiling_z is not None:
        all_points = origins[:, :2] + ((ceiling_z - origins[:, 2]) / directions[:, 2])[:, None] * directions[:, :2]
        for row in segment_records:
            indices = row["indices"]; points = all_points[indices]
            center = np.median(points, axis=0); residuals = np.linalg.norm(points - center, axis=1)
            gate = max(0.08, float(np.percentile(residuals, 80)))
            keep = residuals <= gate
            if int(keep.sum()) < args.min_segment_observations: continue
            center = np.median(points[keep], axis=0); spread = float(np.median(np.linalg.norm(points[keep] - center, axis=1)))
            if spread > 0.08: continue
            track_markers.append({"segment_id": row["segment_id"], "point_xy_m": center.tolist(), "ceiling_z_m": ceiling_z, "observations": int(keep.sum()), "spread_m": spread, "candidate_codes": row["codes"]})
    markers = merge_markers(track_markers, args.merge_radius_m) if track_markers else []
    status = "accepted" if ceiling_z is not None and len(markers) >= 3 else "abstained"
    if status == "abstained": markers = []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "track_candidate_map.png"
    render(png, args.map_dir, trajectory, origins[valid], directions[valid], markers, status, ceiling_z)
    report = {
        "schema": "fliftbot.ceiling_marker_localization.track_candidate.v1", "status": status,
        "runtime_gt_used": False, "runtime_registration_used": False,
        "inputs": {"observations": str(args.observations.resolve()), "trajectory": str(args.trajectory.resolve()), "camera_json": str(args.camera_json.resolve()), "camera_position": str(args.camera_position.resolve()), "lidar_config": str(args.lidar_config.resolve()), "frame_timestamps": frame_source},
        "frame_contract": {"camera_axis_xy_m": camera_axis[:2].tolist(), "lidar_axis_xy_m": lidar[:2].tolist(), "camera_from_lidar_xy_m": camera_from_lidar_xy.tolist(), "camera_height_floor_m": float(camera_axis[2]), "trajectory_is_2d": True, "tracking_frame": "lidar", "origin_formula": "pose_lidar_xy + R(selected_heading) @ camera_from_lidar_xy; origin_z=camera_height_floor", "time_mapping": f"direct same Unix domain; data-only dt={time_offset:+.6f}s", "selected_heading_formula": f"{yaw_sign:+g}*quaternion_yaw + {yaw_offset:+.12g}"},
        "pixel_contract": {"source": "tlbr only", "legacy_order": "[y1,x1,y2,x2]", "center_formula": "[(x1+x2)/2,(y1+y2)/2]", "item_center": "audit_only_wrong_axes", "stored_ray": "audit_only_wrong_pixel"},
        "mount_orientation": mount_meta, "contract_calibration": calibration_meta, "trajectory_metadata": trajectory_meta, "ray_audit": ray_audit,
        "geometry": {"primitive": "continuous track_id segments", "decoder_code_role": "metadata only", "height_method": "closed-form per-track multi-view plane convergence then robust common plane", "merge_radius_m": args.merge_radius_m},
        "height": height_report,
        "counts": {"rows": len(rows), "projected_rows": int(valid.sum()), "track_segments_fitted": len(segment_records), "plausible_height_segments": len(plausible), "track_markers_before_merge": len(track_markers), "physical_markers": len(markers)},
        "markers": markers,
        "outputs": {"map_png": str(png.resolve())},
        "forbidden_inputs": ["GT", "floorplan", "marker 98", "anchor", "SE2 fit", "scale fit", "fixed scene ceiling", "V5/V6 hardcoded angles"],
    }
    output = args.out_dir / "marker_points_track_candidate.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "height": height_report, "markers": len(markers), "output": str(output.resolve()), "png": str(png.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
