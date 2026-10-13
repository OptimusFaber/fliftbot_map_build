#!/usr/bin/env python3
"""Crop, align and render an occupancy map for web/navigation use."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_map_yaml(path: Path) -> dict:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return {"resolution": float(values["resolution"]),
            "origin": [float(v) for v in json.loads(values["origin"].replace("'", '"'))],
            "image": values["image"].strip("'\""), "negate": int(values.get("negate", "0")),
            "occupied_thresh": float(values.get("occupied_thresh", "0.65")),
            "free_thresh": float(values.get("free_thresh", "0.196"))}


def classify(image: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = image.astype(np.float32) / 255.0
    if not meta["negate"]:
        probability = 1.0 - probability
    occupied = probability > meta["occupied_thresh"]
    free = probability < meta["free_thresh"]
    return occupied, free, ~(occupied | free)


def dominant_wall_rotation(occupied: np.ndarray) -> tuple[float, dict]:
    edges = cv2.Canny(occupied.astype(np.uint8) * 255, 40, 120)
    minimum = max(20, round(min(occupied.shape) * 0.025))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=25,
                            minLineLength=minimum, maxLineGap=max(5, minimum // 4))
    samples: list[tuple[float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
            samples.append((((angle + 45.0) % 90.0) - 45.0, length))
    if not samples:
        return 0.0, {"status": "insufficient_lines", "line_count": 0, "rotation_deg": 0.0}
    samples.sort()
    half = sum(weight for _, weight in samples) / 2.0
    cumulative = 0.0
    residual = 0.0
    for value, weight in samples:
        cumulative += weight
        if cumulative >= half:
            residual = value
            break
    return -residual, {"status": "aligned", "line_count": len(samples),
                       "dominant_wall_residual_deg": residual, "rotation_deg": -residual,
                       "median_abs_deviation_deg": float(np.median([abs(v - residual) for v, _ in samples]))}


def rotation_canvas(width: int, height: int, angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    affine = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_deg, 1.0)
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1],
                        [width - 1, height - 1, 1]], dtype=float)
    transformed = (affine @ corners.T).T
    low, high = transformed.min(axis=0), transformed.max(axis=0)
    affine[:, 2] -= low
    return affine, (int(math.ceil(high[0] - low[0] + 1)), int(math.ceil(high[1] - low[1] + 1)))


def matrix3(affine: np.ndarray) -> np.ndarray:
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--web-resolution-m-per-px", type=float, default=0.01)
    parser.add_argument("--margin-m", type=float, default=0.20)
    parser.add_argument("--ray-filter-window-m", type=float, default=1.0,
                        help="density window used only for rejecting isolated distant rays")
    parser.add_argument("--ray-filter-min-cells", type=int, default=0,
                        help="0 selects a density threshold from the strongest bbox contraction")
    parser.add_argument("--no-align", action="store_true")
    args = parser.parse_args()
    if args.web_resolution_m_per_px <= 0:
        raise ValueError("web resolution must be positive")
    source, output = args.map_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta = read_map_yaml(source / "map.yaml")
    source_image = np.asarray(Image.open(source / meta["image"]).convert("L"))
    occupied, _, _ = classify(source_image, meta)
    if not occupied.any():
        raise RuntimeError("occupancy map has no occupied cells")
    window = max(3, round(args.ray_filter_window_m / meta["resolution"]))
    if window % 2 == 0:
        window += 1
    density = cv2.boxFilter(occupied.astype(np.float32), -1, (window, window), normalize=False)
    selected_min_cells = args.ray_filter_min_cells
    if selected_min_cells <= 0:
        scale = window / 21.0
        candidates = sorted({max(1, round(value * scale)) for value in range(20, 81, 10)})
        measurements = []
        for threshold in candidates:
            candidate_y, candidate_x = np.where(density >= threshold)
            if len(candidate_x) < 100:
                continue
            bbox_area = max(1, int(candidate_x.max() - candidate_x.min() + 1)) * max(
                1, int(candidate_y.max() - candidate_y.min() + 1))
            measurements.append((threshold, bbox_area))
        if len(measurements) >= 2:
            selected_min_cells = max(zip(measurements, measurements[1:]),
                                     key=lambda pair: pair[0][1] / pair[1][1])[1][0]
        else:
            selected_min_cells = candidates[0]
    crop_support = density >= selected_min_cells
    if not crop_support.any():
        crop_support = occupied
    alignment_pixels = occupied & crop_support
    angle, alignment = ((0.0, {"status": "disabled", "line_count": 0, "rotation_deg": 0.0})
                        if args.no_align else dominant_wall_rotation(alignment_pixels))
    rotate_affine, rotated_size = rotation_canvas(source_image.shape[1], source_image.shape[0], angle)
    rotated = cv2.warpAffine(source_image, rotate_affine, rotated_size, flags=cv2.INTER_NEAREST, borderValue=128)
    rotated_occupied = cv2.warpAffine(crop_support.astype(np.uint8), rotate_affine, rotated_size,
                                      flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    ys, xs = np.where(rotated_occupied)
    margin = int(math.ceil(args.margin_m / meta["resolution"]))
    x0, x1 = max(0, int(xs.min()) - margin), min(rotated.shape[1] - 1, int(xs.max()) + margin)
    y0, y1 = max(0, int(ys.min()) - margin), min(rotated.shape[0] - 1, int(ys.max()) + margin)
    cropped = rotated[y0:y1 + 1, x0:x1 + 1]
    ratio = meta["resolution"] / args.web_resolution_m_per_px
    web_size = (max(1, round(cropped.shape[1] * ratio)), max(1, round(cropped.shape[0] * ratio)))
    web = cv2.resize(cropped, web_size, interpolation=cv2.INTER_NEAREST)
    web_occ, web_free, _ = classify(web, meta)
    dark, light = web.copy(), web.copy()
    dark[web_occ], dark[web_free] = 20, 245
    light[web_occ], light[web_free] = 235, 25
    Image.fromarray(dark).save(output / "map_dark.jpg", quality=95)
    Image.fromarray(light).save(output / "map_light.jpg", quality=95)
    source_h = source_image.shape[0]
    origin = meta["origin"]
    pb_to_source = np.array([[1 / meta["resolution"], 0, -origin[0] / meta["resolution"]],
                             [0, -1 / meta["resolution"], source_h - 1 + origin[1] / meta["resolution"]],
                             [0, 0, 1]], dtype=float)
    crop_matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=float)
    resize_matrix = np.diag([ratio, ratio, 1.0])
    pb_to_raster = resize_matrix @ crop_matrix @ matrix3(rotate_affine) @ pb_to_source
    web_w, web_h = web_size
    local_to_raster = np.array([[1 / args.web_resolution_m_per_px, 0, 0],
                                [0, -1 / args.web_resolution_m_per_px, web_h - 1], [0, 0, 1]], dtype=float)
    pb_to_local = np.linalg.inv(local_to_raster) @ pb_to_raster
    np.save(output / "map_transform_matrix.npy", local_to_raster)
    np.save(output / "pbstream_to_local_matrix.npy", pb_to_local)
    np.save(output / "pbstream_to_raster_matrix.npy", pb_to_raster)
    origin_pb = (np.linalg.inv(pb_to_local) @ np.array([0.0, 0.0, 1.0]))[:2]
    document = {"schema": "fliftbot.web_map.v1", "resolution_m_per_px": args.web_resolution_m_per_px,
                "image_size_px": [web_w, web_h], "axis_contract": "+X right, +Y up",
                "origin": {"type": "automatic_lower_left", "local_xy_m": [0.0, 0.0],
                           "pbstream_map_xy_m": origin_pb.tolist(), "image_unchanged": True},
                "automatic_alignment": alignment, "crop_rotated_source_px": [x0, y0, x1, y1],
                "distant_ray_filter": {"window_m": args.ray_filter_window_m,
                                        "minimum_occupied_cells": selected_min_cells,
                                        "selection": "automatic_bbox_contraction" if args.ray_filter_min_cells <= 0 else "explicit",
                                        "affects_pixels": False, "affects_crop_only": True},
                "occupancy": {k: meta[k] for k in ("negate", "occupied_thresh", "free_thresh")},
                "matrices": {"local_xy_metres_to_raster_uv_pixels": local_to_raster.tolist(),
                             "pbstream_map_xy_metres_to_local_xy_metres": pb_to_local.tolist(),
                             "pbstream_map_xy_metres_to_raster_uv_pixels": pb_to_raster.tolist()},
                "source_map": str(source)}
    write_json(output / "map_metadata.json", document)
    write_json(output / "edit_recipe.json", {"schema": "fliftbot.web_map_edit_recipe.v1",
               "rotation_deg": angle, "crop_rotated_source_px": [x0, y0, x1, y1],
               "ray_filter_window_m": args.ray_filter_window_m,
               "ray_filter_min_cells": selected_min_cells,
               "margin_m": args.margin_m, "resolution_m_per_px": args.web_resolution_m_per_px,
               "origin": "lower_left"})
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
