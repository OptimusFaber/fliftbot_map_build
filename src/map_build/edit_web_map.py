#!/usr/bin/env python3
"""Interactively create a web map or edit an existing one."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from build_web_map import classify, read_map_yaml


def matrix3(affine: np.ndarray) -> np.ndarray:
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def render_smooth_web_map(image: np.ndarray, meta: dict, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Render antialiased display images without changing map geometry."""
    occupied, free, _ = classify(image, meta)
    occupied_alpha = cv2.resize(occupied.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    free_alpha = cv2.resize(free.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    known = np.clip(occupied_alpha + free_alpha, 0.0, 1.0)
    unknown_alpha = 1.0 - known
    # Neutral unknown space, high contrast walls, and softly antialiased edges.
    dark = occupied_alpha * 22.0 + free_alpha * 247.0 + unknown_alpha * 148.0
    light = occupied_alpha * 238.0 + free_alpha * 24.0 + unknown_alpha * 102.0
    return np.clip(dark, 0, 255).astype(np.uint8), np.clip(light, 0, 255).astype(np.uint8)


def orthogonalize_axes(x_vector: np.ndarray, y_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest right-handed screen-space X/Y pair (Y is 90 degrees from X)."""
    x_length, y_length = np.linalg.norm(x_vector), np.linalg.norm(y_vector)
    if x_length < 1e-9 or y_length < 1e-9:
        raise ValueError("axis direction points must differ from origin")
    x_unit = x_vector / x_length
    # In image coordinates (+V down), right-handed +Y is (-90 degrees) from +X.
    x_suggested_by_y = np.array([-y_vector[1], y_vector[0]], dtype=float) / y_length
    if float(np.dot(x_unit, x_suggested_by_y)) > 0.0:
        combined = x_unit + x_suggested_by_y
        x_unit = combined / np.linalg.norm(combined)
    # If the hints have opposite handedness, X remains authoritative and Y is
    # projected onto its valid perpendicular instead of blocking Enter.
    y_unit = np.array([x_unit[1], -x_unit[0]], dtype=float)
    return x_unit, y_unit


def select_polygon_and_origin(
        image: np.ndarray) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    """Select crop, origin and positive axes on an aspect-safe preview."""
    max_width, max_height = 1400, 850
    scale = min(1.0, max_width / image.shape[1], max_height / image.shape[0])
    preview_size = (max(1, round(image.shape[1] * scale)),
                    max(1, round(image.shape[0] * scale)))
    base = cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    points: list[tuple[int, int]] = []
    cursor = [0, 0]
    phase = ["polygon"]
    origin: list[tuple[int, int]] = []
    axes: list[tuple[int, int]] = []
    custom_axes = [False]
    window = "Web map editor"

    def mouse(event: int, x: int, y: int, flags: int, _data: object) -> None:
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_ALTKEY:
                if phase[0] == "axes" and axes:
                    axes.pop()
                    custom_axes[0] = True
                elif phase[0] == "origin" and origin:
                    origin.clear()
                elif phase[0] == "origin":
                    phase[0] = "polygon"
                    if points:
                        points.pop()
                elif points:
                    points.pop()
            elif not flags & cv2.EVENT_FLAG_SHIFTKEY:
                return
            elif phase[0] == "polygon":
                points.append((x, y))
            elif phase[0] == "origin":
                contour = np.asarray(points, dtype=np.int32)
                if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
                    origin[:] = [(x, y)]
            elif phase[0] == "axes" and origin:
                point = (x, y)
                if not custom_axes[0] or len(axes) >= 2:
                    axes[:] = [point]
                    custom_axes[0] = True
                else:
                    axes.append(point)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    while True:
        canvas = base.copy()
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, (0, 220, 255), 2,
                          cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 4, (0, 255, 255), -1, cv2.LINE_AA)
        if phase[0] == "polygon" and points:
            cv2.line(canvas, points[-1], tuple(cursor), (0, 180, 255), 1, cv2.LINE_AA)
        if phase[0] in ("origin", "axes"):
            cv2.polylines(canvas, [np.asarray(points, np.int32)], True, (0, 220, 255), 2,
                          cv2.LINE_AA)
            if origin:
                x, y = origin[0]
                cv2.drawMarker(canvas, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            if phase[0] == "axes":
                if len(axes) == 1:
                    cv2.arrowedLine(canvas, origin[0], axes[0], (60, 80, 255), 2,
                                    cv2.LINE_AA, tipLength=0.12)
                    cv2.putText(canvas, "+X", axes[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (60, 80, 255), 2, cv2.LINE_AA)
                if len(axes) >= 2:
                    x_hint = np.asarray(axes[0], dtype=float) - origin[0]
                    y_hint = np.asarray(axes[1], dtype=float) - origin[0]
                    x_unit, y_unit = orthogonalize_axes(x_hint, y_hint)
                    display_length = max(50.0, min(np.linalg.norm(x_hint), np.linalg.norm(y_hint)))
                    shown_x = tuple(np.rint(np.asarray(origin[0]) + x_unit * display_length).astype(int))
                    shown_y = tuple(np.rint(np.asarray(origin[0]) + y_unit * display_length).astype(int))
                    cv2.arrowedLine(canvas, origin[0], shown_x, (60, 80, 255), 2,
                                    cv2.LINE_AA, tipLength=0.12)
                    cv2.putText(canvas, "+X", shown_x, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (60, 80, 255), 2, cv2.LINE_AA)
                    cv2.arrowedLine(canvas, origin[0], shown_y, (80, 230, 80), 2,
                                    cv2.LINE_AA, tipLength=0.12)
                    cv2.putText(canvas, "+Y", shown_y, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (80, 230, 80), 2, cv2.LINE_AA)
                message = "AXES: hints are fitted to exact 90 deg; Shift+LMB twice = replace; Enter = save"
            else:
                message = "ORIGIN: Shift+LMB = (0,0), Enter = axes, Backspace = polygon"
        else:
            message = "CROP: Shift+LMB = point, Alt+LMB/U = undo, Enter = close, Esc = cancel"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 30), (30, 30, 30), -1)
        cv2.putText(canvas, message, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("map editing cancelled")
        if phase[0] == "polygon" and key in (8, 127, ord("u"), ord("U")) and points:
            points.pop()
        elif key in (10, 13):
            if phase[0] == "polygon" and len(points) >= 3:
                phase[0] = "origin"
            elif phase[0] == "origin" and origin:
                edges = [(points[(index + 1) % len(points)][0] - point[0],
                          points[(index + 1) % len(points)][1] - point[1])
                         for index, point in enumerate(points)]
                dx, dy = max(edges, key=lambda edge: edge[0] ** 2 + edge[1] ** 2)
                length = max(1.0, math.hypot(dx, dy))
                arrow = min(160.0, max(70.0, length * 0.35))
                ux, uy = dx / length, dy / length
                ox, oy = origin[0]
                axes[:] = [(round(ox + ux * arrow), round(oy + uy * arrow)),
                           (round(ox + uy * arrow), round(oy - ux * arrow))]
                custom_axes[0] = False
                phase[0] = "axes"
            elif phase[0] == "axes" and len(axes) == 2:
                x_vector = (axes[0][0] - origin[0][0], axes[0][1] - origin[0][1])
                y_vector = (axes[1][0] - origin[0][0], axes[1][1] - origin[0][1])
                if math.hypot(*x_vector) >= 10 and math.hypot(*y_vector) >= 10:
                    break
        elif phase[0] == "origin" and key in (8, 127):
            origin.clear()
            phase[0] = "polygon"
        elif phase[0] == "axes" and key in (8, 127):
            axes.clear()
            phase[0] = "origin"
    cv2.destroyWindow(window)

    def source_point(point: tuple[int, int]) -> list[int]:
        return [min(image.shape[1] - 1, max(0, round(point[0] / scale))),
                min(image.shape[0] - 1, max(0, round(point[1] / scale)))]

    return ([source_point(point) for point in points], source_point(origin[0]),
            source_point(axes[0]), source_point(axes[1]))


def create_manual_web_map(map_dir: Path, output: Path, resolution: float) -> None:
    """Create a complete web-map bundle from one interactive selection."""
    source = map_dir.resolve()
    output = output.resolve()
    if resolution <= 0:
        raise ValueError("web resolution must be positive")
    meta = read_map_yaml(source / "map.yaml")
    source_image = np.asarray(Image.open(source / meta["image"]).convert("L"))
    polygon, origin_source, x_axis_source, y_axis_source = select_polygon_and_origin(source_image)
    polygon_array = np.asarray(polygon, dtype=np.int32)
    x_hint = np.asarray(x_axis_source, dtype=float) - origin_source
    y_hint = np.asarray(y_axis_source, dtype=float) - origin_source
    x_vector, y_vector = orthogonalize_axes(x_hint, y_hint)
    rotation_deg = math.degrees(math.atan2(x_vector[1], x_vector[0]))
    rotate_affine, rotated_size = rotation_canvas(source_image.shape[1], source_image.shape[0],
                                                   rotation_deg)
    rotated_source = cv2.warpAffine(source_image, rotate_affine, rotated_size,
                                    flags=cv2.INTER_NEAREST, borderValue=128)
    rotated_polygon = cv2.transform(polygon_array.reshape(1, -1, 2).astype(np.float64),
                                    rotate_affine).reshape(-1, 2)
    rotated_polygon_int = np.rint(rotated_polygon).astype(np.int32)
    x, y, width, height = cv2.boundingRect(rotated_polygon_int)
    x1, y1 = x + width - 1, y + height - 1
    mask = np.zeros((rotated_size[1], rotated_size[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [rotated_polygon_int], 255)
    cropped = rotated_source[y:y1 + 1, x:x1 + 1].copy()
    cropped_mask = mask[y:y1 + 1, x:x1 + 1]
    if len(polygon) != 4:
        cropped[cropped_mask == 0] = 128
    ratio = meta["resolution"] / resolution
    web_size = (max(1, round(cropped.shape[1] * ratio)),
                max(1, round(cropped.shape[0] * ratio)))
    dark, light = render_smooth_web_map(cropped, meta, web_size)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(dark).save(output / "map_dark.jpg", quality=96, optimize=True)
    Image.fromarray(light).save(output / "map_light.jpg", quality=96, optimize=True)

    source_h = source_image.shape[0]
    map_origin = meta["origin"]
    pb_to_source = np.array([
        [1 / meta["resolution"], 0, -map_origin[0] / meta["resolution"]],
        [0, -1 / meta["resolution"], source_h - 1 + map_origin[1] / meta["resolution"]],
        [0, 0, 1],
    ], dtype=float)
    crop_resize = np.diag([ratio, ratio, 1.0]) @ np.array(
        [[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=float)
    pb_to_raster = crop_resize @ matrix3(rotate_affine) @ pb_to_source
    origin_pb = (np.linalg.inv(pb_to_source) @ np.array([*origin_source, 1.0]))[:2]
    effective_x_source = np.asarray(origin_source, dtype=float) + x_vector
    x_axis_pb = (np.linalg.inv(pb_to_source) @ np.array([*effective_x_source, 1.0]))[:2]
    axis_theta = math.atan2(x_axis_pb[1] - origin_pb[1], x_axis_pb[0] - origin_pb[0])
    cosine, sine = math.cos(axis_theta), math.sin(axis_theta)
    pb_to_local = np.array([[cosine, sine, -(cosine * origin_pb[0] + sine * origin_pb[1])],
                            [-sine, cosine, sine * origin_pb[0] - cosine * origin_pb[1]],
                            [0, 0, 1]], dtype=float)
    local_to_raster = pb_to_raster @ np.linalg.inv(pb_to_local)
    np.save(output / "pbstream_to_raster_matrix.npy", pb_to_raster)
    np.save(output / "pbstream_to_local_matrix.npy", pb_to_local)
    np.save(output / "map_transform_matrix.npy", local_to_raster)
    document = {
        "schema": "fliftbot.web_map.v1",
        "resolution_m_per_px": resolution,
        "image_size_px": [web_size[0], web_size[1]],
        "axis_contract": "+X right, +Y up",
        "origin": {"type": "manual_click", "local_xy_m": [0.0, 0.0],
                   "pbstream_map_xy_m": origin_pb.tolist(),
                   "origin_map_yaw_deg": math.degrees(axis_theta),
                   "image_unchanged": False},
        "manual_crop_source_px": polygon,
        "manual_axes_source_px": {"origin": origin_source, "+x": x_axis_source,
                                  "+y_hint": y_axis_source,
                                  "effective_orthogonal_x_unit": x_vector.tolist(),
                                  "effective_orthogonal_y_unit": y_vector.tolist()},
        "manual_rotation_deg": rotation_deg,
        "crop_mode": "aligned_quadrilateral" if len(polygon) == 4 else "aligned_polygon_mask",
        "rendering": {"style": "smooth_semantic_grayscale_v1",
                      "edge_antialiasing": True, "geometry_changed": False},
        "occupancy": {key: meta[key] for key in ("negate", "occupied_thresh", "free_thresh")},
        "matrices": {
            "local_xy_metres_to_raster_uv_pixels": local_to_raster.tolist(),
            "pbstream_map_xy_metres_to_local_xy_metres": pb_to_local.tolist(),
            "pbstream_map_xy_metres_to_raster_uv_pixels": pb_to_raster.tolist(),
        },
        "source_map": str(source),
    }
    metadata_path = output / "map_metadata.json"
    metadata_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    recipe = {"schema": "fliftbot.web_map_manual_edit.v2", "source_map": str(source),
              "crop_polygon_source_px": polygon, "origin_source_px": origin_source,
              "x_axis_source_px": x_axis_source, "y_axis_hint_source_px": y_axis_source,
              "effective_orthogonal_x_unit": x_vector.tolist(),
              "effective_orthogonal_y_unit": y_vector.tolist(),
              "rotation_deg": rotation_deg,
              "origin_pbstream_map_xy_m": origin_pb.tolist(),
              "resolution_m_per_px": resolution}
    (output / "edit_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "origin_reference.json").write_text(json.dumps({
        "schema": "fliftbot.map_origin_reference.v1", "frame": "cartographer_map",
        "origin_name": "manual_click", "origin_map_xy_m": origin_pb.tolist(),
        "origin_map_yaw_rad": axis_theta, "origin_map_yaw_deg": math.degrees(axis_theta),
        "axis_contract": "+X right, +Y up", "source": str(source / "map.pbstream"),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


def rotation_canvas(width: int, height: int, angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    affine = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    corners = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1],
                        [width - 1, height - 1, 1]], dtype=float)
    points = (affine @ corners.T).T
    low, high = points.min(axis=0), points.max(axis=0)
    affine[:, 2] -= low
    return affine, (math.ceil(high[0] - low[0] + 1), math.ceil(high[1] - low[1] + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--map-dir", type=Path,
                              help="original Cartographer map directory (new one-step GUI)")
    source_group.add_argument("--web-map", type=Path,
                              help="existing web-map directory (legacy edit mode)")
    parser.add_argument("--output", type=Path,
                        help="output web-map directory; required with --map-dir")
    parser.add_argument("--web-resolution-m-per-px", type=float, default=0.01)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--crop-px", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--gui", action="store_true", help="select crop rectangle interactively")
    args = parser.parse_args()
    if args.map_dir is not None:
        if args.output is None:
            parser.error("--output is required with --map-dir")
        if not args.gui:
            parser.error("--gui is required with --map-dir")
        create_manual_web_map(args.map_dir, args.output, args.web_resolution_m_per_px)
        return 0
    root = args.web_map.resolve()
    metadata_path = root / "map_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dark = np.asarray(Image.open(root / "map_dark.jpg").convert("L"))
    light = np.asarray(Image.open(root / "map_light.jpg").convert("L"))
    affine, size = rotation_canvas(dark.shape[1], dark.shape[0], args.rotation_deg)
    rotated_dark = cv2.warpAffine(dark, affine, size, flags=cv2.INTER_NEAREST, borderValue=128)
    rotated_light = cv2.warpAffine(light, affine, size, flags=cv2.INTER_NEAREST, borderValue=128)
    crop = args.crop_px
    if args.gui:
        preview = cv2.resize(rotated_dark, None, fx=min(1.0, 1400 / rotated_dark.shape[1]),
                             fy=min(1.0, 900 / rotated_dark.shape[0]))
        scale = preview.shape[1] / rotated_dark.shape[1]
        x, y, width, height = cv2.selectROI("Select web-map crop; Enter accepts", preview,
                                            showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        if width <= 0 or height <= 0:
            raise RuntimeError("crop selection cancelled")
        crop = [round(x / scale), round(y / scale), round((x + width - 1) / scale),
                round((y + height - 1) / scale)]
    if crop is None:
        crop = [0, 0, size[0] - 1, size[1] - 1]
    x0, y0, x1, y1 = crop
    if not (0 <= x0 <= x1 < size[0] and 0 <= y0 <= y1 < size[1]):
        raise ValueError(f"crop outside rotated image {size}: {crop}")
    dark_result = rotated_dark[y0:y1 + 1, x0:x1 + 1]
    light_result = rotated_light[y0:y1 + 1, x0:x1 + 1]
    Image.fromarray(dark_result).save(root / "map_dark.jpg", quality=95)
    Image.fromarray(light_result).save(root / "map_light.jpg", quality=95)
    transform = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=float) @ np.vstack(
        [affine, [0, 0, 1]])
    pb_to_raster = transform @ np.load(root / "pbstream_to_raster_matrix.npy")
    resolution = float(metadata["resolution_m_per_px"])
    height = dark_result.shape[0]
    local_to_raster = np.array([[1 / resolution, 0, 0], [0, -1 / resolution, height - 1],
                                [0, 0, 1]], dtype=float)
    pb_to_local = np.linalg.inv(local_to_raster) @ pb_to_raster
    np.save(root / "pbstream_to_raster_matrix.npy", pb_to_raster)
    np.save(root / "map_transform_matrix.npy", local_to_raster)
    np.save(root / "pbstream_to_local_matrix.npy", pb_to_local)
    origin_pb = (np.linalg.inv(pb_to_local) @ [0, 0, 1])[:2]
    metadata["image_size_px"] = [dark_result.shape[1], dark_result.shape[0]]
    metadata["origin"] = {"type": "manual_edit_lower_left", "local_xy_m": [0.0, 0.0],
                          "pbstream_map_xy_m": origin_pb.tolist(), "image_unchanged": False}
    metadata["matrices"] = {"local_xy_metres_to_raster_uv_pixels": local_to_raster.tolist(),
                            "pbstream_map_xy_metres_to_local_xy_metres": pb_to_local.tolist(),
                            "pbstream_map_xy_metres_to_raster_uv_pixels": pb_to_raster.tolist()}
    metadata["manual_edit"] = {"rotation_deg": args.rotation_deg, "crop_rotated_px": crop}
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    recipe = {"schema": "fliftbot.web_map_manual_edit.v1", "rotation_deg": args.rotation_deg,
              "crop_rotated_px": crop}
    (root / "edit_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
