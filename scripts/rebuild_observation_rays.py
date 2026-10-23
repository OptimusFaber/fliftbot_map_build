#!/usr/bin/env python3
"""Rebuild stored marker rays from exported pixel centers and CameraModel."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fliftbot_navigation.camera_model import CameraModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-json", required=True, type=Path)
    parser.add_argument("--angle-x", required=True, type=float)
    parser.add_argument("--angle-y", required=True, type=float)
    parser.add_argument("--mount-yaw", type=float, default=np.pi / 2.0)
    parser.add_argument("--camera-x", required=True, type=float)
    parser.add_argument("--camera-y", required=True, type=float)
    parser.add_argument("--camera-z", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    model = CameraModel(
        offsets=[args.camera_x, args.camera_y, args.camera_z],
        tilts_rad=[args.angle_x, args.angle_y, args.mount_yaw],
        camera_json_path=str(args.camera_json),
    )

    items = []
    centers = []
    for frame in payload["data"]:
        for item in frame.values():
            center = item.get("center")
            if center is not None:
                items.append(item)
                centers.append(center)

    _, rays = model.pixels_to_rays(
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
    )
    for item, ray in zip(items, rays):
        item["ray"] = [float(value) for value in ray]

    origin, _ = model.pixels_to_rays(
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([[0.0, 0.0]], dtype=np.float64),
    )
    payload["camera_origin"] = [float(value) for value in origin]
    payload["ray_source"] = (
        "CameraModel.pixels_to_rays rebuilt from exported detection pixel centers; "
        f"tilts_rad=[{args.angle_x}, {args.angle_y}, {args.mount_yaw}]"
    )
    payload["ray_rebuild"] = {
        "source_observations": str(args.input.resolve()),
        "camera_json": str(args.camera_json.resolve()),
        "camera_position_m": [args.camera_x, args.camera_y, args.camera_z],
        "camera_model_tilts_rad": [args.angle_x, args.angle_y, args.mount_yaw],
        "rebuilt_rays": len(items),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Rebuilt {len(items)} rays: {args.output}")


if __name__ == "__main__":
    main()
