#!/usr/bin/env python3
"""Validate required map artifacts and all coordinate matrix invariants."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve()
    errors = []
    for name in ("map.pbstream", "map.pgm", "map.yaml", "map.png", "trajectory.json", "track.png"):
        if not (root / "map" / name).is_file():
            errors.append("missing map/" + name)
    web = root / "web_map"
    required_web = ("map_dark.jpg", "map_light.jpg", "map_metadata.json", "edit_recipe.json",
                    "map_transform_matrix.npy", "pbstream_to_local_matrix.npy",
                    "pbstream_to_raster_matrix.npy")
    for name in required_web:
        if not (web / name).is_file():
            errors.append("missing web_map/" + name)
    if not errors:
        local_to = np.load(web / "map_transform_matrix.npy")
        pb_to_local = np.load(web / "pbstream_to_local_matrix.npy")
        pb_to_raster = np.load(web / "pbstream_to_raster_matrix.npy")
        matrices = (local_to, pb_to_local, pb_to_raster)
        if any(matrix.shape != (3, 3) for matrix in matrices):
            errors.append("transform matrix is not 3x3")
        elif not all(np.isfinite(matrix).all() for matrix in matrices):
            errors.append("transform matrix contains non-finite values")
        else:
            if not np.allclose(pb_to_raster, local_to @ pb_to_local, atol=1e-7):
                errors.append("matrix composition mismatch")
            if abs(np.linalg.det(local_to)) < 1e-12 or abs(np.linalg.det(pb_to_local)) < 1e-12:
                errors.append("transform matrix is singular")
        metadata = json.loads((web / "map_metadata.json").read_text(encoding="utf-8"))
        actual_size = list(Image.open(web / "map_dark.jpg").size)
        if metadata.get("image_size_px") != actual_size:
            errors.append(f"metadata image size {metadata.get('image_size_px')} != actual {actual_size}")
        stored = metadata.get("matrices", {})
        for key, matrix in (("local_xy_metres_to_raster_uv_pixels", local_to),
                            ("pbstream_map_xy_metres_to_local_xy_metres", pb_to_local),
                            ("pbstream_map_xy_metres_to_raster_uv_pixels", pb_to_raster)):
            if key not in stored or not np.allclose(np.asarray(stored[key]), matrix, atol=1e-9):
                errors.append(f"metadata matrix mismatch: {key}")
    report = {"schema": "fliftbot.bundle_validation.v1",
              "status": "rejected" if errors else "accepted", "errors": errors}
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.json:
        args.json.resolve().write_text(text, encoding="utf-8")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
