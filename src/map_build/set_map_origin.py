#!/usr/bin/env python3
"""Change the logical map origin while keeping every image pixel fixed."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-map", type=Path, required=True)
    origin_group = parser.add_mutually_exclusive_group(required=True)
    origin_group.add_argument("--origin-map-xy", type=float, nargs=2, metavar=("X", "Y"))
    origin_group.add_argument("--marker-id", help="marker_id from markers_map.json")
    parser.add_argument("--markers", type=Path, help="defaults to sibling markers/markers_map.json")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--origin-name", default="manual")
    args = parser.parse_args()
    root = args.web_map.resolve()
    metadata_path = root / "map_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.marker_id:
        markers_path = args.markers.resolve() if args.markers else root.parent / "markers/markers_map.json"
        markers = json.loads(markers_path.read_text(encoding="utf-8")).get("markers", [])
        matches = [marker for marker in markers if str(marker.get("marker_id")) == args.marker_id]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one marker_id={args.marker_id!r}, found {len(matches)}")
        x, y = map(float, matches[0]["pbstream_map_xyz_m"][:2])
        if args.origin_name == "manual":
            args.origin_name = args.marker_id
    else:
        x, y = args.origin_map_xy
    theta = math.radians(args.yaw_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    pb_to_local = np.array([[cosine, sine, -(cosine * x + sine * y)],
                            [-sine, cosine, sine * x - cosine * y], [0.0, 0.0, 1.0]])
    # Rendered pixels stay fixed, therefore this matrix is the invariant.
    pb_to_raster = np.load(root / "pbstream_to_raster_matrix.npy")
    local_to_raster = pb_to_raster @ np.linalg.inv(pb_to_local)
    np.save(root / "pbstream_to_local_matrix.npy", pb_to_local)
    np.save(root / "map_transform_matrix.npy", local_to_raster)
    metadata["origin"] = {"type": "manual", "name": args.origin_name,
                          "local_xy_m": [0.0, 0.0], "pbstream_map_xy_m": [x, y],
                          "origin_map_yaw_deg": args.yaw_deg, "image_unchanged": True}
    metadata["matrices"]["local_xy_metres_to_raster_uv_pixels"] = local_to_raster.tolist()
    metadata["matrices"]["pbstream_map_xy_metres_to_local_xy_metres"] = pb_to_local.tolist()
    metadata["matrices"]["pbstream_map_xy_metres_to_raster_uv_pixels"] = pb_to_raster.tolist()
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reference = {"schema": "fliftbot.map_origin_reference.v1", "frame": "cartographer_map",
                 "origin_name": args.origin_name, "origin_map_xy_m": [x, y],
                 "origin_map_yaw_rad": theta, "origin_map_yaw_deg": args.yaw_deg,
                 "axis_contract": "+X right, +Y up", "source": "map.pbstream"}
    (root / "origin_reference.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
