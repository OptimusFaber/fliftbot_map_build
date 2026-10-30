#!/usr/bin/env python3
import argparse
import itertools, json, math
from pathlib import Path
from PIL import Image, ImageDraw

TARGETS = [7, 4, 18, 22]

def main(folder: Path, targets=None, suffix="7_4_18_22"):
    targets = list(targets or TARGETS)
    data = json.loads((folder / "marker_points_stored_ray_corrected.json").read_text())
    markers = {}
    for marker in data["markers"]:
        marker_id = marker.get("marker_id", "")
        try:
            label = int(str(marker_id).split("_")[-1])
        except ValueError:
            continue
        if label in targets:
            markers[label] = marker["point_xyz_m"]
    distances = {}
    for a, b in itertools.combinations(targets, 2):
        if a in markers and b in markers:
            pa, pb = markers[a], markers[b]
            distances[f"{a}-{b}"] = math.dist(pa[:2], pb[:2])
    (folder / f"marker_distances_{suffix}.json").write_text(json.dumps({
        "labels": targets, "points_xy_m": {str(k): markers[k][:2] for k in sorted(markers)},
        "distances_m": distances,
    }, indent=2) + "\n")

    # The map image is alongside the marker output in the run's map directory.
    map_dir = folder.parent / "map"
    meta = {}
    for line in (map_dir / "map.yaml").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try: meta[k.strip()] = json.loads(v.strip())
            except Exception: meta[k.strip()] = v.strip().strip("'\"")
    image = Image.open(map_dir / str(meta["image"])).convert("RGB")
    draw = ImageDraw.Draw(image)
    res, origin = float(meta["resolution"]), meta["origin"]
    def pixel(p):
        return ((p[0] - origin[0]) / res, image.height - (p[1] - origin[1]) / res)
    pts = {k: pixel(v) for k, v in markers.items()}
    colors = {label: color for label, color in zip(targets, ["red", "blue", "green", "orange"])}
    for a, b in itertools.combinations(sorted(pts), 2):
        draw.line([pts[a], pts[b]], fill="magenta", width=3)
        mx, my = (pts[a][0] + pts[b][0]) / 2, (pts[a][1] + pts[b][1]) / 2
        key = f"{a}-{b}" if f"{a}-{b}" in distances else f"{b}-{a}"
        draw.text((mx, my), f"{a}-{b}: {distances[key]:.2f} m", fill="black", stroke_width=2, stroke_fill="white")
    for label, point in pts.items():
        x, y = point
        r = 8
        draw.ellipse((x-r, y-r, x+r, y+r), fill=colors[label], outline="white", width=2)
        draw.text((x+10, y-10), str(label), fill=colors[label], stroke_width=2, stroke_fill="white")
    image.save(folder / f"marker_distances_{suffix}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate pairwise XY distances between selected marker IDs")
    parser.add_argument("folder", type=Path, help="marker output directory alongside map/")
    parser.add_argument("targets", help="comma-separated numeric suffixes of marker IDs, e.g. 7,4,18,22")
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",")]
    main(args.folder, targets, "_".join(map(str, targets)))
