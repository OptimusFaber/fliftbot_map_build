#!/usr/bin/env python3
"""Build marker overlays for every map in a geometry sweep."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import subprocess
from pathlib import Path


BASE = Path("/home/optimus/ros-projects/ros-data/results/real/20261021/last18m_range16_geometry_sweep")
MAPS = BASE / "maps"
OBS = BASE / "observations_angle_xy.json"
CAMERA_JSON = Path("/home/optimus/ros-projects/camera_backup.json")
LOCALIZER = Path("/home/optimus/ros-projects/fliftbot_map_build/src/fliftbot_mapping/evaluation/ceiling_marker_localization_stored_ray_corrected.py")
ANGLE_X = 0.0026357628829743046
ANGLE_Y = -0.010379049578707875


def make_sidecar(source: Path, target: Path, z_cm: float) -> None:
    lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    x, y, _ = map(float, lines[0].split())
    target.write_text(
        f"{x:.10g} {y:.10g} {z_cm:.1f}\n"
        f"{-0.0026357628829743046} {-0.010379049578707875} {-1.57079632679}\n"
    )


def build_one(run: Path) -> dict:
    results = []
    for z_cm in (124.8, 126.4):
        z_name = int(round(z_cm * 10))
        out = run / f"markers_angle_xy_z{z_name}"
        out.mkdir(parents=True, exist_ok=True)
        sidecar = out / "camera_position.txt"
        make_sidecar(run / "input" / "camera_position.txt", sidecar, z_cm)
        command = [
            "python3", str(LOCALIZER),
            "--observations", str(OBS),
            "--trajectory", str(run / "map" / "trajectory.json"),
            "--map-dir", str(run / "map"),
            "--camera-position", str(sidecar),
            "--camera-json", str(CAMERA_JSON),
            "--lidar-config", str(run / "input" / "lidar.markers.config"),
            "--out-dir", str(out),
            "--time-offset-s", "0.005",
            "--per-track-height",
            "--merge-radius-m", "0.60",
            "--merge-linkage", "connected",
            "--min-segment-observations", "14",
        ]
        log = out / "build.log"
        with log.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
        report = out / "marker_points_stored_ray_corrected.json"
        count = None
        if report.is_file():
            count = len(json.loads(report.read_text())["markers"])
        results.append({"height_m": z_cm / 100.0, "returncode": completed.returncode, "markers": count, "out": str(out)})
    return {"run": run.name, "results": results}


def main() -> None:
    runs = sorted(path for path in MAPS.iterdir() if path.is_dir())
    summary = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(build_one, run) for run in runs]
        for future in as_completed(futures):
            item = future.result()
            summary.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    summary.sort(key=lambda item: item["run"])
    (BASE / "marker_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    failed = [item for item in summary for result in item["results"] if result["returncode"] != 0 or result["markers"] != 43]
    if failed:
        raise SystemExit(f"Unexpected marker result in {len(failed)} run(s)")


if __name__ == "__main__":
    main()
