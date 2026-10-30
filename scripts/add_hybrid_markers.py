#!/usr/bin/env python3
"""Detect and localize ceiling markers from both videos of a hybrid map."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAP_BUILD = ROOT / "src" / "map_build"
sys.path.insert(0, str(MAP_BUILD))
from marker_profiles import add_marker_options, forward_marker_options  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect(video: Path, timestamps: Path, labels: Path | None, camera_json: Path,
           camera_position: Path, work: Path) -> Path:
    observations = work / "observations.json"
    if observations.is_file():
        print(f"Reusing {observations}")
        return observations
    work.mkdir(parents=True, exist_ok=True)
    video_link = work / f"video{video.suffix}"
    timestamps_link = work / "frame_timestamps.csv"
    if not video_link.exists():
        video_link.symlink_to(video.resolve())
    if not timestamps_link.exists():
        timestamps_link.symlink_to(timestamps.resolve())
    command = [
        sys.executable, str(SRC / "auto_map/get_video_data.py"),
        "--video", str(work), "--camera-json", str(camera_json),
        "--camera-position", str(camera_position),
    ]
    if labels is not None:
        command += ["--labels", str(labels)]
    subprocess.run(command, cwd=SRC / "auto_map", check=True)
    if not observations.is_file():
        raise RuntimeError(f"video detector did not produce {observations}")
    return observations


def frames(document: dict, start: float | None = None, end: float | None = None,
           shift: float = 0.0, track_offset: int = 0) -> list[tuple[float, dict]]:
    times = document.get("frame_timestamps")
    data = document.get("data")
    if not isinstance(times, list) or not isinstance(data, list) or len(times) != len(data):
        raise ValueError("observations must have equal-length data and frame_timestamps arrays")
    result = []
    for timestamp, frame in zip(times, data):
        timestamp = float(timestamp)
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        remapped = {}
        if isinstance(frame, dict):
            for track_id, item in frame.items():
                try:
                    new_id = str(int(track_id) + track_offset)
                except ValueError:
                    new_id = f"{track_offset}:{track_id}"
                remapped[new_id] = item
        result.append((timestamp + shift, remapped))
    return result


def maximum_numeric_track(document: dict) -> int:
    maximum = 0
    for frame in document.get("data", []):
        if not isinstance(frame, dict):
            continue
        for track_id in frame:
            try:
                maximum = max(maximum, int(track_id))
            except ValueError:
                pass
    return maximum


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-run", type=Path, required=True)
    p.add_argument("--hybrid-report", type=Path, required=True)
    p.add_argument("--first-video", type=Path, required=True)
    p.add_argument("--first-timestamps", type=Path, required=True)
    p.add_argument("--second-video", type=Path, required=True)
    p.add_argument("--second-timestamps", type=Path, required=True)
    p.add_argument("--labels2coords", type=Path,
                   help="optional decoder codebook; omit to use tracker-based marker numbering")
    add_marker_options(p)
    p.add_argument("--group-by-code", action="store_true")
    args = p.parse_args()

    run = args.map_run.resolve()
    report = read(args.hybrid_report.resolve())
    if report.get("schema") != "fliftbot.canonical_hybrid.v1":
        raise ValueError("--hybrid-report is not fliftbot.canonical_hybrid.v1")
    run_input = run / "input"
    camera_json = run_input / "camera.json"
    camera_position = run_input / "camera_position.txt"
    lidar_config = run_input / "lidar.markers.config"
    trajectory = run / "map/trajectory.json"
    for required in (camera_json, camera_position, lidar_config, trajectory, run / "map/map.yaml"):
        if not required.is_file():
            raise FileNotFoundError(required)

    work = run_input / "hybrid_marker_videos"
    labels = args.labels2coords.resolve() if args.labels2coords else None
    first_path = detect(args.first_video.resolve(), args.first_timestamps.resolve(),
                        labels, camera_json, camera_position, work / "first")
    second_path = detect(args.second_video.resolve(), args.second_timestamps.resolve(),
                         labels, camera_json, camera_position, work / "second")
    first, second = read(first_path), read(second_path)
    cutoff = float(report["cutoff_unix"])
    first_end = cutoff + float(report["first_tail_minutes"]) * 60.0
    shift = float(report["second_time_shift_s"])
    track_offset = maximum_numeric_track(first) + 1
    merged_frames = frames(first, cutoff, first_end)
    merged_frames += frames(second, shift=shift, track_offset=track_offset)
    merged_frames.sort(key=lambda item: item[0])
    if not merged_frames:
        raise RuntimeError("no video frames overlap the hybrid recording")

    merged = {key: value for key, value in first.items()
              if key not in {"video_path", "data", "frame_timestamps"}}
    merged.update({
        "schema": "fliftbot.hybrid_observations.v1",
        "video_path": [str(args.first_video.resolve()), str(args.second_video.resolve())],
        "data": [frame for _, frame in merged_frames],
        "frame_timestamps": [timestamp for timestamp, _ in merged_frames],
        "hybrid_sources": {
            "first": {"observations": str(first_path), "cutoff_unix": cutoff,
                      "end_unix": first_end, "track_offset": 0},
            "second": {"observations": str(second_path), "time_shift_s": shift,
                       "track_offset": track_offset},
        },
    })
    merged_path = run / "observations_hybrid.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    marker_dir = run / "markers"
    command = [sys.executable, str(MAP_BUILD / "process_markers.py"),
               "--run-dir", str(run), "--observations", str(merged_path)]
    forward_marker_options(command, args)
    if args.group_by_code:
        command.append("--group-by-code")
    subprocess.run(command, check=True)
    print(json.dumps({
        "map_run": str(run), "merged_observations": str(merged_path),
        "frames": len(merged_frames), "first_cutoff_unix": cutoff,
        "first_end_unix": first_end, "second_time_shift_s": shift,
        "second_track_offset": track_offset, "markers": str(marker_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
