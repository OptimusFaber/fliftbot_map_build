#!/usr/bin/env python3
"""Create a canonical recording for an explicit Unix-time interval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--start-unix", type=float, required=True)
    parser.add_argument("--end-unix", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.end_unix <= args.start_unix:
        parser.error("--end-unix must be greater than --start-unix")

    source, output = args.recording.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    scans = sorted((source / "horizon").glob("*.npz"))
    selected = []
    for path in scans:
        with np.load(path) as data:
            stamp = float(data["stamp"])
        if args.start_unix <= stamp <= args.end_unix:
            selected.append((path, stamp))
    if not selected:
        raise RuntimeError("no LiDAR scans in requested interval")

    (output / "horizon").mkdir(parents=True)
    for path, _ in selected:
        link_or_copy(path, output / "horizon" / path.name)
    rows = [json.loads(line) for line in (source / "encoder.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if args.start_unix <= float(row["enc_unix"]) <= args.end_unix]
    (output / "encoder.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )
    for name in ("camera.config", "lidar.config", "liftbot.config", "camera_intrinsics.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    report = {
        "schema": "fliftbot.canonical_interval.v1", "source": str(source),
        "requested_start_unix": args.start_unix, "requested_end_unix": args.end_unix,
        "first_scan_unix": selected[0][1], "last_scan_unix": selected[-1][1],
        "duration_s": selected[-1][1] - selected[0][1],
        "scan_count": len(selected), "encoder_count": len(rows),
    }
    (output / "interval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
