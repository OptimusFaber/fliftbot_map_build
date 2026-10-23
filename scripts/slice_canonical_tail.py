#!/usr/bin/env python3
"""Create a canonical recording containing the last N minutes of a run."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--minutes", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src, out = a.recording.resolve(), a.output.resolve()
    if a.minutes <= 0:
        p.error("--minutes must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output is not empty: {out}")
    scans = sorted((src / "horizon").glob("*.npz"))
    if not scans:
        raise FileNotFoundError("no scans found")
    stamps = np.asarray([float(np.load(f)["stamp"]) for f in scans])
    cutoff = float(stamps[-1] - a.minutes * 60.0)
    first = int(np.searchsorted(stamps, cutoff, side="left"))
    scans = scans[first:]
    (out / "horizon").mkdir(parents=True)
    for f in scans:
        link_or_copy(f, out / "horizon" / f.name)
    rows = [json.loads(line) for line in (src / "encoder.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if float(r["enc_unix"]) >= cutoff]
    (out / "encoder.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    for name in ("camera.config", "lidar.config", "liftbot.config", "camera_intrinsics.json"):
        f = src / name
        if f.is_file():
            shutil.copy2(f, out / name)
    report = {
        "schema": "fliftbot.canonical_tail.v1",
        "source": str(src), "requested_minutes": a.minutes,
        "cutoff_unix": cutoff, "first_scan_unix": float(stamps[first]),
        "last_scan_unix": float(stamps[-1]), "duration_s": float(stamps[-1] - stamps[first]),
        "scan_count": len(scans), "encoder_count": len(rows),
    }
    (out / "tail_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
