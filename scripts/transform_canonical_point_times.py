#!/usr/bin/env python3
"""Copy a canonical recording with centered or reversed within-scan point times."""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("center", "reverse"), required=True)
    a = p.parse_args(); source, output = a.recording.resolve(), a.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    (output / "horizon").mkdir(parents=True)
    original_spans, transformed_spans = [], []
    for path in sorted((source / "horizon").glob("*.npz")):
        with np.load(path) as z:
            arrays = {name: np.asarray(z[name]) for name in z.files}
        stamp = float(arrays["stamp"]); times = np.asarray(arrays["t"], dtype=np.float64)
        original_spans.append(float(times.max() - times.min()))
        arrays["t"] = (np.full_like(times, stamp) if a.mode == "center" else 2.0 * stamp - times)
        transformed_spans.append(float(arrays["t"].max() - arrays["t"].min()))
        np.savez_compressed(output / "horizon" / path.name, **arrays)
    for name in ("encoder.jsonl", "camera.config", "lidar.config", "liftbot.config",
                 "camera_intrinsics.json", "interval_report.json", "tail_report.json"):
        path = source / name
        if path.is_file(): link_or_copy(path, output / name)
    report = {
        "schema": "fliftbot.point_time_transform.v1", "source": str(source), "mode": a.mode,
        "scans": len(original_spans),
        "original_span_ms_median": 1000 * float(np.median(original_spans)),
        "transformed_span_ms_median": 1000 * float(np.median(transformed_spans)),
    }
    (output / "point_time_transform_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
