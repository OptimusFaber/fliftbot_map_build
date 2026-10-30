"""Shared, versioned marker settings for every public pipeline entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs/markers"
BOOLEAN_OPTIONS = ("per_track_height", "stationary_height_filter")
NUMERIC_OPTIONS = {
    "min_segment_observations": int,
    "min_camera_baseline_m": float,
    "merge_radius_m": float,
    "max_track_height_uncertainty_m": float,
    "block_distance_m": float,
    "max_block_height_instability_m": float,
    "stationary_group_radius_m": float,
    "stationary_max_group_fraction": float,
    "stationary_min_groups": int,
    "max_marker_height_m": float,
}
PARAMETER_NAMES = (*BOOLEAN_OPTIONS, *NUMERIC_OPTIONS, "height_quality_method", "ceiling_z")


def add_marker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--marker-profile", choices=("warehouse", "test_site"), default="warehouse",
                        help="marker geometry profile (default: warehouse)")
    for name in BOOLEAN_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                            default=None)
    for name, kind in NUMERIC_OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--height-quality-method", choices=("legacy", "block_cv"))
    parser.add_argument("--ceiling-z", type=float, help="measured plane height; overrides profile height mode")


def forward_marker_options(command: list[str], args: argparse.Namespace) -> None:
    command += ["--marker-profile", args.marker_profile]
    for name in PARAMETER_NAMES:
        value = getattr(args, name)
        if value is None:
            continue
        if name in BOOLEAN_OPTIONS:
            command.append("--" + ("" if value else "no-") + name.replace("_", "-"))
        else:
            command += ["--" + name.replace("_", "-"), str(value)]


def resolve_marker_parameters(args: argparse.Namespace, overrides: dict | None = None) -> tuple[dict, dict]:
    path = PROFILE_DIR / (args.marker_profile + ".json")
    raw = path.read_bytes()
    profile = json.loads(raw)
    if profile.get("schema") != "fliftbot.marker_profile.v1":
        raise ValueError(f"invalid marker profile: {path}")
    parameters = dict(profile["parameters"])
    if overrides:
        parameters.update({k: v for k, v in overrides.items() if k in PARAMETER_NAMES})
        if overrides.get("ceiling_z_m") is not None:
            parameters["ceiling_z"] = overrides["ceiling_z_m"]
    parameters.update({name: getattr(args, name) for name in PARAMETER_NAMES
                       if getattr(args, name) is not None})
    if parameters.get("ceiling_z") is not None:
        if args.per_track_height is True:
            raise ValueError("--ceiling-z and --per-track-height are mutually exclusive")
        parameters["per_track_height"] = False
    for name in NUMERIC_OPTIONS:
        value = parameters.get(name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < parameters["stationary_max_group_fraction"] <= 1:
        raise ValueError("stationary_max_group_fraction must be in (0, 1]")
    if parameters.get("ceiling_z") is not None and (
            not math.isfinite(parameters["ceiling_z"]) or parameters["ceiling_z"] <= 0):
        raise ValueError("ceiling_z must be finite and positive")
    provenance = {"name": args.marker_profile, "schema": profile["schema"],
                  "sha256": hashlib.sha256(raw).hexdigest(), "settings": profile}
    return parameters, provenance


def marker_command_options(parameters: dict) -> list[str]:
    command = []
    for name in PARAMETER_NAMES:
        value = parameters.get(name)
        if name in BOOLEAN_OPTIONS:
            if value:
                command.append("--" + name.replace("_", "-"))
        elif value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    return command
