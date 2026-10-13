#!/usr/bin/env python3
"""Compare encoder-integrated LiDAR pose with Cartographer trajectory."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UNIVERSAL_TO_UNIX_SECONDS = 62135596800.0


def load_encoder(path: Path):
    rows = []
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            timestamp = item.get("enc_unix", item.get("t", item.get("timestamp")))
            if timestamp is not None and "enc_v" in item and "enc_dgr" in item:
                rows.append((float(timestamp), float(item["enc_v"]), float(item["enc_dgr"])))
    rows.sort()
    # The standalone runner drops duplicate encoder timestamps.
    unique = []
    for row in rows:
        if unique and row[0] == unique[-1][0]:
            continue
        unique.append(row)
    return np.asarray(unique, dtype=float)


def integrate_encoder(data, wheelbase, steering_offset_deg, speed_sign):
    poses = np.zeros((len(data), 3), dtype=float)
    distance = np.zeros(len(data), dtype=float)
    offset = math.radians(steering_offset_deg)
    for i in range(1, len(data)):
        dt = data[i, 0] - data[i - 1, 0]
        poses[i] = poses[i - 1]
        distance[i] = distance[i - 1]
        if not 0.0 < dt <= 0.5:
            continue
        speed = speed_sign * 0.5 * (data[i, 1] + data[i - 1, 1])
        steering = math.radians(0.5 * (data[i, 2] + data[i - 1, 2])) + offset
        wz = speed * math.sin(steering) / wheelbase
        theta = wz * dt
        vx = speed * math.cos(steering)
        if abs(wz) < 1e-9:
            dx, dy = vx * dt, 0.0
        else:
            dx = vx * math.sin(theta) / wz
            dy = vx * (1.0 - math.cos(theta)) / wz
        yaw = poses[i - 1, 2]
        poses[i, 0] += math.cos(yaw) * dx - math.sin(yaw) * dy
        poses[i, 1] += math.sin(yaw) * dx + math.cos(yaw) * dy
        poses[i, 2] += theta
        distance[i] += abs(speed) * dt
    return poses, distance


def load_cartographer(path: Path):
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows["poses"]
    result = []
    for row in rows:
        timestamp = row["timestamp_universal"] / 1e7 - UNIVERSAL_TO_UNIX_SECONDS
        yaw = 2.0 * math.atan2(row["qz"], row["qw"])
        result.append((timestamp, row["x"], row["y"], yaw))
    result = np.asarray(result, dtype=float)
    result[:, 3] = np.unwrap(result[:, 3])
    return result


def axis_to_lidar(axis, lidar_x, lidar_y, lidar_yaw_deg):
    result = axis.copy()
    mount_yaw = math.radians(lidar_yaw_deg)
    c, s = np.cos(axis[:, 2]), np.sin(axis[:, 2])
    result[:, 0] += c * lidar_x - s * lidar_y
    result[:, 1] += s * lidar_x + c * lidar_y
    result[:, 2] += mount_yaw
    return result


def interpolate(timestamps, source_t, source_values):
    return np.column_stack([
        np.interp(timestamps, source_t, source_values[:, column])
        for column in range(source_values.shape[1])
    ])


def align_first(reference, estimate):
    angle = reference[0, 2] - estimate[0, 2]
    c, s = math.cos(angle), math.sin(angle)
    rotated = estimate.copy()
    rotated[:, 0] = c * estimate[:, 0] - s * estimate[:, 1]
    rotated[:, 1] = s * estimate[:, 0] + c * estimate[:, 1]
    rotated[:, 2] += angle
    rotated[:, :2] += reference[0, :2] - rotated[0, :2]
    return rotated


def analyse(run_dir: Path, output_dir: Path):
    parameters = json.loads((run_dir / "run_parameters.json").read_text())
    stats = json.loads((run_dir / "map/run_stats.json").read_text())
    robot = parameters["robot"]
    lidar = parameters["lidar"]
    position = lidar["position_m"]
    orientation = lidar.get("mapping_orientation_deg", lidar["orientation_deg"])
    encoder = load_encoder(run_dir / "input/encoder.jsonl")
    axis, encoder_distance = integrate_encoder(
        encoder,
        float(stats.get("wheelbase_m", robot["wheelbase_m"])),
        float(stats.get("steering_offset_deg", robot.get("steering_offset_deg", 0.0))),
        float(stats.get("encoder_speed_sign", -1.0)),
    )
    odom_lidar = axis_to_lidar(
        axis, float(position["x"]), float(position["y"]), float(orientation.get("yaw", 0.0))
    )
    cart = load_cartographer(run_dir / "map/trajectory.json")
    mask = (cart[:, 0] >= encoder[0, 0]) & (cart[:, 0] <= encoder[-1, 0])
    cart = cart[mask]
    odom = interpolate(cart[:, 0], encoder[:, 0], odom_lidar)
    odom = align_first(cart[:, 1:4], odom)
    distance_at_nodes = np.interp(cart[:, 0], encoder[:, 0], encoder_distance)

    position_error = np.linalg.norm(odom[:, :2] - cart[:, 1:3], axis=1)
    yaw_error = np.degrees(np.unwrap(odom[:, 2] - cart[:, 3]))
    yaw_error -= yaw_error[0]
    travelled = float(distance_at_nodes[-1] - distance_at_nodes[0])
    cart_path = float(np.linalg.norm(np.diff(cart[:, 1:3], axis=0), axis=1).sum())
    endpoint_error = float(position_error[-1])
    metrics = {
        "run": str(run_dir),
        "mapping_used_odometry": bool(stats.get("odom_samples", 0)),
        "matched_nodes": int(len(cart)),
        "duration_s": float(cart[-1, 0] - cart[0, 0]),
        "encoder_distance_m": travelled,
        "cartographer_path_m": cart_path,
        "position_error_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_error_median_m": float(np.median(position_error)),
        "position_error_p95_m": float(np.percentile(position_error, 95)),
        "position_error_endpoint_m": endpoint_error,
        "endpoint_error_per_encoder_meter": endpoint_error / travelled if travelled else None,
        "yaw_error_rmse_deg": float(np.sqrt(np.mean(yaw_error ** 2))),
        "yaw_error_endpoint_deg": float(yaw_error[-1]),
        "encoder_relative_yaw_deg": float(np.degrees(odom[-1, 2] - odom[0, 2])),
        "cartographer_relative_yaw_deg": float(np.degrees(cart[-1, 3] - cart[0, 3])),
    }

    name = run_dir.name
    with (output_dir / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "cart_x", "cart_y", "cart_yaw_deg", "odom_x", "odom_y", "odom_yaw_deg", "position_error_m", "yaw_error_deg"])
        for index in range(len(cart)):
            writer.writerow([
                cart[index, 0] - cart[0, 0], cart[index, 1], cart[index, 2], math.degrees(cart[index, 3]),
                odom[index, 0], odom[index, 1], math.degrees(odom[index, 2]), position_error[index], yaw_error[index],
            ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(cart[:, 1], cart[:, 2], label="Cartographer", linewidth=2)
    axes[0].plot(odom[:, 0], odom[:, 1], label="Encoder integration", linewidth=1.5)
    axes[0].scatter(cart[0, 1], cart[0, 2], marker="o", color="black", s=25)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(name)
    axes[0].set_xlabel("x, m")
    axes[0].set_ylabel("y, m")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    elapsed = cart[:, 0] - cart[0, 0]
    axes[1].plot(elapsed, position_error, label="position error, m")
    axes[1].plot(elapsed, np.abs(yaw_error) / 10.0, label="|yaw error| / 10, deg")
    axes[1].set_xlabel("time, s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("odometry_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = [analyse(run.resolve(), args.output.resolve()) for run in args.runs]
    (args.output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for item in metrics:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
