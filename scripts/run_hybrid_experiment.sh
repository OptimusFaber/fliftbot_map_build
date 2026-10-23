#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RECORDING="${RECORDING:-/home/optimus/ros-projects/ros-data/recordings/real/20261014/hybrid_141857_tail6m_plus_163021/canonical}"
OUTPUT="${OUTPUT:-/home/optimus/ros-projects/ros-data/results/real/20261014/hybrid_experiment_60m}"
NAME="${NAME:-hybrid_tail6m_plus_fast_60m}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/bs_version}"
CAMERA_CONFIG="${CAMERA_CONFIG:-${CONFIG_DIR}/camera.config}"
LIDAR_CONFIG="${LIDAR_CONFIG:-${CONFIG_DIR}/lidar_60m.config}"
ROBOT_CONFIG="${ROBOT_CONFIG:-${CONFIG_DIR}/liftbot.config}"
WHEELBASE="${WHEELBASE:-1.39}"
ENCODER_SPEED_SIGN="${ENCODER_SPEED_SIGN:-1}"
STEERING_SIGN="${STEERING_SIGN:-1}"
MAPPING_YAW="${MAPPING_YAW:-180}"

for CONFIG in "${CAMERA_CONFIG}" "${LIDAR_CONFIG}" "${ROBOT_CONFIG}"; do
  [[ -f "${CONFIG}" ]] || { echo "Configuration not found: ${CONFIG}" >&2; exit 2; }
done

"${PYTHON}" "${ROOT}/src/map_build/build_from_canonical.py" \
  --recording "${RECORDING}" \
  --camera-config "${CAMERA_CONFIG}" \
  --lidar-config "${LIDAR_CONFIG}" \
  --robot-config "${ROBOT_CONFIG}" \
  --output "${OUTPUT}" \
  --name "${NAME}" \
  --wheelbase "${WHEELBASE}" \
  --encoder-speed-sign "${ENCODER_SPEED_SIGN}" \
  --steering-sign "${STEERING_SIGN}" \
  --mapping-yaw "${MAPPING_YAW}"

echo "Results: ${OUTPUT}"
