#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LIDAR_CONFIG="${ROOT}/configs/bs_version/lidar_18m.config"
export OUTPUT="${OUTPUT:-/home/optimus/ros-projects/ros-data/results/real/20261013/tail_experiments_18m_141857}"
exec bash "${ROOT}/scripts/run_tail_experiments.sh"
