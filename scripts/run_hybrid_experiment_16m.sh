#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LIDAR_CONFIG="${ROOT}/configs/bs_version/lidar_16m.config"
export OUTPUT="${OUTPUT:-/home/optimus/ros-projects/ros-data/results/real/20261014/hybrid_experiment_16m}"
export NAME="${NAME:-hybrid_tail6m_plus_fast_16m}"
exec bash "${ROOT}/scripts/run_hybrid_experiment.sh"
