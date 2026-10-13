#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
OUT=${1:-${OUT:-"$ROOT/results/camera_calibration_experiments"}}

OLD_MAP=${OLD_MAP:-"$PROJECT_ROOT/ros-data/results/real/20260922/170557_wheelbase_sweep/wb_1_41"}
: "${NEW_MAP:?Set NEW_MAP to the new Cartographer map directory}"

: "${OLD_OBS_SOURCE:?Set OLD_OBS_SOURCE to the old observations JSON}"
: "${NEW_OBS:?Set NEW_OBS to the new observations JSON}"

OLD_CAMERA=${OLD_CAMERA:-"$PROJECT_ROOT/ros-data/results/real/20260922/170557_wheelbase_yaw180_ab/inputs/camera_new.txt"}
: "${NEW_CAMERA:?Set NEW_CAMERA to the new camera position file}"
: "${CAMERA_JSON:?Set CAMERA_JSON to the camera intrinsics JSON}"
: "${LIDAR_CONFIG:?Set LIDAR_CONFIG to the marker-localization LiDAR config}"

: "${VIDEO:?Set VIDEO to the source camera video}"
: "${TIMESTAMPS:?Set TIMESTAMPS to frame_timestamps.csv}"
: "${LABELS:?Set LABELS to labels2coords.pickle}"

LOCALIZER="$ROOT/vendor/fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_stored_ray_corrected.py"
AUTO_MAP_DIR="$ROOT/vendor/auto_map"
VIDEO_PROCESSOR="$AUTO_MAP_DIR/get_video_data.py"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file does not exist: $1" >&2
        exit 2
    fi
}

require_map() {
    require_file "$1/trajectory.json"
    require_file "$1/map.yaml"
    require_file "$1/map.pgm"
}

localize() {
    local observations=$1
    local map_dir=$2
    local camera_position=$3
    local output_dir=$4

    echo
    echo "==> Localizing: $(basename "$output_dir")"
    python3 "$LOCALIZER" \
        --observations "$observations" \
        --trajectory "$map_dir/trajectory.json" \
        --map-dir "$map_dir" \
        --camera-position "$camera_position" \
        --lidar-config "$LIDAR_CONFIG" \
        --out-dir "$output_dir" \
        --ceiling-z 3.03 \
        --group-by-code
}

generate_observations() {
    local work_dir=$1
    local camera_position=$2

    mkdir -p "$work_dir"
    ln -sfn "$VIDEO" "$work_dir/video.mp4"
    ln -sfn "$TIMESTAMPS" "$work_dir/frame_timestamps.csv"

    echo
    echo "==> Generating observations: $(basename "$work_dir")"
    (
        cd "$AUTO_MAP_DIR"
        python3 "$VIDEO_PROCESSOR" \
            --video "$work_dir" \
            --camera-json "$CAMERA_JSON" \
            --camera-position "$camera_position" \
            --labels "$LABELS"
    )

    require_file "$work_dir/observations.json"
}

echo "Checking inputs..."
require_map "$OLD_MAP"
require_map "$NEW_MAP"
for path in \
    "$OLD_OBS_SOURCE" "$NEW_OBS" "$OLD_CAMERA" "$NEW_CAMERA" \
    "$CAMERA_JSON" "$LIDAR_CONFIG" "$VIDEO" "$TIMESTAMPS" \
    "$LABELS" "$LOCALIZER" "$VIDEO_PROCESSOR"
do
    require_file "$path"
done

mkdir -p "$OUT/inputs"
cp "$OLD_OBS_SOURCE" "$OUT/inputs/old_observations.json"
OLD_OBS="$OUT/inputs/old_observations.json"

# Same new roll/pitch/yaw, with only Z changed from 1.248 m to 1.270 m.
cp "$NEW_CAMERA" "$OUT/inputs/camera_new_angles_z1248.txt"
awk 'NR == 1 {$3 = 127.0} {print}' "$NEW_CAMERA" \
    > "$OUT/inputs/camera_new_angles_z1270.txt"

# Old angular calibration with the same XYZ (including Z=1.248 m) as the new
# calibration. The old two-angle legacy form intentionally has no mount yaw.
{
    head -n 1 "$NEW_CAMERA"
    sed -n '2p' "$OLD_CAMERA"
} > "$OUT/inputs/camera_old_angles_z1248.txt"

cp "$OLD_CAMERA" "$OUT/inputs/camera_old_original_z1270.txt"
cp "$NEW_CAMERA" "$OUT/inputs/camera_new_original_z1248.txt"
cp "$CAMERA_JSON" "$OUT/inputs/camera.json"
cp "$LIDAR_CONFIG" "$OUT/inputs/lidar.markers.config"

# 1-4: existing observations, crossing old/new maps and calibrations.
localize "$OLD_OBS" "$OLD_MAP" "$OLD_CAMERA" \
    "$OUT/01_old_map_old_obs_z1270"
localize "$NEW_OBS" "$OLD_MAP" "$NEW_CAMERA" \
    "$OUT/02_old_map_new_obs_z1248"
localize "$OLD_OBS" "$NEW_MAP" "$OLD_CAMERA" \
    "$OUT/03_new_map_old_obs_z1270"
localize "$NEW_OBS" "$NEW_MAP" "$NEW_CAMERA" \
    "$OUT/04_new_map_new_obs_z1248"

# 5: isolate camera Z on the same map and the same observations.
localize "$NEW_OBS" "$NEW_MAP" "$OUT/inputs/camera_new_angles_z1248.txt" \
    "$OUT/05a_same_map_z1248"
localize "$NEW_OBS" "$NEW_MAP" "$OUT/inputs/camera_new_angles_z1270.txt" \
    "$OUT/05b_same_map_z1270"

# 6: regenerate observations separately for each angular calibration while
# keeping camera XYZ, video, timestamps and camera intrinsics identical.
generate_observations \
    "$OUT/generated_observations/old_angles_z1248" \
    "$OUT/inputs/camera_old_angles_z1248.txt"
generate_observations \
    "$OUT/generated_observations/new_angles_z1248" \
    "$OUT/inputs/camera_new_angles_z1248.txt"

localize \
    "$OUT/generated_observations/old_angles_z1248/observations.json" \
    "$NEW_MAP" \
    "$OUT/inputs/camera_old_angles_z1248.txt" \
    "$OUT/06a_regenerated_old_angles"
localize \
    "$OUT/generated_observations/new_angles_z1248/observations.json" \
    "$NEW_MAP" \
    "$OUT/inputs/camera_new_angles_z1248.txt" \
    "$OUT/06b_regenerated_new_angles"

echo
echo "All experiments completed successfully."
echo "Results: $OUT"
