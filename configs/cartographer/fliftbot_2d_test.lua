include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",

  odom_frame = "wheel_odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,

  lookup_transform_timeout_sec = 0.2,

  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 0.02,
  trajectory_publish_period_sec = 0.03,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,

  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- ============================================================
-- LOCAL SLAM
-- ============================================================

TRAJECTORY_BUILDER_2D.use_imu_data = false

TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 30.0

-- LiDAR установлен на высоте ~2.03 м.
TRAJECTORY_BUILDER_2D.min_z = -0.5
TRAJECTORY_BUILDER_2D.max_z = 2.5

-- Пока не меняем размер submap.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90


-- ============================================================
-- GLOBAL SLAM / LOOP CLOSURE
-- ============================================================

-- Чаще запускаем pose graph optimization для диагностики.
POSE_GRAPH.optimize_every_n_nodes = 30

-- Проверяем все потенциальные кандидаты.
POSE_GRAPH.constraint_builder.sampling_ratio = 1.0

-- Логируем найденные/отброшенные matches.
POSE_GRAPH.constraint_builder.log_matches = true

-- Было 0.65. Для диагностического запуска снижаем.
POSE_GRAPH.constraint_builder.min_score = 0.55

POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60


-- ============================================================
-- ODOMETRY
-- ============================================================

-- Пока оставляем прежние веса.
-- Сначала хотим проверить именно constraint builder.
POSE_GRAPH.optimization_problem.odometry_translation_weight = 0.25
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 0.25


return options