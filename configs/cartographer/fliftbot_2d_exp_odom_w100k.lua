-- Experiment 3: strongly anchor the pose graph to corrected wheel odometry.
include "fliftbot_2d_offline.lua"

POSE_GRAPH.constraint_builder.sampling_ratio = 1.0
POSE_GRAPH.constraint_builder.max_constraint_distance = 12.0
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 10.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(45.0)
POSE_GRAPH.optimization_problem.odometry_translation_weight = 100000.0
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 100000.0

return options
