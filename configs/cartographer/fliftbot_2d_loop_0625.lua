-- Diagnostic loop-closure profile for the 2026-10-21 warehouse recording.
-- Keep local SLAM identical to the regular offline profile and vary only the
-- global constraint search parameters across the loop-closure sweep.
include "fliftbot_2d_offline.lua"

POSE_GRAPH.constraint_builder.sampling_ratio = 1.0
POSE_GRAPH.constraint_builder.max_constraint_distance = 12.0
POSE_GRAPH.constraint_builder.min_score = 0.625
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 10.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(45.0)

return options
