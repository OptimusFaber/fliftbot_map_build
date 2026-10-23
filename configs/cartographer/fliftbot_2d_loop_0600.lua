-- Same diagnostic profile as fliftbot_2d_loop_0625.lua, with a moderately
-- relaxed loop-closure acceptance threshold.
include "fliftbot_2d_offline.lua"

POSE_GRAPH.constraint_builder.sampling_ratio = 1.0
POSE_GRAPH.constraint_builder.max_constraint_distance = 12.0
POSE_GRAPH.constraint_builder.min_score = 0.60
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 10.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(45.0)

return options
