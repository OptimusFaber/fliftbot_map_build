-- Offline profile for repetitive indoor geometry.  Keep the local matcher and
-- wheel-odometry prior from fliftbot_2d_offline.lua, but reject long-distance
-- aliases between visually similar parallel walls/corridors.
include "fliftbot_2d_offline.lua"

-- The 2026-10-02 audit found accepted 5--7 m aliases at scores 0.66--0.74.
-- Genuine revisits in this recording have a useful odometry prior, so search
-- locally and demand substantially stronger evidence before adding a loop.
POSE_GRAPH.constraint_builder.max_constraint_distance = 2.0
POSE_GRAPH.constraint_builder.min_score = 0.74
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.82
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 2.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(12.)

return options
