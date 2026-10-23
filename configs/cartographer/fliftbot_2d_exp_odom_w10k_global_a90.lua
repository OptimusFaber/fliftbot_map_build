-- Global constraint scan matcher: +/-90 degrees; inspect false matches.
include "fliftbot_2d_exp_odom_w10k.lua"
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(90.0)
return options
