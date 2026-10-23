-- Local correlative matcher: 0.30 m / 10 degrees.
include "fliftbot_2d_exp_odom_w10k_rt_narrow.lua"
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.30
return options
