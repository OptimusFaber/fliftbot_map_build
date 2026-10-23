-- Local correlative matcher: 0.15 m / 5 degrees.
include "fliftbot_2d_exp_odom_w10k_rt_narrow.lua"
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(5.0)
return options
