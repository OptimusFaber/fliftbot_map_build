-- More conservative loop-closure acceptance threshold.
include "fliftbot_2d_exp_odom_w10k.lua"
POSE_GRAPH.constraint_builder.min_score = 0.60
return options
