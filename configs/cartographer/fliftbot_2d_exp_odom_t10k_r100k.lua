-- Separate rotation-weight test: translation 10k, rotation 100k.
include "fliftbot_2d_exp_odom_w10k.lua"
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 100000.0
return options
