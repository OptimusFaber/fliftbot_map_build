-- Constraint candidates up to 30 m from the predicted pose.
include "fliftbot_2d_exp_odom_w10k.lua"
POSE_GRAPH.constraint_builder.max_constraint_distance = 30.0
return options
