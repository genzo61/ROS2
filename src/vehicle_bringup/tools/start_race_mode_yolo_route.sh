#!/usr/bin/env bash
set -euo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"

WORKSPACE="${WORKSPACE:-$HOME/turtlebot3_ws}"

set +u
source /opt/ros/humble/setup.bash
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
  source "${WORKSPACE}/install/setup.bash"
fi
set -u

export ROS2CLI_NO_DAEMON=1
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/roslog_vehicle_bringup}"
unset ROS_DOMAIN_ID
unset ROS_LOCALHOST_ONLY

exec ros2 launch vehicle_bringup yaris_autonomy.launch.py \
  mode:=race_mode \
  enable_lane_yolo:=true \
  lane_model_path:=auto \
  route_enabled:=true \
  route_weight_normal:=0.18 \
  route_weight_single:=0.08 \
  start_max_dist:=8.0 \
  start_heading_weight:=1.0 \
  route_cross_track_gain:=0.85 \
  route_preview_multiplier:=1.8 \
  lane_only_speed:=0.78 \
  "$@"
