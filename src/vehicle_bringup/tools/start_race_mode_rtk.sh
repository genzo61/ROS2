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
  use_local_ekf:=true \
  enable_gps_stack:=true \
  enable_fake_rtk:=true \
  fake_rtk_input_topic:=/gps/fix \
  gps_fix_topic:=/vehicle/gps/fix \
  rtk_status:=FIX \
  waypoint_source:=auto \
  waypoint_weight_with_lane:=0.04 \
  waypoint_weight_no_lane:=0.22 \
  "$@"
