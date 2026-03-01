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

exec ros2 launch vehicle_bringup sim.launch.py \
  mode:=local_nav2_mode \
  enable_nav2:=true \
  enable_gps_stack:=false \
  use_local_ekf:=false \
  nav2_odom_topic:=/odom \
  cleanup_stale_gazebo:=true \
  "$@"
