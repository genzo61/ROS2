#!/usr/bin/env bash
set -euo pipefail

# setup.bash touches AMENT_TRACE_SETUP_FILES; keep it defined under nounset.
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
  "$@"
