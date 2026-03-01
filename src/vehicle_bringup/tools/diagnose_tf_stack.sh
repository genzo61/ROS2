#!/usr/bin/env bash
set -euo pipefail

# setup.bash touches AMENT_TRACE_SETUP_FILES; keep it defined under nounset.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"

set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$HOME/turtlebot3_ws/install/setup.bash" ]]; then
  source "$HOME/turtlebot3_ws/install/setup.bash"
fi
set -u

export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export ROS2CLI_NO_DAEMON=1

# Recover from stale ros2 daemon state (!rclpy.ok() XMLRPC faults).
ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true
sleep 1

echo "=== Node inventory (EKF/NavSat/Nav2) ==="
ros2 node list | rg -n "ekf|navsat|controller_server|planner_server|bt_navigator|behavior_server|lifecycle_manager" || true

echo
for n in /ekf_filter_node_odom /ekf_filter_node_map /navsat_transform /controller_server /planner_server /bt_navigator /behavior_server /lifecycle_manager_navigation; do
  echo "=== use_sim_time: ${n} ==="
  ros2 param get "$n" use_sim_time 2>/dev/null || echo "not available"
done

echo

echo "=== /clock ==="
ros2 topic info /clock -v || true
timeout 12 ros2 topic hz /clock -w 10 || true


echo

echo "=== /tf publishers ==="
ros2 topic info /tf -v || true

echo

echo "=== /tf_static publishers ==="
ros2 topic info /tf_static -v || true

echo

echo "=== TF frame stream snapshot ==="
timeout 8 ros2 topic echo /tf | rg -n "frame_id:|child_frame_id:|map|odom|base_footprint|base_link" | head -n 120 || true

echo

echo "=== TF availability checks ==="
timeout 8 ros2 run tf2_ros tf2_echo odom base_footprint || true
timeout 8 ros2 run tf2_ros tf2_echo base_footprint base_link || true
timeout 8 ros2 run tf2_ros tf2_echo map odom || true

echo

echo "=== Odom topics ==="
ros2 topic list | rg -n "^/odom$|^/odometry/local$|^/odometry/filtered$|^/odometry/gps$" || true
ros2 topic info /odom -v || true
ros2 topic info /odometry/local -v || true
ros2 topic info /odometry/filtered -v || true

echo

echo "=== Nav2 lifecycle states ==="
timeout 5 ros2 lifecycle get /controller_server || true
timeout 5 ros2 lifecycle get /planner_server || true
timeout 5 ros2 lifecycle get /bt_navigator || true
timeout 5 ros2 lifecycle get /behavior_server || true


echo

echo "=== DONE ==="
