#!/usr/bin/env bash
set -euo pipefail

# setup.bash touches AMENT_TRACE_SETUP_FILES; keep it defined under nounset.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"

WORKSPACE="${WORKSPACE:-$HOME/turtlebot3_ws}"
LOG_DIR="${LOG_DIR:-/tmp/vehicle_bringup_smoke}"
mkdir -p "${LOG_DIR}"

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

LAUNCH_PID=""
CURRENT_LOG=""

cleanup_processes() {
  pkill -TERM gzserver || true
  pkill -TERM gzclient || true
  pkill -TERM gazebo || true
  sleep 1
  pkill -KILL gzserver || true
  pkill -KILL gzclient || true
  pkill -KILL gazebo || true
}

stop_launch() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" || true
    timeout 20 bash -c "while kill -0 ${LAUNCH_PID} 2>/dev/null; do sleep 1; done" || true
    kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
  fi
  LAUNCH_PID=""
}

fail() {
  local msg="$1"
  echo "[FAIL] ${msg}"
  if [[ -n "${CURRENT_LOG}" && -f "${CURRENT_LOG}" ]]; then
    echo "=== tail: ${CURRENT_LOG} ==="
    tail -n 120 "${CURRENT_LOG}" || true
  fi
  exit 1
}

start_launch() {
  local name="$1"
  shift
  CURRENT_LOG="${LOG_DIR}/${name}.log"
  stop_launch
  cleanup_processes
  (
    cd "${WORKSPACE}"
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    export ROS2CLI_NO_DAEMON=1
    export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/roslog_vehicle_bringup}"
    unset ROS_DOMAIN_ID
    unset ROS_LOCALHOST_ONLY
    ros2 launch "$@"
  ) >"${CURRENT_LOG}" 2>&1 &
  LAUNCH_PID=$!
  echo "[INFO] started ${name}, pid=${LAUNCH_PID}, log=${CURRENT_LOG}"
}

wait_for_node() {
  local node="$1"
  local timeout_s="$2"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if ros2 node list 2>/dev/null | grep -qx "${node}"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

require_node() {
  local node="$1"
  local timeout_s="$2"
  wait_for_node "${node}" "${timeout_s}" || fail "node ${node} not found in ${timeout_s}s"
}

ensure_node_absent() {
  local node="$1"
  if ros2 node list 2>/dev/null | grep -qx "${node}"; then
    fail "node ${node} should be absent"
  fi
}

check_lifecycle_active() {
  local node="$1"
  local out
  out="$(timeout 6 ros2 lifecycle get "${node}" 2>&1 || true)"
  echo "[INFO] lifecycle ${node}: ${out}"
  [[ "${out}" == *"active [3]"* ]] || fail "lifecycle ${node} is not active"
}

check_tf() {
  local target="$1"
  local source="$2"
  local out
  out="$(timeout 8 ros2 run tf2_ros tf2_echo "${target}" "${source}" 2>&1 || true)"
  echo "[INFO] tf ${target}<-${source}"
  echo "${out}" | head -n 12
  echo "${out}" | rg -q "Translation:" || fail "tf ${target}<-${source} not available"
}

run_local_nav2_diffdrive() {
  echo "=== TEST: local_nav2_mode (diff_drive TF owner) ==="
  start_launch "local_nav2_diffdrive" \
    vehicle_bringup sim.launch.py \
    mode:=local_nav2_mode \
    enable_nav2:=true \
    enable_gps_stack:=false \
    use_local_ekf:=false \
    nav2_odom_topic:=/odom \
    cleanup_stale_gazebo:=true

  require_node /gazebo 60
  require_node /lifecycle_manager_navigation 120
  require_node /controller_server 120
  require_node /planner_server 120
  require_node /behavior_server 120
  require_node /bt_navigator 120

  check_lifecycle_active /controller_server
  check_lifecycle_active /planner_server
  check_lifecycle_active /behavior_server
  check_lifecycle_active /bt_navigator
  check_tf odom base_footprint
  stop_launch
}

run_gps_nav2() {
  echo "=== TEST: gps_nav2_mode (local+global EKF + navsat) ==="
  start_launch "gps_nav2" \
    vehicle_bringup sim.launch.py \
    mode:=gps_nav2_mode \
    enable_nav2:=true \
    enable_gps_stack:=true \
    use_local_ekf:=true \
    nav2_odom_topic:=/odometry/local \
    cleanup_stale_gazebo:=true

  require_node /gazebo 60
  require_node /ekf_filter_node_odom 120
  require_node /ekf_filter_node_map 120
  require_node /navsat_transform 120
  require_node /lifecycle_manager_navigation 180
  require_node /controller_server 180
  require_node /planner_server 180
  require_node /behavior_server 180
  require_node /bt_navigator 180

  check_lifecycle_active /controller_server
  check_lifecycle_active /planner_server
  check_lifecycle_active /behavior_server
  check_lifecycle_active /bt_navigator
  check_tf odom base_footprint
  check_tf map odom
  stop_launch
}

run_race_mode() {
  echo "=== TEST: race_mode (Nav2 disabled) ==="
  start_launch "race_mode" \
    vehicle_bringup yaris_autonomy.launch.py \
    mode:=race_mode

  require_node /gazebo 60
  require_node /yaris_pilotu 90
  require_node /lane_tracker 90

  ensure_node_absent /controller_server
  ensure_node_absent /planner_server
  ensure_node_absent /bt_navigator

  check_tf odom base_footprint
  stop_launch
}

trap 'stop_launch; cleanup_processes' EXIT

run_local_nav2_diffdrive
run_gps_nav2
run_race_mode

echo "[PASS] All smoke tests completed. Logs in ${LOG_DIR}"
