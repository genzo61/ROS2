#!/usr/bin/env bash
set -euo pipefail

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"

WORKSPACE="${WORKSPACE:-$HOME/turtlebot3_ws}"
PACKAGE="${PACKAGE:-vehicle_bringup}"
LOG_FILE="${LOG_FILE:-/tmp/lane_preview.log}"
REPORT_LINES="${REPORT_LINES:-40}"

setup_ros() {
  set +u
  source /opt/ros/humble/setup.bash
  if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
    source "${WORKSPACE}/install/setup.bash"
  fi
  set -u
}

do_build() {
  cd "${WORKSPACE}"
  setup_ros
  colcon build --packages-select "${PACKAGE}"
}

do_run() {
  cd "${WORKSPACE}"
  : > "${LOG_FILE}"
  do_build
  setup_ros
  echo "LOG_FILE=${LOG_FILE}"
  echo "Ctrl-C ile testi bitir, sonra report komutunu calistir."
  ./src/vehicle_bringup/tools/start_race_mode_yolo.sh 2>&1 | tee "${LOG_FILE}"
}

do_run_route() {
  cd "${WORKSPACE}"
  : > "${LOG_FILE}"
  do_build
  setup_ros
  echo "LOG_FILE=${LOG_FILE}"
  echo "Hybrid route+lane modu aciliyor. Ctrl-C ile testi bitir, sonra report komutunu calistir."
  ./src/vehicle_bringup/tools/start_race_mode_yolo_route.sh 2>&1 | tee "${LOG_FILE}"
}

do_report() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Log dosyasi bulunamadi: ${LOG_FILE}" >&2
    exit 1
  fi

  echo "=== PREVIEW ==="
  rg "lane_valid=|near=|far=|curve=|conf=|near_center=|far_center=|near_src=|far_src=" "${LOG_FILE}" | tail -n "${REPORT_LINES}" || true

  echo
  echo "=== ANG_DEBUG ==="
  rg "\\[ANG_DEBUG\\]|parser_conf=|lane_conf=" "${LOG_FILE}" | tail -n "${REPORT_LINES}" || true
}

do_topics() {
  setup_ros
  ros2 topic list | rg "/lane/(debug|near_error|far_error|curve_indicator|confidence|error|valid)$" || true
}

usage() {
  cat <<'EOF'
Kullanim:
  lane_preview_helper.sh run
  lane_preview_helper.sh run-route
  lane_preview_helper.sh report
  lane_preview_helper.sh topics

Ornek:
  ~/turtlebot3_ws/src/vehicle_bringup/tools/lane_preview_helper.sh run
  ~/turtlebot3_ws/src/vehicle_bringup/tools/lane_preview_helper.sh run-route
  ~/turtlebot3_ws/src/vehicle_bringup/tools/lane_preview_helper.sh report
EOF
}

main() {
  local cmd="${1:-run}"
  case "${cmd}" in
    run)
      do_run
      ;;
    run-route)
      do_run_route
      ;;
    report)
      do_report
      ;;
    topics)
      do_topics
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Bilinmeyen komut: ${cmd}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
