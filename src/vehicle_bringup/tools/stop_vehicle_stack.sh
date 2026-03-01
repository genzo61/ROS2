#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f "ros2 launch vehicle_bringup" || true
pkill -TERM -f "ros2 launch .*yaris_autonomy.launch.py" || true
pkill -TERM -f "ros2 launch .*sim.launch.py" || true
pkill -TERM gzserver || true
pkill -TERM gzclient || true
pkill -TERM gazebo || true
sleep 1
pkill -KILL gzserver || true
pkill -KILL gzclient || true
pkill -KILL gazebo || true

echo "Vehicle stack stop sequence completed."
