#!/usr/bin/env python3

import math
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from lifecycle_msgs.srv import GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from robot_localization.srv import FromLL
from tf2_ros import Buffer, TransformException, TransformListener


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class IgvcWaypointNavigator(Node):
    def __init__(self) -> None:
        super().__init__('igvc_waypoint_navigator')

        self.declare_parameter('gps_waypoints', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('map_waypoints', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('waypoint_altitude', 0.0)
        self.declare_parameter('wait_for_nav2_seconds', 60.0)
        self.declare_parameter('wait_for_fromll_seconds', 60.0)
        self.declare_parameter('wait_for_tf_seconds', 5.0)
        self.declare_parameter('feedback_interval_seconds', 2.0)
        self.declare_parameter('anchor_gps', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('anchor_map', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('enforce_goal_distance_check', True)
        self.declare_parameter('max_goal_distance_m', 60.0)
        self.declare_parameter('skip_waypoint_distance_m', 0.6)

        self.navigator = BasicNavigator()
        self.from_ll_clients = {
            '/fromLL': self.create_client(FromLL, '/fromLL'),
            '/navsat_transform/fromLL': self.create_client(FromLL, '/navsat_transform/fromLL'),
        }
        self.active_from_ll_client: Optional[rclpy.client.Client] = None
        self.map_offset_x = 0.0
        self.map_offset_y = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.nav2_state_clients = {
            'planner_server': self.create_client(GetState, '/planner_server/get_state'),
            'bt_navigator': self.create_client(GetState, '/bt_navigator/get_state'),
        }

    def close(self) -> None:
        listener = getattr(self, 'tf_listener', None)
        if listener is None:
            return

        # Stop dedicated TF listener executor before rclpy shutdown to avoid noisy shutdown traces.
        if hasattr(listener, 'executor'):
            listener.executor.shutdown()
        if hasattr(listener, 'dedicated_listener_thread'):
            listener.dedicated_listener_thread.join(timeout=1.0)
        try:
            listener.unregister()
        except Exception:  # pylint: disable=broad-except
            pass
        self.tf_listener = None

    def _read_gps_waypoints(self) -> List[Tuple[float, float]]:
        raw_values = self.get_parameter('gps_waypoints').value
        if not isinstance(raw_values, list):
            raise ValueError('gps_waypoints must be a list of floats: [lat1, lon1, lat2, lon2, ...]')

        values = [float(v) for v in raw_values]
        if len(values) == 0:
            return []
        if len(values) % 2 != 0:
            raise ValueError('gps_waypoints must contain an even number of values (lat/lon pairs).')

        waypoints = []
        for idx in range(0, len(values), 2):
            waypoints.append((values[idx], values[idx + 1]))
        return waypoints

    def _read_map_waypoints(self) -> List[Tuple[float, float]]:
        raw_values = self.get_parameter('map_waypoints').value
        if not isinstance(raw_values, list):
            raise ValueError('map_waypoints must be a list of floats: [x1, y1, x2, y2, ...]')

        values = [float(v) for v in raw_values]
        if len(values) == 0:
            return []
        if len(values) % 2 != 0:
            raise ValueError('map_waypoints must contain an even number of values (x/y pairs).')

        waypoints = []
        for idx in range(0, len(values), 2):
            waypoints.append((values[idx], values[idx + 1]))
        return waypoints

    def _from_ll_to_map(self, latitude: float, longitude: float, altitude: float) -> Tuple[float, float]:
        if self.active_from_ll_client is None:
            raise RuntimeError('fromLL client is not selected.')

        req = FromLL.Request()
        req.ll_point.latitude = latitude
        req.ll_point.longitude = longitude
        req.ll_point.altitude = altitude

        future = self.active_from_ll_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError('fromLL service call failed.')

        result = future.result()
        return float(result.map_point.x), float(result.map_point.y)

    def _read_optional_pair(self, parameter_name: str) -> List[float]:
        values = self.get_parameter(parameter_name).value
        if not isinstance(values, list):
            raise ValueError(f'{parameter_name} must be a list like [a, b].')
        if len(values) == 0:
            return []
        if len(values) != 2:
            raise ValueError(f'{parameter_name} must contain exactly 2 values.')
        return [float(values[0]), float(values[1])]

    def _configure_map_offset(self, altitude: float) -> None:
        anchor_gps = self._read_optional_pair('anchor_gps')
        anchor_map = self._read_optional_pair('anchor_map')

        if not anchor_gps and not anchor_map:
            self.get_logger().warning(
                'No anchor calibration provided. Using raw fromLL outputs without map offset correction.'
            )
            self.map_offset_x = 0.0
            self.map_offset_y = 0.0
            return

        if not anchor_gps or not anchor_map:
            raise ValueError('anchor_gps and anchor_map must be provided together.')

        raw_x, raw_y = self._from_ll_to_map(anchor_gps[0], anchor_gps[1], altitude)
        self.map_offset_x = anchor_map[0] - raw_x
        self.map_offset_y = anchor_map[1] - raw_y
        self.get_logger().info(
            f'Anchor calibration active. map_offset_x={self.map_offset_x:.3f}, map_offset_y={self.map_offset_y:.3f}'
        )

    def _apply_map_offset(self, x: float, y: float) -> Tuple[float, float]:
        return x + self.map_offset_x, y + self.map_offset_y

    def _get_robot_map_xy(self, timeout_sec: float) -> Tuple[float, float]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time()
                )
                return float(transform.transform.translation.x), float(transform.transform.translation.y)
            except TransformException:
                time.sleep(0.1)
        raise RuntimeError('Timed out waiting for transform map -> base_footprint.')

    def _build_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(yaw)
        return pose

    def _compute_yaw(self, current: Tuple[float, float], nxt: Tuple[float, float]) -> float:
        return math.atan2(nxt[1] - current[1], nxt[0] - current[0])

    def _wait_for_nav2_active(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        req = GetState.Request()

        while time.monotonic() < deadline:
            all_active = True
            for node_name, client in self.nav2_state_clients.items():
                if not client.wait_for_service(timeout_sec=0.5):
                    self.get_logger().info(f'Waiting service: /{node_name}/get_state')
                    all_active = False
                    continue

                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                result = future.result()
                if result is None:
                    self.get_logger().info(f'No state response yet: {node_name}')
                    all_active = False
                    continue

                state_id = int(result.current_state.id)
                state_label = str(result.current_state.label)
                if state_id != 3:
                    self.get_logger().info(f'{node_name} state={state_label} [{state_id}]')
                    all_active = False

            if all_active:
                return True

            time.sleep(0.5)

        return False

    def _wait_for_fromll_service(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        next_log_time = 0.0

        while time.monotonic() < deadline:
            for service_name, client in self.from_ll_clients.items():
                if client.wait_for_service(timeout_sec=0.5):
                    self.active_from_ll_client = client
                    self.get_logger().info(f'Using service: {service_name}')
                    return True

            now = time.monotonic()
            if now >= next_log_time:
                self.get_logger().info('Waiting for fromLL service (/fromLL or /navsat_transform/fromLL)...')
                next_log_time = now + 2.0

        return False

    def run(self) -> int:
        waypoints_map = self._read_map_waypoints()
        waypoints_ll = self._read_gps_waypoints()
        if not waypoints_map and not waypoints_ll:
            self.get_logger().error(
                'No waypoint provided. Set map_waypoints or gps_waypoints parameter.'
            )
            return 1

        wait_for_nav2 = float(self.get_parameter('wait_for_nav2_seconds').value)
        wait_for_fromll = float(self.get_parameter('wait_for_fromll_seconds').value)
        wait_for_tf = float(self.get_parameter('wait_for_tf_seconds').value)
        feedback_interval = float(self.get_parameter('feedback_interval_seconds').value)
        enforce_goal_distance = bool(self.get_parameter('enforce_goal_distance_check').value)
        max_goal_distance = float(self.get_parameter('max_goal_distance_m').value)
        skip_waypoint_distance = float(self.get_parameter('skip_waypoint_distance_m').value)
        altitude = float(self.get_parameter('waypoint_altitude').value)

        self.get_logger().info('Waiting for Nav2 lifecycle nodes to become active...')
        if not self._wait_for_nav2_active(wait_for_nav2):
            self.get_logger().error('Nav2 did not become active before timeout.')
            return 1

        waypoints_xy: List[Tuple[float, float]] = []
        if waypoints_map:
            self.get_logger().info('Using map_waypoints directly (GPS conversion disabled).')
            for idx, (x, y) in enumerate(waypoints_map, start=1):
                self.get_logger().info(f'WP{idx}: map x={x:.3f}, y={y:.3f}')
                waypoints_xy.append((x, y))
        else:
            if not self._wait_for_fromll_service(wait_for_fromll):
                self.get_logger().error('fromLL service is unavailable. Is navsat_transform running?')
                return 1

            try:
                self._configure_map_offset(altitude)
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f'Anchor calibration failed: {exc}')
                return 1

            for idx, (lat, lon) in enumerate(waypoints_ll, start=1):
                try:
                    x, y = self._from_ll_to_map(lat, lon, altitude)
                    x, y = self._apply_map_offset(x, y)
                except Exception as exc:  # pylint: disable=broad-except
                    self.get_logger().error(f'Waypoint {idx} conversion failed: {exc}')
                    return 1
                self.get_logger().info(
                    f'WP{idx}: lat={lat:.8f}, lon={lon:.8f} -> map x={x:.3f}, y={y:.3f}'
                )
                waypoints_xy.append((x, y))

        mission_failed = False
        for idx, current_xy in enumerate(waypoints_xy):
            try:
                robot_x, robot_y = self._get_robot_map_xy(wait_for_tf)
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f'Cannot read robot pose in map frame: {exc}')
                return 1

            distance_to_goal = math.hypot(current_xy[0] - robot_x, current_xy[1] - robot_y)
            if distance_to_goal <= skip_waypoint_distance:
                self.get_logger().info(
                    f'WP{idx + 1} skipped: already within {skip_waypoint_distance:.2f} m '
                    f'(distance={distance_to_goal:.2f} m).'
                )
                continue

            if enforce_goal_distance:
                if distance_to_goal > max_goal_distance:
                    self.get_logger().error(
                        f'WP{idx + 1} rejected: distance_to_goal={distance_to_goal:.2f} m exceeds '
                        f'max_goal_distance_m={max_goal_distance:.2f}. Check anchor calibration.'
                    )
                    return 1

            if idx < len(waypoints_xy) - 1:
                yaw = self._compute_yaw(current_xy, waypoints_xy[idx + 1])
            else:
                yaw = 0.0

            goal = self._build_pose(current_xy[0], current_xy[1], yaw)
            self.get_logger().info(
                f'Navigating to WP{idx + 1}/{len(waypoints_xy)} '
                f'(x={current_xy[0]:.3f}, y={current_xy[1]:.3f})'
            )

            self.navigator.goToPose(goal)
            last_feedback_time = self.get_clock().now()

            while not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                if feedback is None:
                    continue

                now = self.get_clock().now()
                elapsed = (now - last_feedback_time).nanoseconds / 1e9
                if elapsed >= feedback_interval:
                    dist_rem = float(getattr(feedback, 'distance_remaining', float('nan')))
                    self.get_logger().info(
                        f'WP{idx + 1} in progress. distance_remaining={dist_rem:.3f} m'
                    )
                    last_feedback_time = now

            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(f'WP{idx + 1} reached successfully.')
            elif result == TaskResult.CANCELED:
                self.get_logger().warning(f'WP{idx + 1} navigation canceled.')
                mission_failed = True
            elif result == TaskResult.FAILED:
                self.get_logger().error(f'WP{idx + 1} navigation failed.')
                mission_failed = True
            else:
                self.get_logger().warning(f'WP{idx + 1} finished with unknown status: {result}')
                mission_failed = True

        if mission_failed:
            self.get_logger().error('Mission finished with failures.')
            return 1

        self.get_logger().info('Mission complete. All waypoints succeeded.')
        return 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IgvcWaypointNavigator()
    code = 1
    try:
        code = node.run()
    except KeyboardInterrupt:
        node.get_logger().warning('Interrupted by user (Ctrl+C).')
        code = 130
    except Exception as exc:  # pylint: disable=broad-except
        node.get_logger().error(f'Unhandled exception: {exc}')
    finally:
        try:
            node.close()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            node.destroy_node()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # pylint: disable=broad-except
            pass
    raise SystemExit(code)


if __name__ == '__main__':
    main()
