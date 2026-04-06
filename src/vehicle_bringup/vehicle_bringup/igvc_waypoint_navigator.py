#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class IgvcWaypointNavigator(Node):
    def __init__(self) -> None:
        super().__init__('igvc_waypoint_navigator')

        self.declare_parameter('map_waypoints', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('gps_waypoints', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('skip_waypoint_distance_m', 1.0)
        self.declare_parameter('arrival_check_distance_m', 0.8)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('max_heading_error_rad', 1.2)
        self.declare_parameter('heading_hint_topic', '/guidance/heading_hint')
        self.declare_parameter('progress_topic', '/guidance/progress')

        self.global_frame = str(self.get_parameter('global_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.skip_waypoint_distance = float(self.get_parameter('skip_waypoint_distance_m').value)
        self.arrival_check_distance = float(self.get_parameter('arrival_check_distance_m').value)
        self.max_heading_error_rad = max(0.1, float(self.get_parameter('max_heading_error_rad').value))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.heading_hint_pub = self.create_publisher(
            Float32,
            str(self.get_parameter('heading_hint_topic').value),
            10,
        )
        self.progress_pub = self.create_publisher(
            Float32,
            str(self.get_parameter('progress_topic').value),
            10,
        )
        self.current_wp_pub = self.create_publisher(PoseStamped, '/igvc/current_waypoint', 10)

        self.waypoints = self.read_map_waypoints()
        self.current_index = 0
        self.last_log_ns = 0

        if not self.waypoints:
            gps_waypoints = self.read_gps_waypoints()
            if gps_waypoints:
                self.get_logger().warning(
                    'gps_waypoints present but GPS->map conversion is disabled in this minimal fusion refactor. '
                    'Use map_waypoints for race_stack.launch.py.'
                )

        publish_hz = max(2.0, float(self.get_parameter('publish_hz').value))
        self.create_timer(1.0 / publish_hz, self.timer_cb)

    def read_map_waypoints(self) -> List[Tuple[float, float]]:
        parameter = self.get_parameter_or(
            'map_waypoints',
            Parameter('map_waypoints', Parameter.Type.DOUBLE_ARRAY, []),
        )
        raw_values = parameter.value
        if raw_values is None or raw_values == []:
            return []
        values = [float(v) for v in raw_values]
        if len(values) % 2 != 0:
            raise ValueError('map_waypoints must contain an even number of x/y values.')
        return [(values[idx], values[idx + 1]) for idx in range(0, len(values), 2)]

    def read_gps_waypoints(self) -> List[Tuple[float, float]]:
        parameter = self.get_parameter_or(
            'gps_waypoints',
            Parameter('gps_waypoints', Parameter.Type.DOUBLE_ARRAY, []),
        )
        raw_values = parameter.value
        if raw_values is None or raw_values == []:
            return []
        values = [float(v) for v in raw_values]
        if len(values) % 2 != 0:
            raise ValueError('gps_waypoints must contain an even number of lat/lon values.')
        return [(values[idx], values[idx + 1]) for idx in range(0, len(values), 2)]

    def publish_zero(self) -> None:
        heading_msg = Float32()
        progress_msg = Float32()
        self.heading_hint_pub.publish(heading_msg)
        self.progress_pub.publish(progress_msg)

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        return float(translation.x), float(translation.y), yaw

    def advance_waypoint_if_needed(self, x: float, y: float) -> None:
        if not self.waypoints:
            return

        while self.current_index < len(self.waypoints) - 1:
            target_x, target_y = self.waypoints[self.current_index]
            distance = math.hypot(target_x - x, target_y - y)
            if distance > self.skip_waypoint_distance:
                break
            self.current_index += 1

        if self.current_index >= len(self.waypoints):
            self.current_index = len(self.waypoints) - 1

    def publish_current_waypoint(self, x: float, y: float) -> None:
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        self.current_wp_pub.publish(pose)

    def timer_cb(self) -> None:
        if not self.waypoints:
            self.publish_zero()
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_log_ns >= int(2e9):
                self.last_log_ns = now_ns
                self.get_logger().warning(
                    f'Waiting for TF {self.global_frame} -> {self.base_frame} before publishing heading hints.'
                )
            self.publish_zero()
            return

        x, y, yaw = robot_pose
        self.advance_waypoint_if_needed(x, y)

        target_x, target_y = self.waypoints[self.current_index]
        distance_to_target = math.hypot(target_x - x, target_y - y)
        if (
            self.current_index == len(self.waypoints) - 1
            and distance_to_target <= self.arrival_check_distance
        ):
            heading_error = 0.0
        else:
            desired_yaw = math.atan2(target_y - y, target_x - x)
            heading_error = normalize_angle(desired_yaw - yaw)

        heading_hint = clamp(
            heading_error / self.max_heading_error_rad,
            -1.0,
            1.0,
        )
        progress = float(self.current_index) / float(max(1, len(self.waypoints) - 1))

        heading_msg = Float32()
        heading_msg.data = float(heading_hint)
        self.heading_hint_pub.publish(heading_msg)

        progress_msg = Float32()
        progress_msg.data = float(progress)
        self.progress_pub.publish(progress_msg)

        self.publish_current_waypoint(target_x, target_y)

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns >= int(1e9):
            self.last_log_ns = now_ns
            self.get_logger().info(
                f'[WAYPOINT_HINT] idx={self.current_index + 1}/{len(self.waypoints)} '
                f'dist={distance_to_target:.2f} heading_hint={heading_hint:+.3f} progress={progress:.2f}'
            )


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = IgvcWaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        listener = getattr(node, 'tf_listener', None)
        if listener is not None:
            try:
                listener.unregister()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
