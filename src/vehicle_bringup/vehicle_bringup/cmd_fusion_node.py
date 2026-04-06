#!/usr/bin/env python3

from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CmdFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('cmd_fusion_node')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('lane_error_topic', '/lane/error')
        self.declare_parameter('lane_heading_error_topic', '/lane/heading_error')
        self.declare_parameter('lane_confidence_topic', '/lane/confidence')
        self.declare_parameter('lane_available_topic', '/lane/available')
        self.declare_parameter('lane_valid_topic', '/lane/valid')
        self.declare_parameter('lane_predicted_topic', '/lane/predicted')
        self.declare_parameter('obstacle_bias_topic', '/obstacle/bias')
        self.declare_parameter('obstacle_speed_scale_topic', '/obstacle/speed_scale')
        self.declare_parameter('emergency_stop_topic', '/obstacle/emergency_stop')
        self.declare_parameter('obstacle_active_topic', '/obstacle/active')
        self.declare_parameter('obstacle_unknown_topic', '/obstacle/unknown')
        self.declare_parameter('waypoint_heading_hint_topic', '/guidance/heading_hint')
        self.declare_parameter('waypoint_progress_topic', '/guidance/progress')

        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('base_speed', 0.22)
        self.declare_parameter('degraded_speed', 0.14)
        self.declare_parameter('lane_lost_speed', 0.08)
        self.declare_parameter('min_conf_speed_scale', 0.35)
        self.declare_parameter('lane_kp', 0.95)
        self.declare_parameter('lane_heading_kp', 0.60)
        self.declare_parameter('obstacle_weight', 0.35)
        self.declare_parameter('obstacle_weight_during_avoid', 1.10)
        self.declare_parameter('lane_weight_during_avoid', 0.55)
        self.declare_parameter('waypoint_weight_with_lane', 0.0)
        self.declare_parameter('waypoint_weight_no_lane', 0.0)
        self.declare_parameter('max_angular_z', 0.55)
        self.declare_parameter('recovery_max_angular_z', 0.35)
        self.declare_parameter('avoid_max_angular_z', 0.72)
        self.declare_parameter('no_lane_crawl_speed', 0.06)
        self.declare_parameter('steering_smoothing_alpha', 0.35)
        self.declare_parameter('degraded_steering_smoothing_alpha', 0.18)
        self.declare_parameter('nominal_confidence_threshold', 0.60)
        self.declare_parameter('recovery_confidence_threshold', 0.30)
        self.declare_parameter('low_conf_threshold', 0.42)
        self.declare_parameter('offlane_error_threshold', 0.16)
        self.declare_parameter('offlane_recovery_gain', 0.35)
        self.declare_parameter('recent_lane_hold_sec', 1.0)
        self.declare_parameter('no_lane_memory_sec', 2.2)
        self.declare_parameter('low_conf_degraded_speed', 0.08)
        self.declare_parameter('unknown_obstacle_speed', 0.06)
        self.declare_parameter('max_steer_low_conf', 0.24)
        self.declare_parameter('single_line_conf_threshold', 0.55)
        self.declare_parameter('single_line_low_conf_speed', 0.10)
        self.declare_parameter('max_steer_single_line', 0.20)
        self.declare_parameter('no_lane_memory_speed', 0.08)
        self.declare_parameter('no_lane_memory_steer_gain', 0.65)
        self.declare_parameter('curve_speed_reduction_max', 0.0)
        self.declare_parameter('curve_heading_full_scale', 0.30)
        self.declare_parameter('curve_error_full_scale', 0.25)
        self.declare_parameter('invert_lane_error', False)
        self.declare_parameter('invert_lane_heading_error', False)
        self.declare_parameter('lane_timeout_sec', 0.6)
        self.declare_parameter('obstacle_timeout_sec', 0.6)
        self.declare_parameter('waypoint_timeout_sec', 1.5)

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        lane_error_topic = str(self.get_parameter('lane_error_topic').value)
        lane_heading_error_topic = str(self.get_parameter('lane_heading_error_topic').value)
        lane_confidence_topic = str(self.get_parameter('lane_confidence_topic').value)
        lane_available_topic = str(self.get_parameter('lane_available_topic').value)
        lane_valid_topic = str(self.get_parameter('lane_valid_topic').value)
        lane_predicted_topic = str(self.get_parameter('lane_predicted_topic').value)
        obstacle_bias_topic = str(self.get_parameter('obstacle_bias_topic').value)
        obstacle_speed_scale_topic = str(self.get_parameter('obstacle_speed_scale_topic').value)
        emergency_stop_topic = str(self.get_parameter('emergency_stop_topic').value)
        obstacle_active_topic = str(self.get_parameter('obstacle_active_topic').value)
        obstacle_unknown_topic = str(self.get_parameter('obstacle_unknown_topic').value)
        waypoint_heading_hint_topic = str(self.get_parameter('waypoint_heading_hint_topic').value)
        waypoint_progress_topic = str(self.get_parameter('waypoint_progress_topic').value)

        self.base_speed = float(self.get_parameter('base_speed').value)
        self.degraded_speed = float(self.get_parameter('degraded_speed').value)
        self.lane_lost_speed = float(self.get_parameter('lane_lost_speed').value)
        self.min_conf_speed_scale = float(self.get_parameter('min_conf_speed_scale').value)
        self.lane_kp = float(self.get_parameter('lane_kp').value)
        self.lane_heading_kp = float(self.get_parameter('lane_heading_kp').value)
        self.obstacle_weight = float(self.get_parameter('obstacle_weight').value)
        self.obstacle_weight_during_avoid = float(self.get_parameter('obstacle_weight_during_avoid').value)
        self.lane_weight_during_avoid = float(self.get_parameter('lane_weight_during_avoid').value)
        self.waypoint_weight_with_lane = float(self.get_parameter('waypoint_weight_with_lane').value)
        self.waypoint_weight_no_lane = float(self.get_parameter('waypoint_weight_no_lane').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.recovery_max_angular_z = float(self.get_parameter('recovery_max_angular_z').value)
        self.avoid_max_angular_z = float(self.get_parameter('avoid_max_angular_z').value)
        self.no_lane_crawl_speed = float(self.get_parameter('no_lane_crawl_speed').value)
        self.steering_smoothing_alpha = float(self.get_parameter('steering_smoothing_alpha').value)
        self.degraded_steering_smoothing_alpha = float(
            self.get_parameter('degraded_steering_smoothing_alpha').value
        )
        self.nominal_confidence_threshold = float(self.get_parameter('nominal_confidence_threshold').value)
        self.recovery_confidence_threshold = float(self.get_parameter('recovery_confidence_threshold').value)
        self.low_conf_threshold = float(self.get_parameter('low_conf_threshold').value)
        self.offlane_error_threshold = float(self.get_parameter('offlane_error_threshold').value)
        self.offlane_recovery_gain = float(self.get_parameter('offlane_recovery_gain').value)
        self.recent_lane_hold_ns = int(float(self.get_parameter('recent_lane_hold_sec').value) * 1e9)
        self.no_lane_memory_ns = int(float(self.get_parameter('no_lane_memory_sec').value) * 1e9)
        self.low_conf_degraded_speed = float(self.get_parameter('low_conf_degraded_speed').value)
        self.unknown_obstacle_speed = float(self.get_parameter('unknown_obstacle_speed').value)
        self.max_steer_low_conf = float(self.get_parameter('max_steer_low_conf').value)
        self.single_line_conf_threshold = float(self.get_parameter('single_line_conf_threshold').value)
        self.single_line_low_conf_speed = float(self.get_parameter('single_line_low_conf_speed').value)
        self.max_steer_single_line = float(self.get_parameter('max_steer_single_line').value)
        self.no_lane_memory_speed = float(self.get_parameter('no_lane_memory_speed').value)
        self.no_lane_memory_steer_gain = float(self.get_parameter('no_lane_memory_steer_gain').value)
        self.curve_speed_reduction_max = float(self.get_parameter('curve_speed_reduction_max').value)
        self.curve_heading_full_scale = max(
            1e-3,
            float(self.get_parameter('curve_heading_full_scale').value),
        )
        self.curve_error_full_scale = max(
            1e-3,
            float(self.get_parameter('curve_error_full_scale').value),
        )
        self.invert_lane_error = bool(self.get_parameter('invert_lane_error').value)
        self.invert_lane_heading_error = bool(self.get_parameter('invert_lane_heading_error').value)
        self.lane_timeout_ns = int(float(self.get_parameter('lane_timeout_sec').value) * 1e9)
        self.obstacle_timeout_ns = int(float(self.get_parameter('obstacle_timeout_sec').value) * 1e9)
        self.waypoint_timeout_ns = int(float(self.get_parameter('waypoint_timeout_sec').value) * 1e9)

        self.lane_error = 0.0
        self.lane_heading_error = 0.0
        self.lane_confidence = 0.0
        self.lane_available = False
        self.lane_valid = False
        self.lane_predicted = False
        self.obstacle_bias = 0.0
        self.obstacle_speed_scale = 1.0
        self.emergency_stop = False
        self.obstacle_active = False
        self.obstacle_unknown = False
        self.heading_hint = 0.0
        self.progress = 0.0

        self.lane_error_stamp_ns = 0
        self.lane_heading_stamp_ns = 0
        self.lane_conf_stamp_ns = 0
        self.lane_available_stamp_ns = 0
        self.lane_valid_stamp_ns = 0
        self.lane_predicted_stamp_ns = 0
        self.obstacle_bias_stamp_ns = 0
        self.obstacle_scale_stamp_ns = 0
        self.emergency_stamp_ns = 0
        self.obstacle_active_stamp_ns = 0
        self.obstacle_unknown_stamp_ns = 0
        self.heading_hint_stamp_ns = 0
        self.progress_stamp_ns = 0
        self.last_log_ns = 0
        self.last_reliable_lane_error = 0.0
        self.last_reliable_lane_heading_error = 0.0
        self.last_reliable_lane_stamp_ns = 0
        self.last_steering_command = 0.0

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(Float32, lane_error_topic, self.lane_error_cb, 10)
        self.create_subscription(Float32, lane_heading_error_topic, self.lane_heading_error_cb, 10)
        self.create_subscription(Float32, lane_confidence_topic, self.lane_confidence_cb, 10)
        self.create_subscription(Bool, lane_available_topic, self.lane_available_cb, 10)
        self.create_subscription(Bool, lane_valid_topic, self.lane_valid_cb, 10)
        self.create_subscription(Bool, lane_predicted_topic, self.lane_predicted_cb, 10)
        self.create_subscription(Float32, obstacle_bias_topic, self.obstacle_bias_cb, 10)
        self.create_subscription(Float32, obstacle_speed_scale_topic, self.obstacle_speed_scale_cb, 10)
        self.create_subscription(Bool, emergency_stop_topic, self.emergency_stop_cb, 10)
        self.create_subscription(Bool, obstacle_active_topic, self.obstacle_active_cb, 10)
        self.create_subscription(Bool, obstacle_unknown_topic, self.obstacle_unknown_cb, 10)
        self.create_subscription(Float32, waypoint_heading_hint_topic, self.heading_hint_cb, 10)
        self.create_subscription(Float32, waypoint_progress_topic, self.progress_cb, 10)

        control_hz = max(5.0, float(self.get_parameter('control_hz').value))
        self.create_timer(1.0 / control_hz, self.timer_cb)

    def lane_error_cb(self, msg: Float32) -> None:
        self.lane_error = float(msg.data)
        self.lane_error_stamp_ns = self.get_clock().now().nanoseconds

    def lane_heading_error_cb(self, msg: Float32) -> None:
        self.lane_heading_error = float(msg.data)
        self.lane_heading_stamp_ns = self.get_clock().now().nanoseconds

    def lane_confidence_cb(self, msg: Float32) -> None:
        self.lane_confidence = clamp(float(msg.data), 0.0, 1.0)
        self.lane_conf_stamp_ns = self.get_clock().now().nanoseconds

    def lane_available_cb(self, msg: Bool) -> None:
        self.lane_available = bool(msg.data)
        self.lane_available_stamp_ns = self.get_clock().now().nanoseconds

    def lane_valid_cb(self, msg: Bool) -> None:
        self.lane_valid = bool(msg.data)
        self.lane_valid_stamp_ns = self.get_clock().now().nanoseconds

    def lane_predicted_cb(self, msg: Bool) -> None:
        self.lane_predicted = bool(msg.data)
        self.lane_predicted_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_bias_cb(self, msg: Float32) -> None:
        self.obstacle_bias = float(msg.data)
        self.obstacle_bias_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_speed_scale_cb(self, msg: Float32) -> None:
        self.obstacle_speed_scale = clamp(float(msg.data), 0.0, 1.0)
        self.obstacle_scale_stamp_ns = self.get_clock().now().nanoseconds

    def emergency_stop_cb(self, msg: Bool) -> None:
        self.emergency_stop = bool(msg.data)
        self.emergency_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_active_cb(self, msg: Bool) -> None:
        self.obstacle_active = bool(msg.data)
        self.obstacle_active_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_unknown_cb(self, msg: Bool) -> None:
        self.obstacle_unknown = bool(msg.data)
        self.obstacle_unknown_stamp_ns = self.get_clock().now().nanoseconds

    def heading_hint_cb(self, msg: Float32) -> None:
        self.heading_hint = float(msg.data)
        self.heading_hint_stamp_ns = self.get_clock().now().nanoseconds

    def progress_cb(self, msg: Float32) -> None:
        self.progress = clamp(float(msg.data), 0.0, 1.0)
        self.progress_stamp_ns = self.get_clock().now().nanoseconds

    def is_recent(self, stamp_ns: int, timeout_ns: int, now_ns: int) -> bool:
        return stamp_ns > 0 and (now_ns - stamp_ns) <= timeout_ns

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def timer_cb(self) -> None:
        now_ns = self.get_clock().now().nanoseconds

        emergency_active = self.emergency_stop and self.is_recent(
            self.emergency_stamp_ns,
            self.obstacle_timeout_ns,
            now_ns,
        )
        if emergency_active:
            self.publish_stop()
            return

        lane_available = False
        if self.is_recent(self.lane_available_stamp_ns, self.lane_timeout_ns, now_ns):
            lane_available = self.lane_available
        elif self.is_recent(self.lane_valid_stamp_ns, self.lane_timeout_ns, now_ns):
            lane_available = self.lane_valid

        lane_error = self.lane_error if self.is_recent(self.lane_error_stamp_ns, self.lane_timeout_ns, now_ns) else 0.0
        lane_heading_error = (
            self.lane_heading_error
            if self.is_recent(self.lane_heading_stamp_ns, self.lane_timeout_ns, now_ns)
            else 0.0
        )
        lane_confidence = (
            self.lane_confidence
            if self.is_recent(self.lane_conf_stamp_ns, self.lane_timeout_ns, now_ns)
            else (0.85 if lane_available else 0.0)
        )

        obstacle_bias = (
            self.obstacle_bias
            if self.is_recent(self.obstacle_bias_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else 0.0
        )
        obstacle_speed_scale = (
            self.obstacle_speed_scale
            if self.is_recent(self.obstacle_scale_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else 1.0
        )
        obstacle_speed_scale = clamp(obstacle_speed_scale, 0.0, 1.0)
        obstacle_active = (
            self.obstacle_active
            if self.is_recent(self.obstacle_active_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )
        obstacle_unknown = (
            self.obstacle_unknown
            if self.is_recent(self.obstacle_unknown_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )

        heading_hint = (
            self.heading_hint
            if self.is_recent(self.heading_hint_stamp_ns, self.waypoint_timeout_ns, now_ns)
            else 0.0
        )

        if self.invert_lane_error:
            lane_error *= -1.0
        if self.invert_lane_heading_error:
            lane_heading_error *= -1.0

        lane_recent = (
            self.is_recent(self.lane_error_stamp_ns, self.lane_timeout_ns, now_ns)
            or self.is_recent(self.lane_valid_stamp_ns, self.lane_timeout_ns, now_ns)
        )
        predicted_recent = self.is_recent(self.lane_predicted_stamp_ns, self.lane_timeout_ns, now_ns)
        predicted_active = predicted_recent and self.lane_predicted
        if lane_available and lane_recent and lane_confidence >= self.nominal_confidence_threshold:
            self.last_reliable_lane_error = lane_error
            self.last_reliable_lane_heading_error = lane_heading_error
            self.last_reliable_lane_stamp_ns = now_ns

        waypoint_weight = (
            self.waypoint_weight_with_lane
            if lane_available and lane_confidence >= 0.35
            else self.waypoint_weight_no_lane
        )
        if obstacle_active:
            waypoint_weight = 0.0
        if obstacle_unknown:
            waypoint_weight = 0.0

        lane_weight = self.lane_weight_during_avoid if obstacle_active else 1.0
        obstacle_weight = self.obstacle_weight_during_avoid if obstacle_active else self.obstacle_weight
        lane_term = lane_weight * ((self.lane_kp * lane_error) + (self.lane_heading_kp * lane_heading_error))
        steering = lane_term + (obstacle_weight * obstacle_bias) + (waypoint_weight * heading_hint)
        speed_limit = self.base_speed
        mode = 'no_lane'
        curve_heading_ratio = clamp(abs(lane_heading_error) / self.curve_heading_full_scale, 0.0, 1.0)
        curve_error_ratio = clamp(abs(lane_error) / self.curve_error_full_scale, 0.0, 1.0)
        curve_ratio = max(curve_heading_ratio, 0.65 * curve_error_ratio)
        curve_speed_scale = 1.0 - (self.curve_speed_reduction_max * curve_ratio)
        curve_speed_scale = clamp(curve_speed_scale, 0.15, 1.0)

        if lane_available and lane_recent and lane_confidence >= self.nominal_confidence_threshold:
            mode = 'avoid' if obstacle_active else 'nominal'
            if abs(lane_error) >= self.offlane_error_threshold:
                steering += self.offlane_recovery_gain * lane_error
            steer_limit = self.avoid_max_angular_z if obstacle_active else self.max_angular_z
            steering = clamp(steering, -steer_limit, steer_limit)
            confidence_scale = clamp(max(self.min_conf_speed_scale, lane_confidence), 0.0, 1.0)
            speed = self.base_speed * confidence_scale * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif lane_available and lane_recent and lane_confidence >= self.recovery_confidence_threshold:
            mode = 'single_line_avoid' if obstacle_active else 'degraded'
            steering += self.offlane_recovery_gain * lane_error
            steer_limit = self.avoid_max_angular_z if obstacle_active else self.recovery_max_angular_z
            steering = clamp(steering, -steer_limit, steer_limit)
            confidence_scale = clamp(max(0.45, lane_confidence), 0.0, 1.0)
            speed_limit = self.degraded_speed
            speed = self.degraded_speed * confidence_scale * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif predicted_active and lane_recent:
            mode = 'predicted_avoid' if obstacle_active else 'predicted'
            steering += 0.55 * self.offlane_recovery_gain * lane_error
            steer_limit = self.avoid_max_angular_z if obstacle_active else self.recovery_max_angular_z
            steering = clamp(steering, -steer_limit, steer_limit)
            speed_limit = self.lane_lost_speed
            speed = self.lane_lost_speed * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif self.is_recent(self.last_reliable_lane_stamp_ns, self.recent_lane_hold_ns, now_ns):
            mode = 'recovery_avoid' if obstacle_active else 'recovery'
            steering = (
                self.lane_kp * self.last_reliable_lane_error
                + self.lane_heading_kp * self.last_reliable_lane_heading_error
                + 0.25 * self.offlane_recovery_gain * self.last_reliable_lane_error
                + obstacle_weight * obstacle_bias
            )
            steer_limit = self.avoid_max_angular_z if obstacle_active else self.recovery_max_angular_z
            steering = clamp(steering, -steer_limit, steer_limit)
            speed_limit = self.lane_lost_speed
            speed = self.lane_lost_speed * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif self.is_recent(self.last_reliable_lane_stamp_ns, self.no_lane_memory_ns, now_ns) and not obstacle_unknown:
            mode = 'no_lane_memory'
            steering = (
                self.no_lane_memory_steer_gain
                * (
                    self.lane_kp * self.last_reliable_lane_error
                    + self.lane_heading_kp * self.last_reliable_lane_heading_error
                )
                + obstacle_weight * obstacle_bias
            )
            steering = clamp(steering, -self.max_steer_single_line, self.max_steer_single_line)
            speed_limit = self.no_lane_memory_speed
            speed = self.no_lane_memory_speed * max(0.6, obstacle_speed_scale)
        else:
            steering = obstacle_weight * obstacle_bias
            steering = clamp(steering, -0.20, 0.20)
            speed_limit = self.no_lane_crawl_speed
            speed = self.no_lane_crawl_speed * obstacle_speed_scale
            if abs(obstacle_bias) < 0.05:
                speed = min(self.no_lane_crawl_speed, 0.04)

        single_line_low_conf = (
            mode in ('degraded', 'single_line_avoid', 'predicted', 'predicted_avoid', 'recovery', 'recovery_avoid')
            and lane_confidence < self.single_line_conf_threshold
        )
        if single_line_low_conf:
            speed_limit = min(speed_limit, self.single_line_low_conf_speed)
            speed = min(speed, self.single_line_low_conf_speed)
            steer_cap = self.max_steer_single_line if not obstacle_active else max(
                self.max_steer_single_line,
                0.26,
            )
            steering = clamp(steering, -steer_cap, steer_cap)
            if obstacle_unknown:
                mode = 'single_line_unknown'
            elif obstacle_active:
                mode = 'single_line_avoid_low_conf'
            else:
                mode = 'single_line_low_conf'

        low_conf_guard = (
            lane_recent
            and lane_confidence < self.low_conf_threshold
            and (predicted_active or lane_available)
        )
        if obstacle_unknown and low_conf_guard:
            mode = 'unknown_low_conf'
            speed_limit = min(speed_limit, self.low_conf_degraded_speed)
            speed = min(speed, self.low_conf_degraded_speed)
            steering = clamp(steering, -self.max_steer_low_conf, self.max_steer_low_conf)
        elif obstacle_unknown and not obstacle_active and not lane_available:
            mode = 'unknown_crawl'
            speed_limit = min(speed_limit, self.unknown_obstacle_speed)
            speed = min(speed, self.unknown_obstacle_speed)
            steering = clamp(steering, -self.max_steer_low_conf, self.max_steer_low_conf)

        smoothing_alpha = (
            self.steering_smoothing_alpha if mode == 'nominal' else self.degraded_steering_smoothing_alpha
        )
        smoothing_alpha = clamp(smoothing_alpha, 0.0, 1.0)
        steering = (
            smoothing_alpha * steering
            + (1.0 - smoothing_alpha) * self.last_steering_command
        )
        self.last_steering_command = steering
        speed = clamp(speed, 0.0, speed_limit)

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = steering
        self.cmd_pub.publish(cmd)

        if now_ns - self.last_log_ns >= int(1e9):
            self.last_log_ns = now_ns
            self.get_logger().info(
                f'[FUSION] mode={mode} lane_available={lane_available} lane_conf={lane_confidence:.2f} '
                f'lane_term={lane_term:+.3f} obstacle_bias={obstacle_bias:+.3f} '
                f'obstacle_active={obstacle_active} obstacle_unknown={obstacle_unknown} '
                f'heading_hint={heading_hint:+.3f} progress={self.progress:.2f} '
                f'curve_scale={curve_speed_scale:.2f} '
                f'speed={speed:.2f} steer={steering:+.3f}'
            )


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = CmdFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
