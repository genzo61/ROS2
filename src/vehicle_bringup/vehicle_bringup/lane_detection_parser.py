#!/usr/bin/env python3

import json
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class LaneDetectionParser(Node):
    def __init__(self) -> None:
        super().__init__('lane_detection_parser')

        self.declare_parameter('detections_topic', '/lane/detections')
        self.declare_parameter('lane_error_topic', '/lane/error')
        self.declare_parameter('lane_valid_topic', '/lane/valid')
        self.declare_parameter('left_lane_topic', '/lane/left')
        self.declare_parameter('right_lane_topic', '/lane/right')
        self.declare_parameter('lane_deviation_px_topic', '/lane/deviation_px')
        self.declare_parameter('expected_lane_width_px', 260.0)
        self.declare_parameter('smoothing_alpha', 0.35)
        self.declare_parameter('single_lane_error_scale', 1.0)
        self.declare_parameter('adaptive_lane_width_alpha', 0.12)
        self.declare_parameter('single_lane_bias_px', 14.0)
        self.declare_parameter('min_lane_width_px', 120.0)
        self.declare_parameter('max_lane_width_px', 360.0)
        self.declare_parameter('single_lane_max_abs_error', 0.22)
        self.declare_parameter('single_lane_valid_abs_error', 0.18)
        self.declare_parameter('two_lane_max_abs_error', 0.45)
        self.declare_parameter('publish_debug_logs', True)

        detections_topic = str(self.get_parameter('detections_topic').value)
        lane_error_topic = str(self.get_parameter('lane_error_topic').value)
        lane_valid_topic = str(self.get_parameter('lane_valid_topic').value)
        left_lane_topic = str(self.get_parameter('left_lane_topic').value)
        right_lane_topic = str(self.get_parameter('right_lane_topic').value)
        deviation_topic = str(self.get_parameter('lane_deviation_px_topic').value)

        self.expected_lane_width_px = float(self.get_parameter('expected_lane_width_px').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.single_lane_error_scale = float(self.get_parameter('single_lane_error_scale').value)
        self.adaptive_lane_width_alpha = float(self.get_parameter('adaptive_lane_width_alpha').value)
        self.single_lane_bias_px = float(self.get_parameter('single_lane_bias_px').value)
        self.min_lane_width_px = float(self.get_parameter('min_lane_width_px').value)
        self.max_lane_width_px = float(self.get_parameter('max_lane_width_px').value)
        self.single_lane_max_abs_error = float(self.get_parameter('single_lane_max_abs_error').value)
        self.single_lane_valid_abs_error = float(self.get_parameter('single_lane_valid_abs_error').value)
        self.two_lane_max_abs_error = float(self.get_parameter('two_lane_max_abs_error').value)
        self.publish_debug_logs = bool(self.get_parameter('publish_debug_logs').value)
        self.adaptive_lane_width_px = self.expected_lane_width_px

        self.sub = self.create_subscription(String, detections_topic, self.detections_callback, 10)
        self.error_pub = self.create_publisher(Float32, lane_error_topic, 10)
        self.valid_pub = self.create_publisher(Bool, lane_valid_topic, 10)
        self.left_pub = self.create_publisher(Float32MultiArray, left_lane_topic, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, right_lane_topic, 10)
        self.deviation_pub = self.create_publisher(Float32, deviation_topic, 10)

        self.smoothed_error: Optional[float] = None
        self.last_log_time_ns = 0

        self.get_logger().info(
            f'Lane parser ready. detections={detections_topic}, lane_error={lane_error_topic}, lane_valid={lane_valid_topic}'
        )

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def parse_detections(payload: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
        image_width = int(payload.get('image_width', 0))
        image_height = int(payload.get('image_height', 0))
        detections = payload.get('detections', [])
        if not isinstance(detections, list):
            detections = []
        return image_width, image_height, detections

    @staticmethod
    def extract_lane_box(det: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, float]]:
        bbox = det.get('bbox_xywh')
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            cx = float(bbox[0])
            cy = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
            conf = float(det.get('confidence', 0.0))
            return cx, cy, w, h, conf
        except (TypeError, ValueError):
            return None

    def select_lane_candidates(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[Tuple[float, float, float, float, float]]]:
        left_best = None
        right_best = None
        left_score = -1.0
        right_score = -1.0

        for det in detections:
            side = str(det.get('side', '')).lower()
            lane_box = self.extract_lane_box(det)
            if lane_box is None:
                continue
            _, cy, _, h, conf = lane_box
            # Prefer detections closer to the bottom (near field), they are more reliable for control.
            bottom_y = cy + 0.5 * h
            score = conf + 0.002 * bottom_y + 0.0002 * h
            if side == 'left' and score > left_score:
                left_best = lane_box
                left_score = score
            elif side == 'right' and score > right_score:
                right_best = lane_box
                right_score = score

        return left_best, right_best

    def publish_lane_box(self, pub, lane_box: Optional[Tuple[float, float, float, float, float]]) -> None:
        msg = Float32MultiArray()
        if lane_box is None:
            msg.data = []
        else:
            cx, cy, w, h, conf = lane_box
            msg.data = [float(cx), float(cy), float(w), float(h), float(conf)]
        pub.publish(msg)

    def detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        image_width, _, detections = self.parse_detections(payload)
        if image_width <= 0:
            return

        left_lane, right_lane = self.select_lane_candidates(detections)

        lane_valid = left_lane is not None or right_lane is not None
        both_lanes = left_lane is not None and right_lane is not None
        image_center = 0.5 * image_width
        lane_center = image_center

        if left_lane is not None and right_lane is not None:
            measured_lane_width = abs(right_lane[0] - left_lane[0])
            if self.min_lane_width_px <= measured_lane_width <= self.max_lane_width_px:
                alpha_w = self.clamp(self.adaptive_lane_width_alpha, 0.0, 1.0)
                self.adaptive_lane_width_px = (
                    (1.0 - alpha_w) * self.adaptive_lane_width_px + alpha_w * measured_lane_width
                )
            lane_center = 0.5 * (left_lane[0] + right_lane[0])
        elif left_lane is not None:
            lane_center = left_lane[0] + 0.5 * self.adaptive_lane_width_px + self.single_lane_bias_px
        elif right_lane is not None:
            lane_center = right_lane[0] - 0.5 * self.adaptive_lane_width_px - self.single_lane_bias_px

        lane_center = self.clamp(lane_center, 0.0, float(image_width))

        deviation_px = image_center - lane_center
        lane_error = deviation_px / max(1.0, image_center)
        if not both_lanes:
            lane_error *= self.single_lane_error_scale
            lane_error = self.clamp(lane_error, -self.single_lane_max_abs_error, self.single_lane_max_abs_error)
            if abs(lane_error) > self.single_lane_valid_abs_error:
                lane_valid = False
        else:
            lane_error = self.clamp(lane_error, -self.two_lane_max_abs_error, self.two_lane_max_abs_error)
        lane_error = self.clamp(lane_error, -1.0, 1.0)

        if self.smoothed_error is None:
            self.smoothed_error = lane_error
        else:
            alpha = self.clamp(self.smoothing_alpha, 0.0, 1.0)
            self.smoothed_error = alpha * lane_error + (1.0 - alpha) * self.smoothed_error

        error_msg = Float32()
        error_msg.data = float(self.smoothed_error if self.smoothed_error is not None else lane_error)
        self.error_pub.publish(error_msg)

        valid_msg = Bool()
        valid_msg.data = bool(lane_valid)
        self.valid_pub.publish(valid_msg)

        dev_msg = Float32()
        dev_msg.data = float(deviation_px)
        self.deviation_pub.publish(dev_msg)

        self.publish_lane_box(self.left_pub, left_lane)
        self.publish_lane_box(self.right_pub, right_lane)

        if self.publish_debug_logs:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_log_time_ns > int(1e9):
                self.last_log_time_ns = now_ns
                self.get_logger().info(
                    f'lane_valid={lane_valid} err={error_msg.data:+.3f} dev_px={dev_msg.data:+.1f} '
                    f'left={left_lane is not None} right={right_lane is not None}'
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneDetectionParser()
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
