#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from typing import List, Optional


class LaneTracker(Node):
    def __init__(self) -> None:
        super().__init__('lane_tracker')

        self.declare_parameter('image_topic', '/front_camera/image_raw')
        self.declare_parameter(
            'image_topic_candidates',
            [
                '/front_camera/image_raw',
                '/camera/image_raw',
                '/teknofest_araci/front_camera/image_raw',
                '/teknofest_araci/camera/image_raw',
            ],
        )
        self.declare_parameter('auto_discover_image_topic', True)
        self.declare_parameter('image_watchdog_seconds', 2.5)
        self.declare_parameter('topic_scan_period_seconds', 2.0)
        self.declare_parameter('lane_error_topic', '/lane/error')
        self.declare_parameter('lane_valid_topic', '/lane/valid')
        self.declare_parameter('debug_image_topic', '/lane/debug')
        self.declare_parameter('publish_debug', True)

        self.declare_parameter('roi_top_ratio', 0.55)
        self.declare_parameter('white_value_min', 170)
        self.declare_parameter('white_sat_max', 80)
        self.declare_parameter('min_peak_pixels', 160)
        self.declare_parameter('expected_lane_width_px', 260)
        self.declare_parameter('smoothing_alpha', 0.35)

        self.image_topic = str(self.get_parameter('image_topic').value)
        raw_candidates = self.get_parameter('image_topic_candidates').value
        if isinstance(raw_candidates, list):
            self.image_topic_candidates = [str(topic) for topic in raw_candidates]
        else:
            self.image_topic_candidates = [self.image_topic]
        self.auto_discover_image_topic = bool(self.get_parameter('auto_discover_image_topic').value)
        self.image_watchdog_seconds = float(self.get_parameter('image_watchdog_seconds').value)
        topic_scan_period_seconds = float(self.get_parameter('topic_scan_period_seconds').value)

        lane_error_topic = str(self.get_parameter('lane_error_topic').value)
        lane_valid_topic = str(self.get_parameter('lane_valid_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.publish_debug = bool(self.get_parameter('publish_debug').value)

        self.roi_top_ratio = float(self.get_parameter('roi_top_ratio').value)
        self.white_value_min = int(self.get_parameter('white_value_min').value)
        self.white_sat_max = int(self.get_parameter('white_sat_max').value)
        self.min_peak_pixels = int(self.get_parameter('min_peak_pixels').value)
        self.expected_lane_width_px = int(self.get_parameter('expected_lane_width_px').value)
        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)

        self.smoothed_error = None

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.image_sub = None
        self.last_image_msg_ns: Optional[int] = None
        self.no_image_warned = False

        self.set_image_subscription(self.image_topic)
        self.error_pub = self.create_publisher(Float32, lane_error_topic, 10)
        self.valid_pub = self.create_publisher(Bool, lane_valid_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.publish_lane_state(0.0, False)

        self.discovery_timer = None
        if self.auto_discover_image_topic:
            self.discovery_timer = self.create_timer(
                max(0.5, topic_scan_period_seconds),
                self.discover_image_topic,
            )

        self.get_logger().info(f'Lane tracker ready. image_topic={self.image_topic}')

    def set_image_subscription(self, topic_name: str) -> None:
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

        self.image_sub = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            self.sensor_qos,
        )
        self.image_topic = topic_name
        self.get_logger().info(f'Subscribed image topic: {self.image_topic}')

    @staticmethod
    def has_image_type(topic_types: List[str]) -> bool:
        return any(
            topic_type in ('sensor_msgs/msg/Image', 'sensor_msgs/Image')
            for topic_type in topic_types
        )

    def discover_image_topic(self) -> None:
        # Keep current topic if images are arriving.
        if self.last_image_msg_ns is not None:
            now_ns = self.get_clock().now().nanoseconds
            age_s = (now_ns - self.last_image_msg_ns) / 1e9
            if age_s <= self.image_watchdog_seconds:
                self.no_image_warned = False
                return

        available = self.get_topic_names_and_types()
        image_topics = [
            topic_name
            for topic_name, topic_types in available
            if self.has_image_type(topic_types)
        ]

        if not image_topics:
            if not self.no_image_warned:
                self.get_logger().warn(
                    'No Image topic available yet. Ensure yaris_autonomy.launch.py is running and Gazebo camera is loaded.'
                )
                self.no_image_warned = True
            return

        ordered_topics: List[str] = []
        for topic in self.image_topic_candidates:
            if topic in image_topics and topic not in ordered_topics:
                ordered_topics.append(topic)

        for topic in image_topics:
            if topic not in ordered_topics and ('camera' in topic or 'image' in topic):
                ordered_topics.append(topic)

        for topic in image_topics:
            if topic not in ordered_topics:
                ordered_topics.append(topic)

        # If no image is flowing, do not stick forever to the first listed topic.
        if self.image_topic not in ordered_topics:
            best_topic = ordered_topics[0]
        else:
            current_idx = ordered_topics.index(self.image_topic)
            if len(ordered_topics) == 1:
                best_topic = self.image_topic
            else:
                best_topic = ordered_topics[(current_idx + 1) % len(ordered_topics)]

        if best_topic != self.image_topic:
            self.get_logger().warn(
                f'Image data missing on {self.image_topic}, switching to detected topic {best_topic}'
            )
            self.set_image_subscription(best_topic)

        self.no_image_warned = False

    def publish_lane_state(self, lane_error: float, lane_valid: bool) -> None:
        error_msg = Float32()
        error_msg.data = float(self.clamp(lane_error, -1.0, 1.0))
        self.error_pub.publish(error_msg)

        valid_msg = Bool()
        valid_msg.data = bool(lane_valid)
        self.valid_pub.publish(valid_msg)

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def image_to_bgr(self, msg: Image):
        encoding = msg.encoding.lower()

        if encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif encoding in ('bgra8', 'rgba8'):
            channels = 4
        elif encoding == 'mono8':
            channels = 1
        else:
            return None

        expected_step = msg.width * channels
        if msg.step < expected_step:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))
        pixels = rows[:, :expected_step]

        if channels == 1:
            gray = pixels.reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = pixels.reshape((msg.height, msg.width, channels))
        if encoding == 'bgr8':
            return img
        if encoding == 'rgb8':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return None

    @staticmethod
    def bgr_to_image_msg(frame: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        return msg

    def find_lane_center(self, mask: np.ndarray):
        height, width = mask.shape
        half = width // 2

        # Histogram over lower half of ROI where lane lines are strongest.
        hist = np.sum(mask[height // 2 :, :] > 0, axis=0)

        left_idx = int(np.argmax(hist[:half]))
        right_idx = int(np.argmax(hist[half:])) + half

        left_ok = int(hist[left_idx]) >= self.min_peak_pixels
        right_ok = int(hist[right_idx]) >= self.min_peak_pixels

        if left_ok and right_ok and (right_idx - left_idx) > 40:
            center = 0.5 * (left_idx + right_idx)
            return center, float(left_idx), float(right_idx)

        if left_ok:
            center = left_idx + 0.5 * self.expected_lane_width_px
            center = self.clamp(center, 0.0, float(width - 1))
            return center, float(left_idx), None

        if right_ok:
            center = right_idx - 0.5 * self.expected_lane_width_px
            center = self.clamp(center, 0.0, float(width - 1))
            return center, None, float(right_idx)

        return None, None, None

    def image_callback(self, msg: Image) -> None:
        self.last_image_msg_ns = self.get_clock().now().nanoseconds
        self.no_image_warned = False

        frame = self.image_to_bgr(msg)
        if frame is None:
            return

        height, width, _ = frame.shape
        roi_top = int(self.clamp(self.roi_top_ratio, 0.1, 0.9) * height)
        roi = frame[roi_top:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.white_value_min),
            (180, self.white_sat_max, 255),
        )
        yellow_mask = cv2.inRange(hsv, (15, 40, 80), (40, 255, 255))
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        center_px, left_px, right_px = self.find_lane_center(mask)
        lane_valid = center_px is not None

        lane_error = 0.0
        if lane_valid:
            raw_error = ((width * 0.5) - center_px) / (width * 0.5)
            if self.smoothed_error is None:
                self.smoothed_error = raw_error
            else:
                alpha = self.clamp(self.smoothing_alpha, 0.0, 1.0)
                self.smoothed_error = alpha * raw_error + (1.0 - alpha) * self.smoothed_error
            lane_error = float(self.smoothed_error)
        else:
            if self.smoothed_error is not None:
                self.smoothed_error *= 0.95
                lane_error = float(self.smoothed_error)

        self.publish_lane_state(lane_error, lane_valid)

        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            debug = frame.copy()
            cv2.rectangle(debug, (0, roi_top), (width - 1, height - 1), (255, 255, 0), 1)

            overlay = np.zeros_like(roi)
            overlay[:, :, 1] = mask
            debug_roi = debug[roi_top:, :]
            cv2.addWeighted(overlay, 0.35, debug_roi, 0.65, 0.0, debug_roi)

            image_center = int(width * 0.5)
            cv2.line(debug, (image_center, roi_top), (image_center, height - 1), (255, 0, 0), 2)

            if center_px is not None:
                cx = int(center_px)
                cv2.line(debug, (cx, roi_top), (cx, height - 1), (0, 255, 0), 2)
            if left_px is not None:
                lx = int(left_px)
                cv2.line(debug, (lx, roi_top), (lx, height - 1), (0, 200, 200), 1)
            if right_px is not None:
                rx = int(right_px)
                cv2.line(debug, (rx, roi_top), (rx, height - 1), (0, 200, 200), 1)

            text = f'valid={lane_valid} err={self.clamp(lane_error, -1.0, 1.0):+.3f}'
            cv2.putText(debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.debug_pub.publish(self.bgr_to_image_msg(debug, msg.header))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneTracker()
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
