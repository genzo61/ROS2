#!/usr/bin/env python3

from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class LaneCameraSubscriber(Node):
    def __init__(self) -> None:
        super().__init__('lane_camera_subscriber')

        self.declare_parameter('input_image_topic', '/front_camera/image_raw')
        self.declare_parameter(
            'input_topic_candidates',
            [
                '/front_camera/image_raw',
                '/camera/image_raw',
                '/teknofest_araci/front_camera/image_raw',
                '/teknofest_araci/camera/image_raw',
            ],
        )
        self.declare_parameter('auto_discover_image_topic', True)
        self.declare_parameter('watchdog_seconds', 2.5)
        self.declare_parameter('topic_scan_period_seconds', 2.0)
        self.declare_parameter('output_image_topic', '/lane/camera/image_raw')
        self.declare_parameter('ignored_topic_prefixes', ['/lane/'])

        self.input_image_topic = str(self.get_parameter('input_image_topic').value)
        raw_candidates = self.get_parameter('input_topic_candidates').value
        self.input_topic_candidates = [str(t) for t in raw_candidates] if isinstance(raw_candidates, list) else []
        self.auto_discover = bool(self.get_parameter('auto_discover_image_topic').value)
        self.watchdog_seconds = float(self.get_parameter('watchdog_seconds').value)
        self.topic_scan_period = float(self.get_parameter('topic_scan_period_seconds').value)
        self.output_image_topic = str(self.get_parameter('output_image_topic').value)
        raw_ignored_prefixes = self.get_parameter('ignored_topic_prefixes').value
        self.ignored_topic_prefixes = (
            [str(prefix) for prefix in raw_ignored_prefixes] if isinstance(raw_ignored_prefixes, list) else []
        )

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(Image, self.output_image_topic, self.sensor_qos)
        self.sub = None
        self.last_image_ns: Optional[int] = None
        self.no_image_warned = False

        self.set_subscription(self.input_image_topic)
        self.discovery_timer = None
        if self.auto_discover:
            self.discovery_timer = self.create_timer(max(0.5, self.topic_scan_period), self.discover_topic)

        self.get_logger().info(
            f'Lane camera bridge ready. in={self.input_image_topic} out={self.output_image_topic}'
        )

    def set_subscription(self, topic_name: str) -> None:
        if self.sub is not None:
            self.destroy_subscription(self.sub)
            self.sub = None
        self.sub = self.create_subscription(Image, topic_name, self.image_callback, self.sensor_qos)
        self.input_image_topic = topic_name
        self.get_logger().info(f'Subscribed camera topic: {self.input_image_topic}')

    @staticmethod
    def has_image_type(topic_types: List[str]) -> bool:
        return any(topic_type in ('sensor_msgs/msg/Image', 'sensor_msgs/Image') for topic_type in topic_types)

    def is_ignored_topic(self, topic_name: str) -> bool:
        if topic_name == self.output_image_topic:
            return True
        return any(topic_name.startswith(prefix) for prefix in self.ignored_topic_prefixes)

    def has_publishers(self, topic_name: str) -> bool:
        try:
            return len(self.get_publishers_info_by_topic(topic_name)) > 0
        except Exception:
            return False

    def discover_topic(self) -> None:
        if self.last_image_ns is not None:
            age_s = (self.get_clock().now().nanoseconds - self.last_image_ns) / 1e9
            if age_s <= self.watchdog_seconds:
                self.no_image_warned = False
                return

        image_topics = []
        for topic_name, topic_types in self.get_topic_names_and_types():
            if self.has_image_type(topic_types):
                if not self.is_ignored_topic(topic_name):
                    image_topics.append(topic_name)

        if not image_topics:
            if not self.no_image_warned:
                self.get_logger().warn('No image topic detected yet for lane camera bridge.')
                self.no_image_warned = True
            return

        ordered = []
        for topic in self.input_topic_candidates:
            if topic in image_topics and topic not in ordered:
                ordered.append(topic)
        for topic in image_topics:
            if topic not in ordered and ('camera' in topic or 'image' in topic):
                ordered.append(topic)
        for topic in image_topics:
            if topic not in ordered:
                ordered.append(topic)

        live_topics = [topic for topic in ordered if self.has_publishers(topic)]
        if not live_topics:
            if not self.no_image_warned:
                self.get_logger().warn(
                    f'No live camera publisher found yet. Current={self.input_image_topic}'
                )
                self.no_image_warned = True
            return

        best_topic = self.input_image_topic
        if self.input_image_topic not in live_topics:
            best_topic = live_topics[0]
        elif self.input_image_topic in self.input_topic_candidates:
            # Stay on preferred topic if it is alive.
            best_topic = self.input_image_topic
        elif len(live_topics) > 1:
            current_idx = live_topics.index(self.input_image_topic)
            best_topic = live_topics[(current_idx + 1) % len(live_topics)]

        if best_topic != self.input_image_topic:
            self.get_logger().warn(f'No camera data on {self.input_image_topic}, switching to {best_topic}')
            self.set_subscription(best_topic)

        self.no_image_warned = False

    def image_callback(self, msg: Image) -> None:
        self.last_image_ns = self.get_clock().now().nanoseconds
        self.no_image_warned = False
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneCameraSubscriber()
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
