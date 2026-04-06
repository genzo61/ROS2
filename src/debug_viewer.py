#!/usr/bin/env python3

import signal
import sys
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class DebugViewer(Node):
    def __init__(self) -> None:
        super().__init__('debug_viewer')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.drivable_window = 'Drivable Mask'
        self.bev_window = 'BEV Debug'

        self.latest_drivable: Optional[np.ndarray] = None
        self.latest_bev: Optional[np.ndarray] = None

        self.create_subscription(
            Image,
            '/perception/drivable_mask',
            self.drivable_callback,
            qos,
        )
        self.create_subscription(
            Image,
            '/drivable_area/debug_bev',
            self.bev_callback,
            qos,
        )

        self.create_timer(1.0 / 30.0, self.render)
        self.get_logger().info(
            'Debug viewer ready. '
            'Topics: /perception/drivable_mask, /drivable_area/debug_bev'
        )

    @staticmethod
    def image_to_numpy(msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.lower()

        if encoding in ('mono8', '8uc1'):
            channels = 1
        elif encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif encoding in ('bgra8', 'rgba8'):
            channels = 4
        else:
            return None

        expected_step = msg.width * channels
        if msg.step < expected_step:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))
        pixels = rows[:, :expected_step]

        if channels == 1:
            return pixels.reshape((msg.height, msg.width))

        image = pixels.reshape((msg.height, msg.width, channels))
        if encoding == 'bgr8':
            return image
        if encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return None

    def drivable_callback(self, msg: Image) -> None:
        image = self.image_to_numpy(msg)
        if image is None:
            self.get_logger().warn(f'Unsupported drivable mask encoding: {msg.encoding}')
            return
        self.latest_drivable = image

    def bev_callback(self, msg: Image) -> None:
        image = self.image_to_numpy(msg)
        if image is None:
            self.get_logger().warn(f'Unsupported BEV debug encoding: {msg.encoding}')
            return
        self.latest_bev = image

    def render(self) -> None:
        if self.latest_drivable is not None:
            cv2.imshow(self.drivable_window, self.latest_drivable)

        if self.latest_bev is not None:
            cv2.imshow(self.bev_window, self.latest_bev)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            raise KeyboardInterrupt


def main() -> None:
    rclpy.init()
    node = DebugViewer()

    def handle_signal(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
