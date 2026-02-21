#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_filter_node')
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        self.get_logger().info(f"Received odometry: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}")

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

