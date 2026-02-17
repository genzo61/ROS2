import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry

BASE_LAT = 41.0
BASE_LON = 29.0


class FakeGPSNode(Node):

    def __init__(self):
        super().__init__('fake_gps_node')

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(
            NavSatFix,
            '/gps/fix',
            10
        )

    def odom_callback(self, msg):

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        gps_msg = NavSatFix()

        # Basit metre → lat/lon dönüşümü
        gps_msg.latitude = BASE_LAT + (y / 111111.0)
        gps_msg.longitude = BASE_LON + (x / 111111.0)

        self.publisher.publish(gps_msg)


def main():
    rclpy.init()
    node = FakeGPSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

