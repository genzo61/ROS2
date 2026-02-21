import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class FakeGPS(Node):
    def __init__(self):
        super().__init__('fake_gps_pub')
        self.pub = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz

    def timer_callback(self):
        msg = NavSatFix()
        msg.latitude = 41.0
        msg.longitude = 29.0
        msg.altitude = 0.0
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeGPS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
