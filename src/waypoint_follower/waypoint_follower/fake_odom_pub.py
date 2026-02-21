import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class FakeOdom(Node):
    def __init__(self):
        super().__init__('fake_odom_pub')
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.x = 0.0
        self.y = 0.0

    def timer_callback(self):
        msg = Odometry()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.w = 1.0
        self.pub.publish(msg)
        self.x += 0.05  # her 0.1 s’de x 0.05 artar (simülasyon)
        self.y += 0.02  # her 0.1 s’de y 0.02 artar

def main(args=None):
    rclpy.init(args=args)
    node = FakeOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

