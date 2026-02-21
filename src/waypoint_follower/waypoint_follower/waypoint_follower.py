import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # EKF’in global pozisyonunu dinle
        self.pose_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.pose_callback,
            10
        )

        # Robot komutları
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Waypoint listesi (x, y)
        self.waypoints = [(2.0, 2.0), (4.0, 2.0), (4.0, 4.0)]
        self.current_index = 0

    def pose_callback(self, msg):
        if self.current_index >= len(self.waypoints):
            self.get_logger().info('Tüm waypointler tamamlandı.')
            return

        # Mevcut pozisyon
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Hedef waypoint
        target_x, target_y = self.waypoints[self.current_index]

        # Hedefe uzaklık ve yön
        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)

        # Twist mesajı hazırla
        cmd = Twist()
        if distance > 0.1:  # Hedefe yakınsa dur
            cmd.linear.x = 0.5
            cmd.angular.z = angle_to_target
        else:
            self.get_logger().info(f'Waypoint {self.current_index} tamamlandı.')
            self.current_index += 1

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
