import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
import random

class FakeRTKNode(Node):
    def __init__(self):
        super().__init__('fake_rtk_node')
        self.declare_parameter('rtk_status', 'FIX') 
        
        # Gazebo'dan gelen GPS topic ismini buraya yaz (Genelde /gps/fix olur, pluginine bagli)
        # Eger senin robotta GPS plugini yoksa bunu daha sonra ekleriz.
        # Simdilik varsayilan /gps/fix dinliyor.
        self.sim_gps_topic = '/gps/fix' 
        self.vehicle_gps_topic = '/vehicle/gps/fix'
        self.status_topic = '/vehicle/gps/status'

        self.create_subscription(NavSatFix, self.sim_gps_topic, self.gps_callback, 10)
        self.gps_pub = self.create_publisher(NavSatFix, self.vehicle_gps_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.get_logger().info('Fake RTK Node Hazir.')

    def gps_callback(self, msg):
        status = self.get_parameter('rtk_status').get_parameter_value().string_value
        if status == 'FIX':
            msg.position_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            msg.status.status = 0 
            self.gps_pub.publish(msg)
        elif status == 'FLOAT':
            noise = random.uniform(-0.00002, 0.00002)
            msg.latitude += noise
            msg.longitude += noise
            msg.position_covariance = [2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5]
            msg.status.status = 1 
            self.gps_pub.publish(msg)
        
        stat_msg = String()
        stat_msg.data = status
        self.status_pub.publish(stat_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeRTKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
