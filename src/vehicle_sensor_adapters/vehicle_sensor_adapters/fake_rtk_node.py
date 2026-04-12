import copy
import random
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String


class FakeRTKNode(Node):
    def __init__(self) -> None:
        super().__init__('fake_rtk_node')

        self.declare_parameter('rtk_status', 'FIX')
        self.declare_parameter('input_fix_topic', '/gps/fix')
        self.declare_parameter('output_fix_topic', '/vehicle/gps/fix')
        self.declare_parameter('status_topic', '/vehicle/gps/status')
        self.declare_parameter('fix_covariance_m2', 0.01)
        self.declare_parameter('float_covariance_m2', 2.5)
        self.declare_parameter('float_noise_deg', 0.00002)
        self.declare_parameter('frame_id', '')

        input_fix_topic = str(self.get_parameter('input_fix_topic').value)
        output_fix_topic = str(self.get_parameter('output_fix_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)

        self.create_subscription(NavSatFix, input_fix_topic, self.gps_callback, 10)
        self.gps_pub = self.create_publisher(NavSatFix, output_fix_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.get_logger().info(
            f'Fake RTK adapter ready: {input_fix_topic} -> {output_fix_topic}'
        )

    def gps_callback(self, msg: NavSatFix) -> None:
        status = str(self.get_parameter('rtk_status').value).strip().upper()
        output = copy.deepcopy(msg)

        frame_id = str(self.get_parameter('frame_id').value)
        if frame_id:
            output.header.frame_id = frame_id

        if status == 'FIX':
            covariance = float(self.get_parameter('fix_covariance_m2').value)
            output.status.status = NavSatStatus.STATUS_GBAS_FIX
        elif status == 'FLOAT':
            noise = float(self.get_parameter('float_noise_deg').value)
            output.latitude += random.uniform(-noise, noise)
            output.longitude += random.uniform(-noise, noise)
            covariance = float(self.get_parameter('float_covariance_m2').value)
            output.status.status = NavSatStatus.STATUS_FIX
        else:
            covariance = float(self.get_parameter('float_covariance_m2').value)
            output.status.status = NavSatStatus.STATUS_NO_FIX

        output.position_covariance = [
            covariance, 0.0, 0.0,
            0.0, covariance, 0.0,
            0.0, 0.0, covariance,
        ]
        output.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        if output.status.status != NavSatStatus.STATUS_NO_FIX:
            self.gps_pub.publish(output)

        stat_msg = String()
        stat_msg.data = status
        self.status_pub.publish(stat_msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = FakeRTKNode()
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
