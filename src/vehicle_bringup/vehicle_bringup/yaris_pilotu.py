#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import sensor_msgs_py.point_cloud2 as pc2


# User-provided golden route
ROTA: List[Tuple[float, float]] = [
    (17.081, 8.626), (17.076, 8.324), (17.062, 8.021), (17.063, 7.718), (17.096, 7.417),
    (17.137, 7.120), (17.180, 6.819), (17.221, 6.521), (17.258, 6.221), (17.285, 5.921),
    (17.303, 5.620), (17.312, 5.318), (17.318, 5.017), (17.323, 4.716), (17.329, 4.415),
    (17.334, 4.114), (17.340, 3.813), (17.340, 3.511), (17.332, 3.210), (17.314, 2.909),
    (17.294, 2.608), (17.275, 2.308), (17.263, 2.008), (17.261, 1.708), (17.260, 1.407),
    (17.258, 1.102), (17.257, 0.798), (17.256, 0.494), (17.255, 0.189), (17.254, -0.115),
    (17.253, -0.419), (17.253, -0.722), (17.252, -1.027), (17.252, -1.332), (17.251, -1.636),
    (17.251, -1.941), (17.250, -2.246), (17.250, -2.548), (17.249, -2.852), (17.249, -3.157),
    (17.248, -3.466), (17.248, -3.766), (17.247, -4.066), (17.247, -4.366), (17.246, -4.666),
    (17.246, -4.966), (17.246, -5.275), (17.253, -5.585), (17.267, -5.894), (17.287, -6.202),
    (17.313, -6.511), (17.339, -6.810), (17.366, -7.109), (17.394, -7.417), (17.420, -7.716),
    (17.447, -8.015), (17.474, -8.318), (17.501, -8.620), (17.528, -8.924), (17.555, -9.223),
    (17.581, -9.522), (17.608, -9.821), (17.626, -10.122), (17.633, -10.424), (17.619, -10.726),
    (17.570, -11.023), (17.486, -11.312), (17.370, -11.589), (17.216, -11.848), (17.031, -12.085),
    (16.819, -12.298), (16.575, -12.479), (16.303, -12.613), (16.010, -12.680), (15.728, -12.576),
    (15.449, -12.460), (15.176, -12.336), (14.899, -12.218), (14.614, -12.121), (14.316, -12.074),
    (14.017, -12.118), (13.733, -12.218), (13.450, -12.328), (13.170, -12.437), (12.885, -12.537),
    (12.594, -12.623), (12.300, -12.685), (11.995, -12.676), (11.701, -12.599), (11.439, -12.447),
    (11.220, -12.240), (11.046, -11.992), (10.912, -11.722), (10.791, -11.447), (10.668, -11.167),
    (10.547, -10.893), (10.426, -10.618), (10.304, -10.341), (10.181, -10.067), (10.048, -9.792),
    (9.894, -9.529), (9.720, -9.279), (9.522, -9.047), (9.297, -8.842), (9.049, -8.671),
    (8.780, -8.528), (8.500, -8.413), (8.209, -8.325), (7.907, -8.272), (7.607, -8.247),
    (7.307, -8.225), (7.007, -8.203), (6.707, -8.181), (6.407, -8.158), (6.106, -8.136),
    (5.806, -8.114), (5.506, -8.092), (5.206, -8.070), (4.906, -8.048), (4.605, -8.026),
    (4.305, -8.004), (4.006, -7.987), (3.705, -7.975), (3.404, -7.963), (3.103, -7.950),
    (2.803, -7.938), (2.502, -7.926), (2.201, -7.914), (1.900, -7.902), (1.600, -7.890),
    (1.299, -7.877), (0.998, -7.865), (0.697, -7.853), (0.396, -7.841), (0.096, -7.829),
    (-0.205, -7.816), (-0.506, -7.805), (-0.806, -7.800), (-1.106, -7.802), (-1.407, -7.806),
    (-1.708, -7.809), (-2.009, -7.812), (-2.310, -7.817), (-2.610, -7.828), (-2.915, -7.849),
    (-3.220, -7.887), (-3.516, -7.936), (-3.811, -7.991), (-4.107, -8.048), (-4.404, -8.101),
    (-4.701, -8.150), (-4.998, -8.200), (-5.295, -8.249), (-5.592, -8.299), (-5.889, -8.349),
    (-6.186, -8.398), (-6.482, -8.449), (-6.776, -8.508), (-7.075, -8.575), (-7.363, -8.664),
    (-7.636, -8.804), (-7.870, -8.996), (-8.079, -9.218), (-8.260, -9.462), (-8.412, -9.721),
    (-8.538, -9.997), (-8.636, -10.286), (-8.690, -10.587), (-8.685, -10.891), (-8.632, -11.192),
    (-8.569, -11.486), (-8.514, -11.785), (-8.492, -12.088), (-8.544, -12.388), (-8.682, -12.660),
    (-8.869, -12.896), (-9.085, -13.107), (-9.326, -13.291), (-9.587, -13.446), (-9.864, -13.576),
    (-10.146, -13.679), (-10.435, -13.761), (-10.736, -13.814), (-11.039, -13.826), (-11.340, -13.795),
    (-11.633, -13.721), (-11.913, -13.610), (-12.178, -13.465), (-12.427, -13.294), (-12.659, -13.101),
    (-12.869, -12.884), (-13.054, -12.645), (-13.215, -12.388), (-13.353, -12.117), (-13.480, -11.844),
    (-13.615, -11.569), (-13.764, -11.301), (-13.921, -11.045), (-14.087, -10.788), (-14.255, -10.538),
    (-14.422, -10.288), (-14.580, -10.029), (-14.715, -9.754), (-14.817, -9.464), (-14.894, -9.171),
    (-14.919, -8.870), (-14.870, -8.573), (-14.781, -8.284), (-14.680, -7.999), (-14.580, -7.715),
    (-14.487, -7.425), (-14.414, -7.128), (-14.369, -6.826), (-14.372, -6.520), (-14.400, -6.221),
    (-14.429, -5.921), (-14.457, -5.621), (-14.479, -5.320), (-14.491, -5.019), (-14.495, -4.717),
    (-14.490, -4.415), (-14.476, -4.114), (-14.453, -3.813), (-14.421, -3.513), (-14.381, -3.213),
    (-14.337, -2.916), (-14.296, -2.618), (-14.263, -2.320), (-14.240, -2.020), (-14.225, -1.720),
    (-14.210, -1.419), (-14.195, -1.119), (-14.181, -0.818), (-14.166, -0.517), (-14.152, -0.217),
    (-14.137, 0.084), (-14.122, 0.385), (-14.108, 0.685), (-14.093, 0.986), (-14.079, 1.286),
    (-14.064, 1.587), (-14.049, 1.888), (-14.035, 2.188), (-14.020, 2.489), (-14.006, 2.790),
    (-13.991, 3.090), (-13.977, 3.391), (-13.962, 3.692), (-13.947, 3.997), (-13.933, 4.300),
    (-13.926, 4.604), (-13.927, 4.907), (-13.935, 5.210), (-13.951, 5.513), (-13.975, 5.815),
    (-14.004, 6.118), (-14.033, 6.421), (-14.063, 6.723), (-14.095, 7.025), (-14.134, 7.325),
    (-14.181, 7.624), (-14.243, 7.921), (-14.311, 8.217), (-14.379, 8.513), (-14.443, 8.812),
    (-14.493, 9.115), (-14.522, 9.419), (-14.505, 9.724), (-14.441, 10.021), (-14.341, 10.308),
    (-14.214, 10.586), (-14.068, 10.853), (-13.903, 11.109), (-13.720, 11.352), (-13.525, 11.587),
    (-13.321, 11.812), (-13.106, 12.028), (-12.882, 12.235), (-12.650, 12.433), (-12.412, 12.622),
    (-12.167, 12.804), (-11.914, 12.975), (-11.646, 13.126), (-11.369, 13.243), (-11.073, 13.324),
    (-10.769, 13.345), (-10.471, 13.296), (-10.189, 13.187), (-9.923, 13.039), (-9.672, 12.866),
    (-9.444, 12.665), (-9.248, 12.432), (-9.086, 12.172), (-8.961, 11.893), (-8.897, 11.593),
    (-8.909, 11.289), (-8.971, 10.991), (-9.039, 10.695), (-9.091, 10.398), (-9.097, 10.091),
    (-9.034, 9.793), (-8.931, 9.511), (-8.804, 9.239), (-8.657, 8.976), (-8.491, 8.724),
    (-8.306, 8.486), (-8.100, 8.266), (-7.874, 8.067), (-7.623, 7.889), (-7.360, 7.743),
    (-7.077, 7.634), (-6.773, 7.588), (-6.469, 7.634), (-6.193, 7.759), (-5.936, 7.918),
    (-5.681, 8.084), (-5.423, 8.246), (-5.152, 8.391), (-4.877, 8.511), (-4.584, 8.608),
    (-4.286, 8.668), (-3.984, 8.680), (-3.686, 8.633), (-3.395, 8.550), (-3.106, 8.455),
    (-2.817, 8.360), (-2.528, 8.265), (-2.240, 8.170), (-1.951, 8.075), (-1.662, 7.980),
    (-1.373, 7.884), (-1.085, 7.789), (-0.796, 7.694), (-0.507, 7.599), (-0.218, 7.505),
    (0.071, 7.419), (0.365, 7.346), (0.663, 7.288), (0.962, 7.238), (1.263, 7.205),
    (1.565, 7.185), (1.868, 7.174), (2.171, 7.171), (2.475, 7.175), (2.778, 7.188),
    (3.080, 7.208), (3.382, 7.236), (3.682, 7.271), (3.980, 7.321), (4.277, 7.383),
    (4.574, 7.448), (4.871, 7.514), (5.168, 7.579), (5.465, 7.645), (5.762, 7.710),
    (6.058, 7.776), (6.355, 7.841), (6.652, 7.907), (6.949, 7.972), (7.246, 8.038),
    (7.543, 8.103), (7.839, 8.170), (8.133, 8.242), (8.427, 8.321), (8.719, 8.401),
    (9.010, 8.499), (9.287, 8.618), (9.555, 8.763), (9.800, 8.946), (10.015, 9.166),
    (10.189, 9.419), (10.316, 9.695), (10.390, 9.989), (10.420, 10.290), (10.436, 10.593),
    (10.457, 10.898), (10.504, 11.198), (10.596, 11.486), (10.727, 11.759), (10.894, 12.013),
    (11.090, 12.246), (11.314, 12.453), (11.566, 12.626), (11.841, 12.756), (12.135, 12.845),
    (12.438, 12.851), (12.732, 12.767), (12.996, 12.617), (13.243, 12.439), (13.484, 12.253),
    (13.725, 12.068), (13.972, 11.895), (14.230, 11.738), (14.498, 11.596), (14.772, 11.467),
    (15.046, 11.334), (15.314, 11.188), (15.565, 11.016), (15.794, 10.816), (15.992, 10.586),
    (16.157, 10.328), (16.277, 10.053), (16.346, 9.755), (16.337, 9.453), (16.252, 9.164),
]


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class YarisPilotu(Node):
    def __init__(self) -> None:
        super().__init__('yaris_pilotu')

        self.rota = ROTA
        self.hedef_index = 0
        self.tamamlandi = False

        # Driving settings
        self.gps_hiz = 2.0
        self.lookahead_dist = 0.9
        self.yaw_k = 1.5
        self.keskin_viraj_hiz = 0.4
        self.keskin_viraj_esik = 0.4

        # Obstacle settings (PointCloud)
        self.duba_algilama_mesafesi = 1.3
        self.duba_kacis_sertligi = 2.5
        self.duba_min_z = 0.05
        self.duba_y_sinir = 0.6
        self.duba_min_nokta = 5
        self.duba_hiz = 0.3

        self.duba_var = False
        self.duba_konumu = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Lane settings (camera lane tracker feedback)
        self.lane_kp = 2.5
        self.lane_max_correction = 2.0
        self.lane_timeout_sec = 0.6
        self.lane_speed_penalty = 0.8
        self.lane_error = 0.0
        self.lane_valid = False
        self.lane_stamp_ns = 0

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/points', self.lidar_callback, 10)
        self.sub_lane_error = self.create_subscription(Float32, '/lane/error', self.lane_error_callback, 10)
        self.sub_lane_valid = self.create_subscription(Bool, '/lane/valid', self.lane_valid_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('Yaris Pilotu aktif. Altin rota yüklendi, duba kacis aktif.')

    def odom_callback(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def lidar_callback(self, msg: PointCloud2) -> None:
        sayi = 0
        toplam_y = 0.0

        for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            if (
                x > 0.05
                and x < self.duba_algilama_mesafesi
                and y > -self.duba_y_sinir
                and y < self.duba_y_sinir
                and z > self.duba_min_z
            ):
                sayi += 1
                toplam_y += float(y)

        if sayi > self.duba_min_nokta:
            self.duba_var = True
            self.duba_konumu = toplam_y / float(sayi)
        else:
            self.duba_var = False

        self.sur()

    def lane_error_callback(self, msg: Float32) -> None:
        self.lane_error = float(msg.data)
        self.lane_stamp_ns = self.get_clock().now().nanoseconds

    def lane_valid_callback(self, msg: Bool) -> None:
        self.lane_valid = bool(msg.data)
        self.lane_stamp_ns = self.get_clock().now().nanoseconds

    def lane_is_recent_and_valid(self) -> bool:
        if not self.lane_valid:
            return False
        now_ns = self.get_clock().now().nanoseconds
        age_sec = (now_ns - self.lane_stamp_ns) / 1e9
        return age_sec <= self.lane_timeout_sec

    def sur(self) -> None:
        twist = Twist()

        if not hasattr(self, 'baslangic_bulundu'):
            self.baslangic_bulundu = False

        if not self.baslangic_bulundu:
            # Find the closest point in the route to our current position to start from.
            # Only do this once we have an odometry reading (x/y != 0.0).
            if self.x != 0.0 or self.y != 0.0:
                min_dist = float('inf')
                best_idx = 0
                for i, (hx, hy) in enumerate(self.rota):
                    dist = math.hypot(hx - self.x, hy - self.y)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                self.hedef_index = best_idx
                self.baslangic_bulundu = True
                self.get_logger().info(f'Baslangic noktasi bulundu: Index {best_idx}')
            else:
                return # Wait for odometry before moving

        if self.tamamlandi:
            self.pub.publish(twist)
            return

        # Scenario 1: Obstacle exists -> avoid
        if self.duba_var:
            twist.linear.x = self.duba_hiz
            twist.angular.z = -1.0 * self.duba_kacis_sertligi * self.duba_konumu

        # Scenario 2: Follow route (Pure Pursuit style)
        else:
            if self.hedef_index >= len(self.rota) - 1:
                self.tamamlandi = True
                self.pub.publish(twist)
                self.get_logger().info('Parkur tamamlandi!')
                return

            while self.hedef_index < len(self.rota) - 1:
                hx, hy = self.rota[self.hedef_index]
                dist = math.hypot(hx - self.x, hy - self.y)
                if dist < self.lookahead_dist:
                    self.hedef_index += 1
                else:
                    break

            target_x, target_y = self.rota[self.hedef_index]
            hedef_yaw = math.atan2(target_y - self.y, target_x - self.x)
            hata_yaw = normalize_angle(hedef_yaw - self.yaw)

            twist.linear.x = self.gps_hiz
            twist.angular.z = self.yaw_k * hata_yaw

            if abs(hata_yaw) > self.keskin_viraj_esik:
                twist.linear.x = self.keskin_viraj_hiz

            # Apply lane centering correction when lane tracker is healthy.
            if self.lane_is_recent_and_valid():
                # Reduce pure pursuit influence if severely off-center
                pp_weight = max(0.1, 1.0 - abs(self.lane_error) * 2.0)
                twist.angular.z *= pp_weight

                lane_correction = self.lane_kp * self.lane_error
                lane_correction = max(-self.lane_max_correction, min(self.lane_max_correction, lane_correction))
                twist.angular.z += lane_correction

                speed_scale = max(0.3, 1.0 - self.lane_speed_penalty * abs(self.lane_error))
                twist.linear.x *= speed_scale

        self.pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YarisPilotu()
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
