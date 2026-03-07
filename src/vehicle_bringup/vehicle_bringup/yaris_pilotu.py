#!/usr/bin/env python3

import math
from enum import Enum, auto
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray
import sensor_msgs_py.point_cloud2 as pc2


# ---------------------------------------------------------------------------
# Lane State Machine
# ---------------------------------------------------------------------------

class LaneState(Enum):
    NORMAL_LANE = auto()      # İki şerit, sağlıklı kontrol
    DEGRADED_LANE = auto()    # Tek şerit veya düşük güvenilirlik
    NO_LANE_COAST = auto()    # Lane kayboldu, son komutla coast et
    NO_LANE_SLOW = auto()     # Coast süresi bitti, yavaşla
    BLOCKED_STOP = auto()     # Gerçek merkez blokaj veya timeout → dur


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


class YarisPilotu(Node):
    def __init__(self) -> None:
        super().__init__('yaris_pilotu')
        self.declare_parameter('route_enabled', True)
        self.declare_parameter('lane_only_speed', 0.40)

        self.rota = ROTA
        self.hedef_index = 0
        self.tamamlandi = False
        self.route_enabled = as_bool(self.get_parameter('route_enabled').value)
        self.lane_only_speed = float(self.get_parameter('lane_only_speed').value)

        # Driving settings
        self.gps_hiz = 1.1
        self.lookahead_dist = 0.75
        self.yaw_k = 1.25
        self.keskin_viraj_hiz = 0.4
        self.keskin_viraj_esik = 0.4
        self.start_max_dist = 4.0
        self.start_heading_weight = 0.8

        # Obstacle settings (PointCloud)
        self.duba_algilama_mesafesi = 1.0
        self.duba_kacis_sertligi = 1.6
        self.duba_min_z = 0.10
        self.duba_max_z = 1.2
        self.duba_y_sinir = 0.32
        self.duba_min_nokta = 12
        self.duba_cikis_min_nokta = 5
        self.duba_hiz = 0.28
        self.duba_hold_sec = 0.45
        self.duba_filtre_alpha = 0.35
        self.duba_y_deadband = 0.03
        self.duba_max_angular = 1.0

        self.duba_var = False
        self.duba_konumu = 0.0
        self.duba_filtreli_konum = 0.0
        self.duba_last_seen_ns = 0

        # Depth camera based gap selection (RealSense-like behavior in sim)
        self.depth_enabled = True
        self.depth_topic = '/front_depth_camera/depth/image_raw'
        self.depth_alt_topic = '/front_camera/depth/image_raw'
        self.depth_near_m = 0.15
        self.depth_far_m = 4.0
        self.depth_close_m = 0.90
        self.depth_close_px_threshold = 700
        self.depth_close_ratio_threshold = 0.025
        self.depth_center_close_px_threshold = 180
        self.depth_center_gate_px_threshold = 90
        self.depth_emergency_center_px_threshold = 280
        self.depth_emergency_max_center_clearance = 0.55
        self.depth_center_roi_half_width_px = 80
        self.depth_roi_top_ratio = 0.50
        self.depth_roi_bottom_ratio = 0.82
        self.depth_side_margin_px = 120
        self.depth_clearance_percentile = 35.0
        self.depth_clearance_margin_m = 0.10
        self.depth_center_adv_margin_m = 0.22
        self.depth_turn_gain = 0.55
        self.depth_turn_bias = 0.08
        self.depth_hold_sec = 0.5
        self.depth_obstacle = False
        self.depth_emergency = False
        self.depth_min_dist = 99.0
        self.depth_avoid_dir = 1.0
        self.depth_last_seen_ns = 0
        self.depth_left_close_px = 0
        self.depth_right_close_px = 0
        self.depth_center_close_px = 0
        self.depth_close_ratio = 0.0
        self.depth_left_clearance = 99.0
        self.depth_right_clearance = 99.0
        self.depth_center_clearance = 99.0
        # NEW: center block ratio for gating depth avoidance
        self.center_block_ratio = 0.30
        self.stop_distance_threshold = 0.35

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Lane settings (camera lane tracker feedback)
        self.lane_kp = 1.2                          # ↓ was 1.7 — less aggressive P
        self.lane_max_correction = 0.55
        self.lane_timeout_sec = 0.8
        self.lane_hold_sec = 0.9
        self.lane_hold_max_abs_error = 0.12
        self.lane_speed_penalty = 0.9
        self.lane_loss_speed = 0.30
        self.lane_centering_speed = 0.45
        self.route_steer_weight_with_lane = 0.0
        self.lane_deadband = 0.02
        self.lane_error_clip = 0.35
        self.lane_kd = 0.25                         # ↓ was 0.35
        self.lane_ki = 0.90
        self.lane_integral = 0.0
        self.lane_integral_limit = 0.45
        self.lane_single_line_scale = 0.45
        self.lane_single_line_center_push = 0.015
        self.lane_single_line_speed_cap = 0.22
        self.lane_single_line_correction_limit = 0.20   # ↓ was 0.25
        self.lane_single_line_unreliable_err = 0.16
        self.lane_single_line_unreliable_steer = 0.12
        self.lane_single_line_unreliable_speed = 0.14
        self.lane_jump_reject = 0.12
        self.corner_enter_abs_err = 0.10
        self.corner_exit_abs_err = 0.045
        self.corner_hold_sec = 1.2
        self.corner_speed_cap = 0.18
        self.corner_correction_limit = 0.55
        self.corner_max_angular = 0.95
        self.corner_mode = False
        self.corner_hold_until_ns = 0
        self.obstacle_recovery_sec = 1.5
        self.obstacle_recovery_until_ns = 0
        self.prev_duba_var = False
        self.last_lane_control_error = 0.0
        self.last_lane_control_ns = 0

        # ── NEW: angular z safety parameters ──
        self.max_angular_z = 1.0                    # ↓ was 2.2 — absolute hard limit
        self.angular_clamp = 0.8                    # normal operation clamp
        self.angular_rate_limit = 1.5               # max rad/s change per second
        self.angular_smoothing = 0.40               # ↓ was 0.45 — slightly quicker response

        # ── NEW: lane error smoothing ──
        self.lane_error_smoothing_alpha = 0.30
        self.smoothed_lane_error = 0.0

        # ── NEW: single lane confidence gain ──
        self.single_lane_confidence_gain = 0.50

        # ── NEW: state machine parameters ──
        self.coast_duration = 1.5                   # seconds in NO_LANE_COAST
        self.coast_speed = 0.20                     # m/s during coast
        self.slow_speed = 0.08                      # m/s during NO_LANE_SLOW
        self.no_lane_timeout = 3.0                  # kept for compatibility; no longer triggers BLOCKED_STOP alone
        self.lane_recover_debounce_sec = 0.3        # lane must stay valid this long
        self.obstacle_persistence_time = 0.8        # sec required for BLOCKED_STOP latch

        # ── NEW: state machine variables ──
        self.lane_state = LaneState.NO_LANE_COAST   # start safe until lane is found
        self.lane_lost_ns = 0                       # when lane was first lost
        self.lane_recover_start_ns = 0              # when lane started recovering
        self.lane_state_log_ns = 0                  # throttle state logging
        self.center_block_start_ns = 0              # persistence tracker for center block
        self.blocked_persistent = False
        self.angular_debug_log_ns = 0

        # Kept from original
        self.lane_error = 0.0
        self.lane_valid = False
        self.lane_stamp_ns = 0
        self.lane_last_seen_ns = 0
        self.lane_last_sign = 1.0
        self.lane_last_valid_error = 0.0
        self.lane_last_valid_ns = 0
        self.left_lane_seen = False
        self.right_lane_seen = False
        self.last_cmd_angular = 0.0

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/points', self.lidar_callback, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.sub_depth_alt = self.create_subscription(Image, self.depth_alt_topic, self.depth_callback, 10)
        self.sub_lane_error = self.create_subscription(Float32, '/lane/error', self.lane_error_callback, 10)
        self.sub_lane_valid = self.create_subscription(Bool, '/lane/valid', self.lane_valid_callback, 10)
        self.sub_lane_left = self.create_subscription(Float32MultiArray, '/lane/left', self.lane_left_callback, 10)
        self.sub_lane_right = self.create_subscription(Float32MultiArray, '/lane/right', self.lane_right_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'Yaris Pilotu aktif. route_enabled={self.route_enabled} '
            f'angular_clamp={self.angular_clamp} max_angular_z={self.max_angular_z} '
            f'state_machine=ON duba kacis aktif.'
        )

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks (unchanged logic, added error smoothing)
    # ──────────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def lidar_callback(self, msg: PointCloud2) -> None:
        sayi = 0
        toplam_y = 0.0

        try:
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                if (
                    x > 0.05
                    and x < self.duba_algilama_mesafesi
                    and y > -self.duba_y_sinir
                    and y < self.duba_y_sinir
                    and z > self.duba_min_z
                    and z < self.duba_max_z
                ):
                    sayi += 1
                    toplam_y += float(y)
        except RuntimeError:
            return

        now_ns = self.get_clock().now().nanoseconds
        ort_y = (toplam_y / float(sayi)) if sayi > 0 else 0.0

        if sayi >= self.duba_min_nokta:
            self.duba_var = True
            self.duba_last_seen_ns = now_ns
            self.duba_filtreli_konum = (
                self.duba_filtre_alpha * ort_y
                + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
            )
        elif self.duba_var and sayi >= self.duba_cikis_min_nokta:
            self.duba_last_seen_ns = now_ns
            self.duba_filtreli_konum = (
                self.duba_filtre_alpha * ort_y
                + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
            )
        else:
            age_sec = (now_ns - self.duba_last_seen_ns) / 1e9 if self.duba_last_seen_ns > 0 else float('inf')
            self.duba_var = age_sec <= self.duba_hold_sec

        self.duba_konumu = self.duba_filtreli_konum if self.duba_var else 0.0

        self.sur()

    def depth_callback(self, msg: Image) -> None:
        if not self.depth_enabled:
            return

        encoding = msg.encoding.lower()
        if encoding not in ('32fc1', '16uc1'):
            return

        bpp = 4 if encoding == '32fc1' else 2
        if msg.step < msg.width * bpp:
            return

        try:
            raw = np.frombuffer(msg.data, dtype=np.float32 if encoding == '32fc1' else np.uint16)
            row_width = msg.step // bpp
            if row_width <= 0:
                return
            depth = raw.reshape((msg.height, row_width))[:, : msg.width].astype(np.float32)
        except Exception:
            return

        if encoding == '16uc1':
            depth *= 0.001  # mm -> m

        roi_top = int(max(0.2, min(0.9, self.depth_roi_top_ratio)) * msg.height)
        roi_bottom = int(max(0.3, min(0.98, self.depth_roi_bottom_ratio)) * msg.height)
        roi_bottom = max(roi_bottom, roi_top + 1)
        roi = depth[roi_top:roi_bottom, :]
        if roi.size == 0:
            return

        valid = np.isfinite(roi) & (roi > self.depth_near_m) & (roi < self.depth_far_m)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            return

        close = valid & (roi < self.depth_close_m)
        half = roi.shape[1] // 2
        center_half = max(10, int(self.depth_center_roi_half_width_px))
        center_min = max(0, half - center_half)
        center_max = min(roi.shape[1], half + center_half)

        left_close = int(np.count_nonzero(close[:, :half]))
        right_close = int(np.count_nonzero(close[:, half:]))
        center_close = int(np.count_nonzero(close[:, center_min:center_max]))
        total_close = left_close + right_close

        total_px = max(1, close.size)
        close_ratio = float(total_close) / float(total_px)

        left_valid = roi[:, :half][valid[:, :half]]
        right_valid = roi[:, half:][valid[:, half:]]
        center_valid = roi[:, center_min:center_max][valid[:, center_min:center_max]]
        left_clearance = (
            float(np.nanpercentile(left_valid, self.depth_clearance_percentile))
            if left_valid.size > 0
            else self.depth_near_m
        )
        right_clearance = (
            float(np.nanpercentile(right_valid, self.depth_clearance_percentile))
            if right_valid.size > 0
            else self.depth_near_m
        )
        center_clearance = (
            float(np.nanpercentile(center_valid, self.depth_clearance_percentile))
            if center_valid.size > 0
            else self.depth_near_m
        )

        self.depth_left_close_px = left_close
        self.depth_right_close_px = right_close
        self.depth_center_close_px = center_close
        self.depth_close_ratio = close_ratio
        self.depth_left_clearance = left_clearance
        self.depth_right_clearance = right_clearance
        self.depth_center_clearance = center_clearance
        self.depth_min_dist = float(np.nanpercentile(roi[valid], 15))

        now_ns = self.get_clock().now().nanoseconds
        side_min_clearance = min(left_clearance, right_clearance)
        center_is_closer = center_clearance + self.depth_center_adv_margin_m < side_min_clearance

        # ── NEW: compute center block ratio for gating ──
        center_roi_px = max(1, close[:, center_min:center_max].size)
        actual_center_block_ratio = float(center_close) / float(center_roi_px)

        strong_center_block = (
            center_close >= self.depth_center_close_px_threshold
            and self.depth_min_dist <= self.depth_close_m
            and center_is_closer
            and actual_center_block_ratio >= self.center_block_ratio   # NEW gate
        )
        wide_block = (
            total_close >= self.depth_close_px_threshold
            and close_ratio >= self.depth_close_ratio_threshold
            and self.depth_min_dist <= 0.95
            and center_is_closer
        )
        center_gate = center_close >= self.depth_center_gate_px_threshold
        obstacle_now = (
            strong_center_block or (wide_block and center_gate)
        )
        emergency_now = (
            center_close >= self.depth_emergency_center_px_threshold
            and center_clearance <= self.depth_emergency_max_center_clearance
            and center_is_closer
            and actual_center_block_ratio >= self.center_block_ratio   # NEW gate
        )

        # BLOCKED_STOP trigger must be persistent, not single-frame.
        center_block_frame = (
            center_is_closer
            and self.depth_min_dist <= self.stop_distance_threshold
            and actual_center_block_ratio >= self.center_block_ratio
        )
        if center_block_frame:
            if self.center_block_start_ns <= 0:
                self.center_block_start_ns = now_ns
            held_sec = (now_ns - self.center_block_start_ns) / 1e9
            self.blocked_persistent = held_sec >= self.obstacle_persistence_time
        else:
            self.center_block_start_ns = 0
            self.blocked_persistent = False

        if obstacle_now or emergency_now:
            self.get_logger().info(
                "DEPTH OBSTACLE: "
                f"obs={obstacle_now} emerg={emergency_now} "
                f"min_dist={self.depth_min_dist:.2f} "
                f"center={center_clearance:.2f} left={left_clearance:.2f} right={right_clearance:.2f} "
                f"close={center_close} ratio={actual_center_block_ratio:.2f}"
            )
        if obstacle_now:
            self.depth_obstacle = True
            self.depth_emergency = emergency_now
            self.depth_last_seen_ns = now_ns

            clearance_delta = left_clearance - right_clearance
            if abs(clearance_delta) >= self.depth_clearance_margin_m:
                self.depth_avoid_dir = 1.0 if clearance_delta > 0.0 else -1.0
            elif left_close > right_close + self.depth_side_margin_px:
                self.depth_avoid_dir = -1.0  # left side blocked -> go right
            elif right_close > left_close + self.depth_side_margin_px:
                self.depth_avoid_dir = 1.0  # right side blocked -> go left
        else:
            age_sec = (now_ns - self.depth_last_seen_ns) / 1e9 if self.depth_last_seen_ns > 0 else float('inf')
            held = age_sec <= self.depth_hold_sec
            self.depth_obstacle = held
            if not held:
                self.depth_emergency = False

    def lane_error_callback(self, msg: Float32) -> None:
        raw_err = float(msg.data)
        # NEW: EMA smoothing on incoming lane error
        self.smoothed_lane_error = (
            self.lane_error_smoothing_alpha * raw_err
            + (1.0 - self.lane_error_smoothing_alpha) * self.smoothed_lane_error
        )
        self.lane_error = self.smoothed_lane_error
        self.lane_stamp_ns = self.get_clock().now().nanoseconds

    def lane_valid_callback(self, msg: Bool) -> None:
        self.lane_valid = bool(msg.data)
        now_ns = self.get_clock().now().nanoseconds
        if self.lane_valid:
            self.lane_last_valid_error = self.lane_error
            self.lane_last_valid_ns = now_ns
            self.lane_stamp_ns = now_ns
            self.lane_last_seen_ns = now_ns
            if abs(self.lane_error) >= self.lane_deadband:
                self.lane_last_sign = 1.0 if self.lane_error >= 0.0 else -1.0

    def lane_left_callback(self, msg: Float32MultiArray) -> None:
        self.left_lane_seen = len(msg.data) >= 5

    def lane_right_callback(self, msg: Float32MultiArray) -> None:
        self.right_lane_seen = len(msg.data) >= 5

    # ──────────────────────────────────────────────────────────────────────
    # Lane validity & hold helpers
    # ──────────────────────────────────────────────────────────────────────

    def lane_is_recent_and_valid(self) -> bool:
        if not self.lane_valid:
            return False
        now_ns = self.get_clock().now().nanoseconds
        age_sec = (now_ns - self.lane_stamp_ns) / 1e9
        return age_sec <= self.lane_timeout_sec

    def lane_hold_available(self) -> bool:
        if self.lane_last_valid_ns <= 0:
            return False
        if abs(self.lane_last_valid_error) > self.lane_hold_max_abs_error:
            return False
        now_ns = self.get_clock().now().nanoseconds
        age_sec = (now_ns - self.lane_last_valid_ns) / 1e9
        return age_sec <= self.lane_hold_sec

    # ──────────────────────────────────────────────────────────────────────
    # State machine update
    # ──────────────────────────────────────────────────────────────────────

    def update_lane_state(self, now_ns: int) -> None:
        """Deterministic lane state machine: NORMAL → DEGRADED → COAST → SLOW → STOP."""
        lane_recent = self.lane_is_recent_and_valid()
        both_lanes = self.left_lane_seen and self.right_lane_seen

        # ── Lane is active ──
        if lane_recent or self.lane_hold_available():
            # Debounce recovery: lane must stay valid for recover_debounce_sec
            if self.lane_state in (LaneState.NO_LANE_COAST, LaneState.NO_LANE_SLOW, LaneState.BLOCKED_STOP):
                if self.lane_recover_start_ns <= 0:
                    self.lane_recover_start_ns = now_ns
                recover_age = (now_ns - self.lane_recover_start_ns) / 1e9
                if recover_age < self.lane_recover_debounce_sec:
                    return  # keep current no-lane state until debounce passes
                # Debounce passed → recover
                self.lane_integral = 0.0  # reset accumulated integral
                self.lane_recover_start_ns = 0

            if both_lanes:
                self.lane_state = LaneState.NORMAL_LANE
            else:
                self.lane_state = LaneState.DEGRADED_LANE
            self.lane_lost_ns = 0
            return

        # ── Lane is NOT active ──
        self.lane_recover_start_ns = 0  # reset recovery debounce

        if self.lane_lost_ns <= 0:
            self.lane_lost_ns = now_ns

        no_lane_age = (now_ns - self.lane_lost_ns) / 1e9

        if no_lane_age <= self.coast_duration:
            self.lane_state = LaneState.NO_LANE_COAST
        elif self.blocked_persistent:
            self.lane_state = LaneState.BLOCKED_STOP
        else:
            self.lane_state = LaneState.NO_LANE_SLOW

    # ──────────────────────────────────────────────────────────────────────
    # Route / Pure Pursuit helpers
    # ──────────────────────────────────────────────────────────────────────

    def tangent_yaw_at(self, idx: int) -> float:
        if idx < len(self.rota) - 1:
            nx, ny = self.rota[idx + 1]
            cx, cy = self.rota[idx]
            return math.atan2(ny - cy, nx - cx)
        if idx > 0:
            cx, cy = self.rota[idx]
            px, py = self.rota[idx - 1]
            return math.atan2(cy - py, cx - px)
        return self.yaw

    def select_start_index(self) -> tuple[int, float, float]:
        best_idx = 0
        best_score = float('inf')
        best_dist = float('inf')
        best_heading = math.pi

        for i, (hx, hy) in enumerate(self.rota):
            dist = math.hypot(hx - self.x, hy - self.y)
            tangent_yaw = self.tangent_yaw_at(i)
            heading_err = abs(normalize_angle(tangent_yaw - self.yaw))
            score = dist + self.start_heading_weight * heading_err

            if score < best_score:
                best_score = score
                best_idx = i
                best_dist = dist
                best_heading = heading_err

        return best_idx, best_dist, best_heading

    # ──────────────────────────────────────────────────────────────────────
    # Main control loop
    # ──────────────────────────────────────────────────────────────────────

    def sur(self) -> None:
        twist = Twist()
        now_ns = self.get_clock().now().nanoseconds
        lane_term = 0.0
        route_term = 0.0
        avoid_term = 0.0

        # After leaving obstacle avoidance, briefly calm steering and reacquire lane.
        if self.prev_duba_var and not self.duba_var:
            self.obstacle_recovery_until_ns = now_ns + int(self.obstacle_recovery_sec * 1e9)
            self.lane_integral = 0.0
            self.last_cmd_angular *= 0.3
        self.prev_duba_var = self.duba_var
        in_obstacle_recovery = now_ns < self.obstacle_recovery_until_ns

        if not hasattr(self, 'baslangic_bulundu'):
            self.baslangic_bulundu = False

        if self.route_enabled and not self.baslangic_bulundu:
            if self.x != 0.0 or self.y != 0.0:
                best_idx, min_dist, heading_err = self.select_start_index()
                if min_dist > self.start_max_dist:
                    self.get_logger().warn(
                        f'Rota baslangici bulunamadi (en yakin={min_dist:.2f}m). Arac bekliyor.'
                    )
                    self.pub.publish(twist)
                    return

                self.hedef_index = best_idx
                self.baslangic_bulundu = True
                self.get_logger().info(
                    f'Baslangic noktasi bulundu: Index {best_idx} dist={min_dist:.2f}m heading_err={heading_err:.2f}rad'
                )
            else:
                return  # Wait for odometry before moving

        if self.route_enabled and self.tamamlandi:
            self.pub.publish(twist)
            return

        # ── Update lane state machine ──
        self.update_lane_state(now_ns)

        # Throttled state logging (once per second)
        if now_ns - self.lane_state_log_ns > int(1e9):
            self.lane_state_log_ns = now_ns
            self.get_logger().info(
                f'[STATE] {self.lane_state.name} lane_valid={self.lane_valid} '
                f'err={self.lane_error:+.3f} angular={self.last_cmd_angular:+.3f} '
                f'L={self.left_lane_seen} R={self.right_lane_seen}'
            )

        # ═══════════════════════════════════════════════════════════════════
        # Scenario 1: Obstacle exists -> avoid (UNLESS in no-lane states)
        # ═══════════════════════════════════════════════════════════════════
        depth_should_override_lane = self.depth_obstacle and self.depth_emergency
        
        # Requirement: in no-lane states, angular must be zero (no spin).
        lane_is_dead = self.lane_state in (
            LaneState.NO_LANE_COAST,
            LaneState.NO_LANE_SLOW,
            LaneState.BLOCKED_STOP,
        )
        obstacle_active = (self.duba_var or depth_should_override_lane) and not lane_is_dead
        
        if obstacle_active:
            twist.linear.x = self.duba_hiz
            desired_angular = 0.0

            duba_y = self.duba_konumu
            if abs(duba_y) < self.duba_y_deadband:
                duba_y = 0.0

            if self.duba_var:
                desired_angular += -1.0 * self.duba_kacis_sertligi * duba_y

            if depth_should_override_lane:
                depth_turn = self.depth_turn_gain * self.depth_avoid_dir
                if self.depth_center_close_px >= self.depth_center_close_px_threshold:
                    depth_turn += self.depth_turn_bias * self.depth_avoid_dir
                desired_angular += depth_turn
                if self.depth_min_dist < self.stop_distance_threshold:
                    # Emergency stop: obstacle too close at center
                    twist.linear.x = 0.0
                    desired_angular = 0.0
                elif self.depth_min_dist < 0.40:
                    twist.linear.x = min(twist.linear.x, 0.10)
                elif self.depth_min_dist < 0.65:
                    twist.linear.x = min(twist.linear.x, 0.14)
                elif self.depth_min_dist < 0.85:
                    twist.linear.x = min(twist.linear.x, 0.18)
                else:
                    twist.linear.x = min(twist.linear.x, 0.24)
                if not self.duba_var:
                    desired_angular = clamp(desired_angular, -0.45, 0.45)

            desired_angular = clamp(desired_angular, -self.duba_max_angular, self.duba_max_angular)
            avoid_term = desired_angular

        # ═══════════════════════════════════════════════════════════════════
        # Scenario 2: Follow route + lane (STATE MACHINE DRIVEN)
        # ═══════════════════════════════════════════════════════════════════
        if not obstacle_active:
            # Compute pure pursuit angular (used by multiple states)
            pure_pursuit_angular = 0.0
            if self.route_enabled:
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
                pure_pursuit_angular = self.yaw_k * hata_yaw

                if abs(hata_yaw) > self.keskin_viraj_esik:
                    twist.linear.x = self.keskin_viraj_hiz
            else:
                twist.linear.x = self.lane_only_speed

            desired_angular = 0.0
            route_term = pure_pursuit_angular if self.route_enabled else 0.0

            # ─────────────────────────────────────
            # NORMAL_LANE: two lanes, full control
            # ─────────────────────────────────────
            if self.lane_state == LaneState.NORMAL_LANE:
                desired_angular, lane_term, route_term = self._compute_lane_correction(
                    now_ns, pure_pursuit_angular, twist, in_obstacle_recovery,
                    both_lanes=True
                )

            # ─────────────────────────────────────
            # DEGRADED_LANE: single lane, calmed gains
            # ─────────────────────────────────────
            elif self.lane_state == LaneState.DEGRADED_LANE:
                desired_angular, lane_term, route_term = self._compute_lane_correction(
                    now_ns, pure_pursuit_angular, twist, in_obstacle_recovery,
                    both_lanes=False
                )

            # ─────────────────────────────────────
            # NO_LANE_COAST: lane lost, coast forward
            # ─────────────────────────────────────
            elif self.lane_state == LaneState.NO_LANE_COAST:
                self.lane_integral *= 0.8
                self.corner_mode = False
                # Strict no-lane behavior: coast straight, no steering.
                desired_angular = 0.0
                route_term = 0.0
                lane_term = 0.0
                twist.linear.x = min(twist.linear.x, self.coast_speed)

            # ─────────────────────────────────────
            # NO_LANE_SLOW: coast expired, creep forward
            # ─────────────────────────────────────
            elif self.lane_state == LaneState.NO_LANE_SLOW:
                self.lane_integral = 0.0
                self.corner_mode = False
                # Go straight, no route influence
                desired_angular = 0.0
                route_term = 0.0
                lane_term = 0.0
                twist.linear.x = self.slow_speed

            # ─────────────────────────────────────
            # BLOCKED_STOP: full stop, zero everything
            # ─────────────────────────────────────
            elif self.lane_state == LaneState.BLOCKED_STOP:
                self.lane_integral = 0.0
                self.corner_mode = False
                desired_angular = 0.0
                route_term = 0.0
                lane_term = 0.0
                twist.linear.x = 0.0
                # Force angular to zero immediately (bypass rate limiting below)
                self.last_cmd_angular = 0.0

        # ═══════════════════════════════════════════════════════════════════
        # Final angular z — clamp + rate limit + smoothing
        # ═══════════════════════════════════════════════════════════════════

        # Safety rule: if linear speed is effectively zero, angular MUST be zero.
        if abs(twist.linear.x) <= 0.015:
            desired_angular = 0.0
            
            # In no-lane states, bypass rate limiting to kill spin immediately.
            if self.lane_state in (LaneState.NO_LANE_COAST, LaneState.NO_LANE_SLOW, LaneState.BLOCKED_STOP):
                self.last_cmd_angular = 0.0

        # Hard requirement: no steering in no-lane states.
        if self.lane_state in (LaneState.NO_LANE_COAST, LaneState.NO_LANE_SLOW, LaneState.BLOCKED_STOP):
            desired_angular = 0.0
            self.last_cmd_angular = 0.0

        # Step 1: normal operation clamp
        desired_angular = clamp(desired_angular, -self.angular_clamp, self.angular_clamp)

        # Step 2: rate limiting — prevent angular spikes
        dt = 0.05  # approximate control period (lidar callback ~20 Hz)
        max_delta = self.angular_rate_limit * dt
        delta = desired_angular - self.last_cmd_angular
        delta = clamp(delta, -max_delta, max_delta)
        rate_limited_angular = self.last_cmd_angular + delta

        # Step 3: EMA smoothing
        self.last_cmd_angular = (
            self.angular_smoothing * rate_limited_angular
            + (1.0 - self.angular_smoothing) * self.last_cmd_angular
        )

        # Step 4: absolute hard limit (safety)
        self.last_cmd_angular = clamp(self.last_cmd_angular, -self.max_angular_z, self.max_angular_z)

        twist.angular.z = self.last_cmd_angular

        if now_ns - self.angular_debug_log_ns > int(1e9):
            self.angular_debug_log_ns = now_ns
            self.get_logger().info(
                f'[ANG_DEBUG] state={self.lane_state.name} lane_term={lane_term:+.3f} '
                f'route_term={route_term:+.3f} avoid_term={avoid_term:+.3f} '
                f'final_angular={self.last_cmd_angular:+.3f}'
            )

        self.pub.publish(twist)

    # ──────────────────────────────────────────────────────────────────────
    # Lane correction helper (used by NORMAL_LANE and DEGRADED_LANE)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_lane_correction(
        self,
        now_ns: int,
        pure_pursuit_angular: float,
        twist: Twist,
        in_obstacle_recovery: bool,
        both_lanes: bool,
    ) -> tuple[float, float, float]:
        """Compute desired angular from lane error.
        Returns: (desired_angular, lane_term, route_term_used)
        """

        lane_err = clamp(self.lane_error, -self.lane_error_clip, self.lane_error_clip)

        if not both_lanes:
            # ── DEGRADED gains: scale down P, kill I, reduce D ──
            effective_kp = self.lane_kp * self.single_lane_confidence_gain
            effective_kd = self.lane_kd * 0.5
            effective_ki = 0.0  # no integral in single-lane

            lane_err *= self.lane_single_line_scale
            if self.left_lane_seen and not self.right_lane_seen:
                lane_err -= self.lane_single_line_center_push
            elif self.right_lane_seen and not self.left_lane_seen:
                lane_err += self.lane_single_line_center_push
            lane_err = clamp(lane_err, -0.10, 0.10)
            if abs(lane_err) > self.lane_single_line_unreliable_err:
                lane_err = clamp(
                    lane_err,
                    -self.lane_single_line_unreliable_steer,
                    self.lane_single_line_unreliable_steer,
                )
        else:
            effective_kp = self.lane_kp
            effective_kd = self.lane_kd
            effective_ki = self.lane_ki

        if abs(lane_err) < self.lane_deadband:
            lane_err = 0.0

        dt = (now_ns - self.last_lane_control_ns) / 1e9 if self.last_lane_control_ns > 0 else 0.0
        lane_err_rate = 0.0
        if dt > 1e-4:
            if abs(lane_err - self.last_lane_control_error) > self.lane_jump_reject:
                lane_err = 0.75 * self.last_lane_control_error + 0.25 * lane_err
            lane_err_rate = (lane_err - self.last_lane_control_error) / dt
            lane_err_rate = clamp(lane_err_rate, -1.5, 1.5)
        self.last_lane_control_error = lane_err
        self.last_lane_control_ns = now_ns

        if dt > 1e-4 and effective_ki > 0.0:
            self.lane_integral += lane_err * dt * effective_ki
            self.lane_integral = clamp(self.lane_integral, -self.lane_integral_limit, self.lane_integral_limit)

        # Corner mode
        if (not both_lanes) and abs(lane_err) >= self.corner_enter_abs_err:
            self.corner_mode = True
            self.corner_hold_until_ns = now_ns + int(self.corner_hold_sec * 1e9)
        elif both_lanes and abs(lane_err) <= self.corner_exit_abs_err and now_ns > self.corner_hold_until_ns:
            self.corner_mode = False
        elif self.corner_mode and now_ns <= self.corner_hold_until_ns:
            pass
        elif self.corner_mode and both_lanes and abs(lane_err) <= self.corner_exit_abs_err:
            self.corner_mode = False

        lane_correction = effective_kp * lane_err + self.lane_integral + effective_kd * lane_err_rate
        lane_correction_limit = self.lane_max_correction if both_lanes else self.lane_single_line_correction_limit
        if self.corner_mode:
            lane_correction_limit = min(lane_correction_limit, self.corner_correction_limit)
        lane_correction = clamp(lane_correction, -lane_correction_limit, lane_correction_limit)

        route_weight = self.route_steer_weight_with_lane if both_lanes else 0.15
        route_term_used = route_weight * pure_pursuit_angular
        lane_term = lane_correction
        desired_angular = route_term_used + lane_term

        speed_scale = max(0.35, 1.0 - self.lane_speed_penalty * abs(lane_err))
        twist.linear.x = min(twist.linear.x, self.lane_centering_speed * speed_scale)
        if not both_lanes:
            twist.linear.x = min(twist.linear.x, self.lane_single_line_speed_cap)
            if abs(lane_err) >= self.lane_single_line_unreliable_err:
                twist.linear.x = min(twist.linear.x, self.lane_single_line_unreliable_speed)
            if in_obstacle_recovery:
                twist.linear.x = min(twist.linear.x, 0.16)
                desired_angular *= 0.45
        if self.corner_mode:
            twist.linear.x = min(twist.linear.x, self.corner_speed_cap)
            desired_angular = clamp(desired_angular, -self.corner_max_angular, self.corner_max_angular)
            self.lane_integral *= 0.98

        return desired_angular, lane_term, route_term_used


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
