#!/usr/bin/env python3

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
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


class ControlAuthority(Enum):
    CRITICAL_AVOID = auto()
    IN_LANE_AVOID = auto()
    LANE_FOLLOW = auto()
    CORRIDOR_GAP = auto()
    NO_LANE_COAST = auto()
    NO_LANE_SLOW = auto()
    BLOCKED_STOP = auto()


@dataclass
class ControlCommand:
    authority: ControlAuthority
    speed: float = 0.0
    desired_angular: float = 0.0
    lane_term: float = 0.0
    route_term_raw: float = 0.0
    route_term_used: float = 0.0
    gap_term: float = 0.0
    corridor_term: float = 0.0
    avoid_term: float = 0.0
    lane_conf: float = 0.0
    reason: str = 'idle'


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


def gap_label(offset: float) -> str:
    if offset <= -0.60:
        return 'LEFT'
    if offset <= -0.20:
        return 'CENTER_LEFT'
    if offset >= 0.60:
        return 'RIGHT'
    if offset >= 0.20:
        return 'CENTER_RIGHT'
    return 'CENTER'


class YarisPilotu(Node):
    def __init__(self) -> None:
        super().__init__('yaris_pilotu')
        self.declare_parameter('publish_cmd_vel', True)
        self.declare_parameter('point_cloud_topic', '/points')
        self.declare_parameter('point_cloud_alt_topic', '/front_depth_camera/points')
        self.declare_parameter('debug_target_frame', 'base_footprint')
        self.declare_parameter('camera_frame_axis_mode', 'auto')
        self.declare_parameter('obstacle_bias_topic', '/obstacle/bias')
        self.declare_parameter('obstacle_speed_scale_topic', '/obstacle/speed_scale')
        self.declare_parameter('emergency_stop_topic', '/obstacle/emergency_stop')
        self.declare_parameter('obstacle_active_topic', '/obstacle/active')
        self.declare_parameter('obstacle_unknown_topic', '/obstacle/unknown')
        self.declare_parameter('route_enabled', True)
        self.declare_parameter('lane_only_speed', 0.78)
        self.declare_parameter('route_weight_normal', 0.05)
        self.declare_parameter('route_weight_single', 0.02)
        self.declare_parameter('start_max_dist', 4.0)
        self.declare_parameter('start_heading_weight', 0.8)
        self.declare_parameter('lookahead_dist', 1.8)
        self.declare_parameter('route_cross_track_gain', 0.85)
        self.declare_parameter('route_preview_multiplier', 1.8)
        self.declare_parameter('avoid_weight_normal', 0.20)
        self.declare_parameter('avoid_weight_emergency', 1.00)
        self.declare_parameter('lane_weight_during_avoid', 0.58)
        self.declare_parameter('gap_weight_during_avoid', 0.22)
        self.declare_parameter('recover_duration', 0.85)
        self.declare_parameter('recover_lane_gain', 1.15)
        self.declare_parameter('non_emergency_avoid_clamp', 0.28)
        self.declare_parameter('gap_lateral_penalty', 0.14)
        self.declare_parameter('gap_center_bias', 0.12)
        self.declare_parameter('gap_switch_margin', 0.10)
        self.declare_parameter('corridor_memory_gain', 0.28)
        self.declare_parameter('continuity_bonus', 0.18)
        self.declare_parameter('corridor_follow_gain', 0.42)
        self.declare_parameter('obstacle_corridor_weight', 0.32)
        self.declare_parameter('recover_corridor_weight', 0.22)
        self.declare_parameter('lateral_jump_penalty', 0.16)
        self.declare_parameter('gap_unlock_on_emergency', True)
        self.declare_parameter('vehicle_half_width_m', 0.23)
        self.declare_parameter('avoidance_clearance_margin_m', 0.12)
        self.declare_parameter('lane_corridor_cap', 0.28)
        self.declare_parameter('duba_center_trigger_m', 1.00)
        self.declare_parameter('duba_center_escape_y', 0.12)
        self.declare_parameter('duba_escape_gain_close', 2.8)
        self.declare_parameter('critical_roi_forward_min_m', 0.10)
        self.declare_parameter('critical_roi_forward_max_m', 1.50)
        self.declare_parameter('critical_roi_half_width_m', 0.60)
        self.declare_parameter('critical_roi_min_points', 8)
        self.declare_parameter('pointcloud_self_filter_forward_m', 0.28)
        self.declare_parameter('critical_center_ratio_min', 0.36)
        self.declare_parameter('critical_center_dominance_min', 0.92)
        self.declare_parameter('critical_commit_sec', 1.00)
        self.declare_parameter('critical_release_forward_margin_m', 0.18)
        self.declare_parameter('critical_release_lateral_margin_m', 0.08)
        self.declare_parameter('critical_escape_offset_m', 0.42)
        self.declare_parameter('critical_avoid_gain', 1.35)
        self.declare_parameter('critical_avoid_target_limit', 0.34)
        self.declare_parameter('critical_avoid_min_turn', 0.22)
        self.declare_parameter('critical_avoid_ramp_alpha', 0.32)
        self.declare_parameter('depth_frame_timeout_sec', 0.35)
        self.declare_parameter('duba_algilama_mesafesi', 1.35)
        self.declare_parameter('duba_y_sinir', 0.35)
        self.declare_parameter('duba_min_z', -0.35)
        self.declare_parameter('duba_max_z', 0.70)
        self.declare_parameter('duba_min_nokta', 12)
        self.declare_parameter('avoid_bias_gain', 1.15)
        self.declare_parameter('avoid_bias_limit', 0.38)
        self.declare_parameter('obstacle_hold_time_sec', 0.85)
        self.declare_parameter('return_to_center_sec', 1.10)
        self.declare_parameter('return_to_center_decay', 0.72)

        self.rota = ROTA
        self.hedef_index = 0
        self.baslangic_bulundu = False
        self.tamamlandi = False
        self.publish_cmd_vel = as_bool(self.get_parameter('publish_cmd_vel').value)
        self.route_enabled = as_bool(self.get_parameter('route_enabled').value)
        self.point_cloud_topic = str(self.get_parameter('point_cloud_topic').value)
        self.point_cloud_alt_topic = str(self.get_parameter('point_cloud_alt_topic').value)
        self.debug_target_frame = str(self.get_parameter('debug_target_frame').value)
        self.camera_frame_axis_mode = str(self.get_parameter('camera_frame_axis_mode').value).strip().lower()
        self.lane_only_speed = float(self.get_parameter('lane_only_speed').value)
        self.route_weight_normal = clamp(float(self.get_parameter('route_weight_normal').value), 0.0, 1.0)
        self.route_weight_single = clamp(float(self.get_parameter('route_weight_single').value), 0.0, 1.0)
        self.start_max_dist = max(1.0, float(self.get_parameter('start_max_dist').value))
        self.start_heading_weight = max(0.0, float(self.get_parameter('start_heading_weight').value))
        self.lookahead_dist = max(0.8, float(self.get_parameter('lookahead_dist').value))
        self.route_cross_track_gain = max(0.0, float(self.get_parameter('route_cross_track_gain').value))
        self.route_preview_multiplier = max(1.0, float(self.get_parameter('route_preview_multiplier').value))
        # Legacy parameters stay declared so existing launch files keep working.
        self.avoid_weight_normal = clamp(float(self.get_parameter('avoid_weight_normal').value), 0.0, 1.0)
        self.avoid_weight_emergency = clamp(float(self.get_parameter('avoid_weight_emergency').value), 0.0, 1.0)
        self.lane_weight_during_avoid = clamp(float(self.get_parameter('lane_weight_during_avoid').value), 0.0, 1.0)
        self.gap_weight_during_avoid = clamp(float(self.get_parameter('gap_weight_during_avoid').value), 0.0, 1.0)
        self.obstacle_corridor_weight = clamp(float(self.get_parameter('obstacle_corridor_weight').value), 0.0, 1.0)
        self.recover_corridor_weight = clamp(float(self.get_parameter('recover_corridor_weight').value), 0.0, 1.0)
        self.gap_weight_during_avoid = self.obstacle_corridor_weight
        self.recover_duration = max(0.10, float(self.get_parameter('recover_duration').value))
        self.recover_lane_gain = max(1.0, float(self.get_parameter('recover_lane_gain').value))
        self.non_emergency_avoid_clamp = max(0.05, float(self.get_parameter('non_emergency_avoid_clamp').value))
        self.gap_lateral_penalty = max(0.0, float(self.get_parameter('gap_lateral_penalty').value))
        self.gap_center_bias = max(0.0, float(self.get_parameter('gap_center_bias').value))
        self.gap_switch_margin = max(0.0, float(self.get_parameter('gap_switch_margin').value))
        self.corridor_memory_gain = clamp(float(self.get_parameter('corridor_memory_gain').value), 0.05, 0.95)
        self.continuity_bonus_gain = max(0.0, float(self.get_parameter('continuity_bonus').value))
        self.corridor_follow_gain = max(0.05, float(self.get_parameter('corridor_follow_gain').value))
        self.lateral_jump_penalty = max(0.0, float(self.get_parameter('lateral_jump_penalty').value))
        self.gap_unlock_on_emergency = as_bool(self.get_parameter('gap_unlock_on_emergency').value)
        self.vehicle_half_width_m = max(0.12, float(self.get_parameter('vehicle_half_width_m').value))
        self.avoidance_clearance_margin_m = max(0.04, float(self.get_parameter('avoidance_clearance_margin_m').value))
        self.lane_corridor_cap = clamp(float(self.get_parameter('lane_corridor_cap').value), 0.12, 0.45)
        self.required_gap_clearance_m = self.vehicle_half_width_m + self.avoidance_clearance_margin_m
        self.footprint_half_width_m = self.required_gap_clearance_m
        self.tight_gap_clearance_m = self.required_gap_clearance_m + 0.12
        self.duba_center_trigger_m = max(0.35, float(self.get_parameter('duba_center_trigger_m').value))
        self.duba_center_escape_y = max(0.04, float(self.get_parameter('duba_center_escape_y').value))
        self.duba_escape_gain_close = max(1.2, float(self.get_parameter('duba_escape_gain_close').value))
        self.critical_roi_forward_min_m = clamp(float(self.get_parameter('critical_roi_forward_min_m').value), 0.05, 0.60)
        self.critical_roi_forward_max_m = max(
            self.critical_roi_forward_min_m + 0.20,
            float(self.get_parameter('critical_roi_forward_max_m').value),
        )
        self.critical_roi_half_width_m = max(
            self.footprint_half_width_m + 0.08,
            float(self.get_parameter('critical_roi_half_width_m').value),
        )
        self.critical_roi_min_points = max(4, int(self.get_parameter('critical_roi_min_points').value))
        self.pointcloud_self_filter_forward_m = clamp(
            float(self.get_parameter('pointcloud_self_filter_forward_m').value),
            0.08,
            0.80,
        )
        self.critical_center_ratio_min = clamp(
            float(self.get_parameter('critical_center_ratio_min').value),
            0.10,
            0.90,
        )
        self.critical_center_dominance_min = clamp(
            float(self.get_parameter('critical_center_dominance_min').value),
            0.10,
            2.00,
        )
        self.critical_commit_sec = max(0.30, float(self.get_parameter('critical_commit_sec').value))
        self.critical_release_forward_m = self.critical_roi_forward_max_m + max(
            0.05,
            float(self.get_parameter('critical_release_forward_margin_m').value),
        )
        self.critical_release_lateral_margin_m = max(
            0.02,
            float(self.get_parameter('critical_release_lateral_margin_m').value),
        )
        self.critical_escape_offset_m = max(0.18, float(self.get_parameter('critical_escape_offset_m').value))
        self.critical_avoid_gain = max(0.40, float(self.get_parameter('critical_avoid_gain').value))
        self.critical_avoid_target_limit = max(0.18, float(self.get_parameter('critical_avoid_target_limit').value))
        self.critical_avoid_min_turn = max(0.0, float(self.get_parameter('critical_avoid_min_turn').value))
        self.critical_avoid_ramp_alpha = clamp(float(self.get_parameter('critical_avoid_ramp_alpha').value), 0.05, 1.0)
        self.depth_frame_timeout_sec = max(0.10, float(self.get_parameter('depth_frame_timeout_sec').value))
        self.duba_algilama_mesafesi = max(0.60, float(self.get_parameter('duba_algilama_mesafesi').value))
        self.duba_y_sinir = max(0.15, float(self.get_parameter('duba_y_sinir').value))
        self.duba_min_z = float(self.get_parameter('duba_min_z').value)
        self.duba_max_z = max(self.duba_min_z + 0.05, float(self.get_parameter('duba_max_z').value))
        self.duba_min_nokta = max(4, int(self.get_parameter('duba_min_nokta').value))
        self.avoid_bias_gain = max(0.30, float(self.get_parameter('avoid_bias_gain').value))
        self.avoid_bias_limit = max(0.12, float(self.get_parameter('avoid_bias_limit').value))
        self.obstacle_hold_time_sec = max(0.20, float(self.get_parameter('obstacle_hold_time_sec').value))
        self.return_to_center_sec = max(0.20, float(self.get_parameter('return_to_center_sec').value))
        self.return_to_center_decay = clamp(float(self.get_parameter('return_to_center_decay').value), 0.10, 0.95)

        self.control_period = 0.05
        self.gps_hiz = 1.25
        self.straight_speed = 1.00
        self.normal_lane_speed = max(self.lane_only_speed, 0.82)
        self.single_lane_speed = 0.58
        self.single_lane_corner_speed = 0.44
        self.single_lane_recovery_speed = 0.70
        self.single_lane_invalid_speed = 0.38
        self.obstacle_speed = 0.44
        self.coast_speed = 0.26
        self.slow_speed = 0.16

        self.yaw_k = 1.25
        self.keskin_viraj_hiz = 0.58
        self.keskin_viraj_esik = 0.40
        self.route_weight_no_lane_coast = 0.18
        self.route_weight_no_lane_slow = 0.30
        self.no_lane_route_limit_coast = 0.20
        self.no_lane_route_limit_slow = 0.28
        self.no_lane_rate_limit = 1.8
        self.no_lane_smoothing = 0.55

        self.lane_error_clip = 0.35
        self.normal_deadband = 0.012
        self.single_deadband = 0.008
        self.normal_kp = 1.55
        self.normal_kd = 0.18
        self.normal_limit = 0.42
        self.single_kp = 1.85
        self.single_kd = 0.12
        self.single_limit = 0.62
        self.single_corner_trigger = 0.080
        self.single_corner_gain = 1.55
        self.single_corner_limit = 0.92
        self.single_corner_bias = 0.030
        self.single_center_bias = 0.018
        self.single_corner_hold_sec = 1.35

        self.angular_clamp = 0.95
        self.max_angular_z = 1.10
        self.normal_rate_limit = 2.4
        self.single_rate_limit = 4.4
        self.obstacle_rate_limit = 8.5
        self.normal_smoothing = 0.64
        self.single_smoothing = 0.76
        self.obstacle_smoothing = 0.86
        self.speed_angular_gain = 0.38

        self.lane_timeout_sec = 0.75
        self.single_boundary_timeout_sec = 0.90
        self.coast_duration = 1.6
        self.recover_debounce_sec = 0.25
        self.obstacle_recovery_sec = self.recover_duration
        self.blocked_hold_sec = 0.8
        self.obstacle_context_sec = 1.2

        self.duba_cikis_min_nokta = 5
        self.duba_hold_sec = 0.60
        self.duba_filtre_alpha = 0.40
        self.duba_kacis_gain = 2.0
        self.duba_y_deadband = 0.03
        self.duba_max_angular = 1.0

        self.depth_enabled = True
        self.depth_topic = '/front_depth_camera/depth/image_raw'
        self.depth_alt_topic = '/front_camera/depth/image_raw'
        self.depth_near_m = 0.15
        self.depth_far_m = 4.0
        self.depth_close_m = 0.95
        self.depth_stop_m = 0.34
        self.depth_emergency_m = 0.42
        self.depth_hold_sec = 0.55
        self.depth_roi_top_ratio = 0.48
        self.depth_roi_bottom_ratio = 0.84
        self.depth_center_half_width_px = 78
        self.depth_clearance_percentile = 40.0
        self.depth_upper_band_ratio = 0.55
        self.depth_upper_ratio_threshold = 0.018
        self.depth_gap_sector_count = 7
        self.depth_gap_min_offset = 0.18
        self.depth_gap_min_score = 0.62
        self.depth_gap_gain = 0.46
        self.depth_gap_limit = 0.34
        self.depth_gap_corner_gain = 0.28
        self.depth_gap_corner_limit = 0.16
        self.depth_gap_center_penalty = 0.10
        self.depth_gap_heading_penalty = 0.08
        self.depth_gap_hysteresis = 0.04
        self.depth_gap_smoothing = 0.50
        self.obstacle_center_clearance_m = 0.78
        self.obstacle_center_ratio_threshold = 0.08

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.have_odom = False

        self.lane_error = 0.0
        self.raw_lane_error = 0.0
        self.smoothed_lane_error = 0.0
        self.lane_valid = False
        self.lane_stamp_ns = 0
        self.lane_last_valid_error = 0.0
        self.lane_last_valid_ns = 0

        self.left_lane_box = None
        self.right_lane_box = None
        self.left_lane_seen = False
        self.right_lane_seen = False
        self.left_lane_last_seen_ns = 0
        self.right_lane_last_seen_ns = 0

        self.duba_var = False
        self.duba_konumu = 0.0
        self.duba_filtreli_konum = 0.0
        self.duba_mesafe = 99.0
        self.duba_last_seen_ns = 0
        self.duba_nokta_sayisi = 0
        self.pointcloud_obstacle_supported = False
        self.critical_roi_point_count = 0
        self.critical_roi_mean_y = 0.0
        self.critical_roi_min_x = 99.0
        self.critical_roi_min_abs_y = 99.0
        self.critical_roi_intrusion_m = 0.0
        self.critical_center_ratio = 0.0
        self.critical_center_supported = False
        self.critical_obstacle_now = False
        self.critical_obstacle_last_seen_ns = 0
        self.critical_avoid_until_ns = 0
        self.critical_escape_offset = 0.0
        self.critical_avoid_smoothed = 0.0

        self.depth_gap_offset = 0.0
        self.depth_gap_raw_offset = 0.0
        self.depth_selected_gap_label = 'CENTER'
        self.depth_selected_gap_score = 0.0
        self.depth_selected_gap_offset = 0.0
        self.depth_selected_gap_clearance = 99.0
        self.depth_gap_continuity_bonus = 0.0
        self.depth_gap_switch_reason = 'init'
        self.corridor_target_offset = 0.0
        self.corridor_active_until_ns = 0
        self.corridor_enabled_state = False
        self.corridor_gating_reason = 'init'
        self.corridor_reset_reason = 'init'
        self.raw_gap_offset = 0.0
        self.mapped_corridor_target = 0.0
        self.smoothed_corridor_target = 0.0
        self.corridor_error = 0.0
        self.corridor_term_preclamp = 0.0
        self.corridor_term_postclamp = 0.0
        self.depth_left_clearance = 99.0
        self.depth_right_clearance = 99.0
        self.depth_left_risk = 0.0
        self.depth_right_risk = 0.0
        self.depth_left_gap_score = 0.0
        self.depth_right_gap_score = 0.0
        self.depth_center_clearance = 99.0
        self.depth_min_dist = 99.0
        self.depth_obstacle = False
        self.depth_emergency = False
        self.depth_last_seen_ns = 0
        self.depth_frame_stamp_ns = 0
        self.depth_context_last_ns = 0
        self.depth_center_ratio = 0.0
        self.depth_upper_ratio = 0.0
        self.pointcloud_last_ns = 0
        self.pointcloud_total_points = 0
        self.pointcloud_roi_points = 0
        self.pointcloud_front_left_count = 0
        self.pointcloud_front_center_count = 0
        self.pointcloud_front_right_count = 0
        self.pointcloud_front_min_distance = 99.0
        self.pointcloud_filtered_height = 0
        self.pointcloud_filtered_distance = 0
        self.pointcloud_filtered_corridor = 0
        self.pointcloud_source_frame = ''
        self.pointcloud_source_name = ''
        self.pointcloud_filter_frame = ''
        self.pointcloud_axis_mode = 'unknown'
        self.obstacle_reason_code = 'startup'
        self.depth_reason_code = 'startup'
        self.obstacle_unknown = True

        self.lane_state = LaneState.NO_LANE_COAST
        self.lane_lost_ns = 0
        self.recover_start_ns = 0
        self.blocked_start_ns = 0
        self.blocked_persistent = False
        self.corner_mode = False
        self.corner_until_ns = 0
        self.obstacle_recovery_until_ns = 0
        self.return_to_center_until_ns = 0
        self.prev_obstacle_active = False

        self.last_cmd_angular = 0.0
        self.last_normal_error = 0.0
        self.last_single_error = 0.0
        self.last_lane_confidence = 0.0
        self.parser_lane_confidence = 0.0
        self.parser_lane_confidence_ns = 0
        self.last_gap_assist_active = False
        self.last_obstacle_active_ns = 0
        self.control_authority = ControlAuthority.NO_LANE_COAST
        self.control_reason = 'init'
        self.last_mode_name = 'INIT'
        self.summary_obstacle_active = False
        self.summary_obstacle_unknown = True
        self.summary_avoid_latched = False

        self.lane_state_log_ns = 0
        self.angular_debug_log_ns = 0
        self.depth_debug_log_ns = 0
        self.lidar_debug_log_ns = 0

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_lidar = self.create_subscription(
            PointCloud2,
            self.point_cloud_topic,
            lambda msg: self.lidar_callback(msg, 'primary'),
            10,
        )
        self.sub_lidar_alt = None
        if self.point_cloud_alt_topic and self.point_cloud_alt_topic != self.point_cloud_topic:
            self.sub_lidar_alt = self.create_subscription(
                PointCloud2,
                self.point_cloud_alt_topic,
                lambda msg: self.lidar_callback(msg, 'alternate'),
                10,
            )
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.sub_depth_alt = self.create_subscription(Image, self.depth_alt_topic, self.depth_callback, 10)
        self.sub_lane_error = self.create_subscription(Float32, '/lane/error', self.lane_error_callback, 10)
        self.sub_lane_valid = self.create_subscription(Bool, '/lane/valid', self.lane_valid_callback, 10)
        self.sub_lane_confidence = self.create_subscription(Float32, '/lane/confidence', self.lane_confidence_callback, 10)
        self.sub_lane_left = self.create_subscription(Float32MultiArray, '/lane/left', self.lane_left_callback, 10)
        self.sub_lane_right = self.create_subscription(Float32MultiArray, '/lane/right', self.lane_right_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10) if self.publish_cmd_vel else None
        self.obstacle_bias_pub = self.create_publisher(
            Float32,
            str(self.get_parameter('obstacle_bias_topic').value),
            10,
        )
        self.obstacle_speed_scale_pub = self.create_publisher(
            Float32,
            str(self.get_parameter('obstacle_speed_scale_topic').value),
            10,
        )
        self.emergency_stop_pub = self.create_publisher(
            Bool,
            str(self.get_parameter('emergency_stop_topic').value),
            10,
        )
        self.obstacle_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter('obstacle_active_topic').value),
            10,
        )
        self.obstacle_unknown_pub = self.create_publisher(
            Bool,
            str(self.get_parameter('obstacle_unknown_topic').value),
            10,
        )
        self.debug_front_min_distance_pub = self.create_publisher(Float32, '/obstacle/debug/front_min_distance', 10)
        self.debug_roi_z_min_pub = self.create_publisher(Float32, '/obstacle/debug/roi_z_min', 10)
        self.debug_roi_z_max_pub = self.create_publisher(Float32, '/obstacle/debug/roi_z_max', 10)
        self.debug_front_left_count_pub = self.create_publisher(Int32, '/obstacle/debug/front_left_count', 10)
        self.debug_front_center_count_pub = self.create_publisher(Int32, '/obstacle/debug/front_center_count', 10)
        self.debug_front_right_count_pub = self.create_publisher(Int32, '/obstacle/debug/front_right_count', 10)
        self.debug_total_roi_points_pub = self.create_publisher(Int32, '/obstacle/debug/total_roi_points', 10)
        self.debug_clearance_left_pub = self.create_publisher(Float32, '/obstacle/debug/clearance_left', 10)
        self.debug_clearance_right_pub = self.create_publisher(Float32, '/obstacle/debug/clearance_right', 10)
        self.debug_risk_left_pub = self.create_publisher(Float32, '/obstacle/debug/risk_left', 10)
        self.debug_risk_right_pub = self.create_publisher(Float32, '/obstacle/debug/risk_right', 10)
        self.debug_gap_score_left_pub = self.create_publisher(Float32, '/obstacle/debug/gap_score_left', 10)
        self.debug_gap_score_right_pub = self.create_publisher(Float32, '/obstacle/debug/gap_score_right', 10)
        self.debug_source_frame_pub = self.create_publisher(String, '/obstacle/debug/source_frame', 10)
        self.debug_reason_code_pub = self.create_publisher(String, '/obstacle/debug/reason_code', 10)
        self.debug_selected_gap_pub = self.create_publisher(String, '/obstacle/debug/selected_gap', 10)
        self.debug_avoid_latched_pub = self.create_publisher(Bool, '/obstacle/debug/avoid_latched', 10)
        self.debug_return_to_center_pub = self.create_publisher(Bool, '/obstacle/debug/return_to_center_active', 10)
        self.control_timer = self.create_timer(self.control_period, self.sur)

        self.get_logger().info(
            f'Yaris Pilotu aktif. route_enabled={self.route_enabled} '
            f'angular_clamp={self.angular_clamp} max_angular_z={self.max_angular_z} '
            f'state_machine=ON duba kacis aktif. publish_cmd_vel={self.publish_cmd_vel}'
        )
        self.get_logger().info(
            '[OBS_STARTUP] '
            f'point_cloud_topic={self.point_cloud_topic} alt_point_cloud_topic={self.point_cloud_alt_topic} '
            f'depth_topic={self.depth_topic} debug_target_frame={self.debug_target_frame} '
            f'camera_frame_axis_mode={self.camera_frame_axis_mode} '
            f'critical_forward=[{self.critical_roi_forward_min_m:.2f},{self.critical_roi_forward_max_m:.2f}] '
            f'self_filter_forward={self.pointcloud_self_filter_forward_m:.2f} '
            f'critical_center_ratio_min={self.critical_center_ratio_min:.2f} '
            f'critical_center_dominance_min={self.critical_center_dominance_min:.2f} '
            f'corridor_half_width={self.critical_roi_half_width_m:.2f} '
            f'z_limits=[{self.duba_min_z:.2f},{self.duba_max_z:.2f}] '
            f'duba_distance={self.duba_algilama_mesafesi:.2f} min_points={self.duba_min_nokta} '
            f'critical_min_points={self.critical_roi_min_points}'
        )

    def parse_lane_box(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            return None
        return tuple(float(v) for v in msg.data[:5])

    def odom_callback(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.have_odom = True

    def lane_error_callback(self, msg: Float32) -> None:
        raw = clamp(float(msg.data), -self.lane_error_clip, self.lane_error_clip)
        self.raw_lane_error = raw
        self.smoothed_lane_error = 0.35 * raw + 0.65 * self.smoothed_lane_error
        self.lane_error = self.smoothed_lane_error
        self.lane_stamp_ns = self.get_clock().now().nanoseconds
        if self.lane_valid:
            self.lane_last_valid_error = self.lane_error
            self.lane_last_valid_ns = self.lane_stamp_ns

    def lane_valid_callback(self, msg: Bool) -> None:
        self.lane_valid = bool(msg.data)
        now_ns = self.get_clock().now().nanoseconds
        if self.lane_valid:
            self.lane_stamp_ns = now_ns
            self.lane_last_valid_error = self.lane_error
            self.lane_last_valid_ns = now_ns

    def lane_confidence_callback(self, msg: Float32) -> None:
        self.parser_lane_confidence = clamp(float(msg.data), 0.0, 1.0)
        self.parser_lane_confidence_ns = self.get_clock().now().nanoseconds

    def lane_left_callback(self, msg: Float32MultiArray) -> None:
        lane_box = self.parse_lane_box(msg)
        self.left_lane_box = lane_box
        self.left_lane_seen = lane_box is not None
        if lane_box is not None:
            self.left_lane_last_seen_ns = self.get_clock().now().nanoseconds

    def lane_right_callback(self, msg: Float32MultiArray) -> None:
        lane_box = self.parse_lane_box(msg)
        self.right_lane_box = lane_box
        self.right_lane_seen = lane_box is not None
        if lane_box is not None:
            self.right_lane_last_seen_ns = self.get_clock().now().nanoseconds

    def point_in_critical_roi(self, x: float, y: float, z: float) -> bool:
        min_forward = max(self.critical_roi_forward_min_m, self.pointcloud_self_filter_forward_m)
        return (
            min_forward <= x <= self.critical_roi_forward_max_m
            and abs(y) <= self.critical_roi_half_width_m
            and self.duba_min_z < z < self.duba_max_z
        )

    def normalize_point_for_vehicle_frame(
        self,
        x: float,
        y: float,
        z: float,
        frame_id: str,
    ) -> Tuple[float, float, float, str, str]:
        frame = (frame_id or '').lower()
        if self.camera_frame_axis_mode == 'optical':
            return float(z), float(-x), float(-y), self.debug_target_frame, 'forced_optical'
        if self.camera_frame_axis_mode == 'native':
            return float(x), float(y), float(z), frame_id or self.debug_target_frame, 'native'
        if 'optical' in frame or 'camera' in frame or 'depth' in frame:
            # Approximate optical camera axes to vehicle axes: forward=z, left=-x, up=-y.
            return float(z), float(-x), float(-y), self.debug_target_frame, 'camera_optical_auto'
        return float(x), float(y), float(z), frame_id or self.debug_target_frame, 'native'

    def footprint_overlap(self, abs_lateral_m: float, extra_margin_m: float = 0.0) -> bool:
        return abs_lateral_m <= (self.footprint_half_width_m + extra_margin_m)

    def publish_obstacle_debug_topics(self) -> None:
        msg_float = Float32()
        msg_float.data = float(self.pointcloud_front_min_distance)
        self.debug_front_min_distance_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.duba_min_z)
        self.debug_roi_z_min_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.duba_max_z)
        self.debug_roi_z_max_pub.publish(msg_float)

        msg_int = Int32()
        msg_int.data = int(self.pointcloud_front_left_count)
        self.debug_front_left_count_pub.publish(msg_int)
        msg_int = Int32()
        msg_int.data = int(self.pointcloud_front_center_count)
        self.debug_front_center_count_pub.publish(msg_int)
        msg_int = Int32()
        msg_int.data = int(self.pointcloud_front_right_count)
        self.debug_front_right_count_pub.publish(msg_int)
        msg_int = Int32()
        msg_int.data = int(self.pointcloud_roi_points)
        self.debug_total_roi_points_pub.publish(msg_int)
        msg_float = Float32()
        msg_float.data = float(self.depth_left_clearance)
        self.debug_clearance_left_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.depth_right_clearance)
        self.debug_clearance_right_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.depth_left_risk)
        self.debug_risk_left_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.depth_right_risk)
        self.debug_risk_right_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.depth_left_gap_score)
        self.debug_gap_score_left_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.depth_right_gap_score)
        self.debug_gap_score_right_pub.publish(msg_float)

        msg_text = String()
        msg_text.data = self.pointcloud_source_frame or 'unknown'
        self.debug_source_frame_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.obstacle_reason_code
        self.debug_reason_code_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.depth_selected_gap_label
        self.debug_selected_gap_pub.publish(msg_text)
        msg_bool = Bool()
        msg_bool.data = bool(self.summary_avoid_latched)
        self.debug_avoid_latched_pub.publish(msg_bool)
        msg_bool = Bool()
        msg_bool.data = bool(self.return_to_center_active(self.get_clock().now().nanoseconds))
        self.debug_return_to_center_pub.publish(msg_bool)

    def lidar_callback(self, msg: PointCloud2, source_name: str = 'primary') -> None:
        count = 0
        sum_y = 0.0
        min_x = float('inf')
        critical_count = 0
        critical_sum_y = 0.0
        critical_min_x = float('inf')
        critical_min_abs_y = float('inf')
        total_points = 0
        filtered_height = 0
        filtered_distance = 0
        filtered_corridor = 0
        roi_points = 0
        front_left_count = 0
        front_center_count = 0
        front_right_count = 0
        front_min_distance = float('inf')
        frame_id = str(msg.header.frame_id)
        filter_frame = frame_id or self.debug_target_frame
        axis_mode = 'native'
        sector_split = max(0.08, 0.33 * self.critical_roi_half_width_m)
        try:
            for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
                total_points += 1
                xf, yf, zf, filter_frame, axis_mode = self.normalize_point_for_vehicle_frame(
                    float(x),
                    float(y),
                    float(z),
                    frame_id,
                )
                if zf <= self.duba_min_z or zf >= self.duba_max_z:
                    filtered_height += 1
                    continue
                if xf <= self.pointcloud_self_filter_forward_m or xf >= self.duba_algilama_mesafesi:
                    filtered_distance += 1
                    continue
                if abs(yf) >= self.critical_roi_half_width_m:
                    filtered_corridor += 1
                    continue
                roi_points += 1
                front_min_distance = min(front_min_distance, xf)
                if yf > sector_split:
                    front_left_count += 1
                elif yf < -sector_split:
                    front_right_count += 1
                else:
                    front_center_count += 1
                if (
                    self.pointcloud_self_filter_forward_m < xf < self.duba_algilama_mesafesi
                    and -self.duba_y_sinir < yf < self.duba_y_sinir
                ):
                    count += 1
                    sum_y += yf
                    min_x = min(min_x, xf)
                if self.point_in_critical_roi(xf, yf, zf):
                    critical_count += 1
                    critical_sum_y += yf
                    critical_min_x = min(critical_min_x, xf)
                    critical_min_abs_y = min(critical_min_abs_y, abs(yf))
        except RuntimeError:
            self.obstacle_reason_code = 'pointcloud_runtime_error'
            self.publish_obstacle_debug_topics()
            return

        now_ns = self.get_clock().now().nanoseconds
        self.pointcloud_last_ns = now_ns
        self.pointcloud_total_points = total_points
        self.pointcloud_roi_points = roi_points
        self.pointcloud_front_left_count = front_left_count
        self.pointcloud_front_center_count = front_center_count
        self.pointcloud_front_right_count = front_right_count
        self.pointcloud_front_min_distance = front_min_distance if math.isfinite(front_min_distance) else 99.0
        self.pointcloud_filtered_height = filtered_height
        self.pointcloud_filtered_distance = filtered_distance
        self.pointcloud_filtered_corridor = filtered_corridor
        self.pointcloud_source_frame = frame_id
        self.pointcloud_source_name = source_name
        self.pointcloud_filter_frame = filter_frame
        self.pointcloud_axis_mode = axis_mode
        self.duba_nokta_sayisi = count
        mean_y = (sum_y / float(count)) if count > 0 else 0.0
        critical_mean_y = (critical_sum_y / float(critical_count)) if critical_count > 0 else 0.0
        critical_center_ratio = float(front_center_count) / float(max(1, roi_points))
        critical_center_supported = (
            roi_points > 0
            and critical_center_ratio >= self.critical_center_ratio_min
            and float(front_center_count) >= self.critical_center_dominance_min * float(max(front_left_count, front_right_count, 1))
        )
        self.critical_center_ratio = critical_center_ratio
        self.critical_center_supported = critical_center_supported

        if total_points <= 0:
            self.obstacle_reason_code = 'no_points_received'
        elif not frame_id:
            self.obstacle_reason_code = 'missing_frame'
        elif roi_points <= 0:
            if filtered_height >= filtered_distance and filtered_height >= filtered_corridor:
                self.obstacle_reason_code = 'filtered_out_by_height'
            elif filtered_distance >= filtered_corridor:
                self.obstacle_reason_code = 'filtered_out_by_distance'
            else:
                self.obstacle_reason_code = 'outside_corridor'
        elif count < self.duba_min_nokta and critical_count < self.critical_roi_min_points:
            self.obstacle_reason_code = 'not_enough_points'
        else:
            self.obstacle_reason_code = 'roi_points_detected'

        if count >= self.duba_min_nokta:
            self.duba_var = True
            self.duba_last_seen_ns = now_ns
            self.duba_filtreli_konum = self.duba_filtre_alpha * mean_y + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
            self.duba_mesafe = min_x if math.isfinite(min_x) else self.duba_algilama_mesafesi
        elif self.duba_var and count >= self.duba_cikis_min_nokta:
            self.duba_last_seen_ns = now_ns
            self.duba_filtreli_konum = self.duba_filtre_alpha * mean_y + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
            self.duba_mesafe = min_x if math.isfinite(min_x) else self.duba_mesafe
        else:
            held_sec = (now_ns - self.duba_last_seen_ns) / 1e9 if self.duba_last_seen_ns > 0 else float('inf')
            self.duba_var = held_sec <= self.duba_hold_sec

        self.duba_konumu = self.duba_filtreli_konum if self.duba_var else 0.0
        if not self.duba_var:
            self.duba_mesafe = 99.0
        self.critical_roi_point_count = critical_count
        self.critical_roi_mean_y = critical_mean_y if critical_count > 0 else 0.0
        self.critical_roi_min_x = critical_min_x if math.isfinite(critical_min_x) else 99.0
        self.critical_roi_min_abs_y = critical_min_abs_y if math.isfinite(critical_min_abs_y) else 99.0
        self.critical_roi_intrusion_m = max(0.0, self.footprint_half_width_m - self.critical_roi_min_abs_y)
        self.critical_obstacle_now = (
            critical_center_supported
            and
            critical_count >= self.critical_roi_min_points
            and self.critical_roi_min_x <= self.critical_roi_forward_max_m
            and self.footprint_overlap(self.critical_roi_min_abs_y)
            and (
                abs(self.critical_roi_mean_y) >= max(0.10, 0.80 * self.duba_center_escape_y)
                or (
                    self.depth_frame_recent(now_ns)
                    and self.depth_center_clearance < self.tight_gap_clearance_m
                )
            )
        )
        self.pointcloud_obstacle_supported = self.pointcloud_corridor_signal_active()
        if self.pointcloud_obstacle_supported:
            self.depth_context_last_ns = now_ns
        if self.critical_obstacle_now:
            self.critical_obstacle_last_seen_ns = now_ns
            self.depth_context_last_ns = now_ns

        self.publish_obstacle_debug_topics()
        if self.duba_var and now_ns - self.lidar_debug_log_ns > int(1e9):
            self.lidar_debug_log_ns = now_ns
            self.get_logger().info(
                f'[LIDAR_DEBUG] source={self.pointcloud_source_name} frame={self.pointcloud_source_frame} '
                f'filter_frame={self.pointcloud_filter_frame} axis_mode={self.pointcloud_axis_mode} '
                f'reason={self.obstacle_reason_code} roi={self.pointcloud_roi_points} '
                f'L={self.pointcloud_front_left_count} C={self.pointcloud_front_center_count} R={self.pointcloud_front_right_count} '
                f'front_min={self.pointcloud_front_min_distance:.2f} '
                f'duba=True points={self.duba_nokta_sayisi} y={self.duba_konumu:+.3f} dist={self.duba_mesafe:.2f} '
                f'critical_points={self.critical_roi_point_count} critical_dist={self.critical_roi_min_x:.2f} '
                f'critical_y={self.critical_roi_mean_y:+.3f} intrusion={self.critical_roi_intrusion_m:.2f} '
                f'critical_center_ratio={self.critical_center_ratio:.2f} critical_center_supported={self.critical_center_supported} '
                f'corridor_supported={self.pointcloud_obstacle_supported}'
            )

    def depth_callback(self, msg: Image) -> None:
        if not self.depth_enabled:
            self.depth_reason_code = 'depth_disabled'
            return

        encoding = msg.encoding.lower()
        if encoding not in ('32fc1', '16uc1'):
            self.depth_reason_code = 'depth_bad_encoding'
            return

        bpp = 4 if encoding == '32fc1' else 2
        if msg.step < msg.width * bpp:
            self.depth_reason_code = 'depth_bad_step'
            return

        try:
            raw = np.frombuffer(msg.data, dtype=np.float32 if encoding == '32fc1' else np.uint16)
            row_width = msg.step // bpp
            depth = raw.reshape((msg.height, row_width))[:, : msg.width].astype(np.float32)
        except Exception:
            self.depth_reason_code = 'depth_decode_error'
            return

        if encoding == '16uc1':
            depth *= 0.001

        y0 = int(max(0.2, min(0.9, self.depth_roi_top_ratio)) * msg.height)
        y1 = int(max(0.3, min(0.98, self.depth_roi_bottom_ratio)) * msg.height)
        if y1 <= y0:
            self.depth_reason_code = 'depth_bad_roi'
            return
        roi = depth[y0:y1, :]
        valid = np.isfinite(roi) & (roi > self.depth_near_m) & (roi < self.depth_far_m)
        if not np.any(valid):
            self.depth_reason_code = 'depth_no_valid_pixels'
            self.publish_obstacle_debug_topics()
            return

        close = valid & (roi < self.depth_close_m)
        width = roi.shape[1]
        half = width // 2
        center_half = max(10, int(self.depth_center_half_width_px))
        c0 = max(0, half - center_half)
        c1 = min(width, half + center_half)

        left_valid = roi[:, :half][valid[:, :half]]
        right_valid = roi[:, half:][valid[:, half:]]
        center_valid = roi[:, c0:c1][valid[:, c0:c1]]
        left_clear = float(np.nanpercentile(left_valid, self.depth_clearance_percentile)) if left_valid.size else self.depth_near_m
        right_clear = float(np.nanpercentile(right_valid, self.depth_clearance_percentile)) if right_valid.size else self.depth_near_m
        center_clear = float(np.nanpercentile(center_valid, self.depth_clearance_percentile)) if center_valid.size else self.depth_near_m
        left_close = int(np.count_nonzero(close[:, :half]))
        right_close = int(np.count_nonzero(close[:, half:]))
        left_close_ratio = float(left_close) / float(max(1, close[:, :half].size))
        right_close_ratio = float(right_close) / float(max(1, close[:, half:].size))

        total_close = int(np.count_nonzero(close))
        center_close = int(np.count_nonzero(close[:, c0:c1]))
        center_px = max(1, close[:, c0:c1].size)
        total_px = max(1, close.size)
        close_ratio = float(total_close) / float(total_px)
        center_ratio = float(center_close) / float(center_px)

        upper_rows = max(1, int(close.shape[0] * self.depth_upper_band_ratio))
        upper_close = close[:upper_rows, :]
        upper_ratio = float(np.count_nonzero(upper_close)) / float(max(1, upper_close.size))

        now_ns = self.get_clock().now().nanoseconds
        self.depth_frame_stamp_ns = now_ns
        obstacle_now = (
            center_clear < 0.72
            and center_ratio > 0.12
            and upper_ratio > self.depth_upper_ratio_threshold
            and center_clear + 0.12 < min(left_clear, right_clear)
        )
        emergency_now = (
            center_clear < self.depth_emergency_m
            and center_ratio > 0.18
            and upper_ratio > self.depth_upper_ratio_threshold
        )
        blocked_frame = (
            center_clear < self.depth_stop_m
            and center_ratio > 0.20
            and upper_ratio > self.depth_upper_ratio_threshold
        )
        self.depth_reason_code = 'clear_path'

        recent_obstacle_context = self.signal_recent(self.depth_context_last_ns, self.obstacle_context_sec, now_ns)
        single_side = self.active_single_side(now_ns)
        lane_recent = self.lane_valid_recent(now_ns)
        pointcloud_signal = self.pointcloud_corridor_signal_active()
        pointcloud_lateral_hint = max(abs(self.critical_roi_mean_y), abs(self.duba_konumu))
        pointcloud_confident = pointcloud_signal and (
            obstacle_now
            or blocked_frame
            or self.depth_obstacle
            or pointcloud_lateral_hint >= max(0.10, 0.80 * self.duba_center_escape_y)
            or center_clear + 0.18 < max(left_clear, right_clear)
        )
        hard_corridor_signal = (
            pointcloud_confident
            or self.critical_obstacle_now
            or obstacle_now
            or self.depth_obstacle
            or blocked_frame
        )
        corridor_memory_allowed = not lane_recent or hard_corridor_signal or now_ns < self.critical_avoid_until_ns
        keep_corridor = (
            corridor_memory_allowed
            and (
                hard_corridor_signal
                or recent_obstacle_context
                or now_ns < self.obstacle_recovery_until_ns
                or now_ns < self.corridor_active_until_ns
            )
        )

        sector_edges = np.linspace(0, width, self.depth_gap_sector_count + 1, dtype=int)
        sector_offsets = np.linspace(-1.0, 1.0, self.depth_gap_sector_count)
        best_detail = None
        previous_label = self.depth_selected_gap_label
        side_bias = clamp((left_clear - right_clear) / max(self.depth_close_m, 0.1), -1.0, 1.0)
        lane_gap_target = clamp(-self.lane_error / max(self.lane_error_clip, 1e-3), -0.75, 0.75)
        corridor_anchor = self.corridor_target_offset if keep_corridor and abs(self.corridor_target_offset) > 0.08 else 0.0
        continuity_scale = 0.0 if (emergency_now and self.gap_unlock_on_emergency) else 1.0
        center_open_bonus_scale = 0.12 * clamp(
            (center_clear - self.depth_stop_m) / max(self.depth_close_m - self.depth_stop_m, 1e-3),
            0.0,
            1.25,
        )
        required_gap_clearance = self.required_gap_clearance_m
        tight_gap_clearance = self.tight_gap_clearance_m
        sector_details = []
        for idx, offset in enumerate(sector_offsets):
            x0 = int(sector_edges[idx])
            x1 = int(sector_edges[idx + 1])
            if x1 <= x0:
                continue
            sector_mask = valid[:, x0:x1]
            sector_vals = roi[:, x0:x1][sector_mask]
            if sector_vals.size == 0:
                clearance = self.depth_near_m
            else:
                clearance = float(np.nanpercentile(sector_vals, self.depth_clearance_percentile))
            sector_close = int(np.count_nonzero(close[:, x0:x1]))
            sector_ratio = float(sector_close) / float(max(1, close[:, x0:x1].size))
            continuity = max(0.0, 1.0 - abs(offset - corridor_anchor) / 0.65)
            continuity_bonus = continuity_scale * self.continuity_bonus_gain * continuity
            jump_penalty = continuity_scale * self.lateral_jump_penalty * max(0.0, abs(offset - corridor_anchor) - 0.18)
            score = 1.5 * clearance
            score -= 1.2 * sector_ratio
            score -= self.depth_gap_heading_penalty * abs(offset)
            score -= self.gap_lateral_penalty * abs(offset)
            score -= self.gap_center_bias * abs(offset - 0.45 * lane_gap_target)
            score += center_open_bonus_scale * max(0.0, 1.0 - abs(offset))
            score += continuity_bonus
            score -= jump_penalty
            score += (-offset) * side_bias * 0.30
            clearance_shortfall = max(0.0, required_gap_clearance - clearance)
            if clearance_shortfall > 0.0:
                score -= 1.15 + 2.8 * clearance_shortfall
            elif clearance < tight_gap_clearance:
                score -= 0.35 * (tight_gap_clearance - clearance)
            else:
                score += 0.08 * min(clearance - tight_gap_clearance, 0.40)
            if single_side == 'left' and offset < -0.12:
                score -= 0.85 + 0.45 * abs(offset)
            elif single_side == 'right' and offset > 0.12:
                score -= 0.85 + 0.45 * abs(offset)
            if abs(offset) < 0.15:
                score -= self.depth_gap_center_penalty
            if gap_label(offset) == previous_label:
                score += self.depth_gap_hysteresis
            detail = {
                'offset': float(offset),
                'score': score,
                'continuity_bonus': continuity_bonus,
                'clearance': clearance,
                'ratio': sector_ratio,
            }
            sector_details.append(detail)
            if best_detail is None or score > best_detail['score']:
                best_detail = detail

        if best_detail is None:
            self.depth_reason_code = 'gap_not_computed'
            self.publish_obstacle_debug_topics()
            return

        left_detail = max(
            (detail for detail in sector_details if detail['offset'] <= -0.18),
            key=lambda item: item['score'],
            default=None,
        )
        right_detail = max(
            (detail for detail in sector_details if detail['offset'] >= 0.18),
            key=lambda item: item['score'],
            default=None,
        )
        center_detail = min(
            sector_details,
            key=lambda item: abs(item['offset']),
        )
        previous_side = previous_label if previous_label in ('LEFT', 'RIGHT') else ''
        obstacle_side_hint = 0.0
        if self.critical_roi_point_count >= self.critical_roi_min_points and abs(self.critical_roi_mean_y) >= 0.05:
            obstacle_side_hint = self.critical_roi_mean_y
        elif self.duba_var and abs(self.duba_konumu) >= 0.05:
            obstacle_side_hint = self.duba_konumu
        left_risk = (
            1.15 * left_close_ratio
            + 2.60 * max(0.0, required_gap_clearance - left_clear)
            + 0.30 * max(0.0, center_clear - left_clear)
        )
        right_risk = (
            1.15 * right_close_ratio
            + 2.60 * max(0.0, required_gap_clearance - right_clear)
            + 0.30 * max(0.0, center_clear - right_clear)
        )
        left_score = left_clear - left_risk
        right_score = right_clear - right_risk
        if previous_side == 'LEFT':
            left_score += self.gap_switch_margin
        elif previous_side == 'RIGHT':
            right_score += self.gap_switch_margin
        self.depth_left_risk = left_risk
        self.depth_right_risk = right_risk
        self.depth_left_gap_score = left_score
        self.depth_right_gap_score = right_score
        side_clearance_advantage = max(left_clear, right_clear) - center_clear
        side_preferred = (
            abs(obstacle_side_hint) >= 0.10
            or obstacle_now
            or blocked_frame
            or side_clearance_advantage >= 0.10
        )

        current_offset_ref = self.depth_selected_gap_offset if abs(self.depth_selected_gap_offset) > 1e-3 else corridor_anchor
        has_corridor_memory = abs(current_offset_ref) >= 0.10 or now_ns < self.corridor_active_until_ns
        current_detail = min(sector_details, key=lambda item: abs(item['offset'] - current_offset_ref))
        chosen_detail = best_detail if not has_corridor_memory else current_detail
        switch_reason = 'acquire' if not has_corridor_memory else 'continuity_hold'
        current_unsafe = keep_corridor and has_corridor_memory and (
            current_detail['clearance'] < required_gap_clearance
            or current_detail['ratio'] > 0.36
            or current_detail['score'] < (self.depth_gap_min_score - 0.18)
        )
        if emergency_now and self.gap_unlock_on_emergency:
            chosen_detail = best_detail
            switch_reason = 'emergency_unlock'
        elif current_unsafe:
            chosen_detail = best_detail
            switch_reason = 'current_unsafe'
        elif best_detail['score'] > current_detail['score'] + self.gap_switch_margin:
            chosen_detail = best_detail
            switch_reason = 'better_gap'
        elif abs(best_detail['offset'] - current_detail['offset']) <= 0.18:
            chosen_detail = best_detail
            switch_reason = 'continuous_refine'

        raw_gap_offset = best_detail['offset']
        center_occupied = blocked_frame or obstacle_now or pointcloud_confident or self.critical_obstacle_now
        side_only_mode = center_occupied or (hard_corridor_signal and side_preferred)
        if side_only_mode:
            chosen_side_label = 'BLOCKED'
            chosen_side_detail = None
            forced_side_label = ''
            left_safe = (
                left_detail is not None
                and left_clear >= required_gap_clearance
                and left_score >= (self.depth_gap_min_score - 0.10)
            )
            right_safe = (
                right_detail is not None
                and right_clear >= required_gap_clearance
                and right_score >= (self.depth_gap_min_score - 0.10)
            )
            if abs(obstacle_side_hint) >= 0.10:
                forced_side_label = 'RIGHT' if obstacle_side_hint > 0.0 else 'LEFT'
            if forced_side_label == 'LEFT' and left_safe and left_score >= right_score - 0.25:
                chosen_side_label = 'LEFT'
                chosen_side_detail = left_detail
                switch_reason = 'obstacle_side_flip_left'
            elif forced_side_label == 'RIGHT' and right_safe and right_score >= left_score - 0.25:
                chosen_side_label = 'RIGHT'
                chosen_side_detail = right_detail
                switch_reason = 'obstacle_side_flip_right'
            elif previous_side == 'LEFT' and left_safe and left_score >= right_score - self.gap_switch_margin:
                chosen_side_label = 'LEFT'
                chosen_side_detail = left_detail
                switch_reason = 'side_commit_left'
            elif previous_side == 'RIGHT' and right_safe and right_score >= left_score - self.gap_switch_margin:
                chosen_side_label = 'RIGHT'
                chosen_side_detail = right_detail
                switch_reason = 'side_commit_right'
            elif left_safe and (not right_safe or left_score >= right_score):
                chosen_side_label = 'LEFT'
                chosen_side_detail = left_detail
                switch_reason = 'clearance_left'
            elif right_safe:
                chosen_side_label = 'RIGHT'
                chosen_side_detail = right_detail
                switch_reason = 'clearance_right'
            if chosen_side_detail is not None:
                chosen_detail = chosen_side_detail
                raw_gap_offset = chosen_side_detail['offset']
            else:
                chosen_detail = best_detail
                raw_gap_offset = 0.0
        else:
            chosen_side_label = ''
        mapped_corridor_target = self.map_gap_offset_to_corridor_target(chosen_detail['offset'])
        candidate_strong = (
            hard_corridor_signal
            and side_preferred
            and abs(mapped_corridor_target) >= 0.18
            and chosen_detail['clearance'] >= required_gap_clearance
            and chosen_side_label != 'BLOCKED'
            and (
                chosen_detail['clearance'] > center_clear + 0.10
                or obstacle_now
                or blocked_frame
                or pointcloud_confident
            )
        )
        clear_path_preferred = (
            not obstacle_now
            and not blocked_frame
            and not self.depth_obstacle
            and not self.depth_emergency
            and not pointcloud_confident
            and abs(obstacle_side_hint) < max(0.08, 0.60 * self.duba_center_escape_y)
            and center_clear >= max(left_clear, right_clear) - 0.02
            and center_detail['clearance'] >= (required_gap_clearance - 0.02)
            and center_ratio < 0.05
            and upper_ratio < max(0.02, 1.5 * self.depth_upper_ratio_threshold)
        )
        if clear_path_preferred:
            chosen_detail = center_detail
            chosen_side_label = ''
            raw_gap_offset = 0.0
            mapped_corridor_target = 0.0
            candidate_strong = False
        corridor_enabled = candidate_strong or (
            keep_corridor and abs(self.corridor_target_offset) >= 0.10
        )
        corridor_gating_reason = 'disabled_no_signal'
        reset_reason = 'hold'
        if clear_path_preferred:
            corridor_enabled = False
            self.corridor_target_offset = 0.0
            self.corridor_active_until_ns = 0
            self.return_to_center_until_ns = 0
            corridor_gating_reason = 'clear_path_center'
            reset_reason = 'clear_path_center'
            switch_reason = 'clear_path_center'
        if corridor_enabled:
            if hard_corridor_signal:
                self.depth_context_last_ns = now_ns
            if candidate_strong:
                self.corridor_active_until_ns = now_ns + int(max(self.obstacle_context_sec, self.recover_duration) * 1e9)
                corridor_gating_reason = 'candidate_active'
            elif hard_corridor_signal:
                self.corridor_active_until_ns = now_ns + int(0.5 * max(self.obstacle_context_sec, self.recover_duration) * 1e9)
                corridor_gating_reason = 'signal_hold'
            elif now_ns < self.corridor_active_until_ns:
                corridor_gating_reason = 'active_hold'
            else:
                corridor_gating_reason = 'context_hold'
            memory_target = self.corridor_target_offset
            memory_gain = 0.0
            if candidate_strong:
                memory_target = mapped_corridor_target
                memory_gain = self.corridor_memory_gain
            elif hard_corridor_signal and abs(mapped_corridor_target) >= 0.10:
                memory_target = mapped_corridor_target
                memory_gain = min(self.corridor_memory_gain, 0.18)
            if emergency_now and self.gap_unlock_on_emergency:
                memory_gain = max(memory_gain, 0.60)
            if memory_gain > 0.0:
                self.corridor_target_offset = (
                    (1.0 - memory_gain) * self.corridor_target_offset + memory_gain * memory_target
                )
            elif corridor_gating_reason == 'context_hold':
                self.corridor_target_offset *= 0.94
            reset_reason = 'hold'
        else:
            decay = 0.10 if abs(self.corridor_target_offset) >= 0.20 else 0.18
            self.corridor_target_offset *= (1.0 - decay)
            if abs(self.corridor_target_offset) < 0.02:
                self.corridor_target_offset = 0.0
                reset_reason = 'decayed_to_zero'
            else:
                reset_reason = 'soft_decay_no_signal'
        if (
            lane_recent
            and not hard_corridor_signal
            and not keep_corridor
            and now_ns >= self.corridor_active_until_ns
            and not self.return_to_center_active(now_ns)
        ):
            self.corridor_target_offset = 0.0
            self.corridor_active_until_ns = 0
            corridor_enabled = False
            corridor_gating_reason = 'lane_priority'
            reset_reason = 'lane_visible'
        self.corridor_target_offset = clamp(self.corridor_target_offset, -1.0, 1.0)
        left_recent, right_recent = self.active_boundaries(now_ns)
        if left_recent and right_recent:
            self.corridor_target_offset = clamp(self.corridor_target_offset, -self.lane_corridor_cap, self.lane_corridor_cap)
        elif left_recent or right_recent:
            self.corridor_target_offset = clamp(self.corridor_target_offset, -0.36, 0.36)
        self.raw_gap_offset = raw_gap_offset
        self.mapped_corridor_target = mapped_corridor_target
        self.smoothed_corridor_target = self.corridor_target_offset
        self.corridor_enabled_state = corridor_enabled
        self.corridor_gating_reason = corridor_gating_reason
        self.corridor_reset_reason = reset_reason
        self.depth_gap_raw_offset = raw_gap_offset
        self.depth_gap_offset = self.corridor_target_offset
        self.depth_selected_gap_offset = chosen_detail['offset']
        self.depth_selected_gap_clearance = chosen_detail['clearance']
        label_offset = self.corridor_target_offset if abs(self.corridor_target_offset) >= 0.10 else mapped_corridor_target
        if chosen_side_label in ('LEFT', 'RIGHT'):
            self.depth_selected_gap_label = chosen_side_label
        elif center_occupied and abs(label_offset) < 0.10:
            self.depth_selected_gap_label = 'BLOCKED'
        else:
            label_offset = label_offset if abs(label_offset) >= 0.10 else 0.0
            self.depth_selected_gap_label = gap_label(label_offset)
        self.depth_selected_gap_score = chosen_detail['score']
        self.depth_gap_continuity_bonus = chosen_detail['continuity_bonus']
        self.depth_gap_switch_reason = switch_reason
        self.depth_left_clearance = left_clear
        self.depth_right_clearance = right_clear
        self.depth_center_clearance = center_clear
        self.depth_center_ratio = center_ratio
        self.depth_upper_ratio = upper_ratio
        self.depth_min_dist = float(np.nanpercentile(roi[valid], 15))

        if blocked_frame:
            self.depth_reason_code = 'blocked_frame'
            if self.blocked_start_ns <= 0:
                self.blocked_start_ns = now_ns
            held_sec = (now_ns - self.blocked_start_ns) / 1e9
            self.blocked_persistent = held_sec >= self.blocked_hold_sec
        else:
            self.blocked_start_ns = 0
            self.blocked_persistent = False

        if obstacle_now:
            self.depth_reason_code = 'center_blocked'
            self.depth_obstacle = True
            self.depth_emergency = emergency_now
            self.depth_last_seen_ns = now_ns
            self.depth_context_last_ns = now_ns
        else:
            held_sec = (now_ns - self.depth_last_seen_ns) / 1e9 if self.depth_last_seen_ns > 0 else float('inf')
            self.depth_obstacle = held_sec <= self.depth_hold_sec
            if not self.depth_obstacle:
                self.depth_emergency = False
            elif corridor_enabled:
                self.depth_reason_code = 'context_hold'

        if candidate_strong and not blocked_frame:
            self.depth_reason_code = 'candidate_active'
        elif corridor_enabled and not obstacle_now and not blocked_frame:
            self.depth_reason_code = 'corridor_hold'
        elif not corridor_enabled and best_detail['clearance'] < required_gap_clearance:
            self.depth_reason_code = 'clearance_too_small'
        elif not corridor_enabled and best_detail['score'] < self.depth_gap_min_score:
            self.depth_reason_code = 'gap_score_too_low'
        self.publish_obstacle_debug_topics()

        if now_ns - self.depth_debug_log_ns > int(1e9) and (self.depth_obstacle or self.blocked_persistent or abs(self.depth_gap_offset) > 0.25):
            self.depth_debug_log_ns = now_ns
            self.get_logger().info(
                '[OBS_DEBUG] '
                f'reason={self.depth_reason_code} '
                f'blocked={self.blocked_persistent} obs={self.depth_obstacle} emerg={self.depth_emergency} '
                f'gap={self.depth_selected_gap_label} raw_gap_offset={self.raw_gap_offset:+.2f} '
                f'mapped_corridor_target={self.mapped_corridor_target:+.2f} '
                f'smoothed_corridor_target={self.smoothed_corridor_target:+.2f} '
                f'corridor_enabled={self.corridor_enabled_state} gating={self.corridor_gating_reason} '
                f'reset={self.corridor_reset_reason} '
                f'gap_offset={self.depth_gap_offset:+.2f} score={self.depth_selected_gap_score:+.2f} '
                f'gap_clear={self.depth_selected_gap_clearance:.2f} '
                f'cont_bonus={self.depth_gap_continuity_bonus:+.2f} switch={self.depth_gap_switch_reason} '
                f'center={self.depth_center_clearance:.2f} left={self.depth_left_clearance:.2f} right={self.depth_right_clearance:.2f} '
                f'center_ratio={self.depth_center_ratio:.2f} upper_ratio={self.depth_upper_ratio:.2f} min_dist={self.depth_min_dist:.2f}'
            )

    def signal_recent(self, stamp_ns: int, timeout_sec: float, now_ns: int) -> bool:
        return stamp_ns > 0 and (now_ns - stamp_ns) / 1e9 <= timeout_sec

    def lane_valid_recent(self, now_ns: int) -> bool:
        return self.lane_valid and self.signal_recent(self.lane_stamp_ns, self.lane_timeout_sec, now_ns)

    def active_boundaries(self, now_ns: int) -> Tuple[bool, bool]:
        left_recent = self.signal_recent(self.left_lane_last_seen_ns, self.single_boundary_timeout_sec, now_ns)
        right_recent = self.signal_recent(self.right_lane_last_seen_ns, self.single_boundary_timeout_sec, now_ns)
        return left_recent, right_recent

    def obstacle_context_active(self, now_ns: int) -> bool:
        recent_depth = self.signal_recent(self.depth_context_last_ns, self.obstacle_context_sec, now_ns)
        recent_duba = self.signal_recent(self.duba_last_seen_ns, self.obstacle_context_sec, now_ns)
        near_duba = recent_duba and self.duba_mesafe <= (self.duba_center_trigger_m + 0.25)
        recent_side_intrusion = (
            self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
            and self.critical_roi_point_count >= self.critical_roi_min_points
            and self.critical_roi_min_x <= (self.critical_roi_forward_max_m + 0.25)
            and abs(self.critical_roi_mean_y) >= max(0.08, 0.75 * self.duba_center_escape_y)
        )
        visible_side_gap = (
            self.depth_frame_recent(now_ns)
            and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
            and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.02)
            and abs(self.depth_selected_gap_offset) >= self.depth_gap_min_offset
        )
        return (
            self.pointcloud_corridor_signal_active()
            or self.depth_obstacle
            or recent_depth
            or near_duba
            or recent_side_intrusion
            or visible_side_gap
            or now_ns < self.obstacle_recovery_until_ns
        )

    def map_gap_offset_to_corridor_target(self, offset: float) -> float:
        magnitude = abs(offset)
        if magnitude < 0.18:
            return 0.0
        if magnitude >= 0.80:
            target_mag = 0.62
        elif magnitude >= 0.50:
            target_mag = 0.42
        else:
            target_mag = 0.28
        return math.copysign(target_mag, offset)

    def lane_control_available(self) -> bool:
        return self.lane_state in (LaneState.NORMAL_LANE, LaneState.DEGRADED_LANE)

    def active_single_side(self, now_ns: int) -> str:
        left_recent, right_recent = self.active_boundaries(now_ns)
        if left_recent and not right_recent:
            return 'left'
        if right_recent and not left_recent:
            return 'right'
        return ''

    def depth_frame_recent(self, now_ns: int) -> bool:
        return self.signal_recent(self.depth_frame_stamp_ns, self.depth_frame_timeout_sec, now_ns)

    def pointcloud_corridor_signal_active(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        depth_recent = self.depth_frame_recent(now_ns)
        lateral_hint = max(abs(self.critical_roi_mean_y), abs(self.duba_konumu))
        lateral_threshold = max(0.10, 0.80 * self.duba_center_escape_y)
        if self.critical_center_supported:
            if depth_recent:
                return (
                    self.critical_roi_min_x <= (self.duba_center_trigger_m + 0.15)
                    and (
                        lateral_hint >= lateral_threshold
                        or self.depth_center_clearance < self.tight_gap_clearance_m
                    )
                )
            return (
                self.critical_roi_min_x <= max(0.80, self.duba_center_trigger_m - 0.20)
                and lateral_hint >= lateral_threshold
            )
        if (
            self.critical_roi_point_count >= self.critical_roi_min_points
            and self.critical_roi_min_x <= (self.duba_center_trigger_m + 0.20)
            and abs(self.critical_roi_mean_y) >= lateral_threshold
        ):
            return True
        if not self.duba_var:
            return False
        return (
            self.duba_mesafe <= self.duba_center_trigger_m
            and (
                abs(self.duba_konumu) >= lateral_threshold
                or (
                    depth_recent
                    and self.pointcloud_front_center_count >= self.critical_roi_min_points
                    and self.depth_center_clearance < self.tight_gap_clearance_m
                )
            )
        )

    def reset_corridor_state(self, reason: str) -> None:
        self.corridor_target_offset = 0.0
        self.corridor_active_until_ns = 0
        self.corridor_enabled_state = False
        self.corridor_gating_reason = 'authority_reset'
        self.corridor_reset_reason = reason
        self.smoothed_corridor_target = 0.0
        self.depth_gap_offset = 0.0
        self.corridor_error = 0.0
        self.corridor_term_preclamp = 0.0
        self.corridor_term_postclamp = 0.0
        self.return_to_center_until_ns = 0

    def corridor_gap_available(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        if not (
            self.avoidance_required(now_ns)
            or self.return_to_center_active(now_ns)
            or abs(self.corridor_target_offset) >= 0.10
        ):
            return False
        if self.blocked_persistent and self.depth_center_clearance <= self.depth_stop_m:
            return False
        center_open = (
            self.depth_center_clearance >= self.tight_gap_clearance_m
            and self.depth_center_ratio < max(0.12, self.obstacle_center_ratio_threshold)
        )
        scored_gap = (
            self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.05)
            and self.depth_selected_gap_score >= (self.depth_gap_min_score - 0.10)
        )
        return center_open or scored_gap

    def side_bypass_available(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        mapped_gap = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        remembered_gap = self.corridor_target_offset
        gap_target = remembered_gap if abs(remembered_gap) >= 0.10 else mapped_gap
        return (
            abs(gap_target) >= 0.10
            and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.05)
            and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
        )

    def avoidance_required(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        lateral_threshold = max(0.10, 0.80 * self.duba_center_escape_y)
        lateral_hint = max(abs(self.critical_roi_mean_y), abs(self.duba_konumu))
        side_clearance_advantage = max(self.depth_left_clearance, self.depth_right_clearance) - self.depth_center_clearance
        clear_side_available = (
            self.side_bypass_available(now_ns)
            and side_clearance_advantage >= 0.10
        )
        center_compressed = (
            self.depth_obstacle
            or self.depth_emergency
            or self.blocked_persistent
            or self.depth_center_clearance < self.obstacle_center_clearance_m
            or self.depth_center_ratio > self.obstacle_center_ratio_threshold
        )
        lateral_obstacle = (
            lateral_hint >= lateral_threshold
            and self.critical_roi_min_x <= (self.duba_center_trigger_m + 0.20)
        )
        return (
            (clear_side_available and (center_compressed or lateral_obstacle))
            or now_ns < self.critical_avoid_until_ns
        )

    def return_to_center_active(self, now_ns: int) -> bool:
        return now_ns < self.return_to_center_until_ns and abs(self.corridor_target_offset) >= 0.02

    def start_return_to_center(self, now_ns: int) -> None:
        self.return_to_center_until_ns = now_ns + int(self.return_to_center_sec * 1e9)

    def decay_return_to_center(self, now_ns: int, strong: bool = False) -> None:
        if not self.return_to_center_active(now_ns):
            return
        decay = self.return_to_center_decay if not strong else min(0.88, self.return_to_center_decay + 0.10)
        self.corridor_target_offset *= (1.0 - decay)
        self.smoothed_corridor_target = self.corridor_target_offset
        self.depth_gap_offset = self.corridor_target_offset
        if abs(self.corridor_target_offset) < 0.02:
            self.corridor_target_offset = 0.0
            self.smoothed_corridor_target = 0.0
            self.depth_gap_offset = 0.0
            self.return_to_center_until_ns = 0
            self.corridor_enabled_state = False
            self.corridor_gating_reason = 'return_to_center_done'
            self.corridor_reset_reason = 'return_to_center_done'
            self.depth_selected_gap_label = 'CENTER'

    def in_lane_bypass_active(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        if not self.obstacle_context_active(now_ns):
            return False
        if self.side_bypass_available(now_ns) and self.avoidance_required(now_ns):
            return True
        return False

    def critical_obstacle_blocking(self, forward_limit_m: float, lateral_margin_m: float = 0.0) -> bool:
        return (
            self.critical_center_supported
            and
            self.critical_roi_point_count >= self.critical_roi_min_points
            and self.critical_roi_min_x <= forward_limit_m
            and self.footprint_overlap(self.critical_roi_min_abs_y, lateral_margin_m)
        )

    def select_critical_escape_offset(self) -> float:
        gap_target = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        if (
            abs(gap_target) >= 0.18
            and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.06)
            and self.depth_selected_gap_label in ('LEFT', 'RIGHT')
        ):
            return gap_target
        if self.depth_right_gap_score > self.depth_left_gap_score + 0.05:
            return self.critical_escape_offset_m
        if self.depth_left_gap_score > self.depth_right_gap_score + 0.05:
            return -self.critical_escape_offset_m
        obstacle_side = self.critical_roi_mean_y if self.critical_roi_point_count > 0 else self.duba_konumu
        if abs(obstacle_side) > 1e-3:
            return math.copysign(self.critical_escape_offset_m, obstacle_side)
        if abs(self.critical_escape_offset) > 0.05:
            return self.critical_escape_offset
        return self.critical_escape_offset_m if self.depth_right_gap_score >= self.depth_left_gap_score else -self.critical_escape_offset_m

    def update_critical_avoid_state(self, now_ns: int) -> bool:
        if not self.critical_center_supported and not self.depth_emergency:
            self.critical_avoid_until_ns = 0
            self.critical_escape_offset = 0.0
            self.critical_avoid_smoothed = 0.0
            return False
        critical_now = (
            self.depth_emergency
            or self.critical_obstacle_blocking(self.critical_roi_forward_max_m)
        )
        if critical_now:
            if abs(self.critical_escape_offset) < 0.05:
                self.critical_escape_offset = self.select_critical_escape_offset()
            self.critical_obstacle_last_seen_ns = now_ns
            self.critical_avoid_until_ns = now_ns + int(self.critical_commit_sec * 1e9)
            self.depth_context_last_ns = now_ns
            return True
        still_blocking = (
            self.depth_emergency
            or self.critical_obstacle_blocking(
                self.critical_release_forward_m,
                self.critical_release_lateral_margin_m,
            )
        )
        if now_ns < self.critical_avoid_until_ns or still_blocking:
            if abs(self.critical_escape_offset) < 0.05:
                self.critical_escape_offset = self.select_critical_escape_offset()
            return True
        self.critical_avoid_until_ns = 0
        self.critical_escape_offset = 0.0
        return False

    def compute_corridor_authority_term(self, target_offset: float, limit: float) -> float:
        self.corridor_error = target_offset
        self.corridor_term_preclamp = -self.corridor_follow_gain * self.corridor_error
        self.corridor_term_postclamp = clamp(self.corridor_term_preclamp, -limit, limit)
        return self.corridor_term_postclamp

    def build_stop_command(self, reason: str, route_term_raw: float) -> ControlCommand:
        self.reset_corridor_state(reason)
        self.corner_mode = False
        self.last_gap_assist_active = False
        return ControlCommand(
            authority=ControlAuthority.BLOCKED_STOP,
            speed=0.0,
            desired_angular=0.0,
            route_term_raw=route_term_raw,
            reason=reason,
        )

    def build_no_lane_command(self, authority: ControlAuthority, route_term_raw: float) -> ControlCommand:
        route_term_used = 0.0
        desired_angular = 0.0
        self.reset_corridor_state('no_gap_fallback')
        if authority == ControlAuthority.NO_LANE_COAST:
            if self.route_enabled:
                route_term_used = clamp(
                    route_term_raw * self.route_weight_no_lane_coast,
                    -self.no_lane_route_limit_coast,
                    self.no_lane_route_limit_coast,
                )
                desired_angular = route_term_used
            speed = self.coast_speed
        else:
            if self.route_enabled:
                route_term_used = clamp(
                    route_term_raw * self.route_weight_no_lane_slow,
                    -self.no_lane_route_limit_slow,
                    self.no_lane_route_limit_slow,
                )
                desired_angular = route_term_used
            speed = self.slow_speed
        self.corner_mode = False
        self.last_gap_assist_active = False
        return ControlCommand(
            authority=authority,
            speed=speed,
            desired_angular=desired_angular,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            lane_conf=0.0,
            reason=self.lane_state.name.lower(),
        )

    def build_lane_follow_command(self, now_ns: int, route_term_raw: float, route_speed: float, side: str) -> ControlCommand:
        desired, speed, lane_term, _gap_term, lane_conf, route_term_used, _route_weight = self.compute_lane_state_command(
            now_ns,
            route_term_raw,
            side,
            allow_gap_assist=False,
        )
        if self.route_enabled and route_speed > 0.0 and speed > 0.0:
            speed = min(speed, route_speed)
        if not self.in_lane_bypass_active(now_ns):
            if self.return_to_center_active(now_ns):
                self.decay_return_to_center(now_ns, strong=True)
                self.corridor_gating_reason = 'return_to_center'
                self.corridor_reset_reason = 'decay'
            elif (
                self.obstacle_context_active(now_ns)
                and self.depth_frame_recent(now_ns)
                and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
                and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.02)
            ):
                self.corridor_active_until_ns = max(
                    self.corridor_active_until_ns,
                    now_ns + int(0.35 * 1e9),
                )
                self.corridor_enabled_state = True
                self.corridor_gating_reason = 'lane_overlap_hold'
                self.corridor_reset_reason = 'hold'
            else:
                self.reset_corridor_state('lane_priority')
            self.last_gap_assist_active = False
        return ControlCommand(
            authority=ControlAuthority.LANE_FOLLOW,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            lane_conf=lane_conf,
            reason=self.lane_state.name.lower(),
        )

    def build_in_lane_avoid_command(
        self,
        now_ns: int,
        route_term_raw: float,
        route_speed: float,
        side: str,
    ) -> ControlCommand:
        desired, speed, lane_term, gap_term, lane_conf, route_term_used, _route_weight = self.compute_lane_state_command(
            now_ns,
            route_term_raw,
            side,
            allow_gap_assist=True,
        )
        target_offset = self.corridor_target_offset
        if abs(target_offset) < 0.05:
            target_offset = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        limit = self.depth_gap_limit if self.depth_center_clearance < self.tight_gap_clearance_m else self.depth_gap_corner_limit
        corridor_term = self.compute_corridor_authority_term(target_offset, limit)
        desired = (
            self.lane_weight_during_avoid * lane_term
            + corridor_term
            + gap_term
            + route_term_used
        )
        desired = clamp(desired, -self.duba_max_angular, self.duba_max_angular)
        if self.route_enabled and route_speed > 0.0 and speed > 0.0:
            speed = min(speed, route_speed)
        if self.lane_state == LaneState.NORMAL_LANE:
            speed = min(speed, self.obstacle_speed)
        else:
            speed = min(speed, max(self.single_lane_invalid_speed, 0.34))
        if self.depth_center_clearance < self.depth_stop_m + 0.12:
            speed = min(speed, 0.18)
        self.last_gap_assist_active = abs(corridor_term) > 1e-3 or abs(gap_term) > 1e-3
        return ControlCommand(
            authority=ControlAuthority.IN_LANE_AVOID,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            gap_term=gap_term,
            corridor_term=corridor_term,
            lane_conf=lane_conf,
            reason='single_line_bypass' if self.lane_state == LaneState.DEGRADED_LANE else 'in_lane_bypass',
        )

    def build_corridor_gap_command(self, now_ns: int, route_term_raw: float) -> ControlCommand:
        self.corner_mode = False
        target_offset = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        if abs(target_offset) < 0.05:
            target_offset = self.map_gap_offset_to_corridor_target(self.raw_gap_offset)
        if abs(target_offset) < 0.10:
            return self.build_no_lane_command(ControlAuthority.NO_LANE_SLOW, route_term_raw)
        blend = 0.45 if abs(target_offset) >= 0.10 else 0.20
        self.corridor_target_offset = (
            (1.0 - blend) * self.corridor_target_offset + blend * target_offset
        )
        if abs(target_offset) < 0.05:
            self.corridor_target_offset *= 0.65
        if abs(self.corridor_target_offset) < 0.03:
            self.corridor_target_offset = 0.0
        self.corridor_target_offset = clamp(self.corridor_target_offset, -0.52, 0.52)
        self.corridor_active_until_ns = now_ns + int(self.depth_frame_timeout_sec * 1e9)
        self.corridor_enabled_state = True
        self.corridor_gating_reason = 'lane_missing_fallback'
        self.corridor_reset_reason = 'active'
        self.smoothed_corridor_target = self.corridor_target_offset
        self.depth_gap_offset = self.corridor_target_offset
        limit = self.depth_gap_limit if self.depth_center_clearance < self.tight_gap_clearance_m else self.depth_gap_corner_limit
        corridor_term = self.compute_corridor_authority_term(self.corridor_target_offset, limit)
        speed = 0.24 if self.depth_center_clearance > self.tight_gap_clearance_m else 0.18
        speed = min(speed, self.coast_speed if self.lane_state == LaneState.NO_LANE_COAST else max(self.slow_speed, speed))
        self.last_gap_assist_active = abs(corridor_term) > 1e-3
        return ControlCommand(
            authority=ControlAuthority.CORRIDOR_GAP,
            speed=speed,
            desired_angular=corridor_term,
            route_term_raw=route_term_raw,
            corridor_term=corridor_term,
            lane_conf=0.0,
            reason=self.depth_gap_switch_reason,
        )

    def build_critical_avoid_command(self, now_ns: int, route_term_raw: float) -> ControlCommand:
        self.corner_mode = False
        if abs(self.critical_escape_offset) < 0.05:
            self.critical_escape_offset = self.select_critical_escape_offset()
        escape_offset = self.critical_escape_offset
        closest_forward = self.critical_roi_min_x if self.critical_roi_min_x < 90.0 else self.critical_roi_forward_max_m
        if self.depth_frame_recent(now_ns):
            closest_forward = min(closest_forward, self.depth_center_clearance)
        if self.depth_center_clearance <= self.depth_stop_m and not self.side_bypass_available(now_ns):
            return self.build_stop_command('critical_stop', route_term_raw)
        intrusion_scale = clamp(
            self.critical_roi_intrusion_m / max(self.footprint_half_width_m, 1e-3),
            0.0,
            1.0,
        )
        distance_scale = clamp(
            (self.critical_roi_forward_max_m - closest_forward)
            / max(self.critical_roi_forward_max_m - self.critical_roi_forward_min_m, 1e-3),
            0.0,
            1.0,
        )
        turn_mag = self.critical_avoid_gain * abs(escape_offset) * (
            0.75 + 0.55 * distance_scale + 0.35 * intrusion_scale
        )
        target = -math.copysign(turn_mag, escape_offset)
        target = clamp(target, -self.critical_avoid_target_limit, self.critical_avoid_target_limit)
        if abs(target) < self.critical_avoid_min_turn:
            target = math.copysign(
                self.critical_avoid_min_turn,
                target if abs(target) > 1e-3 else -escape_offset,
            )
        self.critical_avoid_smoothed = (
            self.critical_avoid_ramp_alpha * target
            + (1.0 - self.critical_avoid_ramp_alpha) * self.critical_avoid_smoothed
        )
        desired = clamp(self.critical_avoid_smoothed, -self.duba_max_angular, self.duba_max_angular)
        if closest_forward < 0.45:
            speed = 0.18
        elif closest_forward < 0.70:
            speed = 0.28
        else:
            speed = min(self.obstacle_speed, 0.42)
        if self.depth_selected_gap_clearance < self.tight_gap_clearance_m:
            speed = min(speed, 0.20)
        self.corridor_target_offset = escape_offset
        self.corridor_active_until_ns = now_ns + int(self.critical_commit_sec * 1e9)
        self.corridor_enabled_state = True
        self.corridor_gating_reason = 'critical_avoid'
        self.corridor_reset_reason = 'committed'
        self.smoothed_corridor_target = self.corridor_target_offset
        self.depth_gap_offset = self.corridor_target_offset
        self.depth_selected_gap_label = 'RIGHT' if self.corridor_target_offset > 0.0 else 'LEFT'
        self.corridor_error = escape_offset
        self.corridor_term_preclamp = 0.0
        self.corridor_term_postclamp = 0.0
        self.last_gap_assist_active = True
        return ControlCommand(
            authority=ControlAuthority.CRITICAL_AVOID,
            speed=speed,
            desired_angular=desired,
            route_term_raw=route_term_raw,
            avoid_term=desired,
            reason='critical_roi',
        )

    def select_control_command(self, now_ns: int, route_term_raw: float, route_speed: float, side: str) -> ControlCommand:
        critical_active = self.update_critical_avoid_state(now_ns)
        if critical_active:
            return self.build_critical_avoid_command(now_ns, route_term_raw)
        if self.prev_obstacle_active:
            self.obstacle_recovery_until_ns = now_ns + int(self.obstacle_recovery_sec * 1e9)
        if not self.in_lane_bypass_active(now_ns):
            self.decay_return_to_center(now_ns, strong=self.lane_control_available())
        lane_overlap_hold = (
            self.lane_control_available()
            and self.avoidance_required(now_ns)
            and self.depth_frame_recent(now_ns)
            and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
            and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.02)
            and (
                abs(self.corridor_target_offset) >= 0.10
                or abs(self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)) >= 0.18
            )
        )
        if self.lane_control_available() and (self.in_lane_bypass_active(now_ns) or lane_overlap_hold):
            return self.build_in_lane_avoid_command(now_ns, route_term_raw, route_speed, side)
        if self.lane_control_available():
            return self.build_lane_follow_command(now_ns, route_term_raw, route_speed, side)
        if self.corridor_gap_available(now_ns):
            return self.build_corridor_gap_command(now_ns, route_term_raw)
        if self.lane_state == LaneState.BLOCKED_STOP:
            return self.build_stop_command('blocked_stop', route_term_raw)
        if self.lane_state == LaneState.NO_LANE_COAST:
            return self.build_no_lane_command(ControlAuthority.NO_LANE_COAST, route_term_raw)
        return self.build_no_lane_command(ControlAuthority.NO_LANE_SLOW, route_term_raw)

    def filter_speed_for_authority(self, speed: float, desired_angular: float, lane_conf: float, authority: ControlAuthority) -> float:
        speed = max(0.0, speed)
        if authority in (ControlAuthority.BLOCKED_STOP, ControlAuthority.NO_LANE_COAST, ControlAuthority.NO_LANE_SLOW, ControlAuthority.CORRIDOR_GAP):
            return speed
        speed *= max(0.60, 1.0 - self.speed_angular_gain * abs(desired_angular))
        if authority == ControlAuthority.CRITICAL_AVOID:
            return clamp(speed, 0.0, self.obstacle_speed)
        if self.lane_state == LaneState.DEGRADED_LANE:
            speed *= max(0.78, lane_conf + 0.35)
            if self.corner_mode:
                speed = max(speed, 0.34)
        else:
            speed *= max(0.82, lane_conf)
        if self.obstacle_context_active(self.get_clock().now().nanoseconds) and self.lane_state == LaneState.NORMAL_LANE:
            speed = min(speed, 0.70)
        return clamp(speed, 0.0, self.gps_hiz)

    def filter_angular_for_authority(self, desired: float, speed: float, authority: ControlAuthority) -> float:
        if authority == ControlAuthority.BLOCKED_STOP or speed <= 0.015:
            self.last_cmd_angular = 0.0
            return 0.0
        desired = clamp(desired, -self.angular_clamp, self.angular_clamp)
        if authority in (ControlAuthority.NO_LANE_COAST, ControlAuthority.NO_LANE_SLOW):
            if abs(desired) < 1e-3:
                self.last_cmd_angular = 0.0
                return 0.0
            rate_limit = self.no_lane_rate_limit
            smoothing = self.no_lane_smoothing
        elif authority == ControlAuthority.CRITICAL_AVOID:
            rate_limit = self.obstacle_rate_limit
            smoothing = self.obstacle_smoothing
        elif authority == ControlAuthority.CORRIDOR_GAP:
            rate_limit = self.single_rate_limit
            smoothing = self.single_smoothing
        elif self.lane_state == LaneState.DEGRADED_LANE or self.corner_mode:
            rate_limit = self.single_rate_limit
            smoothing = self.single_smoothing
        else:
            rate_limit = self.normal_rate_limit
            smoothing = self.normal_smoothing
        if desired * self.last_cmd_angular < 0.0 and authority in (ControlAuthority.CRITICAL_AVOID, ControlAuthority.CORRIDOR_GAP):
            self.last_cmd_angular *= 0.20
        max_delta = rate_limit * self.control_period
        delta = clamp(desired - self.last_cmd_angular, -max_delta, max_delta)
        rate_limited = self.last_cmd_angular + delta
        self.last_cmd_angular = smoothing * rate_limited + (1.0 - smoothing) * self.last_cmd_angular
        self.last_cmd_angular = clamp(self.last_cmd_angular, -self.max_angular_z, self.max_angular_z)
        return self.last_cmd_angular

    def publish_control_command(self, command: ControlCommand, twist: Twist) -> float:
        speed = self.filter_speed_for_authority(
            command.speed,
            command.desired_angular,
            command.lane_conf,
            command.authority,
        )
        angular = self.filter_angular_for_authority(command.desired_angular, speed, command.authority)
        twist.linear.x = speed
        twist.angular.z = angular
        return angular

    def publish_obstacle_summary(self, now_ns: int, command: ControlCommand) -> None:
        bias = 0.0
        speed_scale = 1.0
        pointcloud_recent = self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
        depth_recent = self.depth_frame_recent(now_ns)
        obstacle_active = (
            command.authority in (
                ControlAuthority.CRITICAL_AVOID,
                ControlAuthority.IN_LANE_AVOID,
                ControlAuthority.CORRIDOR_GAP,
            )
            or self.in_lane_bypass_active(now_ns)
        )
        emergency_stop = (
            command.authority == ControlAuthority.BLOCKED_STOP
            or self.blocked_persistent
        )
        if self.depth_emergency and not self.corridor_gap_available(now_ns):
            emergency_stop = True

        if command.authority == ControlAuthority.CRITICAL_AVOID:
            bias = clamp(command.desired_angular, -self.max_angular_z, self.max_angular_z)
            speed_scale = clamp(
                command.speed / max(self.normal_lane_speed, 1e-3),
                0.48,
                0.76,
            )
        elif command.authority == ControlAuthority.IN_LANE_AVOID:
            bypass_bias = -self.avoid_bias_gain * self.corridor_target_offset
            if abs(command.corridor_term) > 1e-3:
                bypass_bias += 0.55 * command.corridor_term
            if abs(command.gap_term) > 1e-3:
                bypass_bias += 0.45 * command.gap_term
            bias = clamp(bypass_bias, -self.avoid_bias_limit, self.avoid_bias_limit)
            speed_scale = clamp(
                command.speed / max(self.normal_lane_speed, 1e-3),
                0.60,
                0.86,
            )
        elif command.authority == ControlAuthority.CORRIDOR_GAP:
            bias = clamp(command.desired_angular, -self.depth_gap_limit, self.depth_gap_limit)
            speed_scale = clamp(
                command.speed / max(self.normal_lane_speed, 1e-3),
                0.60,
                0.90,
            )
        elif self.return_to_center_active(now_ns):
            decay_ratio = clamp(
                (self.return_to_center_until_ns - now_ns) / max(self.return_to_center_sec * 1e9, 1.0),
                0.0,
                1.0,
            )
            bias = clamp(-0.40 * self.depth_gap_offset * decay_ratio, -0.12, 0.12)
            speed_scale = clamp(
                min(max(command.speed, self.obstacle_speed), 0.96) / max(self.normal_lane_speed, 1e-3),
                0.72,
                1.00,
            )
        elif self.obstacle_context_active(now_ns) and not self.lane_control_available():
            bias = clamp(-0.55 * self.depth_gap_offset, -0.22, 0.22)
            speed_scale = clamp(
                min(max(command.speed, self.obstacle_speed), 0.88) / max(self.normal_lane_speed, 1e-3),
                0.60,
                0.96,
            )

        if emergency_stop:
            speed_scale = 0.0
            obstacle_active = True

        self.obstacle_unknown = (
            not obstacle_active
            and (
                (not pointcloud_recent and not depth_recent)
                or self.obstacle_reason_code in ('no_points_received', 'missing_frame', 'pointcloud_runtime_error')
                or self.obstacle_reason_code == 'filtered_out_by_height'
                or self.depth_reason_code in ('depth_no_valid_pixels', 'depth_bad_encoding', 'depth_decode_error')
            )
        )
        if obstacle_active:
            self.last_obstacle_active_ns = now_ns
        else:
            obstacle_active = self.signal_recent(self.last_obstacle_active_ns, self.obstacle_hold_time_sec, now_ns)
        if self.obstacle_unknown and not obstacle_active:
            self.obstacle_reason_code = self.depth_reason_code if depth_recent else self.obstacle_reason_code
        self.summary_obstacle_active = bool(obstacle_active)
        self.summary_obstacle_unknown = bool(self.obstacle_unknown)
        self.summary_avoid_latched = bool(
            now_ns < self.critical_avoid_until_ns or self.return_to_center_active(now_ns)
        )

        bias_msg = Float32()
        bias_msg.data = float(bias)
        self.obstacle_bias_pub.publish(bias_msg)

        speed_scale_msg = Float32()
        speed_scale_msg.data = float(clamp(speed_scale, 0.0, 1.0))
        self.obstacle_speed_scale_pub.publish(speed_scale_msg)

        emergency_msg = Bool()
        emergency_msg.data = bool(emergency_stop)
        self.emergency_stop_pub.publish(emergency_msg)

        active_msg = Bool()
        active_msg.data = bool(obstacle_active)
        self.obstacle_active_pub.publish(active_msg)

        unknown_msg = Bool()
        unknown_msg.data = bool(self.obstacle_unknown)
        self.obstacle_unknown_pub.publish(unknown_msg)
        self.publish_obstacle_debug_topics()

    def update_lane_state(self, now_ns: int) -> None:
        if self.blocked_persistent:
            self.lane_state = LaneState.BLOCKED_STOP
            return

        left_recent, right_recent = self.active_boundaries(now_ns)
        both_recent = left_recent and right_recent
        any_recent = left_recent or right_recent
        lane_recent = self.lane_valid_recent(now_ns)

        desired_state = None
        if both_recent:
            desired_state = LaneState.NORMAL_LANE
        elif any_recent or lane_recent:
            desired_state = LaneState.DEGRADED_LANE

        if desired_state is not None:
            if self.lane_state in (LaneState.NO_LANE_COAST, LaneState.NO_LANE_SLOW, LaneState.BLOCKED_STOP):
                if self.recover_start_ns <= 0:
                    self.recover_start_ns = now_ns
                recover_age = (now_ns - self.recover_start_ns) / 1e9
                if recover_age < self.recover_debounce_sec:
                    return
            self.recover_start_ns = 0
            self.lane_lost_ns = 0
            self.lane_state = desired_state
            return

        self.recover_start_ns = 0
        if self.lane_lost_ns <= 0:
            self.lane_lost_ns = now_ns
        lost_age = (now_ns - self.lane_lost_ns) / 1e9
        if lost_age <= self.coast_duration:
            self.lane_state = LaneState.NO_LANE_COAST
        else:
            self.lane_state = LaneState.NO_LANE_SLOW

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

    def select_start_index(self) -> Tuple[int, float, float]:
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

    def compute_route_term(self) -> Tuple[float, float]:
        if not self.route_enabled:
            return 0.0, self.lane_only_speed
        if self.tamamlandi:
            return 0.0, 0.0
        if not self.baslangic_bulundu:
            best_idx, min_dist, _ = self.select_start_index()
            if min_dist > self.start_max_dist:
                return 0.0, 0.0
            self.hedef_index = best_idx
            self.baslangic_bulundu = True
        if self.hedef_index >= len(self.rota) - 1:
            self.tamamlandi = True
            return 0.0, 0.0

        while self.hedef_index < len(self.rota) - 1:
            hx, hy = self.rota[self.hedef_index]
            if math.hypot(hx - self.x, hy - self.y) < self.lookahead_dist:
                self.hedef_index += 1
            else:
                break

        anchor_idx = max(0, self.hedef_index - 1)
        preview_idx = anchor_idx
        preview_target_dist = self.lookahead_dist * self.route_preview_multiplier
        preview_dist = 0.0
        while preview_idx < len(self.rota) - 1 and preview_dist < preview_target_dist:
            x0, y0 = self.rota[preview_idx]
            x1, y1 = self.rota[preview_idx + 1]
            preview_dist += math.hypot(x1 - x0, y1 - y0)
            preview_idx += 1

        anchor_x, anchor_y = self.rota[anchor_idx]
        preview_x, preview_y = self.rota[preview_idx]
        path_yaw = math.atan2(preview_y - anchor_y, preview_x - anchor_x)

        dx = self.x - anchor_x
        dy = self.y - anchor_y
        cross_track = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy
        cross_track = clamp(cross_track, -1.5, 1.5)
        correction = math.atan2(
            self.route_cross_track_gain * cross_track,
            max(self.lookahead_dist, 0.5),
        )
        desired_yaw = normalize_angle(path_yaw - correction)
        yaw_err = normalize_angle(desired_yaw - self.yaw)
        speed = self.gps_hiz
        if abs(yaw_err) > self.keskin_viraj_esik:
            speed = self.keskin_viraj_hiz
        return self.yaw_k * yaw_err, speed

    def update_corner_mode(self, now_ns: int, raw_err: float, side: str) -> None:
        trigger = abs(raw_err) >= self.single_corner_trigger
        trigger = trigger or (self.obstacle_context_active(now_ns) and abs(self.depth_gap_offset) >= 0.22)
        if trigger and side:
            self.corner_until_ns = now_ns + int(self.single_corner_hold_sec * 1e9)
        self.corner_mode = side != '' and now_ns < self.corner_until_ns

    def compute_lane_state_command(self, now_ns: int, route_term: float, side: str, allow_gap_assist: bool = False) -> Tuple[float, float, float, float, float, float, float]:
        if self.lane_state == LaneState.NORMAL_LANE:
            route_weight = min(self.route_weight_normal, 0.05) if self.route_enabled else 0.0
            desired, speed, lane_term, gap_term, lane_conf = self.compute_normal_lane_command(now_ns, route_term, route_weight)
            route_term_used = route_term * route_weight if self.route_enabled else 0.0
            return desired, speed, lane_term, gap_term, lane_conf, route_term_used, route_weight
        if self.lane_state == LaneState.DEGRADED_LANE:
            route_weight = min(self.route_weight_single, 0.02) if self.route_enabled else 0.0
            desired, speed, lane_term, gap_term, lane_conf = self.compute_single_lane_command(
                now_ns,
                route_term,
                side,
                route_weight,
                allow_gap_assist=allow_gap_assist,
            )
            route_term_used = route_term * route_weight if self.route_enabled else 0.0
            return desired, speed, lane_term, gap_term, lane_conf, route_term_used, route_weight
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def compute_normal_lane_command(self, now_ns: int, route_term: float, route_weight: float) -> Tuple[float, float, float, float, float]:
        err = clamp(self.lane_error, -self.lane_error_clip, self.lane_error_clip)
        if abs(err) < self.normal_deadband:
            err = 0.0
        deriv = (err - self.last_normal_error) / max(self.control_period, 1e-3)
        self.last_normal_error = err
        lane_term = clamp(self.normal_kp * err + self.normal_kd * deriv, -self.normal_limit, self.normal_limit)
        desired = lane_term + route_term * route_weight
        lane_conf = clamp(1.0 - 0.55 * abs(err) / max(self.lane_error_clip, 1e-3), 0.78, 1.0)
        speed = self.straight_speed if abs(err) < 0.030 and abs(desired) < 0.10 else self.normal_lane_speed
        speed *= max(0.72, 1.0 - 0.90 * abs(err))
        if abs(desired) > 0.25:
            speed = min(speed, 0.68)
        if self.obstacle_context_active(now_ns):
            speed = min(speed, 0.64)
        return desired, speed, lane_term, 0.0, lane_conf

    def compute_single_lane_gap_term(self, now_ns: int, lane_term: float, side: str) -> float:
        if abs(self.depth_gap_offset) < self.depth_gap_min_offset:
            return 0.0
        if self.depth_selected_gap_score < self.depth_gap_min_score:
            return 0.0
        if self.depth_center_clearance < self.depth_stop_m + 0.06:
            return 0.0

        obstacle_context = self.obstacle_context_active(now_ns)
        gain = self.depth_gap_gain if obstacle_context else self.depth_gap_corner_gain
        limit = self.depth_gap_limit if obstacle_context else self.depth_gap_corner_limit
        gap_term = clamp(-gain * self.depth_gap_offset, -limit, limit)

        if lane_term != 0.0 and gap_term * lane_term < 0.0:
            if obstacle_context:
                gap_term *= 0.25
            else:
                gap_term = 0.0

        if not obstacle_context:
            if side == 'left' and gap_term > 0.0:
                gap_term *= 0.18
            elif side == 'right' and gap_term < 0.0:
                gap_term *= 0.18

        return gap_term

    def compute_single_lane_command(self, now_ns: int, route_term: float, side: str, route_weight: float, allow_gap_assist: bool = False) -> Tuple[float, float, float, float, float]:
        raw_err = clamp(self.raw_lane_error if self.lane_stamp_ns > 0 else self.lane_last_valid_error, -self.lane_error_clip, self.lane_error_clip)
        if not self.lane_valid and self.lane_last_valid_ns > 0:
            raw_err = 0.75 * self.lane_last_valid_error + 0.25 * raw_err

        self.update_corner_mode(now_ns, raw_err, side)

        err = raw_err * 0.62
        if side == 'left':
            err -= self.single_center_bias
        elif side == 'right':
            err += self.single_center_bias
        if self.corner_mode:
            err *= self.single_corner_gain
            if side == 'left':
                err -= self.single_corner_bias
            elif side == 'right':
                err += self.single_corner_bias
        if abs(err) < self.single_deadband:
            err = 0.0
        err = clamp(err, -0.30, 0.30)

        deriv = (err - self.last_single_error) / max(self.control_period, 1e-3)
        self.last_single_error = err
        lane_limit = self.single_corner_limit if self.corner_mode else self.single_limit
        lane_term = clamp(self.single_kp * err + self.single_kd * deriv, -lane_limit, lane_limit)
        gap_term = self.compute_single_lane_gap_term(now_ns, lane_term, side) if allow_gap_assist else 0.0
        desired = lane_term + gap_term + route_term * route_weight

        base_speed = self.single_lane_corner_speed if self.corner_mode else self.single_lane_speed
        if not self.corner_mode and abs(err) < 0.040:
            base_speed = self.single_lane_recovery_speed
        if not self.lane_valid:
            base_speed = min(base_speed, self.single_lane_invalid_speed)
        if abs(desired) > 0.38:
            base_speed = min(base_speed, 0.40)
        if self.obstacle_context_active(now_ns):
            base_speed = max(base_speed, 0.38)
        lane_conf = clamp(0.58 - 0.30 * abs(raw_err) / max(self.lane_error_clip, 1e-3), 0.22, 0.62)
        return desired, base_speed, lane_term, gap_term, lane_conf

    def sur(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self.update_lane_state(now_ns)

        left_recent, right_recent = self.active_boundaries(now_ns)
        side = self.active_single_side(now_ns)
        route_term_raw, route_speed = self.compute_route_term()
        twist = Twist()
        was_obstacle_active = self.summary_obstacle_active
        command = self.select_control_command(now_ns, route_term_raw, route_speed, side)
        angular = self.publish_control_command(command, twist)
        self.publish_obstacle_summary(now_ns, command)
        obstacle_active = self.summary_obstacle_active
        if was_obstacle_active and not obstacle_active:
            self.start_return_to_center(now_ns)
            self.last_cmd_angular *= 0.55
            self.critical_avoid_smoothed = 0.0
        self.prev_obstacle_active = obstacle_active
        self.last_lane_confidence = command.lane_conf
        self.control_authority = command.authority
        self.control_reason = command.reason
        self.last_mode_name = command.authority.name

        if now_ns - self.lane_state_log_ns > int(1e9):
            self.lane_state_log_ns = now_ns
            self.get_logger().info(
                f'[STATE] lane={self.lane_state.name} authority={command.authority.name} '
                f'reason={command.reason} lane_valid={self.lane_valid} '
                f'err={self.lane_error:+.3f} angular={self.last_cmd_angular:+.3f} '
                f'L={left_recent} R={right_recent} blocked={self.blocked_persistent}'
            )

        if now_ns - self.angular_debug_log_ns > int(1e9):
            self.angular_debug_log_ns = now_ns
            self.get_logger().info(
                f'[CTRL_DEBUG] lane_state={self.lane_state.name} authority={command.authority.name} '
                f'reason={command.reason} lane_term={command.lane_term:+.3f} '
                f'route_term_raw={command.route_term_raw:+.3f} route_term_used={command.route_term_used:+.3f} '
                f'gap_term={command.gap_term:+.3f} corridor_term={command.corridor_term:+.3f} '
                f'avoid_term={command.avoid_term:+.3f} final_angular={angular:+.3f} '
                f'speed={twist.linear.x:.2f} lane_conf={command.lane_conf:.2f} parser_conf={self.parser_lane_confidence:.2f} '
                f'selected_gap={self.depth_selected_gap_label} raw_gap_offset={self.raw_gap_offset:+.2f} '
                f'mapped_corridor_target={self.mapped_corridor_target:+.2f} '
                f'smoothed_corridor_target={self.smoothed_corridor_target:+.2f} '
                f'corridor_target_offset={self.corridor_target_offset:+.2f} corridor_error={self.corridor_error:+.2f} '
                f'corridor_term_preclamp={self.corridor_term_preclamp:+.3f} corridor_term_postclamp={self.corridor_term_postclamp:+.3f} '
                f'gap_score={self.depth_selected_gap_score:+.2f} continuity_bonus={self.depth_gap_continuity_bonus:+.2f} '
                f'gap_switch_reason={self.depth_gap_switch_reason} '
                f'corridor_enabled={self.corridor_enabled_state} corridor_gating_reason={self.corridor_gating_reason} '
                f'corridor_reset_reason={self.corridor_reset_reason} obstacle_active={obstacle_active} '
                f'duba_dist={self.duba_mesafe:.2f} critical_dist={self.critical_roi_min_x:.2f} '
                f'critical_points={self.critical_roi_point_count} footprint_intrusion={self.critical_roi_intrusion_m:.2f} '
                f'avoid_latched={self.summary_avoid_latched} blocked={self.blocked_persistent}'
            )

        if self.pub is not None:
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
