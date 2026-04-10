#!/usr/bin/env python3

import json
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
    PRE_AVOID = auto()
    POST_AVOID_HOLD = auto()
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
        self.declare_parameter('pass_state_topic', '/obstacle/pass_state')
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
        self.declare_parameter('critical_intrusion_min_persistence_cycles', 3)
        self.declare_parameter('critical_intrusion_persistence_min_cycles', 3)
        self.declare_parameter('critical_geometry_consistency_tolerance', 0.18)
        self.declare_parameter('false_critical_demote_timeout_sec', 0.30)
        self.declare_parameter('emergency_demote_timeout_sec', 0.20)
        self.declare_parameter('center_corridor_override_priority_weight', 1.35)
        self.declare_parameter('center_corridor_stabilizer_weight', 0.28)
        self.declare_parameter('no_commit_center_stabilizer_weight', 0.22)
        self.declare_parameter('critical_override_block_center_margin', 0.08)
        self.declare_parameter('critical_lane_term_min_weight', 0.35)
        self.declare_parameter('critical_corridor_term_min_weight', 0.22)
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
        self.declare_parameter('duba_pass_freeze_distance_m', 0.60)
        self.declare_parameter('duba_pass_freeze_lateral_m', 0.10)
        self.declare_parameter('duba_pass_hold_sec', 0.55)
        self.declare_parameter('close_side_avoid_distance_m', 1.05)
        self.declare_parameter('close_side_avoid_full_distance_m', 0.80)
        self.declare_parameter('close_side_avoid_lateral_m', 0.08)
        self.declare_parameter('close_side_avoid_min_offset_m', 0.14)
        self.declare_parameter('close_side_avoid_offset_m', 0.42)
        self.declare_parameter('close_side_avoid_speed_mps', 0.24)
        self.declare_parameter('close_side_avoid_lane_weight_min', 0.14)
        self.declare_parameter('pre_avoid_trigger_m', 1.80)
        self.declare_parameter('near_avoid_trigger_m', 1.15)
        self.declare_parameter('emergency_avoid_trigger_m', 0.72)
        self.declare_parameter('obstacle_release_distance_m', 2.10)
        self.declare_parameter('obstacle_latch_hold_sec', 0.90)
        self.declare_parameter('obstacle_preempt_intrusion_m', 0.04)
        self.declare_parameter('obstacle_preempt_center_ratio', 0.18)
        self.declare_parameter('pre_avoid_min_offset_m', 0.08)
        self.declare_parameter('pre_avoid_max_offset_m', 0.22)
        self.declare_parameter('pre_avoid_lane_weight', 0.84)
        self.declare_parameter('pre_avoid_corridor_blend', 0.65)
        self.declare_parameter('pre_avoid_speed_scale_far', 0.84)
        self.declare_parameter('pre_avoid_speed_scale_near', 0.62)
        self.declare_parameter('pre_avoid_speed_scale_emergency', 0.28)
        self.declare_parameter('center_gap_penalty_gain', 1.45)
        self.declare_parameter('center_gap_penalty_max', 1.80)
        self.declare_parameter('duba_preempt_max_age_sec', 0.30)
        self.declare_parameter('stale_obstacle_release_sec', 0.22)
        self.declare_parameter('avoid_bias_lane_attenuation', 0.62)
        self.declare_parameter('avoid_corridor_limit_degraded', 0.24)
        self.declare_parameter('tracked_obstacle_persist_sec', 1.20)
        self.declare_parameter('tracked_obstacle_match_distance_m', 0.90)
        self.declare_parameter('tracked_obstacle_lateral_gate_m', 0.90)
        self.declare_parameter('avoid_pass_longitudinal_margin_m', 0.55)
        self.declare_parameter('avoid_pass_lateral_clearance_m', 0.18)
        self.declare_parameter('force_odom_pass_latch', True)
        self.declare_parameter('avoid_pass_min_progress_m', 1.50)
        self.declare_parameter('avoid_pass_max_hold_sec', 6.00)
        self.declare_parameter('pass_latch_duration_sec', 1.20)
        self.declare_parameter('pass_latch_distance_m', 0.60)
        self.declare_parameter('fallback_side_selection_timeout_sec', 0.30)
        self.declare_parameter('min_corridor_hold_sec', 0.45)
        self.declare_parameter('corridor_gating_hysteresis_sec', 0.35)
        self.declare_parameter('obstacle_local_y_filter_alpha', 0.35)
        self.declare_parameter('obstacle_local_y_deadband', 0.10)
        self.declare_parameter('side_selection_persistence_cycles', 3)
        self.declare_parameter('side_score_margin_min', 0.05)
        self.declare_parameter('side_block_persistence_cycles', 4)
        self.declare_parameter('commit_exit_clearance_distance_m', 0.80)
        self.declare_parameter('commit_exit_clear_cycles', 3)
        self.declare_parameter('tracked_memory_ttl_sec', 0.60)
        self.declare_parameter('commit_stall_timeout_sec', 0.80)
        self.declare_parameter('min_progress_delta_for_active_commit', 0.05)
        self.declare_parameter('min_tracked_local_x_change_for_active_commit', 0.10)
        self.declare_parameter('fallback_commit_score_margin_min', 0.12)
        self.declare_parameter('startup_straight_corridor_guard_sec', 2.50)
        self.declare_parameter('startup_straight_corridor_min_clearance_m', 1.10)
        self.declare_parameter('startup_straight_corridor_max_score_delta', 0.12)
        self.declare_parameter('startup_straight_corridor_side_balance_ratio', 0.22)
        self.declare_parameter('progress_completion_threshold', 0.92)
        self.declare_parameter('tracked_memory_require_strong_source', True)
        self.declare_parameter('post_avoid_straight_distance_m', 1.20)
        self.declare_parameter('post_avoid_hold_sec', 2.50)
        self.declare_parameter('post_avoid_lane_weight', 0.22)
        self.declare_parameter('post_avoid_corridor_weight', 0.58)
        self.declare_parameter('single_lane_transition_frames', 3)
        self.declare_parameter('no_lane_transition_frames', 6)
        self.declare_parameter('blocked_persistence_sec', 0.45)
        self.declare_parameter('advisory_side_gap_max_weight', 0.22)
        self.declare_parameter('precommit_speed_scale', 0.58)
        self.declare_parameter('center_reject_min_score', 0.95)
        self.declare_parameter('center_reject_persistence_cycles', 3)
        self.declare_parameter('lane_edge_safety_margin', 0.08)
        self.declare_parameter('corridor_target_lane_clip_margin', 0.04)
        self.declare_parameter('lane_hard_constraint_margin', 0.05)
        self.declare_parameter('no_commit_side_bias_cap', 0.10)
        self.declare_parameter('center_corridor_priority_weight', 1.15)
        self.declare_parameter('false_emergency_reset_cycles', 2)

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
        self.pass_state_topic = str(self.get_parameter('pass_state_topic').value)
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
        self.critical_intrusion_min_persistence_cycles = max(
            1,
            int(self.get_parameter('critical_intrusion_min_persistence_cycles').value),
        )
        self.critical_intrusion_persistence_min_cycles = max(
            self.critical_intrusion_min_persistence_cycles,
            int(self.get_parameter('critical_intrusion_persistence_min_cycles').value),
        )
        self.critical_intrusion_min_persistence_cycles = self.critical_intrusion_persistence_min_cycles
        self.critical_geometry_consistency_tolerance = max(
            0.05,
            float(self.get_parameter('critical_geometry_consistency_tolerance').value),
        )
        self.false_critical_demote_timeout_sec = max(
            0.0,
            float(self.get_parameter('false_critical_demote_timeout_sec').value),
        )
        self.false_critical_demote_timeout_ns = int(self.false_critical_demote_timeout_sec * 1e9)
        self.emergency_demote_timeout_sec = max(
            0.0,
            float(self.get_parameter('emergency_demote_timeout_sec').value),
        )
        self.emergency_demote_timeout_ns = int(self.emergency_demote_timeout_sec * 1e9)
        self.center_corridor_override_priority_weight = max(
            1.0,
            float(self.get_parameter('center_corridor_override_priority_weight').value),
        )
        self.center_corridor_stabilizer_weight = clamp(
            float(self.get_parameter('center_corridor_stabilizer_weight').value),
            0.0,
            1.0,
        )
        self.no_commit_center_stabilizer_weight = clamp(
            float(self.get_parameter('no_commit_center_stabilizer_weight').value),
            0.0,
            1.0,
        )
        self.critical_override_block_center_margin = max(
            0.0,
            float(self.get_parameter('critical_override_block_center_margin').value),
        )
        self.critical_lane_term_min_weight = clamp(
            float(self.get_parameter('critical_lane_term_min_weight').value),
            0.0,
            1.0,
        )
        self.critical_corridor_term_min_weight = clamp(
            float(self.get_parameter('critical_corridor_term_min_weight').value),
            0.0,
            1.0,
        )
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
        self.duba_pass_freeze_distance_m = max(
            self.pointcloud_self_filter_forward_m + 0.05,
            float(self.get_parameter('duba_pass_freeze_distance_m').value),
        )
        self.duba_pass_freeze_lateral_m = max(
            0.04,
            float(self.get_parameter('duba_pass_freeze_lateral_m').value),
        )
        self.duba_pass_hold_sec = max(0.10, float(self.get_parameter('duba_pass_hold_sec').value))
        self.close_side_avoid_distance_m = max(
            self.pointcloud_self_filter_forward_m + 0.10,
            float(self.get_parameter('close_side_avoid_distance_m').value),
        )
        self.close_side_avoid_full_distance_m = clamp(
            float(self.get_parameter('close_side_avoid_full_distance_m').value),
            self.pointcloud_self_filter_forward_m + 0.05,
            self.close_side_avoid_distance_m - 0.05,
        )
        self.close_side_avoid_lateral_m = max(
            0.04,
            float(self.get_parameter('close_side_avoid_lateral_m').value),
        )
        self.close_side_avoid_min_offset_m = clamp(
            float(self.get_parameter('close_side_avoid_min_offset_m').value),
            0.06,
            self.lane_corridor_cap,
        )
        self.close_side_avoid_offset_m = clamp(
            float(self.get_parameter('close_side_avoid_offset_m').value),
            self.close_side_avoid_min_offset_m,
            self.lane_corridor_cap,
        )
        self.close_side_avoid_speed_mps = max(
            0.10,
            float(self.get_parameter('close_side_avoid_speed_mps').value),
        )
        self.close_side_avoid_lane_weight_min = clamp(
            float(self.get_parameter('close_side_avoid_lane_weight_min').value),
            0.0,
            1.0,
        )
        self.pre_avoid_trigger_m = max(
            self.close_side_avoid_distance_m,
            float(self.get_parameter('pre_avoid_trigger_m').value),
        )
        self.near_avoid_trigger_m = clamp(
            float(self.get_parameter('near_avoid_trigger_m').value),
            self.close_side_avoid_full_distance_m + 0.05,
            self.pre_avoid_trigger_m - 0.10,
        )
        self.emergency_avoid_trigger_m = clamp(
            float(self.get_parameter('emergency_avoid_trigger_m').value),
            self.pointcloud_self_filter_forward_m + 0.08,
            self.near_avoid_trigger_m - 0.08,
        )
        self.obstacle_release_distance_m = max(
            self.pre_avoid_trigger_m + 0.15,
            float(self.get_parameter('obstacle_release_distance_m').value),
        )
        self.obstacle_latch_hold_sec = max(
            0.20,
            float(self.get_parameter('obstacle_latch_hold_sec').value),
        )
        self.obstacle_preempt_intrusion_m = max(
            0.0,
            float(self.get_parameter('obstacle_preempt_intrusion_m').value),
        )
        self.obstacle_preempt_center_ratio = clamp(
            float(self.get_parameter('obstacle_preempt_center_ratio').value),
            0.05,
            0.95,
        )
        raw_pre_avoid_max_offset = clamp(
            float(self.get_parameter('pre_avoid_max_offset_m').value),
            0.04,
            self.lane_corridor_cap,
        )
        self.pre_avoid_min_offset_m = clamp(
            float(self.get_parameter('pre_avoid_min_offset_m').value),
            0.04,
            raw_pre_avoid_max_offset,
        )
        self.pre_avoid_max_offset_m = clamp(
            raw_pre_avoid_max_offset,
            self.pre_avoid_min_offset_m,
            self.lane_corridor_cap,
        )
        self.pre_avoid_lane_weight = clamp(
            float(self.get_parameter('pre_avoid_lane_weight').value),
            0.0,
            1.0,
        )
        self.pre_avoid_corridor_blend = clamp(
            float(self.get_parameter('pre_avoid_corridor_blend').value),
            0.0,
            1.0,
        )
        self.pre_avoid_speed_scale_far = clamp(
            float(self.get_parameter('pre_avoid_speed_scale_far').value),
            0.20,
            1.0,
        )
        raw_pre_avoid_speed_scale_near = clamp(
            float(self.get_parameter('pre_avoid_speed_scale_near').value),
            0.0,
            self.pre_avoid_speed_scale_far,
        )
        self.pre_avoid_speed_scale_near = clamp(
            raw_pre_avoid_speed_scale_near,
            0.0,
            self.pre_avoid_speed_scale_far,
        )
        self.pre_avoid_speed_scale_emergency = clamp(
            float(self.get_parameter('pre_avoid_speed_scale_emergency').value),
            0.0,
            self.pre_avoid_speed_scale_near,
        )
        self.center_gap_penalty_gain = max(
            0.0,
            float(self.get_parameter('center_gap_penalty_gain').value),
        )
        self.center_gap_penalty_max = max(
            0.0,
            float(self.get_parameter('center_gap_penalty_max').value),
        )
        self.duba_preempt_max_age_sec = max(
            0.05,
            float(self.get_parameter('duba_preempt_max_age_sec').value),
        )
        self.stale_obstacle_release_sec = max(
            0.05,
            float(self.get_parameter('stale_obstacle_release_sec').value),
        )
        self.avoid_bias_lane_attenuation = clamp(
            float(self.get_parameter('avoid_bias_lane_attenuation').value),
            0.20,
            1.0,
        )
        self.avoid_corridor_limit_degraded = clamp(
            float(self.get_parameter('avoid_corridor_limit_degraded').value),
            0.10,
            self.depth_gap_limit if hasattr(self, 'depth_gap_limit') else 0.34,
        )
        self.tracked_obstacle_persist_sec = max(
            0.20,
            float(self.get_parameter('tracked_obstacle_persist_sec').value),
        )
        self.tracked_obstacle_match_distance_m = max(
            0.10,
            float(self.get_parameter('tracked_obstacle_match_distance_m').value),
        )
        self.tracked_obstacle_lateral_gate_m = max(
            self.vehicle_half_width_m,
            float(self.get_parameter('tracked_obstacle_lateral_gate_m').value),
        )
        self.avoid_pass_longitudinal_margin_m = max(
            0.10,
            float(self.get_parameter('avoid_pass_longitudinal_margin_m').value),
        )
        self.avoid_pass_lateral_clearance_m = max(
            0.0,
            float(self.get_parameter('avoid_pass_lateral_clearance_m').value),
        )
        self.force_odom_pass_latch = as_bool(self.get_parameter('force_odom_pass_latch').value)
        self.avoid_pass_min_progress_m = max(
            0.50,
            float(self.get_parameter('avoid_pass_min_progress_m').value),
        )
        self.avoid_pass_max_hold_sec = max(
            1.0,
            float(self.get_parameter('avoid_pass_max_hold_sec').value),
        )
        self.pass_latch_duration_sec = max(
            0.20,
            float(self.get_parameter('pass_latch_duration_sec').value),
        )
        self.pass_latch_distance_m = max(
            0.50,
            float(self.get_parameter('pass_latch_distance_m').value),
        )
        self.fallback_side_selection_timeout_sec = max(
            0.05,
            float(self.get_parameter('fallback_side_selection_timeout_sec').value),
        )
        self.min_corridor_hold_sec = max(
            0.05,
            float(self.get_parameter('min_corridor_hold_sec').value),
        )
        self.corridor_gating_hysteresis_sec = max(
            0.05,
            float(self.get_parameter('corridor_gating_hysteresis_sec').value),
        )
        self.obstacle_local_y_filter_alpha = clamp(
            float(self.get_parameter('obstacle_local_y_filter_alpha').value),
            0.05,
            1.00,
        )
        self.obstacle_local_y_deadband = max(
            0.01,
            float(self.get_parameter('obstacle_local_y_deadband').value),
        )
        self.side_selection_persistence_cycles = max(
            1,
            int(self.get_parameter('side_selection_persistence_cycles').value),
        )
        self.side_score_margin_min = max(
            0.0,
            float(self.get_parameter('side_score_margin_min').value),
        )
        self.side_block_persistence_cycles = max(
            1,
            int(self.get_parameter('side_block_persistence_cycles').value),
        )
        self.commit_exit_clearance_distance_m = max(
            0.05,
            float(self.get_parameter('commit_exit_clearance_distance_m').value),
        )
        self.commit_exit_clear_cycles = max(
            1,
            int(self.get_parameter('commit_exit_clear_cycles').value),
        )
        self.tracked_memory_ttl_sec = max(
            0.10,
            float(self.get_parameter('tracked_memory_ttl_sec').value),
        )
        self.commit_stall_timeout_sec = max(
            0.20,
            float(self.get_parameter('commit_stall_timeout_sec').value),
        )
        self.min_progress_delta_for_active_commit = clamp(
            float(self.get_parameter('min_progress_delta_for_active_commit').value),
            0.0,
            1.0,
        )
        self.min_tracked_local_x_change_for_active_commit = max(
            0.0,
            float(self.get_parameter('min_tracked_local_x_change_for_active_commit').value),
        )
        self.fallback_commit_score_margin_min = max(
            0.0,
            float(self.get_parameter('fallback_commit_score_margin_min').value),
        )
        self.startup_straight_corridor_guard_sec = max(
            0.0,
            float(self.get_parameter('startup_straight_corridor_guard_sec').value),
        )
        self.startup_straight_corridor_min_clearance_m = max(
            self.required_gap_clearance_m + 0.18,
            float(self.get_parameter('startup_straight_corridor_min_clearance_m').value),
        )
        self.startup_straight_corridor_max_score_delta = clamp(
            float(self.get_parameter('startup_straight_corridor_max_score_delta').value),
            0.02,
            0.40,
        )
        self.startup_straight_corridor_side_balance_ratio = clamp(
            float(self.get_parameter('startup_straight_corridor_side_balance_ratio').value),
            0.05,
            0.80,
        )
        self.progress_completion_threshold = clamp(
            float(self.get_parameter('progress_completion_threshold').value),
            0.50,
            1.00,
        )
        self.tracked_memory_require_strong_source = as_bool(
            self.get_parameter('tracked_memory_require_strong_source').value
        )
        self.post_avoid_straight_distance_m = max(
            0.20,
            float(self.get_parameter('post_avoid_straight_distance_m').value),
        )
        self.post_avoid_hold_sec = max(
            0.20,
            float(self.get_parameter('post_avoid_hold_sec').value),
        )
        self.post_avoid_lane_weight = clamp(
            float(self.get_parameter('post_avoid_lane_weight').value),
            0.0,
            1.0,
        )
        self.post_avoid_corridor_weight = clamp(
            float(self.get_parameter('post_avoid_corridor_weight').value),
            0.0,
            1.0,
        )
        self.single_lane_transition_frames = max(
            1,
            int(self.get_parameter('single_lane_transition_frames').value),
        )
        self.no_lane_transition_frames = max(
            1,
            int(self.get_parameter('no_lane_transition_frames').value),
        )
        self.blocked_hold_sec = max(
            0.05,
            float(self.get_parameter('blocked_persistence_sec').value),
        )
        self.advisory_side_gap_max_weight = clamp(
            float(self.get_parameter('advisory_side_gap_max_weight').value),
            0.05,
            0.50,
        )
        self.precommit_speed_scale = clamp(
            float(self.get_parameter('precommit_speed_scale').value),
            self.pre_avoid_speed_scale_emergency,
            1.0,
        )
        self.center_reject_min_score = max(
            0.10,
            float(self.get_parameter('center_reject_min_score').value),
        )
        self.center_reject_persistence_cycles = max(
            1,
            int(self.get_parameter('center_reject_persistence_cycles').value),
        )
        self.lane_edge_safety_margin = clamp(
            float(self.get_parameter('lane_edge_safety_margin').value),
            0.0,
            max(0.0, self.lane_corridor_cap - 0.02),
        )
        self.lane_hard_constraint_margin = clamp(
            float(self.get_parameter('lane_hard_constraint_margin').value),
            0.0,
            max(0.0, self.lane_corridor_cap - 0.02),
        )
        self.corridor_target_lane_clip_margin = clamp(
            float(self.get_parameter('corridor_target_lane_clip_margin').value),
            0.0,
            max(0.0, self.lane_corridor_cap - 0.02),
        )
        self.no_commit_side_bias_cap = clamp(
            float(self.get_parameter('no_commit_side_bias_cap').value),
            0.02,
            self.lane_corridor_cap,
        )
        self.center_corridor_priority_weight = max(
            1.0,
            float(self.get_parameter('center_corridor_priority_weight').value),
        )
        self.false_emergency_reset_cycles = max(
            1,
            int(self.get_parameter('false_emergency_reset_cycles').value),
        )

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
        self.odom_path_length_m = 0.0
        self.last_odom_x = None
        self.last_odom_y = None

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
        self.duba_pass_hold_until_ns = 0
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
        self.critical_intrusion_persistence_cycles = 0
        self.critical_intrusion_persistence_last_update_ns = 0
        self.false_critical_since_ns = 0
        self.false_critical_override_detected = False
        self.critical_override_blocked_by_center_corridor = False
        self.critical_trigger_consistent_with_tracked_geometry = True
        self.center_corridor_override_priority_applied = False
        self.critical_commit_rejected_reason = 'none'
        self.lane_term_preserved_in_critical = False
        self.corridor_term_preserved_in_critical = False
        self.side_commit_cancelled_due_to_valid_center_corridor = False
        self.false_emergency_demoted = False
        self.emergency_latch_rejected_due_to_low_persistence = False
        self.emergency_latch_rejected_due_to_center_corridor = False
        self.center_corridor_stabilizer_active = False
        self.lane_only_fallback_blocked = False
        self.critical_intrusion_persistence_cycles_used = 0
        self.emergency_latch_kept_reason = 'none'
        self.false_emergency_detected_cycles = 0
        self.false_emergency_since_ns = 0

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
        self.pre_avoid_active = False
        self.obstacle_preempted_by_lane = False
        self.avoid_trigger_source = 'none'
        self.speed_scale_obstacle = 1.0
        self.center_gap_penalty = 0.0
        self.authority_transition_reason = 'init'
        self.obstacle_latch_state = 'idle'
        self.obstacle_release_reason = 'init'
        self.obstacle_latch_until_ns = 0
        self.obstacle_forward_distance = 99.0
        self.last_authority_transition_reason = 'init'
        self.last_command_authority = ControlAuthority.NO_LANE_COAST
        self.tracked_obstacle_valid = False
        self.tracked_obstacle_world_x = 0.0
        self.tracked_obstacle_world_y = 0.0
        self.tracked_obstacle_local_x = 99.0
        self.tracked_obstacle_local_y = 0.0
        self.tracked_obstacle_radius_m = 0.20
        self.tracked_obstacle_last_seen_ns = 0
        self.tracked_obstacle_source = 'none'
        self.tracked_obstacle_last_refresh_ns = 0
        self.tracked_memory_expire_reason = 'init'
        self.pass_latch_active = False
        self.pass_latch_started_ns = 0
        self.pass_latch_obstacle_world_x = 0.0
        self.pass_latch_obstacle_world_y = 0.0
        self.pass_latch_obstacle_radius_m = 0.20
        self.pass_latch_source = 'none'
        self.pass_latch_start_x = 0.0
        self.pass_latch_start_y = 0.0
        self.pass_latch_start_path_m = 0.0
        self.pass_latch_travel_m = 0.0
        self.post_avoid_hold_until_ns = 0
        self.post_avoid_start_path_m = 0.0
        self.post_avoid_travel_m = 0.0
        self.post_avoid_target_offset = 0.0
        self.authoritative_pass_owner = 'yaris_pilotu'
        self.requested_pass_side = 'NONE'
        self.published_pass_side = 'NONE'
        self.pass_side_none_reason = 'startup'
        self.pass_commit_source = 'none'
        self.pass_commit_exit_reason = 'init'
        self.center_reject_reason = 'startup'
        self.pass_commit_until_ns = 0
        self.pass_commit_started_ns = 0
        self.pass_commit_start_path_m = 0.0
        self.pass_side_pending_since_ns = 0
        self.fallback_side_triggered = False
        self.fallback_side_last_triggered_ns = 0
        self.pass_progress = 0.0
        self.commit_remaining_distance_m = 0.0
        self.commit_remaining_sec_value = 0.0
        self.commit_session_sequence = 0
        self.commit_session_id = 0
        self.last_commit_session_id = 0
        self.commit_session_start_reason = 'none'
        self.side_lock_active = False
        self.locked_pass_side = 'NONE'
        self.side_flip_blocked = False
        self.side_switch_reject_reason = 'none'
        self.zombie_commit_state_detected = False
        self.atomic_commit_state_clear_applied = False
        self.critical_reject_forced_state_clear = False
        self.pass_state_validity_ok = True
        self.lane_hard_constraint_active = False
        self.center_corridor_exists = False
        self.center_corridor_preferred = False
        self.center_preferred_reason = 'startup'
        self.center_reject_strength = 0.0
        self.center_reject_persistence = 0
        self.advisory_side_gap_strength = 0.0
        self.side_gap_suppressed_due_to_no_commit = False
        self.side_target_suppressed_reason = 'none'
        self.target_clipped_to_lane_bounds = False
        self.target_clip_reason = 'none'
        self.final_controller_mode = 'lane_center'
        self.lane_corridor_min_offset = -self.lane_corridor_cap
        self.lane_corridor_max_offset = self.lane_corridor_cap
        self.commit_watchdog_last_progress = 0.0
        self.commit_watchdog_last_tracked_local_x = 99.0
        self.commit_watchdog_last_odom_path_m = 0.0
        self.commit_watchdog_last_check_ns = 0
        self.commit_watchdog_progress_delta = 0.0
        self.commit_watchdog_tracked_local_x_delta = 0.0
        self.commit_watchdog_odom_delta = 0.0
        self.commit_stale_detected = False
        self.stale_obstacle_memory_detected = False
        self.stale_commit_hold_until_ns = 0
        self.node_started_ns = self.get_clock().now().nanoseconds
        self.startup_guard_armed_ns = 0
        self.startup_straight_corridor_guard_active_state = False
        self.startup_straight_corridor_guard_reason = 'init'
        self.filtered_obstacle_local_y = 0.0
        self.filtered_obstacle_local_y_valid = False
        self.obstacle_local_y_deadband_active = True
        self.side_selection_candidate = 'NONE'
        self.side_selection_candidate_cycles = 0
        self.side_blocked_cycle_count = 0
        self.commit_exit_clear_count = 0
        self.corridor_force_until_ns = 0
        self.left_gap_safe = False
        self.right_gap_safe = False

        self.lane_state = LaneState.NO_LANE_COAST
        self.lane_lost_ns = 0
        self.recover_start_ns = 0
        self.blocked_start_ns = 0
        self.blocked_persistent = False
        self.blocked_center_now = False
        self.blocked_selected_side_now = False
        self.pending_single_lane_frames = 0
        self.pending_no_lane_frames = 0
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
        self.selected_pass_side = 'NONE'
        self.stop_reason = 'startup'

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
        self.pass_state_pub = self.create_publisher(String, self.pass_state_topic, 10)
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
        self.debug_preempted_by_lane_pub = self.create_publisher(Bool, '/obstacle/debug/obstacle_preempted_by_lane', 10)
        self.debug_pre_avoid_active_pub = self.create_publisher(Bool, '/obstacle/debug/pre_avoid_active', 10)
        self.debug_trigger_source_pub = self.create_publisher(String, '/obstacle/debug/avoid_trigger_source', 10)
        self.debug_speed_scale_obstacle_pub = self.create_publisher(Float32, '/obstacle/debug/speed_scale_obstacle', 10)
        self.debug_center_gap_penalty_pub = self.create_publisher(Float32, '/obstacle/debug/center_gap_penalty', 10)
        self.debug_authority_transition_reason_pub = self.create_publisher(String, '/obstacle/debug/authority_transition_reason', 10)
        self.debug_obstacle_latch_state_pub = self.create_publisher(String, '/obstacle/debug/obstacle_latch_state', 10)
        self.debug_obstacle_release_reason_pub = self.create_publisher(String, '/obstacle/debug/obstacle_release_reason', 10)
        self.debug_lane_state_pub = self.create_publisher(String, '/lane/debug/state', 10)
        self.debug_selected_pass_side_pub = self.create_publisher(String, '/obstacle/debug/selected_pass_side', 10)
        self.debug_authoritative_pass_owner_pub = self.create_publisher(String, '/obstacle/debug/authoritative_pass_owner', 10)
        self.debug_requested_pass_side_pub = self.create_publisher(String, '/obstacle/debug/requested_pass_side', 10)
        self.debug_published_pass_side_pub = self.create_publisher(String, '/obstacle/debug/published_pass_side', 10)
        self.debug_commit_source_pub = self.create_publisher(String, '/obstacle/debug/commit_source', 10)
        self.debug_commit_exit_reason_pub = self.create_publisher(String, '/obstacle/debug/commit_exit_reason', 10)
        self.debug_commit_active_pub = self.create_publisher(Bool, '/obstacle/debug/commit_active', 10)
        self.debug_commit_remaining_pub = self.create_publisher(Float32, '/obstacle/debug/commit_remaining_sec', 10)
        self.debug_commit_remaining_distance_pub = self.create_publisher(Float32, '/obstacle/debug/commit_remaining_distance_m', 10)
        self.debug_progress_pub = self.create_publisher(Float32, '/obstacle/debug/progress', 10)
        self.debug_fallback_side_triggered_pub = self.create_publisher(Bool, '/obstacle/debug/fallback_side_triggered', 10)
        self.debug_pass_side_none_reason_pub = self.create_publisher(String, '/obstacle/debug/pass_side_none_reason', 10)
        self.debug_blocked_center_pub = self.create_publisher(Bool, '/obstacle/debug/blocked_center', 10)
        self.debug_blocked_selected_side_pub = self.create_publisher(Bool, '/obstacle/debug/blocked_selected_side', 10)
        self.debug_stop_reason_pub = self.create_publisher(String, '/obstacle/debug/stop_reason', 10)
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
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        if self.have_odom and self.last_odom_x is not None and self.last_odom_y is not None:
            step_m = math.hypot(new_x - self.last_odom_x, new_y - self.last_odom_y)
            if step_m <= 1.0:
                self.odom_path_length_m += step_m
        self.x = new_x
        self.y = new_y
        self.last_odom_x = new_x
        self.last_odom_y = new_y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.have_odom = True

    def vehicle_to_world(self, local_x: float, local_y: float) -> Tuple[float, float]:
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        world_x = self.x + cos_yaw * local_x - sin_yaw * local_y
        world_y = self.y + sin_yaw * local_x + cos_yaw * local_y
        return world_x, world_y

    def world_to_vehicle(self, world_x: float, world_y: float) -> Tuple[float, float]:
        dx = world_x - self.x
        dy = world_y - self.y
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return local_x, local_y

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
        now_ns = self.get_clock().now().nanoseconds
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
        msg_bool = Bool()
        msg_bool.data = bool(self.obstacle_preempted_by_lane)
        self.debug_preempted_by_lane_pub.publish(msg_bool)
        msg_bool = Bool()
        msg_bool.data = bool(self.pre_avoid_active)
        self.debug_pre_avoid_active_pub.publish(msg_bool)
        msg_float = Float32()
        msg_float.data = float(self.speed_scale_obstacle)
        self.debug_speed_scale_obstacle_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.center_gap_penalty)
        self.debug_center_gap_penalty_pub.publish(msg_float)
        msg_text = String()
        msg_text.data = self.avoid_trigger_source
        self.debug_trigger_source_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.authority_transition_reason
        self.debug_authority_transition_reason_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.obstacle_latch_state
        self.debug_obstacle_latch_state_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.obstacle_release_reason
        self.debug_obstacle_release_reason_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.lane_state.name
        self.debug_lane_state_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.authoritative_pass_owner
        self.debug_authoritative_pass_owner_pub.publish(msg_text)
        self.sanitize_authoritative_pass_commit_state(now_ns)
        msg_text = String()
        msg_text.data = self.requested_pass_side
        self.debug_requested_pass_side_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.published_pass_side
        self.debug_published_pass_side_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.selected_pass_side
        self.debug_selected_pass_side_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.pass_commit_source
        self.debug_commit_source_pub.publish(msg_text)
        msg_text = String()
        msg_text.data = self.pass_commit_exit_reason
        self.debug_commit_exit_reason_pub.publish(msg_text)
        msg_bool = Bool()
        msg_bool.data = bool(self.commit_active(now_ns))
        self.debug_commit_active_pub.publish(msg_bool)
        msg_float = Float32()
        msg_float.data = float(self.commit_remaining_sec(now_ns))
        self.debug_commit_remaining_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.commit_remaining_distance(now_ns))
        self.debug_commit_remaining_distance_pub.publish(msg_float)
        msg_float = Float32()
        msg_float.data = float(self.compute_pass_progress(now_ns))
        self.debug_progress_pub.publish(msg_float)
        msg_bool = Bool()
        msg_bool.data = bool(
            self.fallback_side_triggered
            or self.pass_commit_source.startswith('fallback')
        )
        self.debug_fallback_side_triggered_pub.publish(msg_bool)
        msg_text = String()
        msg_text.data = self.pass_side_none_reason
        self.debug_pass_side_none_reason_pub.publish(msg_text)
        msg_bool = Bool()
        msg_bool.data = bool(self.blocked_center_now)
        self.debug_blocked_center_pub.publish(msg_bool)
        msg_bool = Bool()
        msg_bool.data = bool(self.blocked_selected_side_now)
        self.debug_blocked_selected_side_pub.publish(msg_bool)
        msg_text = String()
        msg_text.data = self.stop_reason
        self.debug_stop_reason_pub.publish(msg_text)
        self.publish_authoritative_pass_state(now_ns)

    def publish_authoritative_pass_state(self, now_ns: int) -> None:
        self.sanitize_authoritative_pass_commit_state(now_ns)
        commit_active = self.commit_active(now_ns)
        commit_remaining_time = self.commit_remaining_sec(now_ns)
        commit_remaining_distance = self.commit_remaining_distance(now_ns)
        progress = self.compute_pass_progress(now_ns)
        state = {
            'stamp_ns': int(now_ns),
            'source_node': self.authoritative_pass_owner,
            'obstacle_active': bool(self.summary_obstacle_active),
            'pre_avoid_active': bool(self.pre_avoid_active),
            'obstacle_latch_state': self.obstacle_latch_state,
            'pass_side': self.locked_pass_side if self.side_lock_active else self.published_pass_side,
            'selected_gap': self.locked_pass_side if self.side_lock_active else self.depth_selected_gap_label,
            'corridor_target': float(self.corridor_target_offset),
            'corridor_enabled': bool(self.corridor_enabled_state),
            'corridor_gating_reason': self.corridor_gating_reason,
            'commit_active': bool(commit_active),
            'commit_session_id': int(self.commit_session_id),
            'side_lock_active': bool(self.side_lock_active),
            'locked_pass_side': self.locked_pass_side,
            'stale_commit_active': bool(self.stale_commit_active(now_ns)),
            'stale_commit_detected': bool(self.commit_stale_detected),
            'stale_obstacle_memory_detected': bool(self.stale_obstacle_memory_detected),
            'commit_remaining_time': float(commit_remaining_time),
            'commit_remaining_distance': float(commit_remaining_distance),
            'progress': float(progress),
            'commit_age': float(max(0.0, (now_ns - self.pass_commit_started_ns) / 1e9)) if self.pass_commit_started_ns > 0 else 0.0,
            'progress_delta': float(self.commit_watchdog_progress_delta),
            'tracked_local_x_delta': float(self.commit_watchdog_tracked_local_x_delta),
            'odom_delta_since_commit': float(
                max(0.0, self.odom_path_length_m - self.pass_commit_start_path_m)
            ) if self.have_odom and self.pass_commit_started_ns > 0 else 0.0,
            'blocked_center': bool(self.blocked_center_now),
            'blocked_selected_side': bool(self.blocked_selected_side_now),
            'exit_reason': self.pass_commit_exit_reason,
            'enter_reason': self.pass_commit_source,
            'lane_hard_constraint_active': bool(self.lane_hard_constraint_active),
            'center_corridor_exists': bool(self.center_corridor_exists),
            'center_corridor_preferred': bool(self.center_corridor_preferred),
            'center_preferred_reason': self.center_preferred_reason,
            'center_reject_reason': self.center_reject_reason,
            'center_reject_strength': float(self.center_reject_strength),
            'center_reject_persistence': int(self.center_reject_persistence),
            'advisory_side_gap_strength': float(self.advisory_side_gap_strength),
            'side_gap_suppressed_due_to_no_commit': bool(self.side_gap_suppressed_due_to_no_commit),
            'side_target_suppressed_reason': self.side_target_suppressed_reason,
            'target_clipped_to_lane_bounds': bool(self.target_clipped_to_lane_bounds),
            'target_clip_reason': self.target_clip_reason,
            'final_controller_mode': self.final_controller_mode,
            'lane_corridor_min_offset': float(self.lane_corridor_min_offset),
            'lane_corridor_max_offset': float(self.lane_corridor_max_offset),
            'filtered_obstacle_local_y': float(self.filtered_obstacle_local_y),
            'obstacle_local_y_deadband_active': bool(self.obstacle_local_y_deadband_active),
            'side_flip_blocked': bool(self.side_flip_blocked),
            'side_switch_reject_reason': self.side_switch_reject_reason,
            'commit_session_start_reason': self.commit_session_start_reason,
            'tracked_memory_expire_reason': self.tracked_memory_expire_reason,
            'tracked_local_x': float(self.tracked_obstacle_local_x),
            'tracked_local_y': float(self.tracked_obstacle_local_y),
            'critical_dist': float(self.critical_roi_min_x),
            'critical_points': int(self.critical_roi_point_count),
            'footprint_intrusion': float(self.critical_roi_intrusion_m),
            'critical_intrusion_persistence_cycles_used': int(self.critical_intrusion_persistence_cycles_used),
            'false_critical_override_detected': bool(self.false_critical_override_detected),
            'critical_override_blocked_by_center_corridor': bool(
                self.critical_override_blocked_by_center_corridor
            ),
            'critical_trigger_consistent_with_tracked_geometry': bool(
                self.critical_trigger_consistent_with_tracked_geometry
            ),
            'center_corridor_override_priority_applied': bool(
                self.center_corridor_override_priority_applied
            ),
            'critical_commit_rejected_reason': self.critical_commit_rejected_reason,
            'zombie_commit_state_detected': bool(self.zombie_commit_state_detected),
            'atomic_commit_state_clear_applied': bool(self.atomic_commit_state_clear_applied),
            'critical_reject_forced_state_clear': bool(self.critical_reject_forced_state_clear),
            'pass_state_validity_ok': bool(self.pass_state_validity_ok),
            'false_emergency_demoted': bool(self.false_emergency_demoted),
            'emergency_latch_rejected_due_to_low_persistence': bool(
                self.emergency_latch_rejected_due_to_low_persistence
            ),
            'emergency_latch_rejected_due_to_center_corridor': bool(
                self.emergency_latch_rejected_due_to_center_corridor
            ),
            'center_corridor_stabilizer_active': bool(self.center_corridor_stabilizer_active),
            'lane_only_fallback_blocked': bool(self.lane_only_fallback_blocked),
            'emergency_latch_kept_reason': self.emergency_latch_kept_reason,
            'lane_term_preserved_in_critical': bool(self.lane_term_preserved_in_critical),
            'corridor_term_preserved_in_critical': bool(self.corridor_term_preserved_in_critical),
            'side_commit_cancelled_due_to_valid_center_corridor': bool(
                self.side_commit_cancelled_due_to_valid_center_corridor
            ),
            'startup_straight_corridor_guard_active': bool(self.startup_straight_corridor_guard_active_state),
            'startup_straight_corridor_guard_reason': self.startup_straight_corridor_guard_reason,
            'pre_avoid_side_selection_timeout': float(self.fallback_side_selection_timeout_sec),
        }
        msg_text = String()
        msg_text.data = json.dumps(state, separators=(',', ':'), sort_keys=True)
        self.pass_state_pub.publish(msg_text)

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
        lateral_hint = critical_mean_y if critical_count >= self.critical_roi_min_points else mean_y
        close_side_pass = (
            math.isfinite(min_x)
            and min_x <= self.duba_pass_freeze_distance_m
            and abs(lateral_hint) >= self.duba_pass_freeze_lateral_m
        )
        if close_side_pass:
            self.duba_pass_hold_until_ns = max(
                self.duba_pass_hold_until_ns,
                now_ns + int(self.duba_pass_hold_sec * 1e9),
            )
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
            if not close_side_pass or self.duba_mesafe >= 90.0:
                self.duba_filtreli_konum = (
                    self.duba_filtre_alpha * mean_y
                    + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
                )
                self.duba_mesafe = min_x if math.isfinite(min_x) else self.duba_algilama_mesafesi
            elif math.isfinite(min_x):
                self.duba_mesafe = min(self.duba_mesafe, min_x)
        elif self.duba_var and count >= self.duba_cikis_min_nokta:
            self.duba_last_seen_ns = now_ns
            if not close_side_pass:
                self.duba_filtreli_konum = (
                    self.duba_filtre_alpha * mean_y
                    + (1.0 - self.duba_filtre_alpha) * self.duba_filtreli_konum
                )
                self.duba_mesafe = min_x if math.isfinite(min_x) else self.duba_mesafe
        else:
            held_sec = (now_ns - self.duba_last_seen_ns) / 1e9 if self.duba_last_seen_ns > 0 else float('inf')
            self.duba_var = held_sec <= max(self.duba_hold_sec, self.duba_pass_hold_sec)

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
                or self.critical_roi_intrusion_m >= self.obstacle_preempt_intrusion_m
                or self.critical_center_ratio >= self.obstacle_preempt_center_ratio
                or self.pointcloud_front_min_distance <= self.near_avoid_trigger_m
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
                f'corridor_supported={self.pointcloud_obstacle_supported} '
                f'pre_avoid_active={self.pre_avoid_active} latch_state={self.obstacle_latch_state} '
                f'trigger_source={self.avoid_trigger_source}'
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
        centered_pointcloud_trigger = self.centered_obstacle_bypass_active(now_ns)
        pointcloud_confident = centered_pointcloud_trigger or (
            pointcloud_signal and (
                obstacle_now
                or blocked_frame
                or self.depth_obstacle
                or pointcloud_lateral_hint >= max(0.10, 0.80 * self.duba_center_escape_y)
                or center_clear + 0.18 < max(left_clear, right_clear)
            )
        )
        hard_corridor_signal = (
            pointcloud_confident
            or centered_pointcloud_trigger
            or self.critical_obstacle_now
            or obstacle_now
            or self.depth_obstacle
            or blocked_frame
        )
        center_gap_penalty = 0.0
        if self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            center_gap_penalty += self.center_gap_penalty_gain * clamp(
                self.critical_center_ratio / max(self.obstacle_preempt_center_ratio, 1e-3),
                0.0,
                1.0,
            )
            center_gap_penalty += self.center_gap_penalty_gain * clamp(
                self.critical_roi_intrusion_m / max(self.footprint_half_width_m, 1e-3),
                0.0,
                1.0,
            )
            if self.pointcloud_front_center_count >= self.critical_roi_min_points:
                center_gap_penalty += 0.35 * self.center_gap_penalty_gain
            if self.pointcloud_front_min_distance <= self.pre_avoid_trigger_m:
                center_gap_penalty += self.center_gap_penalty_gain * clamp(
                    (self.pre_avoid_trigger_m - self.pointcloud_front_min_distance)
                    / max(self.pre_avoid_trigger_m - self.emergency_avoid_trigger_m, 1e-3),
                    0.0,
                    1.0,
                )
        center_gap_penalty = clamp(center_gap_penalty, 0.0, self.center_gap_penalty_max)
        self.center_gap_penalty = center_gap_penalty
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
                score -= center_gap_penalty
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
        gate_override = self.pointcloud_center_gate_override_active(now_ns)
        obstacle_side_hint = 0.0 if gate_override else self.filtered_obstacle_side_hint()
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
        if abs(obstacle_side_hint) < 0.05 and centered_pointcloud_trigger:
            side_score_delta = right_score - left_score
            if abs(side_score_delta) >= 0.04:
                obstacle_side_hint = 0.20 if side_score_delta > 0.0 else -0.20
        side_clearance_advantage = max(left_clear, right_clear) - center_clear
        lane_hard_constraint_active = self.lane_hard_constraints_active(now_ns)
        center_corridor_exists = self.center_corridor_traversable(center_clear, center_ratio, upper_ratio)
        center_reject_strength = 0.0
        if blocked_frame:
            center_reject_strength += 1.10
        if self.depth_emergency:
            center_reject_strength += 1.10
        center_reject_strength += clamp(
            max(0.0, required_gap_clearance - center_clear) / max(required_gap_clearance, 1e-3),
            0.0,
            1.2,
        )
        center_reject_strength += clamp(
            (center_ratio - max(0.12, 1.5 * self.obstacle_center_ratio_threshold))
            / max(0.20, 1.5 * self.obstacle_center_ratio_threshold),
            0.0,
            0.8,
        )
        center_reject_strength += clamp(
            (upper_ratio - max(0.04, 2.0 * self.depth_upper_ratio_threshold))
            / max(0.08, 2.0 * self.depth_upper_ratio_threshold),
            0.0,
            0.6,
        )
        center_reject_strength += 0.55 * clamp(
            center_gap_penalty / max(self.center_gap_penalty_max, 1e-3),
            0.0,
            1.0,
        )
        center_reject_strength += 0.35 * clamp(side_clearance_advantage / 0.30, 0.0, 1.0)
        center_reject_signal = (
            not center_corridor_exists
            and (
                blocked_frame
                or obstacle_now
                or self.depth_obstacle
                or self.depth_emergency
                or pointcloud_confident
                or self.critical_obstacle_now
            )
        )
        if center_reject_signal:
            self.center_reject_persistence = min(
                self.center_reject_persistence + 1,
                self.center_reject_persistence_cycles + 4,
            )
        else:
            self.center_reject_persistence = 0
        center_reject_allowed = (
            blocked_frame
            or self.depth_emergency
            or (
                center_reject_strength >= self.center_reject_min_score
                and self.center_reject_persistence >= self.center_reject_persistence_cycles
            )
        )
        center_gap_recovery = self.center_gap_recovery_preferred(now_ns)
        center_lane_keep_preferred = (
            center_clear >= max(tight_gap_clearance, required_gap_clearance + 0.28)
            and center_ratio <= max(0.08, 1.25 * self.obstacle_center_ratio_threshold)
            and upper_ratio <= max(0.04, 2.0 * self.depth_upper_ratio_threshold)
            and side_clearance_advantage <= 0.12
            and abs(obstacle_side_hint) < max(self.obstacle_local_y_deadband, 0.10)
            and abs(left_score - right_score) <= max(
                self.startup_straight_corridor_max_score_delta,
                self.side_score_margin_min + 0.05,
            )
        )
        center_lane_keep_preferred = center_lane_keep_preferred or center_gap_recovery
        side_preferred = (
            not center_lane_keep_preferred
            and (
                abs(obstacle_side_hint) >= 0.10
                or centered_pointcloud_trigger
                or obstacle_now
                or blocked_frame
                or side_clearance_advantage >= 0.10
            )
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
        center_occupied = (
            center_reject_allowed
            or (
                not center_lane_keep_preferred
                and (obstacle_now or pointcloud_confident or self.critical_obstacle_now)
            )
        )
        side_only_mode = center_occupied or (
            not center_lane_keep_preferred
            and hard_corridor_signal
            and side_preferred
            and center_reject_allowed
        )
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
        commit_side_lock_active = (
            self.side_lock_active
            and self.locked_pass_side in ('LEFT', 'RIGHT')
            and self.commit_session_id > 0
            and self.commit_active(now_ns)
        )
        if commit_side_lock_active:
            requested_side = chosen_side_label if chosen_side_label in ('LEFT', 'RIGHT') else 'NONE'
            locked_side = self.locked_pass_side
            self.side_flip_blocked = requested_side not in ('NONE', locked_side)
            if self.side_flip_blocked:
                self.side_switch_reject_reason = (
                    f'active_commit_lock_reject:{requested_side.lower()}->{locked_side.lower()}'
                )
            elif requested_side == 'NONE':
                self.side_switch_reject_reason = 'active_commit_lock_hold'
            else:
                self.side_switch_reject_reason = 'active_commit_lock_confirm'
            locked_detail = left_detail if locked_side == 'LEFT' else right_detail
            if locked_detail is None:
                locked_detail = {
                    'offset': -0.80 if locked_side == 'LEFT' else 0.80,
                    'score': left_score if locked_side == 'LEFT' else right_score,
                    'continuity_bonus': chosen_detail['continuity_bonus'],
                    'clearance': left_clear if locked_side == 'LEFT' else right_clear,
                    'ratio': left_close_ratio if locked_side == 'LEFT' else right_close_ratio,
                }
            chosen_detail = locked_detail
            chosen_side_label = locked_side
            raw_gap_offset = chosen_detail['offset']
            switch_reason = 'commit_side_lock_hold'
        else:
            self.side_flip_blocked = False
            self.side_switch_reject_reason = 'none'
        mapped_corridor_target = self.map_gap_offset_to_corridor_target(chosen_detail['offset'])
        candidate_strong = (
            hard_corridor_signal
            and not center_lane_keep_preferred
            and not center_corridor_exists
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
            and not centered_pointcloud_trigger
            and self.pointcloud_front_min_distance > self.pre_avoid_trigger_m
            and self.critical_roi_min_x > self.pre_avoid_trigger_m
            and abs(obstacle_side_hint) < max(0.08, 0.60 * self.duba_center_escape_y)
            and center_clear >= max(left_clear, right_clear) - 0.02
            and center_detail['clearance'] >= (required_gap_clearance - 0.02)
            and center_ratio < 0.05
            and upper_ratio < max(0.02, 1.5 * self.depth_upper_ratio_threshold)
            and center_gap_penalty < 0.10
        )
        clear_path_preferred = clear_path_preferred or center_lane_keep_preferred
        center_corridor_preferred = (
            not self.commit_active(now_ns)
            and self.published_pass_side == 'NONE'
            and not blocked_frame
            and center_corridor_exists
            and (
                clear_path_preferred
                or not center_reject_allowed
            )
        )
        if center_corridor_preferred:
            clear_path_preferred = True
        if clear_path_preferred:
            chosen_detail = center_detail
            chosen_side_label = ''
            raw_gap_offset = 0.0
            mapped_corridor_target = 0.0
            candidate_strong = False
            if center_gap_recovery:
                self.center_reject_reason = 'center_gap_recovery'
            elif center_corridor_preferred:
                self.center_reject_reason = 'lane_bounded_center_corridor'
            else:
                self.center_reject_reason = 'clear_path_preferred'
        elif side_only_mode:
            self.center_reject_reason = 'front_obstacle_requires_side_pass'
        elif candidate_strong:
            self.center_reject_reason = 'strong_side_corridor'
        else:
            self.center_reject_reason = 'center_allowed'
        self.lane_hard_constraint_active = lane_hard_constraint_active
        self.center_corridor_exists = center_corridor_exists
        self.center_corridor_preferred = center_corridor_preferred
        self.center_preferred_reason = (
            'center_gap_recovery'
            if center_gap_recovery
            else (
                'lane_bounded_center_corridor'
                if center_corridor_preferred
                else 'not_preferred'
            )
        )
        self.center_reject_strength = center_reject_strength
        forced_hold_active = (
            self.commit_active(now_ns)
            or self.pre_avoid_active
            or now_ns < self.corridor_force_until_ns
        )
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
            if (
                abs(memory_target) >= 0.10
                and abs(self.corridor_target_offset) >= 0.10
                and memory_target * self.corridor_target_offset < 0.0
            ):
                memory_gain = max(memory_gain, 0.72)
                corridor_gating_reason = 'side_flip_commit'
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
            and now_ns >= self.corridor_force_until_ns
            and not self.return_to_center_active(now_ns)
            and not forced_hold_active
        ):
            self.corridor_target_offset = 0.0
            self.corridor_active_until_ns = 0
            corridor_enabled = False
            corridor_gating_reason = 'lane_priority'
            reset_reason = 'lane_visible'
        elif not corridor_enabled and forced_hold_active and abs(self.corridor_target_offset) >= 0.05:
            corridor_enabled = True
            corridor_gating_reason = 'gating_hysteresis_hold'
            reset_reason = 'hold'
        self.corridor_target_offset = clamp(self.corridor_target_offset, -1.0, 1.0)
        self.corridor_target_offset = self.clip_target_to_lane_corridor(
            self.corridor_target_offset,
            now_ns,
            'depth_corridor_target',
        )
        if (
            not self.commit_active(now_ns)
            and self.published_pass_side == 'NONE'
            and not self.center_corridor_preferred
            and self.depth_selected_gap_label in ('LEFT', 'RIGHT')
        ):
            self.corridor_target_offset = self.apply_precommit_side_target_policy(
                now_ns,
                self.corridor_target_offset,
                'depth_precommit_advisory',
            )
        elif self.center_corridor_preferred:
            self.advisory_side_gap_strength = 0.0
            self.side_gap_suppressed_due_to_no_commit = self.depth_selected_gap_label in ('LEFT', 'RIGHT')
            self.side_target_suppressed_reason = (
                'center_corridor_preferred'
                if self.side_gap_suppressed_due_to_no_commit
                else 'none'
            )
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
        self.left_gap_safe = left_safe
        self.right_gap_safe = right_safe

        selected_pass_side = chosen_side_label if chosen_side_label in ('LEFT', 'RIGHT') else 'NONE'
        if selected_pass_side == 'NONE':
            if self.depth_selected_gap_label in ('LEFT', 'RIGHT') and self.commit_active(now_ns):
                selected_pass_side = self.depth_selected_gap_label
            elif left_safe ^ right_safe:
                selected_pass_side = 'LEFT' if left_safe else 'RIGHT'
        self.requested_pass_side = selected_pass_side
        self.selected_pass_side = selected_pass_side
        self.blocked_center_now = bool(not self.center_corridor_exists and center_reject_allowed)
        if commit_side_lock_active and self.locked_pass_side == 'LEFT':
            self.blocked_selected_side_now = not left_safe
        elif commit_side_lock_active and self.locked_pass_side == 'RIGHT':
            self.blocked_selected_side_now = not right_safe
        elif self.selected_pass_side == 'LEFT':
            self.blocked_selected_side_now = not left_safe
        elif self.selected_pass_side == 'RIGHT':
            self.blocked_selected_side_now = not right_safe
        else:
            self.blocked_selected_side_now = not (left_safe or right_safe)

        if self.blocked_center_now and self.blocked_selected_side_now:
            self.depth_reason_code = 'blocked_center_and_side'
            if self.blocked_start_ns <= 0:
                self.blocked_start_ns = now_ns
            held_sec = (now_ns - self.blocked_start_ns) / 1e9
            self.blocked_persistent = held_sec >= self.blocked_hold_sec
        else:
            self.blocked_start_ns = 0
            self.blocked_persistent = False
            if self.blocked_center_now and not self.blocked_selected_side_now:
                self.depth_reason_code = 'center_blocked_side_open'

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
                f'center_ratio={self.depth_center_ratio:.2f} upper_ratio={self.depth_upper_ratio:.2f} min_dist={self.depth_min_dist:.2f} '
                f'center_gap_penalty={self.center_gap_penalty:.2f} pre_avoid={self.pre_avoid_active} '
                f'obstacle_latch_state={self.obstacle_latch_state} trigger_source={self.avoid_trigger_source}'
            )

    def signal_recent(self, stamp_ns: int, timeout_sec: float, now_ns: int) -> bool:
        return stamp_ns > 0 and (now_ns - stamp_ns) / 1e9 <= timeout_sec

    def lane_valid_recent(self, now_ns: int) -> bool:
        return self.lane_valid and self.signal_recent(self.lane_stamp_ns, self.lane_timeout_sec, now_ns)

    def active_boundaries(self, now_ns: int) -> Tuple[bool, bool]:
        left_recent = self.signal_recent(self.left_lane_last_seen_ns, self.single_boundary_timeout_sec, now_ns)
        right_recent = self.signal_recent(self.right_lane_last_seen_ns, self.single_boundary_timeout_sec, now_ns)
        return left_recent, right_recent

    def lane_hard_constraints_active(self, now_ns: int) -> bool:
        left_recent, right_recent = self.active_boundaries(now_ns)
        lane_memory_recent = self.signal_recent(self.lane_last_valid_ns, self.lane_timeout_sec, now_ns)
        return left_recent or right_recent or (self.lane_control_available() and lane_memory_recent)

    def lane_corridor_limits(self, now_ns: int) -> Tuple[float, float]:
        left_recent, right_recent = self.active_boundaries(now_ns)
        lane_memory_recent = self.signal_recent(self.lane_last_valid_ns, self.lane_timeout_sec, now_ns)
        if left_recent and right_recent:
            cap = max(
                0.06,
                self.lane_corridor_cap - max(self.corridor_target_lane_clip_margin, self.lane_hard_constraint_margin),
            )
        elif left_recent or right_recent:
            cap = max(
                0.06,
                min(
                    self.lane_corridor_cap - max(self.corridor_target_lane_clip_margin, self.lane_hard_constraint_margin),
                    self.lane_corridor_cap - max(self.lane_edge_safety_margin, self.lane_hard_constraint_margin),
                    0.22,
                ),
            )
        elif self.lane_control_available() and lane_memory_recent:
            cap = max(
                0.06,
                min(
                    self.lane_corridor_cap - self.lane_hard_constraint_margin,
                    0.24,
                ),
            )
        else:
            cap = self.lane_corridor_cap
        self.lane_corridor_min_offset = -cap
        self.lane_corridor_max_offset = cap
        self.lane_hard_constraint_active = (
            left_recent
            or right_recent
            or (self.lane_control_available() and lane_memory_recent)
        )
        return self.lane_corridor_min_offset, self.lane_corridor_max_offset

    def clip_target_to_lane_corridor(
        self,
        target_offset: float,
        now_ns: int,
        reason: str,
    ) -> float:
        low, high = self.lane_corridor_limits(now_ns)
        clipped = clamp(target_offset, low, high)
        clipped_active = abs(clipped - target_offset) > 1e-3
        self.target_clipped_to_lane_bounds = clipped_active
        self.target_clip_reason = reason if clipped_active else 'none'
        return clipped

    def apply_precommit_side_target_policy(
        self,
        now_ns: int,
        target_offset: float,
        reason: str,
    ) -> float:
        target_offset = self.clip_target_to_lane_corridor(target_offset, now_ns, reason)
        self.advisory_side_gap_strength = 0.0
        self.side_gap_suppressed_due_to_no_commit = False
        self.side_target_suppressed_reason = 'none'
        if self.commit_active(now_ns) or self.published_pass_side in ('LEFT', 'RIGHT'):
            return target_offset
        if self.selected_pass_side not in ('LEFT', 'RIGHT'):
            return target_offset
        if self.center_corridor_preferred:
            self.side_gap_suppressed_due_to_no_commit = True
            self.side_target_suppressed_reason = 'center_corridor_preferred'
            return 0.0
        capped = clamp(target_offset, -self.no_commit_side_bias_cap, self.no_commit_side_bias_cap)
        if abs(capped - target_offset) > 1e-3:
            self.side_gap_suppressed_due_to_no_commit = True
            self.side_target_suppressed_reason = 'no_commit_side_bias_cap'
        self.advisory_side_gap_strength = self.advisory_side_gap_max_weight * clamp(
            abs(capped) / max(self.no_commit_side_bias_cap, 1e-3),
            0.0,
            1.0,
        )
        return capped

    def startup_straight_corridor_guard_active(self, now_ns: int) -> bool:
        if self.startup_straight_corridor_guard_sec <= 0.0:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'disabled'
            return False
        if self.node_started_ns <= 0 and now_ns > 0:
            self.node_started_ns = now_ns
        if self.startup_guard_armed_ns <= 0:
            spawn_ready = self.have_odom and (
                self.pointcloud_last_ns > 0 or self.depth_frame_stamp_ns > 0
            )
            if not spawn_ready:
                self.startup_straight_corridor_guard_active_state = False
                self.startup_straight_corridor_guard_reason = 'waiting_for_spawn_sensors'
                return False
            self.startup_guard_armed_ns = max(
                now_ns,
                self.pointcloud_last_ns,
                self.depth_frame_stamp_ns,
            )
        if self.startup_guard_armed_ns <= 0:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'startup_clock_unset'
            return False
        startup_age = max(0.0, (now_ns - self.startup_guard_armed_ns) / 1e9)
        if startup_age > self.startup_straight_corridor_guard_sec:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'startup_guard_expired'
            return False
        if self.commit_session_id > 0 or self.side_lock_active or self.pass_latch_active:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'active_commit_present'
            return False
        if not self.depth_frame_recent(now_ns):
            if self.startup_pointcloud_symmetry_guard(now_ns):
                self.startup_straight_corridor_guard_active_state = True
                self.startup_straight_corridor_guard_reason = 'startup_pointcloud_symmetry'
                return True
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'depth_not_ready'
            return False
        if self.depth_obstacle or self.depth_emergency:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'depth_center_blocked'
            return False
        center_open = (
            self.depth_center_clearance >= self.startup_straight_corridor_min_clearance_m
            and self.depth_center_ratio <= max(0.05, self.obstacle_center_ratio_threshold)
            and self.center_gap_penalty <= 0.12
        )
        if not center_open:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'center_not_open'
            return False
        centered_hint = max(
            abs(self.filtered_obstacle_local_y),
            abs(self.critical_roi_mean_y),
            abs(self.duba_konumu),
        ) <= max(self.obstacle_local_y_deadband, 0.10)
        if not centered_hint:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'obstacle_has_side_bias'
            return False
        score_delta = abs(self.depth_left_gap_score - self.depth_right_gap_score)
        if score_delta > self.startup_straight_corridor_max_score_delta:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'side_gap_delta_strong'
            return False
        left_right_max = float(max(self.pointcloud_front_left_count, self.pointcloud_front_right_count, 1))
        balanced_pointcloud = (
            abs(self.pointcloud_front_left_count - self.pointcloud_front_right_count)
            <= self.startup_straight_corridor_side_balance_ratio * left_right_max
        )
        if not balanced_pointcloud:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'pointcloud_side_imbalance'
            return False
        if not self.critical_center_supported:
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'critical_center_not_supported'
            return False
        if self.depth_center_clearance + 0.08 < max(self.depth_left_clearance, self.depth_right_clearance):
            self.startup_straight_corridor_guard_active_state = False
            self.startup_straight_corridor_guard_reason = 'center_weaker_than_side_gap'
            return False
        self.startup_straight_corridor_guard_active_state = True
        self.startup_straight_corridor_guard_reason = 'startup_straight_corridor'
        return True

    def startup_pointcloud_symmetry_guard(self, now_ns: int) -> bool:
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.critical_center_supported:
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        if not math.isfinite(self.critical_roi_min_x) or self.critical_roi_min_x >= self.pre_avoid_trigger_m:
            return False
        centered_hint = max(
            abs(self.filtered_obstacle_local_y),
            abs(self.critical_roi_mean_y),
            abs(self.duba_konumu),
        ) <= max(self.obstacle_local_y_deadband, 0.10)
        if not centered_hint:
            return False
        left_right_max = float(max(self.pointcloud_front_left_count, self.pointcloud_front_right_count, 1))
        balanced_sides = (
            abs(self.pointcloud_front_left_count - self.pointcloud_front_right_count)
            <= self.startup_straight_corridor_side_balance_ratio * left_right_max
        )
        center_not_side_biased = self.pointcloud_front_center_count <= 1.20 * left_right_max
        center_ratio_ok = self.critical_center_ratio <= max(
            0.34,
            self.obstacle_preempt_center_ratio + 0.16,
        )
        return balanced_sides and center_not_side_biased and center_ratio_ok

    def pointcloud_center_lane_keep_preferred(self, now_ns: int) -> bool:
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.critical_center_supported:
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        if not math.isfinite(self.critical_roi_min_x) or self.critical_roi_min_x >= self.pre_avoid_trigger_m:
            return False
        centered_hint = max(
            abs(self.filtered_obstacle_local_y),
            abs(self.critical_roi_mean_y),
            abs(self.duba_konumu),
        ) <= max(self.obstacle_local_y_deadband, 0.10)
        if not centered_hint:
            return False
        left_right_max = float(max(self.pointcloud_front_left_count, self.pointcloud_front_right_count, 1))
        balanced_sides = (
            abs(self.pointcloud_front_left_count - self.pointcloud_front_right_count)
            <= self.startup_straight_corridor_side_balance_ratio * left_right_max
        )
        center_not_dominant = self.pointcloud_front_center_count <= 1.15 * left_right_max
        intrusion_ok = self.critical_roi_intrusion_m <= max(0.08, 1.8 * self.obstacle_preempt_intrusion_m)
        center_ratio_ok = self.critical_center_ratio <= max(0.34, self.obstacle_preempt_center_ratio + 0.16)
        return balanced_sides and center_not_dominant and intrusion_ok and center_ratio_ok

    def center_corridor_lane_keep_preferred(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if not self.depth_frame_recent(now_ns):
            return self.pointcloud_center_lane_keep_preferred(now_ns)
        if self.depth_obstacle or self.depth_emergency:
            return False
        center_clear = self.depth_center_clearance
        if not math.isfinite(center_clear):
            return False
        center_open = center_clear >= max(self.tight_gap_clearance_m, self.required_gap_clearance_m + 0.28)
        center_sparse = self.depth_center_ratio <= max(0.08, 1.25 * self.obstacle_center_ratio_threshold)
        upper_sparse = self.depth_upper_ratio <= max(0.04, 2.0 * self.depth_upper_ratio_threshold)
        side_advantage = max(self.depth_left_clearance, self.depth_right_clearance) - center_clear
        centered_hint = max(
            abs(self.filtered_obstacle_local_y),
            abs(self.critical_roi_mean_y),
            abs(self.duba_konumu),
        ) <= max(self.obstacle_local_y_deadband, 0.10)
        left_right_max = float(max(self.pointcloud_front_left_count, self.pointcloud_front_right_count, 1))
        balanced_pointcloud = (
            abs(self.pointcloud_front_left_count - self.pointcloud_front_right_count)
            <= self.startup_straight_corridor_side_balance_ratio * left_right_max
        )
        return (
            center_open
            and center_sparse
            and upper_sparse
            and centered_hint
            and balanced_pointcloud
            and side_advantage <= 0.12
        )

    def center_corridor_traversable(self, center_clear: float, center_ratio: float, upper_ratio: float) -> bool:
        if not math.isfinite(center_clear):
            return False
        return (
            center_clear >= max(self.required_gap_clearance_m - 0.02, self.depth_stop_m + 0.10)
            and center_ratio <= max(0.18, 2.2 * self.obstacle_center_ratio_threshold)
            and upper_ratio <= max(0.08, 3.0 * self.depth_upper_ratio_threshold)
        )

    def pointcloud_center_gate_override_active(self, now_ns: int) -> bool:
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.critical_center_supported:
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        if not math.isfinite(self.critical_roi_min_x):
            return False
        if self.critical_roi_min_x > max(self.near_avoid_trigger_m + 0.20, self.close_side_avoid_distance_m + 0.25):
            return False
        left_count = float(self.pointcloud_front_left_count)
        center_count = float(self.pointcloud_front_center_count)
        right_count = float(self.pointcloud_front_right_count)
        side_max = max(left_count, right_count, 1.0)
        side_min = min(left_count, right_count)
        balanced_sides = (
            side_min >= 0.52 * side_max
            and abs(left_count - right_count) <= 0.35 * side_max
        )
        center_dominant = (
            center_count >= max(
                float(self.critical_roi_min_points * 2),
                1.20 * side_max,
            )
            or self.critical_center_ratio >= max(
                0.40,
                self.obstacle_preempt_center_ratio + 0.18,
            )
        )
        centered_geometry = (
            abs(self.critical_roi_mean_y) <= 0.10
            and abs(self.duba_konumu) <= max(self.close_side_avoid_lateral_m, 0.18)
        )
        return balanced_sides and center_dominant and centered_geometry

    def center_gap_recovery_preferred(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        if self.depth_obstacle or self.depth_emergency or self.blocked_persistent:
            return False
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.critical_center_supported:
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        if not math.isfinite(self.critical_roi_min_x):
            return False
        if self.critical_roi_min_x > max(self.pre_avoid_trigger_m, self.close_side_avoid_distance_m + 0.20):
            return False
        gate_override = self.pointcloud_center_gate_override_active(now_ns)
        if (
            self.pointcloud_front_left_count < self.critical_roi_min_points
            or self.pointcloud_front_right_count < self.critical_roi_min_points
        ):
            return False
        left_right_max = float(max(self.pointcloud_front_left_count, self.pointcloud_front_right_count, 1))
        left_right_min = float(min(self.pointcloud_front_left_count, self.pointcloud_front_right_count))
        min_balance_ratio = 0.52 if gate_override else 0.42
        if left_right_min < min_balance_ratio * left_right_max:
            return False
        centered_hint_limit = max(1.5 * self.obstacle_local_y_deadband, 0.14)
        if gate_override:
            centered_hint = (
                abs(self.critical_roi_mean_y) <= 0.10
                and abs(self.duba_konumu) <= max(self.close_side_avoid_lateral_m, 0.18)
            )
        else:
            centered_hint = max(
                abs(self.filtered_obstacle_local_y),
                abs(self.critical_roi_mean_y),
                abs(self.duba_konumu),
            ) <= centered_hint_limit
        if not centered_hint:
            return False
        center_open = self.depth_center_clearance >= max(
            self.tight_gap_clearance_m,
            self.required_gap_clearance_m + (0.00 if gate_override else 0.04),
        )
        center_sparse = self.depth_center_ratio <= max(
            0.12 if gate_override else 0.10,
            (1.70 if gate_override else 1.50) * self.obstacle_center_ratio_threshold,
        )
        upper_sparse = self.depth_upper_ratio <= max(0.05, 2.0 * self.depth_upper_ratio_threshold)
        side_advantage = max(self.depth_left_clearance, self.depth_right_clearance) - self.depth_center_clearance
        depth_balanced = abs(self.depth_left_clearance - self.depth_right_clearance) <= max(
            0.30 if gate_override else 0.18,
            (0.24 if gate_override else 0.18) * max(self.depth_center_clearance, 1.0),
        )
        return (
            center_open
            and center_sparse
            and upper_sparse
            and side_advantage <= (0.32 if gate_override else 0.18)
            and depth_balanced
        )

    def obstacle_context_active(self, now_ns: int) -> bool:
        self.update_tracked_obstacle_memory(now_ns)
        if self.startup_straight_corridor_guard_active(now_ns):
            return False
        if self.pointcloud_center_lane_keep_preferred(now_ns) and not self.commit_active(now_ns):
            return False
        recent_depth = self.signal_recent(self.depth_context_last_ns, self.obstacle_context_sec, now_ns)
        recent_duba = self.fresh_duba_measurement(now_ns)
        near_duba = recent_duba and self.duba_mesafe <= self.pre_avoid_trigger_m
        pass_hold_active = self.side_pass_hold_active(now_ns)
        recent_side_intrusion = (
            self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
            and self.critical_roi_point_count >= self.critical_roi_min_points
            and self.critical_roi_min_x <= self.pre_avoid_trigger_m
            and (
                abs(self.critical_roi_mean_y) >= max(0.08, 0.75 * self.duba_center_escape_y)
                or self.critical_roi_intrusion_m >= self.obstacle_preempt_intrusion_m
                or self.critical_center_ratio >= self.obstacle_preempt_center_ratio
            )
        )
        visible_side_gap = (
            self.depth_frame_recent(now_ns)
            and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
            and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.02)
            and abs(self.depth_selected_gap_offset) >= self.depth_gap_min_offset
        )
        return (
            self.pre_avoid_active
            or self.pointcloud_corridor_signal_active()
            or self.depth_obstacle
            or recent_depth
            or near_duba
            or pass_hold_active
            or recent_side_intrusion
            or visible_side_gap
            or self.tracked_obstacle_valid
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
        if self.startup_straight_corridor_guard_active(now_ns):
            return False
        if self.center_corridor_lane_keep_preferred():
            return False
        depth_recent = self.depth_frame_recent(now_ns)
        centered_trigger = self.centered_obstacle_bypass_active(now_ns)
        lateral_hint = max(abs(self.critical_roi_mean_y), abs(self.duba_konumu))
        lateral_threshold = max(0.10, 0.80 * self.duba_center_escape_y)
        forward_distance, _ = self.nearest_obstacle_measurement(now_ns)
        centered_pressure = (
            self.critical_roi_intrusion_m >= self.obstacle_preempt_intrusion_m
            or self.critical_center_ratio >= self.obstacle_preempt_center_ratio
            or centered_trigger
        )
        if self.critical_center_supported:
            if centered_pressure and forward_distance <= self.pre_avoid_trigger_m:
                return True
            if depth_recent:
                return (
                    forward_distance <= self.pre_avoid_trigger_m
                    and (
                        lateral_hint >= lateral_threshold
                        or centered_pressure
                        or self.depth_center_clearance < self.tight_gap_clearance_m
                    )
                )
            return (
                forward_distance <= self.pre_avoid_trigger_m
                and (
                    lateral_hint >= lateral_threshold
                    or centered_pressure
                )
            )
        if (
            self.critical_roi_point_count >= self.critical_roi_min_points
            and forward_distance <= self.pre_avoid_trigger_m
            and (
                abs(self.critical_roi_mean_y) >= lateral_threshold
                or centered_pressure
            )
        ):
            return True
        if not self.duba_var:
            return False
        return (
            self.duba_mesafe <= self.pre_avoid_trigger_m
            and (
                abs(self.duba_konumu) >= lateral_threshold
                or centered_pressure
                or (
                    depth_recent
                    and self.pointcloud_front_center_count >= self.critical_roi_min_points
                    and self.depth_center_clearance < self.tight_gap_clearance_m
                )
            )
        )

    def centered_bypass_trigger_active(self, now_ns: int) -> bool:
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.critical_center_supported:
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        forward_limit = max(self.close_side_avoid_distance_m, self.duba_center_trigger_m + 0.10)
        if self.critical_roi_min_x > forward_limit:
            return False
        if self.depth_frame_recent(now_ns):
            side_advantage = max(self.depth_left_clearance, self.depth_right_clearance) - self.depth_center_clearance
            center_dense = self.depth_center_ratio >= max(0.10, 1.2 * self.obstacle_center_ratio_threshold)
            return (
                side_advantage >= 0.05
                or center_dense
                or self.pointcloud_front_center_count >= max(self.critical_roi_min_points, int(1.4 * self.pointcloud_front_left_count))
                or self.pointcloud_front_center_count >= max(self.critical_roi_min_points, int(1.4 * self.pointcloud_front_right_count))
            )
        return self.pointcloud_front_center_count >= max(
            self.critical_roi_min_points * 2,
            max(self.pointcloud_front_left_count, self.pointcloud_front_right_count) + self.critical_roi_min_points,
        )

    def centered_obstacle_bypass_active(self, now_ns: int) -> bool:
        if self.centered_bypass_trigger_active(now_ns):
            return True
        if not self.signal_recent(self.duba_last_seen_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.duba_var or not math.isfinite(self.duba_mesafe):
            return False
        return (
            self.duba_nokta_sayisi >= self.duba_min_nokta
            and self.duba_mesafe <= max(self.close_side_avoid_distance_m, self.duba_center_trigger_m + 0.10)
            and abs(self.duba_konumu) <= max(self.close_side_avoid_lateral_m, 0.12)
        )

    def preferred_bypass_direction(self, now_ns: int) -> float:
        side_hint = self.obstacle_side_hint()
        centered_trigger = self.centered_obstacle_bypass_active(now_ns)
        if abs(side_hint) >= self.close_side_avoid_lateral_m:
            return math.copysign(1.0, side_hint)
        if self.depth_frame_recent(now_ns):
            if self.depth_selected_gap_label == 'LEFT':
                return -1.0
            if self.depth_selected_gap_label == 'RIGHT':
                return 1.0
            score_delta = self.depth_right_gap_score - self.depth_left_gap_score
            if abs(score_delta) >= 0.03:
                return 1.0 if score_delta > 0.0 else -1.0
            clearance_delta = self.depth_right_clearance - self.depth_left_clearance
            if abs(clearance_delta) >= 0.04:
                return 1.0 if clearance_delta > 0.0 else -1.0
            if centered_trigger:
                if abs(score_delta) > 1e-3:
                    return 1.0 if score_delta > 0.0 else -1.0
                if abs(clearance_delta) > 1e-3:
                    return 1.0 if clearance_delta > 0.0 else -1.0
        if abs(self.corridor_target_offset) >= 0.08:
            return math.copysign(1.0, self.corridor_target_offset)
        if centered_trigger and self.lane_control_available():
            return -1.0 if self.lane_error > 0.0 else 1.0
        return 0.0

    def reset_corridor_state(self, reason: str) -> None:
        self.corridor_target_offset = 0.0
        self.corridor_active_until_ns = 0
        self.corridor_force_until_ns = 0
        self.corridor_enabled_state = False
        self.corridor_gating_reason = 'authority_reset'
        self.corridor_reset_reason = reason
        self.smoothed_corridor_target = 0.0
        self.depth_gap_offset = 0.0
        self.corridor_error = 0.0
        self.corridor_term_preclamp = 0.0
        self.corridor_term_postclamp = 0.0
        self.return_to_center_until_ns = 0
        self.advisory_side_gap_strength = 0.0
        self.side_gap_suppressed_due_to_no_commit = False
        self.side_target_suppressed_reason = 'none'
        self.target_clipped_to_lane_bounds = False
        self.target_clip_reason = 'none'

    def corridor_gap_available(self, now_ns: int) -> bool:
        if not self.depth_frame_recent(now_ns):
            return False
        if not (
            self.avoidance_required(now_ns)
            or self.return_to_center_active(now_ns)
            or abs(self.corridor_target_offset) >= 0.10
            or self.pre_avoid_active
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
        if self.close_side_bypass_ratio(now_ns) >= 0.05:
            return True
        if self.side_pass_hold_active(now_ns) and self.hard_obstacle_confirmation(now_ns) and abs(self.corridor_target_offset) >= 0.08:
            return True
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
        if self.close_side_bypass_ratio(now_ns) >= 0.05:
            return True
        if self.side_pass_hold_active(now_ns) and self.hard_obstacle_confirmation(now_ns) and abs(self.corridor_target_offset) >= 0.08:
            return True
        if not self.depth_frame_recent(now_ns) and not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        lateral_threshold = max(0.10, 0.80 * self.duba_center_escape_y)
        lateral_hint = max(abs(self.critical_roi_mean_y), abs(self.duba_konumu))
        side_clearance_advantage = max(self.depth_left_clearance, self.depth_right_clearance) - self.depth_center_clearance
        clear_side_available = (
            self.side_bypass_available(now_ns)
            and side_clearance_advantage >= 0.10
        )
        strong_center = (
            self.critical_center_supported
            and self.critical_roi_point_count >= self.critical_roi_min_points
            and self.obstacle_forward_distance <= self.pre_avoid_trigger_m
        )
        center_compressed = (
            self.depth_obstacle
            or self.depth_emergency
            or self.blocked_persistent
            or self.depth_center_clearance < self.obstacle_center_clearance_m
            or self.depth_center_ratio > self.obstacle_center_ratio_threshold
            or self.critical_roi_intrusion_m >= self.obstacle_preempt_intrusion_m
            or self.critical_center_ratio >= self.obstacle_preempt_center_ratio
        )
        lateral_obstacle = (
            lateral_hint >= lateral_threshold
            and self.obstacle_forward_distance <= self.pre_avoid_trigger_m
        )
        return (
            strong_center
            or (clear_side_available and (center_compressed or lateral_obstacle))
            or now_ns < self.critical_avoid_until_ns
            or self.obstacle_latch_state in ('avoid', 'emergency')
        )

    def return_to_center_active(self, now_ns: int) -> bool:
        return now_ns < self.return_to_center_until_ns and abs(self.corridor_target_offset) >= 0.02

    def side_pass_hold_active(self, now_ns: int) -> bool:
        return now_ns < self.duba_pass_hold_until_ns

    def active_commit_session_id(self) -> int:
        if self.commit_session_id > 0:
            return self.commit_session_id
        return self.last_commit_session_id

    def raw_obstacle_side_hint(self) -> float:
        if self.critical_roi_point_count >= self.critical_roi_min_points and abs(self.critical_roi_mean_y) >= 0.05:
            return self.critical_roi_mean_y
        if self.duba_var and abs(self.duba_konumu) >= 0.05:
            return self.duba_konumu
        return 0.0

    def refresh_filtered_obstacle_side_hint(self) -> float:
        raw_hint = self.raw_obstacle_side_hint()
        alpha = self.obstacle_local_y_filter_alpha
        if not self.filtered_obstacle_local_y_valid:
            self.filtered_obstacle_local_y = raw_hint
            self.filtered_obstacle_local_y_valid = True
        else:
            self.filtered_obstacle_local_y = (
                alpha * raw_hint
                + (1.0 - alpha) * self.filtered_obstacle_local_y
            )
        self.filtered_obstacle_local_y = clamp(self.filtered_obstacle_local_y, -2.0, 2.0)
        self.obstacle_local_y_deadband_active = (
            abs(self.filtered_obstacle_local_y) < self.obstacle_local_y_deadband
        )
        if self.obstacle_local_y_deadband_active:
            return 0.0
        return self.filtered_obstacle_local_y

    def filtered_obstacle_side_hint(self) -> float:
        if not self.filtered_obstacle_local_y_valid:
            return self.refresh_filtered_obstacle_side_hint()
        if self.obstacle_local_y_deadband_active:
            return 0.0
        return self.filtered_obstacle_local_y

    def start_commit_session(
        self,
        now_ns: int,
        side: str,
        source: str,
        reason: str,
    ) -> None:
        if side not in ('LEFT', 'RIGHT'):
            return
        if self.side_lock_active and self.locked_pass_side == side and self.commit_session_id > 0:
            self.apply_atomic_pass_commit_state(
                pass_side=side,
                published_pass_side=side,
                side_lock_active=True,
                locked_pass_side=side,
                commit_source=self.pass_commit_source,
                commit_remaining=self.commit_remaining_sec_value,
                commit_remaining_distance=self.commit_remaining_distance_m,
                progress=self.pass_progress,
                commit_session_id=self.commit_session_id,
            )
            return

        self.commit_session_sequence += 1
        next_session_id = self.commit_session_sequence
        self.last_commit_session_id = next_session_id
        self.commit_session_start_reason = reason
        self.side_flip_blocked = False
        self.side_switch_reject_reason = 'none'
        self.side_blocked_cycle_count = 0
        self.commit_exit_clear_count = 0
        self.pass_commit_exit_reason = 'active'
        self.pass_side_none_reason = 'latched_pass_side'
        self.pass_commit_started_ns = now_ns
        self.pass_commit_start_path_m = self.odom_path_length_m if self.have_odom else 0.0
        commit_remaining_distance = self.pass_latch_distance_m if self.have_odom else 0.0
        self.commit_watchdog_last_progress = 0.0
        self.commit_watchdog_last_tracked_local_x = self.tracked_obstacle_local_x if self.tracked_obstacle_valid else 99.0
        self.commit_watchdog_last_odom_path_m = self.odom_path_length_m if self.have_odom else 0.0
        self.commit_watchdog_last_check_ns = now_ns
        self.commit_watchdog_progress_delta = 0.0
        self.commit_watchdog_tracked_local_x_delta = 0.0
        self.commit_watchdog_odom_delta = 0.0
        self.commit_stale_detected = False
        self.stale_obstacle_memory_detected = False
        self.pass_commit_until_ns = now_ns + int(self.pass_latch_duration_sec * 1e9)
        self.apply_atomic_pass_commit_state(
            pass_side=side,
            published_pass_side=side,
            side_lock_active=True,
            locked_pass_side=side,
            commit_source=source,
            commit_remaining=self.pass_latch_duration_sec,
            commit_remaining_distance=commit_remaining_distance,
            progress=0.0,
            commit_session_id=next_session_id,
        )

    def apply_atomic_pass_commit_state(
        self,
        *,
        pass_side: str,
        published_pass_side: str,
        side_lock_active: bool,
        locked_pass_side: str,
        commit_source: str,
        commit_remaining: float,
        commit_remaining_distance: float,
        progress: float,
        commit_session_id: int,
    ) -> None:
        normalized_side = pass_side if pass_side in ('LEFT', 'RIGHT') else 'NONE'
        normalized_published_side = (
            published_pass_side if published_pass_side in ('LEFT', 'RIGHT') else 'NONE'
        )
        normalized_locked_side = (
            locked_pass_side if side_lock_active and locked_pass_side in ('LEFT', 'RIGHT') else 'NONE'
        )
        self.requested_pass_side = normalized_side
        self.selected_pass_side = normalized_side
        self.published_pass_side = normalized_published_side
        self.side_lock_active = bool(side_lock_active and normalized_locked_side in ('LEFT', 'RIGHT'))
        self.locked_pass_side = normalized_locked_side
        self.pass_commit_source = commit_source if commit_source else 'none'
        self.commit_remaining_sec_value = max(0.0, float(commit_remaining))
        self.commit_remaining_distance_m = max(0.0, float(commit_remaining_distance))
        self.pass_progress = clamp(float(progress), 0.0, 1.0)
        self.commit_session_id = max(0, int(commit_session_id))

    def force_clear_authoritative_pass_commit_state(
        self,
        reason: str,
        *,
        zombie: bool = False,
        critical_reject: bool = False,
        clear_critical_avoid: bool = False,
    ) -> None:
        normalized_reason = reason if reason else 'forced_clear'
        if self.commit_session_id > 0:
            self.last_commit_session_id = self.commit_session_id
        self.apply_atomic_pass_commit_state(
            pass_side='NONE',
            published_pass_side='NONE',
            side_lock_active=False,
            locked_pass_side='NONE',
            commit_source='none',
            commit_remaining=0.0,
            commit_remaining_distance=0.0,
            progress=0.0,
            commit_session_id=0,
        )
        self.pass_commit_exit_reason = normalized_reason
        self.pass_commit_until_ns = 0
        self.pass_commit_started_ns = 0
        self.pass_commit_start_path_m = 0.0
        self.pass_side_none_reason = normalized_reason
        self.commit_session_start_reason = 'none'
        self.side_flip_blocked = False
        self.side_switch_reject_reason = 'none'
        self.side_blocked_cycle_count = 0
        self.commit_exit_clear_count = 0
        self.side_selection_candidate = 'NONE'
        self.side_selection_candidate_cycles = 0
        self.pass_side_pending_since_ns = 0
        self.commit_watchdog_last_check_ns = 0
        self.commit_watchdog_progress_delta = 0.0
        self.commit_watchdog_tracked_local_x_delta = 0.0
        self.commit_watchdog_odom_delta = 0.0
        self.commit_stale_detected = False
        self.pass_latch_clear()
        if clear_critical_avoid:
            self.critical_avoid_until_ns = 0
            self.critical_escape_offset = 0.0
            self.critical_avoid_smoothed = 0.0
        self.zombie_commit_state_detected = bool(zombie)
        self.atomic_commit_state_clear_applied = True
        self.critical_reject_forced_state_clear = bool(critical_reject)
        self.pass_state_validity_ok = False

    def sanitize_authoritative_pass_commit_state(self, now_ns: int) -> bool:
        reject_active = (
            self.critical_override_blocked_by_center_corridor
            or self.critical_commit_rejected_reason != 'none'
        )
        if reject_active:
            clear_reason = (
                f'critical_reject:{self.critical_commit_rejected_reason}'
                if self.critical_commit_rejected_reason != 'none'
                else 'critical_override_blocked_by_center_corridor'
            )
            self.force_clear_authoritative_pass_commit_state(
                clear_reason,
                critical_reject=True,
                clear_critical_avoid=True,
            )
            return False
        commit_active = self.commit_active(now_ns)
        illegal_hybrid = (
            (not commit_active and self.requested_pass_side in ('LEFT', 'RIGHT'))
            or (not commit_active and self.selected_pass_side in ('LEFT', 'RIGHT'))
            or
            (not commit_active and self.published_pass_side in ('LEFT', 'RIGHT'))
            or (not commit_active and self.side_lock_active)
            or (not commit_active and self.locked_pass_side in ('LEFT', 'RIGHT'))
            or (not commit_active and self.pass_commit_source != 'none')
            or (not commit_active and self.commit_remaining_sec_value > 1e-6)
            or (not commit_active and self.commit_remaining_distance_m > 1e-6)
            or (not commit_active and self.pass_progress > 1e-6)
        )
        if illegal_hybrid:
            self.force_clear_authoritative_pass_commit_state(
                'zombie_commit_state',
                zombie=True,
                clear_critical_avoid=False,
            )
            return False
        self.pass_state_validity_ok = True
        return True

    def clear_commit_session(
        self,
        exit_reason: str,
        clear_side: bool = True,
    ) -> None:
        del clear_side
        normalized_reason = exit_reason if exit_reason else 'none'
        if self.commit_session_id > 0:
            self.last_commit_session_id = self.commit_session_id
        self.apply_atomic_pass_commit_state(
            pass_side='NONE',
            published_pass_side='NONE',
            side_lock_active=False,
            locked_pass_side='NONE',
            commit_source='none',
            commit_remaining=0.0,
            commit_remaining_distance=0.0,
            progress=0.0,
            commit_session_id=0,
        )
        self.side_flip_blocked = False
        self.side_switch_reject_reason = 'none'
        self.side_blocked_cycle_count = 0
        self.commit_exit_clear_count = 0
        self.side_selection_candidate = 'NONE'
        self.side_selection_candidate_cycles = 0
        self.pass_commit_exit_reason = normalized_reason
        self.commit_watchdog_last_check_ns = 0
        self.commit_watchdog_progress_delta = 0.0
        self.commit_watchdog_tracked_local_x_delta = 0.0
        self.commit_watchdog_odom_delta = 0.0
        self.pass_commit_until_ns = 0
        self.pass_commit_started_ns = 0
        self.pass_commit_start_path_m = 0.0
        self.pass_side_none_reason = normalized_reason

    def hold_locked_pass_side(self, now_ns: int, reason: str) -> None:
        locked_side = self.locked_pass_side
        if locked_side not in ('LEFT', 'RIGHT'):
            return
        self.apply_atomic_pass_commit_state(
            pass_side=locked_side,
            published_pass_side=locked_side,
            side_lock_active=True,
            locked_pass_side=locked_side,
            commit_source=self.pass_commit_source,
            commit_remaining=self.commit_remaining_sec_value,
            commit_remaining_distance=self.commit_remaining_distance_m,
            progress=self.pass_progress,
            commit_session_id=self.commit_session_id,
        )
        self.pass_side_none_reason = 'latched_pass_side'
        self.side_switch_reject_reason = reason
        target_offset = self.pass_side_target_offset(now_ns, locked_side)
        self.force_corridor_hold(now_ns, target_offset, 'pass_commit_hold')
        self.depth_selected_gap_label = locked_side
        self.depth_selected_gap_offset = -0.80 if locked_side == 'LEFT' else 0.80

    def stable_side_selection(self, candidate_side: str) -> str:
        if candidate_side not in ('LEFT', 'RIGHT'):
            self.side_selection_candidate = 'NONE'
            self.side_selection_candidate_cycles = 0
            return 'NONE'

        immediate_choice = self.left_gap_safe ^ self.right_gap_safe
        score_margin = abs(self.depth_left_gap_score - self.depth_right_gap_score)
        stable_margin = (
            immediate_choice
            or score_margin >= self.side_score_margin_min
            or not self.obstacle_local_y_deadband_active
        )
        if self.side_selection_candidate != candidate_side:
            self.side_selection_candidate = candidate_side
            self.side_selection_candidate_cycles = 1
        else:
            self.side_selection_candidate_cycles += 1
        if stable_margin and self.side_selection_candidate_cycles >= self.side_selection_persistence_cycles:
            return candidate_side
        return 'NONE'

    def locked_side_blocked_persistent(self) -> bool:
        if not self.side_lock_active or self.locked_pass_side not in ('LEFT', 'RIGHT'):
            self.side_blocked_cycle_count = 0
            return False
        side_blocked_now = (
            (self.locked_pass_side == 'LEFT' and not self.left_gap_safe)
            or (self.locked_pass_side == 'RIGHT' and not self.right_gap_safe)
        )
        if side_blocked_now:
            self.side_blocked_cycle_count += 1
        else:
            self.side_blocked_cycle_count = 0
        return self.side_blocked_cycle_count >= self.side_block_persistence_cycles

    def stale_commit_active(self, now_ns: int) -> bool:
        return now_ns < self.stale_commit_hold_until_ns

    def expire_tracked_obstacle_memory(self, reason: str) -> None:
        normalized_reason = reason if reason else 'expired'
        self.tracked_obstacle_valid = False
        self.tracked_obstacle_world_x = 0.0
        self.tracked_obstacle_world_y = 0.0
        self.tracked_obstacle_local_x = 99.0
        self.tracked_obstacle_local_y = 0.0
        self.tracked_obstacle_source = 'none'
        self.tracked_obstacle_last_seen_ns = 0
        self.tracked_obstacle_last_refresh_ns = 0
        self.tracked_memory_expire_reason = normalized_reason
        self.stale_obstacle_memory_detected = normalized_reason != 'none'

    def update_commit_stall_watchdog(self, now_ns: int) -> None:
        if not self.commit_active(now_ns) or self.commit_session_id <= 0:
            self.commit_watchdog_last_check_ns = 0
            self.commit_watchdog_progress_delta = 0.0
            self.commit_watchdog_tracked_local_x_delta = 0.0
            self.commit_watchdog_odom_delta = 0.0
            self.commit_stale_detected = False
            return

        progress_now = self.compute_pass_progress(now_ns)
        tracked_local_x_now = self.tracked_obstacle_local_x if self.tracked_obstacle_valid else 99.0
        odom_now = self.odom_path_length_m if self.have_odom else self.pass_commit_start_path_m
        if self.commit_watchdog_last_check_ns <= 0:
            self.commit_watchdog_last_check_ns = now_ns
            self.commit_watchdog_last_progress = progress_now
            self.commit_watchdog_last_tracked_local_x = tracked_local_x_now
            self.commit_watchdog_last_odom_path_m = odom_now
            self.commit_watchdog_progress_delta = 0.0
            self.commit_watchdog_tracked_local_x_delta = 0.0
            self.commit_watchdog_odom_delta = 0.0
            self.commit_stale_detected = False
            return

        commit_age = max(0.0, (now_ns - self.pass_commit_started_ns) / 1e9) if self.pass_commit_started_ns > 0 else 0.0
        fallback_commit = self.pass_commit_source.startswith('fallback')
        fallback_support_lost = fallback_commit and not (
            self.hard_obstacle_confirmation(now_ns)
            or self.signal_recent(self.tracked_obstacle_last_refresh_ns, self.tracked_memory_ttl_sec, now_ns)
        )
        if fallback_support_lost and commit_age >= 0.20:
            self.commit_stale_detected = True
            self.stale_commit_hold_until_ns = now_ns + int(self.commit_stall_timeout_sec * 1e9)
            self.pass_latch_clear()
            self.expire_tracked_obstacle_memory('stale_obstacle_memory')
            self.obstacle_latch_state = 'idle'
            self.obstacle_latch_until_ns = 0
            self.pre_avoid_active = False
            self.obstacle_release_reason = 'fallback_support_lost'
            self.clear_commit_session('fallback_support_lost')
            return

        window_sec = max(0.0, (now_ns - self.commit_watchdog_last_check_ns) / 1e9)
        if window_sec < self.commit_stall_timeout_sec:
            return

        progress_delta = max(0.0, progress_now - self.commit_watchdog_last_progress)
        tracked_local_x_delta = 0.0
        if math.isfinite(tracked_local_x_now) and math.isfinite(self.commit_watchdog_last_tracked_local_x):
            tracked_local_x_delta = abs(tracked_local_x_now - self.commit_watchdog_last_tracked_local_x)
        odom_delta = max(0.0, odom_now - self.commit_watchdog_last_odom_path_m)
        self.commit_watchdog_progress_delta = progress_delta
        self.commit_watchdog_tracked_local_x_delta = tracked_local_x_delta
        self.commit_watchdog_odom_delta = odom_delta
        stalled = (
            commit_age >= self.commit_stall_timeout_sec
            and progress_delta < self.min_progress_delta_for_active_commit
            and tracked_local_x_delta < self.min_tracked_local_x_change_for_active_commit
            and odom_delta < max(0.05, 0.50 * self.min_tracked_local_x_change_for_active_commit)
        )
        if stalled:
            reason = 'stale_commit_timeout'
            self.commit_stale_detected = True
            self.stale_commit_hold_until_ns = now_ns + int(self.commit_stall_timeout_sec * 1e9)
            self.pass_latch_clear()
            self.expire_tracked_obstacle_memory('stale_obstacle_memory')
            self.obstacle_latch_state = 'idle'
            self.obstacle_latch_until_ns = 0
            self.pre_avoid_active = False
            self.obstacle_release_reason = reason
            self.clear_commit_session(reason)
            return

        self.commit_watchdog_last_check_ns = now_ns
        self.commit_watchdog_last_progress = progress_now
        self.commit_watchdog_last_tracked_local_x = tracked_local_x_now
        self.commit_watchdog_last_odom_path_m = odom_now
        self.commit_stale_detected = False

    def commit_active(self, now_ns: int) -> bool:
        return (
            self.commit_session_id > 0
            and self.locked_pass_side in ('LEFT', 'RIGHT')
            and (
                now_ns < self.pass_commit_until_ns
                or self.obstacle_latch_state in ('avoid', 'emergency')
                or now_ns < self.critical_avoid_until_ns
                or self.side_pass_hold_active(now_ns)
                or self.pass_latch_active
            )
        )

    def commit_remaining_sec(self, now_ns: int) -> float:
        if self.commit_session_id <= 0:
            self.commit_remaining_sec_value = 0.0
            return 0.0
        remaining_ns = 0
        remaining_ns = max(remaining_ns, self.pass_commit_until_ns - now_ns)
        remaining_ns = max(remaining_ns, self.obstacle_latch_until_ns - now_ns)
        remaining_ns = max(remaining_ns, self.critical_avoid_until_ns - now_ns)
        remaining_ns = max(remaining_ns, self.duba_pass_hold_until_ns - now_ns)
        if self.pass_latch_active and self.pass_latch_started_ns > 0:
            max_hold_ns = int(self.avoid_pass_max_hold_sec * 1e9)
            remaining_ns = max(
                remaining_ns,
                (self.pass_latch_started_ns + max_hold_ns) - now_ns,
            )
        remaining_sec = max(0.0, remaining_ns / 1e9)
        if self.commit_session_id > 0:
            if self.commit_remaining_sec_value <= 0.0:
                self.commit_remaining_sec_value = remaining_sec
            else:
                self.commit_remaining_sec_value = min(self.commit_remaining_sec_value, remaining_sec)
            return self.commit_remaining_sec_value
        self.commit_remaining_sec_value = remaining_sec
        return remaining_sec

    def commit_remaining_distance(self, now_ns: int) -> float:
        if self.commit_session_id <= 0:
            self.commit_remaining_distance_m = 0.0
            return 0.0
        if not self.have_odom:
            return self.commit_remaining_distance_m
        traveled_m = 0.0
        if self.pass_latch_active:
            traveled_m = max(0.0, self.odom_path_length_m - self.pass_latch_start_path_m)
        elif self.pass_commit_started_ns > 0:
            traveled_m = max(0.0, self.odom_path_length_m - self.pass_commit_start_path_m)
        remaining_distance = max(0.0, self.pass_latch_distance_m - traveled_m)
        if self.commit_session_id > 0 and self.commit_remaining_distance_m > 0.0:
            self.commit_remaining_distance_m = min(self.commit_remaining_distance_m, remaining_distance)
        else:
            self.commit_remaining_distance_m = remaining_distance
        return self.commit_remaining_distance_m

    def compute_pass_progress(self, now_ns: int) -> float:
        if self.commit_session_id <= 0:
            self.pass_progress = 0.0
            return 0.0
        progress = 0.0
        if self.commit_active(now_ns):
            if self.have_odom and self.pass_commit_started_ns > 0:
                traveled_m = max(0.0, self.odom_path_length_m - self.pass_commit_start_path_m)
                progress = clamp(traveled_m / max(self.pass_latch_distance_m, 1e-3), 0.0, 1.0)
            elif self.pass_commit_started_ns > 0:
                elapsed_sec = max(0.0, (now_ns - self.pass_commit_started_ns) / 1e9)
                progress = clamp(elapsed_sec / max(self.pass_latch_duration_sec, 1e-3), 0.0, 1.0)
            if self.pass_latch_active:
                local_x, local_y = self.world_to_vehicle(
                    self.pass_latch_obstacle_world_x,
                    self.pass_latch_obstacle_world_y,
                )
                behind_threshold = self.pass_latch_obstacle_radius_m + self.avoid_pass_longitudinal_margin_m
                if local_x < -behind_threshold:
                    progress = max(progress, self.progress_completion_threshold)
                if abs(local_y) > (self.pass_latch_obstacle_radius_m + self.avoid_pass_lateral_clearance_m):
                    progress = max(progress, min(1.0, self.progress_completion_threshold + 0.04))
        self.pass_progress = max(self.pass_progress, clamp(progress, 0.0, 1.0))
        return self.pass_progress

    def pick_pass_side(self, now_ns: int) -> str:
        left_score = self.depth_left_gap_score
        right_score = self.depth_right_gap_score
        if self.side_lock_active and self.locked_pass_side in ('LEFT', 'RIGHT'):
            return self.locked_pass_side
        if self.requested_pass_side in ('LEFT', 'RIGHT'):
            return self.requested_pass_side
        if self.left_gap_safe and not self.right_gap_safe:
            return 'LEFT'
        if self.right_gap_safe and not self.left_gap_safe:
            return 'RIGHT'
        if not (self.left_gap_safe or self.right_gap_safe):
            return 'NONE'
        obstacle_side = self.filtered_obstacle_side_hint()
        if abs(obstacle_side) >= 0.05:
            preferred = 'RIGHT' if obstacle_side > 0.0 else 'LEFT'
            if preferred == 'LEFT' and self.left_gap_safe:
                return 'LEFT'
            if preferred == 'RIGHT' and self.right_gap_safe:
                return 'RIGHT'
        if self.selected_pass_side == 'LEFT' and self.left_gap_safe and left_score >= right_score - self.gap_switch_margin:
            return 'LEFT'
        if self.selected_pass_side == 'RIGHT' and self.right_gap_safe and right_score >= left_score - self.gap_switch_margin:
            return 'RIGHT'
        if abs(left_score - right_score) <= 0.03:
            clearance_delta = self.depth_left_clearance - self.depth_right_clearance
            if abs(clearance_delta) >= 0.02:
                return 'LEFT' if clearance_delta > 0.0 else 'RIGHT'
            return 'RIGHT'
        return 'LEFT' if left_score >= right_score else 'RIGHT'

    def pass_side_target_offset(self, now_ns: int, pass_side: str) -> float:
        if pass_side not in ('LEFT', 'RIGHT'):
            return 0.0
        sign = -1.0 if pass_side == 'LEFT' else 1.0
        candidates = []
        mapped_gap = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        if abs(mapped_gap) >= self.pre_avoid_min_offset_m and mapped_gap * sign > 0.0:
            candidates.append(mapped_gap)
        if abs(self.corridor_target_offset) >= self.pre_avoid_min_offset_m and self.corridor_target_offset * sign > 0.0:
            candidates.append(self.corridor_target_offset)
        close_target = self.close_side_bypass_target(now_ns)
        if abs(close_target) >= self.pre_avoid_min_offset_m and close_target * sign > 0.0:
            candidates.append(close_target)
        fallback_mag = clamp(
            max(self.pre_avoid_max_offset_m, self.close_side_avoid_min_offset_m),
            self.pre_avoid_min_offset_m,
            self.lane_corridor_cap,
        )
        candidates.append(math.copysign(fallback_mag, sign))
        best_target = max(candidates, key=lambda value: abs(value))
        return self.clip_target_to_lane_corridor(best_target, now_ns, 'pass_side_target')

    def force_corridor_hold(self, now_ns: int, target_offset: float, reason: str) -> None:
        if abs(target_offset) < 0.05:
            return
        self.corridor_target_offset = self.clip_target_to_lane_corridor(target_offset, now_ns, reason)
        hold_ns = int(self.min_corridor_hold_sec * 1e9)
        hysteresis_ns = int(self.corridor_gating_hysteresis_sec * 1e9)
        self.corridor_active_until_ns = max(self.corridor_active_until_ns, now_ns + hold_ns)
        self.corridor_force_until_ns = max(
            self.corridor_force_until_ns,
            self.corridor_active_until_ns,
            now_ns + hysteresis_ns,
        )
        self.corridor_enabled_state = True
        self.corridor_gating_reason = reason
        self.corridor_reset_reason = 'hold'
        self.smoothed_corridor_target = self.corridor_target_offset
        self.depth_gap_offset = self.corridor_target_offset

    def update_pass_authority(self, now_ns: int) -> None:
        self.refresh_filtered_obstacle_side_hint()
        if not self.sanitize_authoritative_pass_commit_state(now_ns):
            self.depth_selected_gap_label = 'CENTER'
            self.depth_selected_gap_offset = 0.0
            self.corridor_target_offset = 0.0
            return
        if self.startup_straight_corridor_guard_active(now_ns):
            self.requested_pass_side = 'NONE'
            self.selected_pass_side = 'NONE'
            self.published_pass_side = 'NONE'
            self.pass_side_none_reason = 'startup_straight_corridor_guard'
            if self.commit_session_id <= 0 and not self.pass_latch_active:
                self.depth_selected_gap_label = 'CENTER'
            return
        if self.center_gap_recovery_preferred(now_ns):
            self.requested_pass_side = 'NONE'
            self.selected_pass_side = 'NONE'
            self.published_pass_side = 'NONE'
            self.pass_side_none_reason = 'center_gap_recovery'
            self.depth_selected_gap_label = 'CENTER'
            self.depth_selected_gap_offset = 0.0
            self.pass_latch_clear()
            self.reset_corridor_state('center_gap_recovery')
            if self.commit_session_id > 0:
                self.clear_commit_session('center_gap_recovery')
            return
        if self.center_corridor_preferred and not self.commit_active(now_ns):
            self.requested_pass_side = 'NONE'
            self.selected_pass_side = 'NONE'
            self.published_pass_side = 'NONE'
            self.pass_side_none_reason = 'lane_bounded_center_corridor'
            self.depth_selected_gap_label = 'CENTER'
            self.depth_selected_gap_offset = 0.0
            self.corridor_target_offset = 0.0
            return
        if self.center_corridor_lane_keep_preferred() and not self.commit_active(now_ns):
            self.requested_pass_side = 'NONE'
            self.selected_pass_side = 'NONE'
            self.published_pass_side = 'NONE'
            self.pass_side_none_reason = 'center_corridor_lane_keep'
            self.depth_selected_gap_label = 'CENTER'
            return
        needs_side_pass = (
            self.pre_avoid_active
            or self.obstacle_latch_state in ('avoid', 'emergency')
            or self.pass_latch_active
            or self.side_pass_hold_active(now_ns)
            or now_ns < self.pass_commit_until_ns
            or self.centered_obstacle_bypass_active(now_ns)
        )
        self.requested_pass_side = 'NONE'
        if self.depth_selected_gap_label in ('LEFT', 'RIGHT'):
            self.requested_pass_side = self.depth_selected_gap_label
        elif self.left_gap_safe ^ self.right_gap_safe:
            self.requested_pass_side = 'LEFT' if self.left_gap_safe else 'RIGHT'

        candidate_side = self.requested_pass_side
        commit_source = 'depth_selected_gap' if candidate_side in ('LEFT', 'RIGHT') else 'none'
        self.fallback_side_triggered = False
        if self.side_lock_active and self.locked_pass_side in ('LEFT', 'RIGHT'):
            hold_state = (
                needs_side_pass
                or self.post_avoid_hold_active(now_ns)
                or self.return_to_center_active(now_ns)
            )
            if not hold_state:
                exit_reason = self.obstacle_release_reason if self.obstacle_release_reason != 'init' else 'commit_complete'
                self.clear_commit_session(exit_reason)
                return
            if self.locked_side_blocked_persistent():
                self.side_flip_blocked = True
                self.side_switch_reject_reason = 'selected_side_blocked_persistent'
                self.pass_latch_clear()
                self.clear_commit_session('selected_side_blocked_abort')
                return
            if candidate_side not in ('NONE', self.locked_pass_side):
                self.side_flip_blocked = True
                self.side_switch_reject_reason = (
                    f'active_commit_lock_reject:{candidate_side.lower()}->{self.locked_pass_side.lower()}'
                )
            elif candidate_side == 'NONE':
                self.side_flip_blocked = False
                self.side_switch_reject_reason = 'active_commit_lock_hold'
            else:
                self.side_flip_blocked = False
                self.side_switch_reject_reason = 'active_commit_lock_confirm'
            self.hold_locked_pass_side(now_ns, self.side_switch_reject_reason)
            return

        if needs_side_pass and candidate_side == 'NONE':
            if self.pass_side_pending_since_ns <= 0:
                self.pass_side_pending_since_ns = now_ns
            pending_sec = max(0.0, (now_ns - self.pass_side_pending_since_ns) / 1e9)
            if pending_sec >= self.fallback_side_selection_timeout_sec:
                fallback_side = self.pick_pass_side(now_ns)
                side_margin = abs(self.depth_left_gap_score - self.depth_right_gap_score)
                if (
                    fallback_side in ('LEFT', 'RIGHT')
                    and side_margin >= self.fallback_commit_score_margin_min
                ):
                    candidate_side = fallback_side
                    commit_source = 'fallback_gap_score'
                    self.fallback_side_triggered = True
                    self.fallback_side_last_triggered_ns = now_ns
                elif fallback_side in ('LEFT', 'RIGHT'):
                    self.pass_side_none_reason = 'fallback_margin_too_weak'
        else:
            self.pass_side_pending_since_ns = 0

        if needs_side_pass and candidate_side in ('LEFT', 'RIGHT'):
            stable_side = self.stable_side_selection(candidate_side)
            if stable_side in ('LEFT', 'RIGHT'):
                candidate_side = stable_side
            else:
                candidate_side = 'NONE'
                self.pass_side_none_reason = 'awaiting_side_selection_persistence'
        elif needs_side_pass and candidate_side == 'NONE':
            self.pass_side_none_reason = 'awaiting_side_selection'

        if candidate_side in ('LEFT', 'RIGHT') and needs_side_pass:
            self.start_commit_session(
                now_ns,
                candidate_side,
                commit_source,
                f'new_commit:{commit_source}',
            )
            self.hold_locked_pass_side(now_ns, 'new_commit_session')
        else:
            hold_state = (
                self.commit_active(now_ns)
                or self.post_avoid_hold_active(now_ns)
                or self.return_to_center_active(now_ns)
            )
            if not hold_state:
                exit_reason = self.obstacle_release_reason if self.obstacle_release_reason != 'init' else 'no_side_bias'
                self.clear_commit_session(exit_reason)
            elif self.selected_pass_side in ('LEFT', 'RIGHT'):
                self.published_pass_side = self.selected_pass_side
                target_offset = self.pass_side_target_offset(now_ns, self.selected_pass_side)
                self.force_corridor_hold(now_ns, target_offset, 'commit_hysteresis_hold')

    def obstacle_side_hint(self) -> float:
        return self.filtered_obstacle_side_hint()

    def current_obstacle_measurement_local(
        self,
        now_ns: int,
        strong_only: bool = False,
    ) -> Tuple[float, float, float, str]:
        if self.startup_straight_corridor_guard_active(now_ns):
            return 99.0, 0.0, 0.20, 'none'
        if (
            self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
            and self.critical_roi_point_count >= self.critical_roi_min_points
            and math.isfinite(self.critical_roi_min_x)
            and self.critical_roi_min_x < 90.0
        ):
            return (
                self.critical_roi_min_x,
                self.critical_roi_mean_y,
                max(0.18, 0.5 * self.critical_roi_intrusion_m + 0.18),
                'critical_dist',
            )
        live_duba_measurement = (
            self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
            and self.duba_nokta_sayisi >= self.duba_cikis_min_nokta
            and math.isfinite(self.duba_mesafe)
            and self.duba_mesafe < 90.0
            and abs(self.duba_konumu) <= (self.critical_roi_half_width_m + 0.10)
        )
        if live_duba_measurement:
            return (
                self.duba_mesafe,
                self.duba_konumu,
                0.20,
                'dist',
            )
        if not strong_only and self.fresh_duba_measurement(now_ns):
            return (
                self.duba_mesafe,
                self.duba_konumu,
                0.20,
                'dist',
            )
        return 99.0, 0.0, 0.20, 'none'

    def update_tracked_obstacle_memory(self, now_ns: int) -> None:
        if not self.have_odom:
            return
        if self.startup_straight_corridor_guard_active(now_ns):
            if self.tracked_obstacle_valid:
                self.expire_tracked_obstacle_memory('startup_straight_corridor_guard')
            else:
                self.tracked_memory_expire_reason = 'startup_straight_corridor_guard'
                self.tracked_obstacle_local_x = 99.0
                self.tracked_obstacle_local_y = 0.0
            return
        if (
            self.pointcloud_center_lane_keep_preferred(now_ns)
            and not self.pass_latch_active
            and self.commit_session_id <= 0
        ):
            if self.tracked_obstacle_valid:
                self.expire_tracked_obstacle_memory('center_corridor_lane_keep')
            else:
                self.tracked_memory_expire_reason = 'center_corridor_lane_keep'
                self.tracked_obstacle_local_x = 99.0
                self.tracked_obstacle_local_y = 0.0
            return
        if (
            self.center_corridor_lane_keep_preferred()
            and not self.pass_latch_active
            and self.commit_session_id <= 0
        ):
            if self.tracked_obstacle_valid:
                self.expire_tracked_obstacle_memory('center_corridor_lane_keep')
            else:
                self.tracked_memory_expire_reason = 'center_corridor_lane_keep'
                self.tracked_obstacle_local_x = 99.0
                self.tracked_obstacle_local_y = 0.0
            return
        if (
            self.center_gap_recovery_preferred(now_ns)
            and not self.pass_latch_active
            and self.commit_session_id <= 0
        ):
            if self.tracked_obstacle_valid:
                self.expire_tracked_obstacle_memory('center_gap_recovery')
            else:
                self.tracked_memory_expire_reason = 'center_gap_recovery'
                self.tracked_obstacle_local_x = 99.0
                self.tracked_obstacle_local_y = 0.0
            return
        local_x, local_y, radius_m, source = self.current_obstacle_measurement_local(
            now_ns,
            strong_only=self.tracked_memory_require_strong_source,
        )
        if source != 'none' and math.isfinite(local_x) and local_x < 90.0:
            world_x, world_y = self.vehicle_to_world(local_x, local_y)
            if self.tracked_obstacle_valid:
                jump_m = math.hypot(
                    world_x - self.tracked_obstacle_world_x,
                    world_y - self.tracked_obstacle_world_y,
                )
                if jump_m <= self.tracked_obstacle_match_distance_m:
                    blend = 0.55
                    world_x = blend * world_x + (1.0 - blend) * self.tracked_obstacle_world_x
                    world_y = blend * world_y + (1.0 - blend) * self.tracked_obstacle_world_y
            self.tracked_obstacle_world_x = world_x
            self.tracked_obstacle_world_y = world_y
            self.tracked_obstacle_radius_m = max(0.12, radius_m)
            self.tracked_obstacle_last_seen_ns = now_ns
            self.tracked_obstacle_last_refresh_ns = now_ns
            self.tracked_obstacle_source = source
            self.tracked_obstacle_valid = True
            self.tracked_memory_expire_reason = 'fresh_observation'
            self.stale_obstacle_memory_detected = False
        elif not self.signal_recent(self.tracked_obstacle_last_refresh_ns, self.tracked_memory_ttl_sec, now_ns):
            self.expire_tracked_obstacle_memory('tracked_memory_ttl_expired')
        elif not self.signal_recent(self.tracked_obstacle_last_seen_ns, self.tracked_obstacle_persist_sec, now_ns):
            self.expire_tracked_obstacle_memory('tracked_memory_persist_expired')

        if self.tracked_obstacle_valid:
            tracked_local_x, tracked_local_y = self.world_to_vehicle(
                self.tracked_obstacle_world_x,
                self.tracked_obstacle_world_y,
            )
            self.tracked_obstacle_local_x = tracked_local_x
            self.tracked_obstacle_local_y = tracked_local_y
            if (
                abs(tracked_local_y) > (self.tracked_obstacle_lateral_gate_m + self.tracked_obstacle_radius_m)
                or tracked_local_x < -(self.avoid_pass_longitudinal_margin_m + self.tracked_obstacle_radius_m)
                or not self.signal_recent(self.tracked_obstacle_last_seen_ns, self.tracked_obstacle_persist_sec, now_ns)
            ):
                if abs(tracked_local_y) > (self.tracked_obstacle_lateral_gate_m + self.tracked_obstacle_radius_m):
                    self.expire_tracked_obstacle_memory('tracked_memory_lateral_gate')
                elif tracked_local_x < -(self.avoid_pass_longitudinal_margin_m + self.tracked_obstacle_radius_m):
                    self.expire_tracked_obstacle_memory('tracked_memory_passed_vehicle')
                else:
                    self.expire_tracked_obstacle_memory('tracked_memory_persist_expired')
        else:
            self.tracked_obstacle_local_x = 99.0
            self.tracked_obstacle_local_y = 0.0

    def start_pass_latch(self, now_ns: int) -> None:
        if not self.force_odom_pass_latch or not self.have_odom:
            return
        self.pass_latch_active = True
        self.pass_latch_started_ns = now_ns
        if self.pass_commit_started_ns <= 0:
            self.pass_commit_started_ns = now_ns
            self.pass_commit_start_path_m = self.odom_path_length_m
        self.pass_latch_start_x = self.x
        self.pass_latch_start_y = self.y
        self.pass_latch_start_path_m = self.odom_path_length_m
        self.pass_latch_travel_m = 0.0
        self.update_tracked_obstacle_memory(now_ns)
        if self.tracked_obstacle_valid:
            self.pass_latch_obstacle_world_x = self.tracked_obstacle_world_x
            self.pass_latch_obstacle_world_y = self.tracked_obstacle_world_y
            self.pass_latch_obstacle_radius_m = self.tracked_obstacle_radius_m
            self.pass_latch_source = self.tracked_obstacle_source
            return
        local_x, local_y, radius_m, source = self.current_obstacle_measurement_local(now_ns, strong_only=False)
        seed_x = local_x if source != 'none' and math.isfinite(local_x) and local_x < 90.0 else self.near_avoid_trigger_m
        seed_y = local_y if source != 'none' and math.isfinite(local_y) else 0.0
        world_x, world_y = self.vehicle_to_world(seed_x, seed_y)
        self.pass_latch_obstacle_world_x = world_x
        self.pass_latch_obstacle_world_y = world_y
        self.pass_latch_obstacle_radius_m = max(0.16, radius_m)
        self.pass_latch_source = source if source != 'none' else 'progress_only'

    def pass_latch_clear(self) -> None:
        self.pass_latch_active = False
        self.pass_latch_started_ns = 0
        self.pass_latch_source = 'none'
        self.pass_latch_travel_m = 0.0

    def odom_pass_latch_active(self, now_ns: int) -> bool:
        if not self.pass_latch_active or not self.have_odom:
            self.commit_exit_clear_count = 0
            return False
        travel_m = max(0.0, self.odom_path_length_m - self.pass_latch_start_path_m)
        self.pass_latch_travel_m = travel_m
        self.commit_remaining_distance_m = max(0.0, self.pass_latch_distance_m - travel_m)
        elapsed_sec = max(0.0, (now_ns - self.pass_latch_started_ns) / 1e9)
        if elapsed_sec >= self.avoid_pass_max_hold_sec:
            self.pass_latch_clear()
            self.tracked_obstacle_valid = False
            self.tracked_obstacle_source = 'none'
            self.obstacle_release_reason = 'odom_pass_timeout'
            self.clear_commit_session(self.obstacle_release_reason)
            return False
        progress_ratio = clamp(travel_m / max(self.pass_latch_distance_m, 1e-3), 0.0, 1.0)
        if progress_ratio < self.progress_completion_threshold:
            self.obstacle_release_reason = 'odom_progress_hold'
            return True
        local_x, local_y = self.world_to_vehicle(
            self.pass_latch_obstacle_world_x,
            self.pass_latch_obstacle_world_y,
        )
        passed_longitudinal = local_x < -(
            self.pass_latch_obstacle_radius_m + self.avoid_pass_longitudinal_margin_m
        )
        cleared_lateral = abs(local_y) > (
            self.pass_latch_obstacle_radius_m + self.avoid_pass_lateral_clearance_m
        )
        front_critical_clear = (
            self.critical_roi_min_x > self.commit_exit_clearance_distance_m
            and self.pointcloud_front_min_distance > self.commit_exit_clearance_distance_m
            and self.depth_center_clearance > self.commit_exit_clearance_distance_m
            and not self.critical_obstacle_now
            and not self.depth_obstacle
            and not self.depth_emergency
        )
        if front_critical_clear:
            self.commit_exit_clear_count += 1
        else:
            self.commit_exit_clear_count = 0
        front_critical_clear = self.commit_exit_clear_count >= self.commit_exit_clear_cycles
        corridor_recentered_clear = (
            front_critical_clear
            and abs(self.corridor_target_offset) <= max(0.08, self.pre_avoid_min_offset_m)
            and not self.centered_obstacle_bypass_active(now_ns)
        )
        progress_only_source = self.pass_latch_source == 'progress_only'
        if progress_only_source:
            self.pass_latch_clear()
            self.tracked_obstacle_valid = False
            self.tracked_obstacle_source = 'none'
            self.obstacle_release_reason = 'odom_progress_release'
            self.clear_commit_session(self.obstacle_release_reason)
            return False
        if passed_longitudinal and (cleared_lateral or travel_m >= (self.pass_latch_distance_m + 0.20)):
            self.pass_latch_clear()
            self.tracked_obstacle_valid = False
            self.tracked_obstacle_source = 'none'
            self.obstacle_release_reason = 'odom_passed'
            self.clear_commit_session(self.obstacle_release_reason)
            return False
        if progress_ratio >= 1.0 and front_critical_clear:
            self.pass_latch_clear()
            self.tracked_obstacle_valid = False
            self.tracked_obstacle_source = 'none'
            self.obstacle_release_reason = 'front_clear_after_commit_distance'
            self.clear_commit_session(self.obstacle_release_reason)
            return False
        if corridor_recentered_clear:
            self.pass_latch_clear()
            self.tracked_obstacle_valid = False
            self.tracked_obstacle_source = 'none'
            self.obstacle_release_reason = 'corridor_recentered_clear'
            self.clear_commit_session(self.obstacle_release_reason)
            return False
        self.obstacle_release_reason = 'odom_pass_latch'
        return True

    def fresh_duba_measurement(self, now_ns: int) -> bool:
        return (
            self.duba_var
            and self.duba_nokta_sayisi >= self.duba_cikis_min_nokta
            and self.signal_recent(self.duba_last_seen_ns, self.duba_preempt_max_age_sec, now_ns)
        )

    def hard_obstacle_confirmation(self, now_ns: int) -> bool:
        self.update_tracked_obstacle_memory(now_ns)
        if self.startup_straight_corridor_guard_active(now_ns):
            return False
        if self.pointcloud_center_lane_keep_preferred(now_ns):
            return False
        if self.center_corridor_lane_keep_preferred():
            return False
        if self.center_gap_recovery_preferred(now_ns):
            return False
        return (
            (
                self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
                and (
                    (
                        self.critical_roi_point_count >= self.critical_roi_min_points
                        and self.critical_roi_min_x <= self.pre_avoid_trigger_m
                    )
                    or self.pointcloud_front_min_distance <= self.pre_avoid_trigger_m
                )
            )
            or (
                self.depth_frame_recent(now_ns)
                and (
                    self.depth_obstacle
                    or self.depth_emergency
                    or (
                        self.depth_center_clearance <= self.pre_avoid_trigger_m
                        and self.depth_center_ratio >= self.obstacle_center_ratio_threshold
                    )
                )
            )
        )

    def nearest_obstacle_measurement(self, now_ns: int) -> Tuple[float, str]:
        candidates = []
        self.update_tracked_obstacle_memory(now_ns)
        if self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            if (
                self.critical_roi_point_count >= self.critical_roi_min_points
                and math.isfinite(self.critical_roi_min_x)
                and self.critical_roi_min_x < 90.0
            ):
                candidates.append((self.critical_roi_min_x, 'critical_dist'))
            if math.isfinite(self.pointcloud_front_min_distance) and self.pointcloud_front_min_distance < 90.0:
                candidates.append((self.pointcloud_front_min_distance, 'front_min'))
        if self.fresh_duba_measurement(now_ns):
            if self.duba_var and math.isfinite(self.duba_mesafe) and self.duba_mesafe < 90.0:
                candidates.append((self.duba_mesafe, 'dist'))
        if self.depth_frame_recent(now_ns) and self.depth_center_clearance < self.depth_far_m:
            if (
                self.depth_center_ratio >= self.obstacle_center_ratio_threshold
                or self.depth_obstacle
                or self.depth_emergency
            ):
                candidates.append((self.depth_center_clearance, 'depth_center'))
        allow_tracked_memory = (
            self.pass_latch_active
            or self.signal_recent(self.critical_obstacle_last_seen_ns, self.tracked_obstacle_persist_sec, now_ns)
        )
        if (
            allow_tracked_memory
            and
            self.tracked_obstacle_valid
            and 0.0 < self.tracked_obstacle_local_x <= self.obstacle_release_distance_m
            and abs(self.tracked_obstacle_local_y) <= (
                self.tracked_obstacle_lateral_gate_m + self.tracked_obstacle_radius_m
            )
        ):
            candidates.append((self.tracked_obstacle_local_x, 'tracked_memory'))
        if not candidates:
            return 99.0, 'none'
        return min(candidates, key=lambda item: item[0])

    def obstacle_signal_supported(self, now_ns: int) -> bool:
        self.update_tracked_obstacle_memory(now_ns)
        if self.startup_straight_corridor_guard_active(now_ns):
            return False
        if self.pointcloud_center_lane_keep_preferred(now_ns) and not self.commit_active(now_ns):
            return False
        if self.center_corridor_lane_keep_preferred():
            return False
        if self.center_gap_recovery_preferred(now_ns) and not self.commit_active(now_ns):
            return False
        recent_pointcloud = self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
        recent_duba = self.fresh_duba_measurement(now_ns)
        strong_center = (
            recent_pointcloud
            and self.critical_center_supported
            and self.critical_roi_point_count >= self.critical_roi_min_points
        )
        return (
            self.depth_frame_recent(now_ns)
            or self.depth_obstacle
            or self.depth_emergency
            or strong_center
            or recent_duba
            or self.tracked_obstacle_valid
        )

    def compute_obstacle_stage(self, now_ns: int) -> Tuple[str, float, str]:
        distance, distance_source = self.nearest_obstacle_measurement(now_ns)
        self.update_critical_intrusion_persistence()
        if self.startup_straight_corridor_guard_active(now_ns):
            return 'idle', distance, 'startup_straight_corridor_guard'
        if self.pointcloud_center_lane_keep_preferred(now_ns) and not self.commit_active(now_ns):
            return 'idle', distance, 'center_corridor_lane_keep'
        if self.center_corridor_lane_keep_preferred():
            return 'idle', distance, 'center_corridor_lane_keep'
        if self.center_gap_recovery_preferred(now_ns):
            return 'idle', distance, 'center_gap_recovery'
        strong_center = (
            self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
            and self.critical_center_supported
            and self.critical_roi_point_count >= self.critical_roi_min_points
        )
        hard_confirmation = self.hard_obstacle_confirmation(now_ns)
        intrusion_trigger = self.critical_roi_intrusion_m >= self.obstacle_preempt_intrusion_m
        center_ratio_trigger = self.critical_center_ratio >= self.obstacle_preempt_center_ratio
        centered_bypass = self.centered_obstacle_bypass_active(now_ns)
        depth_center_blocked = (
            self.depth_frame_recent(now_ns)
            and (
                self.depth_obstacle
                or self.depth_emergency
                or (
                    self.depth_center_clearance <= self.pre_avoid_trigger_m
                    and self.depth_center_ratio >= self.obstacle_center_ratio_threshold
                )
            )
        )
        if not self.obstacle_signal_supported(now_ns):
            return 'idle', distance, 'none'

        trigger_source = distance_source
        if intrusion_trigger and (trigger_source == 'none' or distance > self.pre_avoid_trigger_m):
            trigger_source = 'intrusion'
        elif center_ratio_trigger and (trigger_source == 'none' or distance > self.pre_avoid_trigger_m):
            trigger_source = 'center_ratio'

        hard_distance_source = distance_source in ('critical_dist', 'front_min', 'depth_center')
        emergency_signal = (
            self.depth_emergency
            or (
                distance <= self.emergency_avoid_trigger_m
                and (hard_distance_source or hard_confirmation or centered_bypass)
            )
        )
        emergency_blocked, emergency_block_reason = self.should_block_emergency_latch(now_ns)
        if emergency_signal and emergency_blocked:
            emergency_signal = False
            if trigger_source in ('critical_dist', 'tracked_memory', 'intrusion', 'center_ratio'):
                trigger_source = emergency_block_reason
        near_signal = (
            emergency_signal
            or (
                distance <= self.near_avoid_trigger_m
                and (hard_distance_source or hard_confirmation or centered_bypass)
            )
            or (
                strong_center
                and intrusion_trigger
                and distance <= (self.near_avoid_trigger_m + 0.12)
            )
        )
        pre_avoid_signal = (
            near_signal
            or distance <= self.pre_avoid_trigger_m
            or centered_bypass
            or (
                strong_center
                and (intrusion_trigger or center_ratio_trigger or depth_center_blocked)
            )
        )

        if emergency_signal:
            return 'emergency', distance, trigger_source if trigger_source != 'none' else 'critical_dist'
        if near_signal and (strong_center or centered_bypass or depth_center_blocked or self.duba_var):
            return 'avoid', distance, trigger_source
        if pre_avoid_signal:
            return 'pre_avoid', distance, trigger_source
        return 'idle', distance, 'none'

    def compute_obstacle_speed_scale(self, distance: float) -> float:
        if distance >= self.pre_avoid_trigger_m:
            return 1.0
        if distance <= self.emergency_avoid_trigger_m:
            return self.pre_avoid_speed_scale_emergency
        if distance <= self.near_avoid_trigger_m:
            ratio = clamp(
                (distance - self.emergency_avoid_trigger_m)
                / max(self.near_avoid_trigger_m - self.emergency_avoid_trigger_m, 1e-3),
                0.0,
                1.0,
            )
            return (
                ratio * self.pre_avoid_speed_scale_near
                + (1.0 - ratio) * self.pre_avoid_speed_scale_emergency
            )
        ratio = clamp(
            (distance - self.near_avoid_trigger_m)
            / max(self.pre_avoid_trigger_m - self.near_avoid_trigger_m, 1e-3),
            0.0,
            1.0,
        )
        return ratio + (1.0 - ratio) * self.pre_avoid_speed_scale_near

    def obstacle_preempt_target(self, now_ns: int) -> float:
        if self.center_corridor_preferred and not self.commit_active(now_ns):
            return 0.0
        if self.depth_selected_gap_label in ('LEFT', 'RIGHT'):
            remembered_gap = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
            if abs(remembered_gap) >= 0.10:
                return remembered_gap
        close_target = self.close_side_bypass_target(now_ns)
        if abs(close_target) >= self.pre_avoid_min_offset_m:
            return clamp(close_target, -self.pre_avoid_max_offset_m, self.pre_avoid_max_offset_m)
        direction = self.preferred_bypass_direction(now_ns)
        if abs(direction) < 1e-3:
            side_hint = self.obstacle_side_hint()
            if abs(side_hint) >= 1e-3:
                direction = math.copysign(1.0, side_hint)
        if abs(direction) < 1e-3:
            return 0.0
        distance = max(self.emergency_avoid_trigger_m, min(self.pre_avoid_trigger_m, self.obstacle_forward_distance))
        ratio = clamp(
            (self.pre_avoid_trigger_m - distance)
            / max(self.pre_avoid_trigger_m - self.emergency_avoid_trigger_m, 1e-3),
            0.0,
            1.0,
        )
        target_mag = (
            self.pre_avoid_min_offset_m
            + ratio * (self.pre_avoid_max_offset_m - self.pre_avoid_min_offset_m)
        )
        return math.copysign(target_mag, direction)

    def update_obstacle_preemption_state(self, now_ns: int) -> None:
        self.false_emergency_demoted = False
        self.emergency_latch_rejected_due_to_low_persistence = False
        self.emergency_latch_rejected_due_to_center_corridor = False
        self.center_corridor_stabilizer_active = False
        self.lane_only_fallback_blocked = False
        self.critical_intrusion_persistence_cycles_used = self.critical_intrusion_persistence_cycles
        self.emergency_latch_kept_reason = 'none'
        previous_state = self.obstacle_latch_state
        stage, distance, trigger_source = self.compute_obstacle_stage(now_ns)
        strong_signal = stage != 'idle'
        hard_confirmation = self.hard_obstacle_confirmation(now_ns)
        odom_pass_hold = self.odom_pass_latch_active(now_ns)
        self.update_commit_stall_watchdog(now_ns)
        if self.commit_stale_detected:
            stage = 'idle'
            strong_signal = False
            hard_confirmation = False
            odom_pass_hold = False
            trigger_source = 'none'
        stale_duba_only = (
            trigger_source == 'dist'
            and not hard_confirmation
            and not self.centered_obstacle_bypass_active(now_ns)
        )
        release_clear = (
            distance >= self.obstacle_release_distance_m
            and not self.depth_obstacle
            and not self.depth_emergency
            and not self.critical_center_supported
            and not self.centered_obstacle_bypass_active(now_ns)
        )
        state_priority = {'idle': 0, 'pre_avoid': 1, 'avoid': 2, 'emergency': 3}
        latched_state = previous_state
        if strong_signal:
            if stale_duba_only and state_priority[stage] > state_priority['pre_avoid']:
                stage = 'pre_avoid'
            if stage == 'emergency':
                emergency_blocked, emergency_block_reason = self.should_block_emergency_latch(now_ns)
                if emergency_blocked:
                    stage = 'avoid' if distance <= self.near_avoid_trigger_m else 'pre_avoid'
                    trigger_source = emergency_block_reason
            if state_priority[stage] >= state_priority[previous_state]:
                latched_state = stage
            self.obstacle_latch_until_ns = max(
                self.obstacle_latch_until_ns,
                now_ns + int(self.obstacle_latch_hold_sec * 1e9),
            )
            self.obstacle_release_reason = 'holding_obstacle'
            if stale_duba_only:
                self.obstacle_latch_until_ns = min(
                    self.obstacle_latch_until_ns,
                    now_ns + int(self.stale_obstacle_release_sec * 1e9),
                )
                self.obstacle_release_reason = 'stale_duba_hold'
        elif stale_duba_only and previous_state != 'idle':
            latched_state = 'idle'
            self.obstacle_release_reason = 'stale_duba_release'
        elif now_ns < self.obstacle_latch_until_ns and not release_clear:
            self.obstacle_release_reason = 'hold_timer'
        elif now_ns < self.obstacle_latch_until_ns and previous_state != 'idle':
            latched_state = previous_state
            self.obstacle_release_reason = 'hold_timer'
        else:
            latched_state = 'idle'
            if previous_state != 'idle':
                self.obstacle_release_reason = 'clear_release' if release_clear else 'timer_elapsed'

        if odom_pass_hold and previous_state in ('avoid', 'emergency'):
            latched_state = 'avoid'
            self.obstacle_release_reason = 'odom_pass_latch'

        if (
            latched_state in ('avoid', 'emergency')
            and not hard_confirmation
            and self.depth_reason_code == 'clear_path'
            and not self.pointcloud_corridor_signal_active()
            and not odom_pass_hold
        ):
            latched_state = 'idle'
            self.obstacle_release_reason = 'clear_path_release'

        if self.commit_stale_detected:
            latched_state = 'idle'
            self.obstacle_release_reason = self.pass_commit_exit_reason

        if latched_state == 'emergency':
            false_emergency, false_reason = self.false_emergency_latch_active(now_ns, latched_state)
            if false_emergency:
                latched_state = 'avoid' if self.center_corridor_exists else 'pre_avoid'
                self.obstacle_release_reason = f'false_emergency_demoted:{false_reason}'
                self.false_emergency_demoted = True
                self.critical_override_blocked_by_center_corridor = (
                    self.center_corridor_exists and not self.blocked_center_now
                )
                self.emergency_latch_kept_reason = 'false_emergency_demoted'
            else:
                self.emergency_latch_kept_reason = false_reason
        else:
            self.false_emergency_detected_cycles = 0
            self.false_emergency_since_ns = 0

        self.obstacle_latch_state = latched_state
        self.pre_avoid_active = latched_state in ('pre_avoid', 'avoid', 'emergency')
        self.speed_scale_obstacle = self.compute_obstacle_speed_scale(distance)
        if latched_state == 'idle':
            self.speed_scale_obstacle = 1.0
        self.avoid_trigger_source = trigger_source if latched_state != 'idle' else 'none'
        self.obstacle_forward_distance = distance
        if previous_state != self.obstacle_latch_state:
            if self.obstacle_latch_state == 'idle':
                self.pass_commit_exit_reason = self.obstacle_release_reason
                self.authority_transition_reason = f'release:{self.obstacle_release_reason}'
            else:
                self.authority_transition_reason = f'{self.obstacle_latch_state}:{self.avoid_trigger_source}'

    def closest_obstacle_forward_distance(self) -> float:
        candidates = []
        if math.isfinite(self.obstacle_forward_distance) and self.obstacle_forward_distance < 90.0:
            candidates.append(self.obstacle_forward_distance)
        if self.duba_var and math.isfinite(self.duba_mesafe):
            candidates.append(self.duba_mesafe)
        if self.critical_roi_point_count >= self.critical_roi_min_points and math.isfinite(self.critical_roi_min_x):
            candidates.append(self.critical_roi_min_x)
        if math.isfinite(self.pointcloud_front_min_distance) and self.pointcloud_front_min_distance < 90.0:
            candidates.append(self.pointcloud_front_min_distance)
        return min(candidates) if candidates else 99.0

    def close_side_bypass_ratio(self, now_ns: int) -> float:
        if self.center_corridor_preferred and not self.commit_active(now_ns):
            return 0.0
        centered_trigger = self.centered_obstacle_bypass_active(now_ns)
        if abs(self.obstacle_side_hint()) < self.close_side_avoid_lateral_m and not centered_trigger:
            return 0.0
        if self.side_pass_hold_active(now_ns) and self.hard_obstacle_confirmation(now_ns):
            return 1.0
        closest_forward = self.closest_obstacle_forward_distance()
        return clamp(
            (self.close_side_avoid_distance_m - closest_forward)
            / max(
                self.close_side_avoid_distance_m - self.close_side_avoid_full_distance_m,
                1e-3,
            ),
            0.0,
            1.0,
        )

    def close_side_bypass_target(self, now_ns: int) -> float:
        direction = self.preferred_bypass_direction(now_ns)
        centered_trigger = self.centered_obstacle_bypass_active(now_ns)
        if abs(direction) < 1e-3 and not centered_trigger:
            return 0.0
        ratio = self.close_side_bypass_ratio(now_ns)
        if ratio <= 1e-3:
            return 0.0
        eased_ratio = math.sqrt(ratio)
        target_mag = max(
            self.close_side_avoid_min_offset_m
            + eased_ratio * (self.close_side_avoid_offset_m - self.close_side_avoid_min_offset_m),
            min(self.lane_corridor_cap, abs(self.corridor_target_offset)),
        )
        return math.copysign(target_mag, direction if abs(direction) >= 1e-3 else 1.0)

    def start_return_to_center(self, now_ns: int) -> None:
        self.return_to_center_until_ns = now_ns + int(self.return_to_center_sec * 1e9)

    def start_post_avoid_hold(self, now_ns: int) -> None:
        if not self.have_odom:
            self.start_return_to_center(now_ns)
            return
        self.post_avoid_hold_until_ns = now_ns + int(self.post_avoid_hold_sec * 1e9)
        self.post_avoid_start_path_m = self.odom_path_length_m
        self.post_avoid_travel_m = 0.0
        remembered_offset = self.corridor_target_offset
        if abs(remembered_offset) < 0.05:
            remembered_offset = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        self.post_avoid_target_offset = clamp(
            remembered_offset,
            -self.lane_corridor_cap,
            self.lane_corridor_cap,
        )
        self.return_to_center_until_ns = 0

    def clear_post_avoid_hold(self) -> None:
        self.post_avoid_hold_until_ns = 0
        self.post_avoid_travel_m = 0.0
        self.post_avoid_target_offset = 0.0

    def post_avoid_hold_active(self, now_ns: int) -> bool:
        if self.post_avoid_hold_until_ns <= 0:
            return False
        if not self.have_odom:
            return now_ns < self.post_avoid_hold_until_ns
        self.post_avoid_travel_m = max(0.0, self.odom_path_length_m - self.post_avoid_start_path_m)
        if (
            now_ns >= self.post_avoid_hold_until_ns
            or self.post_avoid_travel_m >= self.post_avoid_straight_distance_m
        ):
            self.clear_post_avoid_hold()
            if abs(self.corridor_target_offset) >= 0.05:
                self.start_return_to_center(now_ns)
            return False
        return True

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
            self.corridor_force_until_ns = 0
            self.corridor_enabled_state = False
            self.corridor_gating_reason = 'return_to_center_done'
            self.corridor_reset_reason = 'return_to_center_done'
            self.depth_selected_gap_label = 'CENTER'

    def in_lane_bypass_active(self, now_ns: int) -> bool:
        if self.close_side_bypass_ratio(now_ns) >= 0.05:
            return True
        if self.side_pass_hold_active(now_ns) and self.hard_obstacle_confirmation(now_ns) and abs(self.corridor_target_offset) >= 0.08:
            return True
        if not self.depth_frame_recent(now_ns) and not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if not self.obstacle_context_active(now_ns):
            return False
        if self.obstacle_latch_state in ('avoid', 'emergency'):
            return True
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

    def reset_critical_override_debug_state(self) -> None:
        self.false_critical_override_detected = False
        self.critical_override_blocked_by_center_corridor = False
        self.critical_trigger_consistent_with_tracked_geometry = True
        self.center_corridor_override_priority_applied = False
        self.critical_commit_rejected_reason = 'none'
        self.lane_term_preserved_in_critical = False
        self.corridor_term_preserved_in_critical = False
        self.side_commit_cancelled_due_to_valid_center_corridor = False
        self.false_emergency_demoted = False
        self.emergency_latch_rejected_due_to_low_persistence = False
        self.emergency_latch_rejected_due_to_center_corridor = False
        self.center_corridor_stabilizer_active = False
        self.lane_only_fallback_blocked = False
        self.critical_intrusion_persistence_cycles_used = self.critical_intrusion_persistence_cycles
        self.emergency_latch_kept_reason = 'none'

    def clear_critical_avoid_state(
        self,
        reason: str = 'none',
        clear_commit: bool = False,
    ) -> None:
        self.critical_avoid_until_ns = 0
        self.critical_escape_offset = 0.0
        self.critical_avoid_smoothed = 0.0
        self.critical_intrusion_persistence_cycles = 0
        self.false_critical_since_ns = 0
        if clear_commit and self.pass_commit_source == 'critical_escape':
            self.pass_latch_clear()
            self.clear_commit_session(reason)
            self.requested_pass_side = 'NONE'
            self.selected_pass_side = 'NONE'
            self.published_pass_side = 'NONE'
            self.pass_side_none_reason = reason
            self.depth_selected_gap_label = 'CENTER'
            self.depth_selected_gap_offset = 0.0
            self.reset_corridor_state(reason)
            self.side_commit_cancelled_due_to_valid_center_corridor = True

    def critical_center_corridor_progress_possible(self, now_ns: int) -> bool:
        if not self.center_corridor_exists or self.blocked_center_now:
            return False
        if not self.lane_control_available():
            return False
        selected_gap_centerish = self.depth_selected_gap_label in ('CENTER', 'CENTER_LEFT', 'CENTER_RIGHT')
        mapped_target = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        return (
            self.center_corridor_preferred
            or self.center_corridor_lane_keep_preferred()
            or selected_gap_centerish
            or abs(mapped_target) <= (self.no_commit_side_bias_cap + 0.04)
        )

    def valid_lane_bounded_center_corridor_available(self, now_ns: int) -> bool:
        if not self.critical_center_corridor_progress_possible(now_ns):
            return False
        if not self.lane_hard_constraints_active(now_ns):
            return False
        if not math.isfinite(self.depth_center_clearance):
            return False
        return self.depth_center_clearance >= (
            self.required_gap_clearance_m + self.critical_override_block_center_margin
        )

    def update_critical_intrusion_persistence(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns == self.critical_intrusion_persistence_last_update_ns:
            self.critical_intrusion_persistence_cycles_used = self.critical_intrusion_persistence_cycles
            return self.critical_intrusion_persistence_cycles >= self.critical_intrusion_persistence_min_cycles
        self.critical_intrusion_persistence_last_update_ns = now_ns
        strong_intrusion_floor = max(
            self.obstacle_preempt_intrusion_m + 0.08,
            0.72 * self.footprint_half_width_m,
        )
        strong_sample = (
            self.critical_center_supported
            and self.critical_roi_point_count >= self.critical_roi_min_points
            and math.isfinite(self.critical_roi_min_x)
            and self.critical_roi_min_x <= (self.near_avoid_trigger_m + 0.20)
            and self.critical_roi_intrusion_m >= strong_intrusion_floor
        )
        if self.center_corridor_preferred:
            preferred_floor = self.center_corridor_override_priority_weight * max(
                self.obstacle_preempt_intrusion_m + 0.04,
                0.14,
            )
            strong_sample = strong_sample and self.critical_roi_intrusion_m >= preferred_floor
        if strong_sample:
            self.critical_intrusion_persistence_cycles = min(
                self.critical_intrusion_persistence_cycles + 1,
                self.critical_intrusion_persistence_min_cycles + 6,
            )
        else:
            self.critical_intrusion_persistence_cycles = 0
        self.critical_intrusion_persistence_cycles_used = self.critical_intrusion_persistence_cycles
        return self.critical_intrusion_persistence_cycles >= self.critical_intrusion_persistence_min_cycles

    def should_block_emergency_latch(self, now_ns: int) -> Tuple[bool, str]:
        persistence_ready = self.update_critical_intrusion_persistence()
        if not persistence_ready and not self.depth_emergency:
            self.emergency_latch_rejected_due_to_low_persistence = True
            self.emergency_latch_kept_reason = 'low_persistence_rejected'
            return True, 'low_persistence'
        center_corridor_valid = (
            self.center_corridor_exists
            and self.center_corridor_preferred
            and not self.blocked_center_now
        )
        if center_corridor_valid and not self.depth_emergency:
            requested_side = 'RIGHT' if self.select_critical_escape_offset() > 0.0 else 'LEFT'
            geometry_consistent = self.critical_trigger_matches_tracked_geometry(now_ns, requested_side)
            self.critical_trigger_consistent_with_tracked_geometry = geometry_consistent
            if not (persistence_ready and geometry_consistent):
                self.critical_override_blocked_by_center_corridor = True
                self.emergency_latch_rejected_due_to_center_corridor = True
                self.emergency_latch_kept_reason = 'center_corridor_rejected'
                return True, 'center_corridor'
        return False, 'keep'

    def false_emergency_latch_active(self, now_ns: int, latch_state: str = '') -> Tuple[bool, str]:
        effective_state = latch_state if latch_state else self.obstacle_latch_state
        if effective_state != 'emergency':
            self.false_emergency_detected_cycles = 0
            self.false_emergency_since_ns = 0
            return False, 'not_emergency'
        low_persistence = (
            self.critical_intrusion_persistence_cycles
            < self.critical_intrusion_persistence_min_cycles
        )
        valid_center = self.center_corridor_exists and not self.blocked_center_now
        no_authoritative_side_commit = self.published_pass_side == 'NONE' and not self.commit_active(now_ns)
        if not valid_center:
            self.false_emergency_detected_cycles = 0
            self.false_emergency_since_ns = 0
            return False, 'center_invalid'
        if not (low_persistence or no_authoritative_side_commit):
            self.false_emergency_detected_cycles = 0
            self.false_emergency_since_ns = 0
            return False, 'emergency_persistent'
        self.false_emergency_detected_cycles = min(
            self.false_emergency_detected_cycles + 1,
            self.false_emergency_reset_cycles + 2,
        )
        if self.false_emergency_since_ns <= 0:
            self.false_emergency_since_ns = now_ns
        timeout_hit = (
            self.emergency_demote_timeout_ns > 0
            and (now_ns - self.false_emergency_since_ns) >= self.emergency_demote_timeout_ns
        )
        cycles_hit = self.false_emergency_detected_cycles >= self.false_emergency_reset_cycles
        reason = 'no_authoritative_side_commit' if no_authoritative_side_commit else 'low_persistence'
        return (timeout_hit or cycles_hit), reason

    def critical_trigger_matches_tracked_geometry(self, now_ns: int, requested_side: str) -> bool:
        tolerance = max(0.05, self.critical_geometry_consistency_tolerance)
        if self.depth_emergency:
            return True
        if not self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns):
            return False
        if self.critical_roi_point_count < self.critical_roi_min_points:
            return False
        if not math.isfinite(self.critical_roi_min_x) or self.critical_roi_min_x >= 90.0:
            return False
        if self.valid_lane_bounded_center_corridor_available(now_ns):
            return False
        tracked_valid = (
            self.tracked_obstacle_valid
            and math.isfinite(self.tracked_obstacle_local_x)
            and self.tracked_obstacle_local_x < 90.0
        )
        if not tracked_valid:
            if self.depth_selected_gap_label in ('LEFT', 'RIGHT') and self.depth_selected_gap_label == requested_side:
                return True
            return abs(self.filtered_obstacle_local_y) >= max(self.obstacle_local_y_deadband, tolerance)
        x_tol = max(0.30, 2.5 * tolerance)
        y_tol = max(0.18, 1.5 * tolerance)
        x_consistent = (
            abs(self.tracked_obstacle_local_x - self.critical_roi_min_x) <= x_tol
            or self.tracked_obstacle_local_x <= (self.critical_roi_forward_max_m + x_tol)
        )
        y_delta = abs(self.tracked_obstacle_local_y - self.critical_roi_mean_y)
        y_consistent = (
            y_delta <= max(0.30, 2.0 * y_tol)
            or (
                abs(self.tracked_obstacle_local_y) <= y_tol
                and abs(self.critical_roi_mean_y) <= y_tol
            )
        )
        if not y_consistent:
            y_consistent = (
                abs(self.tracked_obstacle_local_y) > y_tol
                and abs(self.critical_roi_mean_y) > y_tol
                and math.copysign(1.0, self.tracked_obstacle_local_y)
                == math.copysign(1.0, self.critical_roi_mean_y)
            )
        if not (x_consistent and y_consistent):
            return False
        if self.depth_selected_gap_label in ('LEFT', 'RIGHT'):
            return self.depth_selected_gap_label == requested_side
        return True

    def false_critical_override_active(self, now_ns: int) -> bool:
        if self.depth_emergency:
            self.false_critical_since_ns = 0
            return False
        if not self.center_corridor_exists or self.blocked_center_now:
            self.false_critical_since_ns = 0
            return False
        if not self.critical_center_corridor_progress_possible(now_ns):
            self.false_critical_since_ns = 0
            return False
        strong_and_consistent = (
            self.update_critical_intrusion_persistence()
            and self.critical_trigger_matches_tracked_geometry(
                now_ns,
                'RIGHT' if self.critical_escape_offset > 0.0 else 'LEFT',
            )
        )
        if strong_and_consistent:
            self.false_critical_since_ns = 0
            return False
        if self.false_critical_since_ns <= 0:
            self.false_critical_since_ns = now_ns
            return False
        return (now_ns - self.false_critical_since_ns) >= self.false_critical_demote_timeout_ns

    def critical_escape_commit_allowed(self, now_ns: int) -> bool:
        self.update_tracked_obstacle_memory(now_ns)
        self.center_corridor_override_priority_applied = (
            self.center_corridor_preferred
            or self.valid_lane_bounded_center_corridor_available(now_ns)
        )
        if self.depth_emergency:
            self.critical_trigger_consistent_with_tracked_geometry = True
            self.critical_commit_rejected_reason = 'none'
            return True
        if self.center_gap_recovery_preferred(now_ns):
            self.critical_commit_rejected_reason = 'center_gap_recovery'
            self.force_clear_authoritative_pass_commit_state(
                'critical_reject:center_gap_recovery',
                critical_reject=True,
                clear_critical_avoid=True,
            )
            return False
        if self.valid_lane_bounded_center_corridor_available(now_ns):
            self.critical_override_blocked_by_center_corridor = True
            self.critical_trigger_consistent_with_tracked_geometry = False
            self.critical_commit_rejected_reason = 'valid_center_corridor'
            self.force_clear_authoritative_pass_commit_state(
                'critical_reject:valid_center_corridor',
                critical_reject=True,
                clear_critical_avoid=True,
            )
            return False
        requested_side = 'RIGHT' if self.select_critical_escape_offset() > 0.0 else 'LEFT'
        persistent_intrusion = self.update_critical_intrusion_persistence()
        self.critical_trigger_consistent_with_tracked_geometry = (
            self.critical_trigger_matches_tracked_geometry(now_ns, requested_side)
        )
        center_collapsed = (
            not self.center_corridor_exists
            or self.blocked_center_now
            or not math.isfinite(self.depth_center_clearance)
            or self.depth_center_clearance < (
                self.required_gap_clearance_m + 0.02
            )
            or self.depth_selected_gap_label == 'BLOCKED'
        )
        if center_collapsed:
            self.critical_commit_rejected_reason = 'none'
            return True
        if not persistent_intrusion:
            self.critical_commit_rejected_reason = 'intrusion_not_persistent'
            self.force_clear_authoritative_pass_commit_state(
                'critical_reject:intrusion_not_persistent',
                critical_reject=True,
                clear_critical_avoid=True,
            )
            return False
        if not self.critical_trigger_consistent_with_tracked_geometry:
            self.critical_commit_rejected_reason = 'geometry_inconsistent'
            self.force_clear_authoritative_pass_commit_state(
                'critical_reject:geometry_inconsistent',
                critical_reject=True,
                clear_critical_avoid=True,
            )
            return False
        self.critical_commit_rejected_reason = 'none'
        return True

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
        self.reset_critical_override_debug_state()
        if self.startup_straight_corridor_guard_active(now_ns):
            self.clear_critical_avoid_state('startup_straight_corridor_guard')
            return False
        if self.pointcloud_center_lane_keep_preferred(now_ns) and not self.depth_emergency:
            self.clear_critical_avoid_state('center_corridor_lane_keep')
            return False
        if self.center_corridor_lane_keep_preferred() and not self.depth_emergency:
            self.clear_critical_avoid_state('center_corridor_lane_keep')
            return False
        if self.center_gap_recovery_preferred(now_ns) and not self.depth_emergency:
            self.clear_critical_avoid_state('center_gap_recovery')
            return False
        if not self.critical_center_supported and not self.depth_emergency:
            self.clear_critical_avoid_state('critical_center_not_supported')
            return False
        critical_now = (
            self.depth_emergency
            or self.critical_obstacle_blocking(self.critical_roi_forward_max_m)
        )
        if critical_now and not self.critical_escape_commit_allowed(now_ns):
            critical_now = False
        if critical_now:
            if abs(self.critical_escape_offset) < 0.05:
                self.critical_escape_offset = self.select_critical_escape_offset()
            self.critical_obstacle_last_seen_ns = now_ns
            self.critical_avoid_until_ns = now_ns + int(self.critical_commit_sec * 1e9)
            self.depth_context_last_ns = now_ns
            self.false_critical_since_ns = 0
            return True
        if (
            self.critical_avoid_until_ns > 0
            and not self.depth_emergency
            and not self.critical_escape_commit_allowed(now_ns)
        ):
            self.clear_critical_avoid_state(self.critical_commit_rejected_reason)
            return False
        still_blocking = (
            self.depth_emergency
            or self.critical_obstacle_blocking(
                self.critical_release_forward_m,
                self.critical_release_lateral_margin_m,
            )
        )
        if now_ns < self.critical_avoid_until_ns or still_blocking:
            if self.false_critical_override_active(now_ns):
                self.false_critical_override_detected = True
                self.critical_commit_rejected_reason = 'false_critical_valid_center_corridor'
                self.authority_transition_reason = 'critical_avoid->in_lane_avoid:false_critical'
                self.clear_critical_avoid_state(
                    'false_critical_valid_center_corridor',
                    clear_commit=True,
                )
                return False
            if abs(self.critical_escape_offset) < 0.05:
                self.critical_escape_offset = self.select_critical_escape_offset()
            return True
        self.clear_critical_avoid_state('critical_release')
        return False

    def compute_corridor_authority_term(self, target_offset: float, limit: float) -> float:
        self.corridor_error = target_offset
        self.corridor_term_preclamp = -self.corridor_follow_gain * self.corridor_error
        self.corridor_term_postclamp = clamp(self.corridor_term_preclamp, -limit, limit)
        return self.corridor_term_postclamp

    def activate_center_corridor_stabilizer(self, now_ns: int, reason: str) -> None:
        self.center_corridor_stabilizer_active = True
        self.corridor_enabled_state = True
        self.corridor_target_offset = 0.0
        self.smoothed_corridor_target *= 0.70
        if abs(self.smoothed_corridor_target) < 0.02:
            self.smoothed_corridor_target = 0.0
        self.depth_gap_offset = 0.0
        self.corridor_active_until_ns = max(
            self.corridor_active_until_ns,
            now_ns + int(max(self.emergency_demote_timeout_sec, self.min_corridor_hold_sec) * 1e9),
        )
        self.corridor_force_until_ns = max(self.corridor_force_until_ns, self.corridor_active_until_ns)
        self.corridor_gating_reason = reason
        self.corridor_reset_reason = 'center_corridor_stabilizer'

    def compute_center_corridor_stabilizer_term(self, limit: float, no_commit: bool = True) -> float:
        if not self.center_corridor_exists or self.blocked_center_now:
            return 0.0
        lateral_hint = self.filtered_obstacle_local_y
        if abs(lateral_hint) < 1e-3 and self.critical_roi_point_count > 0:
            lateral_hint = self.critical_roi_mean_y
        if abs(lateral_hint) < 1e-3 and self.tracked_obstacle_valid:
            lateral_hint = self.tracked_obstacle_local_y
        if abs(lateral_hint) < max(self.obstacle_local_y_deadband, 0.03):
            return 0.0
        weight = (
            self.no_commit_center_stabilizer_weight
            if no_commit
            else self.center_corridor_stabilizer_weight
        )
        return clamp(
            -weight * lateral_hint,
            -min(limit, self.lane_corridor_cap),
            min(limit, self.lane_corridor_cap),
        )

    def build_stop_command(self, reason: str, route_term_raw: float) -> ControlCommand:
        self.reset_corridor_state(reason)
        self.corner_mode = False
        self.last_gap_assist_active = False
        self.stop_reason = reason
        self.final_controller_mode = 'lane_center'
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
        self.final_controller_mode = 'lane_center'
        return ControlCommand(
            authority=authority,
            speed=speed,
            desired_angular=desired_angular,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            lane_conf=0.0,
            reason=self.lane_state.name.lower(),
        )

    def build_pre_avoid_command(
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
        target_offset = self.obstacle_preempt_target(now_ns)
        if self.selected_pass_side in ('LEFT', 'RIGHT'):
            target_offset = self.pass_side_target_offset(now_ns, self.selected_pass_side)
        target_offset = self.apply_precommit_side_target_policy(
            now_ns,
            target_offset,
            'pre_avoid_target',
        )
        corridor_limit = min(self.depth_gap_limit, self.pre_avoid_max_offset_m)
        corridor_term = self.compute_corridor_authority_term(target_offset, corridor_limit)
        if not self.commit_active(now_ns) and self.published_pass_side == 'NONE':
            corridor_term *= max(0.0, self.advisory_side_gap_max_weight)
        if self.center_corridor_preferred and self.depth_selected_gap_label == 'CENTER' and self.published_pass_side == 'NONE':
            self.activate_center_corridor_stabilizer(now_ns, 'pre_avoid_center_corridor')
            corridor_term += self.compute_center_corridor_stabilizer_term(corridor_limit, no_commit=True)
            self.lane_only_fallback_blocked = True
        avoid_term = self.pre_avoid_corridor_blend * corridor_term + 0.45 * gap_term
        desired = (
            self.pre_avoid_lane_weight * lane_term
            + avoid_term
            + route_term_used
        )
        desired = clamp(desired, -self.duba_max_angular, self.duba_max_angular)
        speed_cap = self.normal_lane_speed * self.speed_scale_obstacle
        if self.selected_pass_side in ('LEFT', 'RIGHT') and self.published_pass_side == 'NONE':
            speed_cap = min(speed_cap, self.normal_lane_speed * self.precommit_speed_scale)
        speed = min(speed, speed_cap)
        if self.route_enabled and route_speed > 0.0 and speed > 0.0:
            speed = min(speed, route_speed)
        if abs(target_offset) >= 0.05:
            self.force_corridor_hold(now_ns, target_offset, 'pre_avoid_hold')
        self.last_gap_assist_active = abs(corridor_term) > 1e-3 or abs(gap_term) > 1e-3
        self.final_controller_mode = (
            'center_corridor'
            if self.center_corridor_preferred
            else (
                'advisory_side_gap'
                if self.selected_pass_side in ('LEFT', 'RIGHT') and self.published_pass_side == 'NONE'
                else 'lane_center'
            )
        )
        return ControlCommand(
            authority=ControlAuthority.PRE_AVOID,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            gap_term=gap_term,
            corridor_term=corridor_term,
            avoid_term=avoid_term,
            lane_conf=lane_conf,
            reason=f'pre_avoid_{self.avoid_trigger_source}',
        )

    def build_post_avoid_hold_command(
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
        hold_target = self.post_avoid_target_offset
        if abs(hold_target) < 0.05:
            hold_target = self.corridor_target_offset
        hold_target = self.clip_target_to_lane_corridor(hold_target, now_ns, 'post_avoid_hold')
        corridor_term = self.compute_corridor_authority_term(
            hold_target,
            min(self.depth_gap_limit, self.lane_corridor_cap),
        )
        desired = (
            self.post_avoid_lane_weight * lane_term
            + self.post_avoid_corridor_weight * corridor_term
            + 0.18 * gap_term
            + 0.20 * route_term_used
        )
        desired = clamp(desired, -self.duba_max_angular, self.duba_max_angular)
        speed = min(speed, max(self.single_lane_invalid_speed, 0.36))
        speed = min(speed, self.normal_lane_speed * max(0.72, self.speed_scale_obstacle))
        if self.route_enabled and route_speed > 0.0:
            speed = min(speed, route_speed)
        self.corridor_target_offset = hold_target
        self.corridor_enabled_state = abs(hold_target) >= 0.05
        self.corridor_gating_reason = 'post_avoid_hold'
        self.corridor_reset_reason = 'hold'
        self.last_gap_assist_active = abs(corridor_term) > 1e-3 or abs(gap_term) > 1e-3
        self.final_controller_mode = 'committed_side_pass'
        return ControlCommand(
            authority=ControlAuthority.POST_AVOID_HOLD,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            gap_term=gap_term,
            corridor_term=corridor_term,
            lane_conf=lane_conf,
            reason='post_avoid_hold',
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
                self.center_corridor_exists
                and self.depth_selected_gap_label == 'CENTER'
                and self.published_pass_side == 'NONE'
            ):
                self.activate_center_corridor_stabilizer(now_ns, 'lane_follow_center_corridor')
                self.lane_only_fallback_blocked = True
            elif self.commit_active(now_ns) and self.selected_pass_side in ('LEFT', 'RIGHT'):
                hold_target = self.pass_side_target_offset(now_ns, self.selected_pass_side)
                self.force_corridor_hold(now_ns, hold_target, 'commit_forced_on')
            elif (
                self.obstacle_context_active(now_ns)
                and self.depth_frame_recent(now_ns)
                and self.depth_selected_gap_label not in ('CENTER', 'BLOCKED')
                and self.depth_selected_gap_clearance >= (self.required_gap_clearance_m + 0.02)
            ):
                self.force_corridor_hold(
                    now_ns,
                    self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset),
                    'lane_overlap_hold',
                )
            elif self.pre_avoid_active or self.obstacle_latch_state in ('avoid', 'emergency'):
                self.force_corridor_hold(now_ns, self.corridor_target_offset, 'preempt_hold')
            elif now_ns < self.corridor_force_until_ns and abs(self.corridor_target_offset) >= 0.05:
                self.force_corridor_hold(now_ns, self.corridor_target_offset, 'gating_hysteresis_hold')
            else:
                self.reset_corridor_state('lane_priority')
            self.last_gap_assist_active = False
        self.final_controller_mode = (
            'center_corridor'
            if self.center_corridor_preferred
            else 'lane_center'
        )
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
        if self.selected_pass_side in ('LEFT', 'RIGHT'):
            target_offset = self.pass_side_target_offset(now_ns, self.selected_pass_side)
        if abs(target_offset) < 0.05:
            target_offset = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        if abs(target_offset) < 0.05:
            target_offset = self.obstacle_preempt_target(now_ns)
        close_bypass_ratio = self.close_side_bypass_ratio(now_ns)
        strong_bypass_target = self.close_side_bypass_target(now_ns)
        preferred_gap_target = self.map_gap_offset_to_corridor_target(self.depth_selected_gap_offset)
        if (
            self.depth_selected_gap_label in ('LEFT', 'RIGHT')
            and abs(preferred_gap_target) >= 0.18
            and abs(target_offset) >= 0.10
            and preferred_gap_target * target_offset < 0.0
        ):
            target_offset = preferred_gap_target
            self.corridor_target_offset = preferred_gap_target
            self.corridor_active_until_ns = max(
                self.corridor_active_until_ns,
                now_ns + int(max(self.duba_pass_hold_sec, 0.35) * 1e9),
            )
            self.corridor_enabled_state = True
            self.corridor_gating_reason = 'side_flip_commit'
            self.corridor_reset_reason = 'hold'
        if abs(strong_bypass_target) > abs(target_offset):
            target_offset = strong_bypass_target
            self.corridor_target_offset = strong_bypass_target
            self.corridor_active_until_ns = max(
                self.corridor_active_until_ns,
                now_ns + int(max(self.duba_pass_hold_sec, 0.35) * 1e9),
            )
            self.corridor_enabled_state = True
            self.corridor_gating_reason = 'close_side_override'
            self.corridor_reset_reason = 'hold'
        target_offset = self.apply_precommit_side_target_policy(
            now_ns,
            target_offset,
            'in_lane_avoid_target',
        )
        bypass_push_ratio = math.sqrt(close_bypass_ratio) if close_bypass_ratio > 0.0 else 0.0
        limit = self.depth_gap_limit if self.depth_center_clearance < self.tight_gap_clearance_m else self.depth_gap_corner_limit
        if bypass_push_ratio > 0.0:
            limit = max(
                limit,
                self.depth_gap_corner_limit + bypass_push_ratio * (self.depth_gap_limit - self.depth_gap_corner_limit),
            )
        if self.lane_state == LaneState.DEGRADED_LANE:
            degraded_limit = self.avoid_corridor_limit_degraded
            if self.obstacle_latch_state == 'emergency':
                degraded_limit = min(self.depth_gap_limit, degraded_limit + 0.04)
            limit = min(limit, degraded_limit)
        corridor_term = self.compute_corridor_authority_term(target_offset, limit)
        if not self.commit_active(now_ns) and self.published_pass_side == 'NONE':
            corridor_term *= max(0.0, self.advisory_side_gap_max_weight)
        if self.center_corridor_exists and self.depth_selected_gap_label == 'CENTER' and self.published_pass_side == 'NONE':
            self.activate_center_corridor_stabilizer(now_ns, 'in_lane_center_corridor')
            corridor_term += self.compute_center_corridor_stabilizer_term(limit, no_commit=True)
            self.lane_only_fallback_blocked = True
        lane_weight = self.lane_weight_during_avoid
        if bypass_push_ratio > 0.0:
            lane_weight = min(
                lane_weight,
                (1.0 - bypass_push_ratio) * self.lane_weight_during_avoid
                    + bypass_push_ratio * self.close_side_avoid_lane_weight_min,
            )
        if self.center_corridor_preferred:
            lane_weight = max(lane_weight, self.center_corridor_priority_weight)
        avoid_term = corridor_term + gap_term
        desired = lane_weight * lane_term + avoid_term + route_term_used
        desired = clamp(desired, -self.duba_max_angular, self.duba_max_angular)
        if self.route_enabled and route_speed > 0.0 and speed > 0.0:
            speed = min(speed, route_speed)
        speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
        if self.selected_pass_side in ('LEFT', 'RIGHT') and self.published_pass_side == 'NONE':
            speed = min(speed, self.normal_lane_speed * self.precommit_speed_scale)
        if self.lane_state == LaneState.NORMAL_LANE:
            speed = min(speed, max(self.obstacle_speed, 0.56))
        else:
            speed = min(speed, max(self.single_lane_invalid_speed, 0.42))
        if bypass_push_ratio > 0.0:
            anticipatory_speed_cap = self.close_side_avoid_speed_mps + (1.0 - bypass_push_ratio) * 0.16
            speed = min(speed, anticipatory_speed_cap)
        if self.side_pass_hold_active(now_ns):
            speed = min(speed, max(0.30, self.close_side_avoid_speed_mps))
            self.corridor_active_until_ns = max(
                self.corridor_active_until_ns,
                self.duba_pass_hold_until_ns,
            )
            self.force_corridor_hold(now_ns, target_offset, 'side_pass_hold')
        if self.depth_center_clearance < self.depth_stop_m + 0.12:
            speed = min(speed, 0.24)
        if abs(target_offset) >= 0.05:
            self.force_corridor_hold(now_ns, target_offset, 'commit_corridor_hold')
        self.last_gap_assist_active = abs(corridor_term) > 1e-3 or abs(gap_term) > 1e-3
        self.final_controller_mode = (
            'committed_side_pass'
            if self.commit_active(now_ns) or self.published_pass_side in ('LEFT', 'RIGHT')
            else (
                'center_corridor'
                if self.center_corridor_preferred
                else 'advisory_side_gap'
            )
        )
        return ControlCommand(
            authority=ControlAuthority.IN_LANE_AVOID,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            gap_term=gap_term,
            corridor_term=corridor_term,
            avoid_term=avoid_term,
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
        self.corridor_target_offset = self.clip_target_to_lane_corridor(
            self.corridor_target_offset,
            now_ns,
            'corridor_gap_fallback',
        )
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
        speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
        self.last_gap_assist_active = abs(corridor_term) > 1e-3
        self.final_controller_mode = 'advisory_side_gap'
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
        escape_offset = self.clip_target_to_lane_corridor(
            self.critical_escape_offset,
            now_ns,
            'critical_escape_target',
        )
        self.critical_escape_offset = escape_offset
        closest_forward = self.critical_roi_min_x if self.critical_roi_min_x < 90.0 else self.critical_roi_forward_max_m
        if self.depth_frame_recent(now_ns):
            closest_forward = min(closest_forward, self.depth_center_clearance)
        if self.depth_center_clearance <= self.depth_stop_m and not self.side_bypass_available(now_ns):
            return self.build_stop_command('critical_stop', route_term_raw)
        critical_side = self.active_single_side(now_ns)
        _lane_desired, _lane_speed, lane_term_raw, gap_term, lane_conf, route_term_used, _route_weight = (
            self.compute_lane_state_command(
                now_ns,
                route_term_raw,
                critical_side,
                allow_gap_assist=True,
            )
        )
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
        critical_turn_limit = self.critical_avoid_target_limit
        if self.lane_control_available() and self.side_bypass_available(now_ns):
            critical_turn_limit = min(critical_turn_limit, self.depth_gap_limit, 0.24)
        target = clamp(target, -critical_turn_limit, critical_turn_limit)
        if abs(target) < self.critical_avoid_min_turn:
            target = math.copysign(
                self.critical_avoid_min_turn,
                target if abs(target) > 1e-3 else -escape_offset,
            )
        self.critical_avoid_smoothed = (
            self.critical_avoid_ramp_alpha * target
            + (1.0 - self.critical_avoid_ramp_alpha) * self.critical_avoid_smoothed
        )
        corridor_limit = (
            self.depth_gap_limit
            if self.depth_center_clearance < self.tight_gap_clearance_m
            else self.depth_gap_corner_limit
        )
        raw_corridor_term = self.compute_corridor_authority_term(escape_offset, corridor_limit)
        lane_support_weight = self.critical_lane_term_min_weight
        if self.center_corridor_preferred or self.valid_lane_bounded_center_corridor_available(now_ns):
            lane_support_weight = clamp(
                lane_support_weight * self.center_corridor_override_priority_weight,
                self.critical_lane_term_min_weight,
                1.0,
            )
            self.center_corridor_override_priority_applied = True
        lane_term = lane_support_weight * lane_term_raw
        corridor_term = self.critical_corridor_term_min_weight * raw_corridor_term
        self.lane_term_preserved_in_critical = abs(lane_term) > 1e-3
        self.corridor_term_preserved_in_critical = abs(corridor_term) > 1e-3
        critical_avoid_term = self.critical_avoid_smoothed
        desired = clamp(
            critical_avoid_term + lane_term + corridor_term + 0.12 * route_term_used,
            -self.duba_max_angular,
            self.duba_max_angular,
        )
        if closest_forward < 0.45:
            speed = 0.18
        elif closest_forward < 0.70:
            speed = 0.28
        else:
            speed = min(self.obstacle_speed, 0.42)
        speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
        if self.depth_selected_gap_clearance < self.tight_gap_clearance_m:
            speed = min(speed, 0.20)
        self.corridor_target_offset = escape_offset
        self.corridor_active_until_ns = now_ns + int(self.critical_commit_sec * 1e9)
        self.force_corridor_hold(now_ns, escape_offset, 'critical_avoid')
        self.smoothed_corridor_target = self.corridor_target_offset
        self.depth_gap_offset = self.corridor_target_offset
        requested_side = 'RIGHT' if self.corridor_target_offset > 0.0 else 'LEFT'
        if self.side_lock_active and self.locked_pass_side in ('LEFT', 'RIGHT') and requested_side != self.locked_pass_side:
            self.side_flip_blocked = True
            self.side_switch_reject_reason = (
                f'critical_escape_abort:{requested_side.lower()}->{self.locked_pass_side.lower()}'
            )
            self.force_clear_authoritative_pass_commit_state(
                'hard_safety_abort',
                critical_reject=True,
                clear_critical_avoid=False,
            )
        self.start_commit_session(now_ns, requested_side, 'critical_escape', 'critical_escape')
        self.hold_locked_pass_side(now_ns, 'critical_escape_hold')
        self.pass_commit_until_ns = max(
            self.pass_commit_until_ns,
            now_ns + int(self.critical_commit_sec * 1e9),
        )
        self.corridor_error = escape_offset
        self.last_gap_assist_active = True
        self.final_controller_mode = 'committed_side_pass'
        return ControlCommand(
            authority=ControlAuthority.CRITICAL_AVOID,
            speed=speed,
            desired_angular=desired,
            lane_term=lane_term,
            route_term_raw=route_term_raw,
            route_term_used=route_term_used,
            gap_term=gap_term,
            corridor_term=corridor_term,
            avoid_term=critical_avoid_term,
            lane_conf=lane_conf,
            reason='critical_roi',
        )

    def select_control_command(self, now_ns: int, route_term_raw: float, route_speed: float, side: str) -> ControlCommand:
        critical_active = self.update_critical_avoid_state(now_ns)
        if critical_active:
            return self.build_critical_avoid_command(now_ns, route_term_raw)
        if self.prev_obstacle_active:
            self.obstacle_recovery_until_ns = now_ns + int(self.obstacle_recovery_sec * 1e9)
        if not self.in_lane_bypass_active(now_ns) and not self.post_avoid_hold_active(now_ns):
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
        commit_hold_active = self.commit_active(now_ns) and self.selected_pass_side in ('LEFT', 'RIGHT')
        if self.lane_control_available() and (self.in_lane_bypass_active(now_ns) or lane_overlap_hold or commit_hold_active):
            return self.build_in_lane_avoid_command(now_ns, route_term_raw, route_speed, side)
        if self.lane_control_available() and self.pre_avoid_active:
            return self.build_pre_avoid_command(now_ns, route_term_raw, route_speed, side)
        if self.lane_control_available() and self.post_avoid_hold_active(now_ns):
            return self.build_post_avoid_hold_command(now_ns, route_term_raw, route_speed, side)
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
            return min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
        speed *= max(0.60, 1.0 - self.speed_angular_gain * abs(desired_angular))
        if authority == ControlAuthority.CRITICAL_AVOID:
            speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
            return clamp(speed, 0.0, self.obstacle_speed)
        if authority == ControlAuthority.PRE_AVOID:
            speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
            return clamp(speed, 0.0, self.gps_hiz)
        if authority == ControlAuthority.POST_AVOID_HOLD:
            speed = min(speed, self.normal_lane_speed * max(0.72, self.speed_scale_obstacle))
            return clamp(speed, 0.0, self.gps_hiz)
        if self.lane_state == LaneState.DEGRADED_LANE:
            speed *= max(0.78, lane_conf + 0.35)
            if self.corner_mode:
                speed = max(speed, 0.34)
        else:
            speed *= max(0.82, lane_conf)
        if self.obstacle_context_active(self.get_clock().now().nanoseconds) and self.lane_state == LaneState.NORMAL_LANE:
            speed = min(speed, 0.70)
        speed = min(speed, self.normal_lane_speed * self.speed_scale_obstacle)
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
        elif authority == ControlAuthority.PRE_AVOID:
            rate_limit = self.single_rate_limit
            smoothing = self.obstacle_smoothing
        elif authority == ControlAuthority.POST_AVOID_HOLD:
            rate_limit = self.single_rate_limit
            smoothing = self.single_smoothing
        elif authority == ControlAuthority.CORRIDOR_GAP:
            rate_limit = self.single_rate_limit
            smoothing = self.single_smoothing
        elif self.lane_state == LaneState.DEGRADED_LANE or self.corner_mode:
            rate_limit = self.single_rate_limit
            smoothing = self.single_smoothing
        else:
            rate_limit = self.normal_rate_limit
            smoothing = self.normal_smoothing
        if desired * self.last_cmd_angular < 0.0 and authority in (ControlAuthority.CRITICAL_AVOID, ControlAuthority.PRE_AVOID, ControlAuthority.CORRIDOR_GAP):
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
        speed_scale = self.speed_scale_obstacle
        no_authoritative_side = self.published_pass_side == 'NONE' and not self.commit_active(now_ns)
        pointcloud_recent = self.signal_recent(self.pointcloud_last_ns, self.obstacle_context_sec, now_ns)
        depth_recent = self.depth_frame_recent(now_ns)
        obstacle_active = (
            command.authority in (
                ControlAuthority.CRITICAL_AVOID,
                ControlAuthority.IN_LANE_AVOID,
                ControlAuthority.PRE_AVOID,
                ControlAuthority.CORRIDOR_GAP,
            )
            or self.in_lane_bypass_active(now_ns)
            or self.pre_avoid_active
        )
        emergency_stop = (
            command.authority == ControlAuthority.BLOCKED_STOP
            or self.blocked_persistent
        )
        if self.depth_emergency and not self.corridor_gap_available(now_ns):
            emergency_stop = True

        if command.authority == ControlAuthority.CRITICAL_AVOID:
            critical_bias = command.avoid_term
            support_bias = 0.35 * command.corridor_term + 0.25 * command.lane_term
            if (
                self.center_corridor_preferred
                or self.false_critical_override_detected
                or self.critical_override_blocked_by_center_corridor
                or not self.critical_trigger_consistent_with_tracked_geometry
            ):
                self.center_corridor_override_priority_applied = True
                critical_bias = 0.60 * critical_bias + support_bias
            else:
                critical_bias = critical_bias + 0.20 * support_bias
            bias = clamp(critical_bias, -self.max_angular_z, self.max_angular_z)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_emergency,
                self.pre_avoid_speed_scale_near,
            )
        elif command.authority == ControlAuthority.PRE_AVOID:
            preempt_target = self.obstacle_preempt_target(now_ns)
            if self.selected_pass_side in ('LEFT', 'RIGHT'):
                preempt_target = self.pass_side_target_offset(now_ns, self.selected_pass_side)
            preempt_target = self.apply_precommit_side_target_policy(
                now_ns,
                preempt_target,
                'summary_pre_avoid_target',
            )
            bias = clamp(
                -self.avoid_bias_gain * self.avoid_bias_lane_attenuation * preempt_target,
                -0.18,
                0.18,
            )
            if abs(command.corridor_term) > 1e-3:
                bias += 0.35 * command.corridor_term
            if abs(command.gap_term) > 1e-3:
                bias += 0.18 * command.gap_term
            bias = clamp(bias, -self.avoid_bias_limit, self.avoid_bias_limit)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_near,
                0.96,
            )
        elif command.authority == ControlAuthority.IN_LANE_AVOID:
            bypass_bias = -self.avoid_bias_gain * self.corridor_target_offset
            if abs(command.corridor_term) > 1e-3:
                bypass_bias += 0.55 * command.corridor_term
            if abs(command.gap_term) > 1e-3:
                bypass_bias += 0.45 * command.gap_term
            if self.lane_control_available():
                bypass_bias *= self.avoid_bias_lane_attenuation
            bias = clamp(bypass_bias, -self.avoid_bias_limit, self.avoid_bias_limit)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_near,
                0.86,
            )
        elif command.authority == ControlAuthority.CORRIDOR_GAP:
            bias = clamp(command.desired_angular, -self.depth_gap_limit, self.depth_gap_limit)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_near,
                0.90,
            )
        elif command.authority == ControlAuthority.POST_AVOID_HOLD:
            hold_bias = -0.60 * self.post_avoid_target_offset
            if abs(command.corridor_term) > 1e-3:
                hold_bias += 0.45 * command.corridor_term
            bias = clamp(hold_bias, -0.14, 0.14)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                0.74,
                0.98,
            )
        elif self.pre_avoid_active:
            preempt_target = self.obstacle_preempt_target(now_ns)
            if self.selected_pass_side in ('LEFT', 'RIGHT'):
                preempt_target = self.pass_side_target_offset(now_ns, self.selected_pass_side)
            bias = clamp(-0.85 * self.avoid_bias_gain * preempt_target, -0.18, 0.18)
            speed_scale = clamp(
                min(speed_scale, command.speed / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_near,
                0.98,
            )
        elif self.return_to_center_active(now_ns):
            decay_ratio = clamp(
                (self.return_to_center_until_ns - now_ns) / max(self.return_to_center_sec * 1e9, 1.0),
                0.0,
                1.0,
            )
            bias = clamp(-0.40 * self.depth_gap_offset * decay_ratio, -0.12, 0.12)
            speed_scale = clamp(
                min(speed_scale, min(max(command.speed, self.obstacle_speed), 0.96) / max(self.normal_lane_speed, 1e-3)),
                0.72,
                1.00,
            )
        elif self.obstacle_context_active(now_ns) and not self.lane_control_available():
            bias = clamp(-0.55 * self.depth_gap_offset, -0.22, 0.22)
            speed_scale = clamp(
                min(speed_scale, min(max(command.speed, self.obstacle_speed), 0.88) / max(self.normal_lane_speed, 1e-3)),
                self.pre_avoid_speed_scale_near,
                0.96,
            )
        else:
            speed_scale = clamp(speed_scale, self.pre_avoid_speed_scale_emergency, 1.0)

        if no_authoritative_side:
            if self.center_corridor_preferred:
                bias = 0.0
                self.side_gap_suppressed_due_to_no_commit = True
                if self.side_target_suppressed_reason == 'none':
                    self.side_target_suppressed_reason = 'center_corridor_preferred'
            elif self.selected_pass_side in ('LEFT', 'RIGHT'):
                bias = clamp(bias, -self.no_commit_side_bias_cap, self.no_commit_side_bias_cap)
            if self.selected_pass_side in ('LEFT', 'RIGHT') or abs(bias) > 1e-3:
                speed_scale = min(speed_scale, self.precommit_speed_scale)

        if emergency_stop:
            speed_scale = 0.0
            obstacle_active = True

        self.obstacle_unknown = (
            not self.pre_avoid_active
            and
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
            self.pre_avoid_active
            or now_ns < self.critical_avoid_until_ns
            or self.post_avoid_hold_active(now_ns)
            or self.return_to_center_active(now_ns)
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
            self.pending_no_lane_frames = 0
            if self.lane_state in (LaneState.NO_LANE_COAST, LaneState.NO_LANE_SLOW, LaneState.BLOCKED_STOP):
                if self.recover_start_ns <= 0:
                    self.recover_start_ns = now_ns
                recover_age = (now_ns - self.recover_start_ns) / 1e9
                if recover_age < self.recover_debounce_sec:
                    return
            if desired_state == LaneState.DEGRADED_LANE and self.lane_state == LaneState.NORMAL_LANE:
                self.pending_single_lane_frames += 1
                if self.pending_single_lane_frames < self.single_lane_transition_frames:
                    return
            self.recover_start_ns = 0
            self.lane_lost_ns = 0
            self.pending_single_lane_frames = 0
            self.lane_state = desired_state
            return

        self.recover_start_ns = 0
        if self.lane_state == LaneState.NORMAL_LANE:
            self.pending_single_lane_frames += 1
            if self.pending_single_lane_frames < self.single_lane_transition_frames:
                return
            self.pending_single_lane_frames = 0
            self.lane_state = LaneState.DEGRADED_LANE
        if self.lane_state == LaneState.DEGRADED_LANE:
            self.pending_no_lane_frames += 1
            if self.pending_no_lane_frames < self.no_lane_transition_frames:
                return
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
        self.zombie_commit_state_detected = False
        self.atomic_commit_state_clear_applied = False
        self.critical_reject_forced_state_clear = False
        self.pass_state_validity_ok = True
        self.update_lane_state(now_ns)
        self.update_obstacle_preemption_state(now_ns)
        self.update_pass_authority(now_ns)

        left_recent, right_recent = self.active_boundaries(now_ns)
        side = self.active_single_side(now_ns)
        route_term_raw, route_speed = self.compute_route_term()
        twist = Twist()
        was_obstacle_active = self.summary_obstacle_active
        command = self.select_control_command(now_ns, route_term_raw, route_speed, side)
        if (
            command.authority in (ControlAuthority.IN_LANE_AVOID, ControlAuthority.CRITICAL_AVOID)
            and self.last_command_authority not in (ControlAuthority.IN_LANE_AVOID, ControlAuthority.CRITICAL_AVOID)
        ):
            self.start_pass_latch(now_ns)
        self.obstacle_preempted_by_lane = bool(
            self.pre_avoid_active and command.authority == ControlAuthority.LANE_FOLLOW
        )
        if command.authority != self.last_command_authority:
            if self.pre_avoid_active:
                self.authority_transition_reason = (
                    f'{self.last_command_authority.name.lower()}->{command.authority.name.lower()}:{self.avoid_trigger_source}'
                )
            else:
                self.authority_transition_reason = (
                    f'{self.last_command_authority.name.lower()}->{command.authority.name.lower()}:{command.reason}'
                )
            self.last_command_authority = command.authority
        angular = self.publish_control_command(command, twist)
        if twist.linear.x <= 1e-3:
            if command.authority == ControlAuthority.BLOCKED_STOP:
                self.stop_reason = command.reason
            elif self.blocked_center_now and self.blocked_selected_side_now:
                self.stop_reason = 'blocked_center_and_side'
            else:
                self.stop_reason = f'zero_speed_{command.authority.name.lower()}'
        else:
            self.stop_reason = 'none'
        self.publish_obstacle_summary(now_ns, command)
        obstacle_active = self.summary_obstacle_active
        if was_obstacle_active and not obstacle_active:
            self.start_post_avoid_hold(now_ns)
            self.last_cmd_angular *= 0.70
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
                f'L={left_recent} R={right_recent} blocked={self.blocked_persistent} '
                f'blocked_center={self.blocked_center_now} blocked_side={self.blocked_selected_side_now} '
                f'authoritative_pass_owner={self.authoritative_pass_owner} '
                f'requested_pass_side={self.requested_pass_side} published_pass_side={self.published_pass_side} '
                f'pass_side={self.selected_pass_side} commit_active={self.commit_active(now_ns)} '
                f'commit_session_id={self.commit_session_id} side_lock_active={self.side_lock_active} '
                f'locked_pass_side={self.locked_pass_side} side_flip_blocked={self.side_flip_blocked} '
                f'filtered_obstacle_local_y={self.filtered_obstacle_local_y:+.3f} '
                f'obstacle_local_y_deadband_active={self.obstacle_local_y_deadband_active} '
                f'startup_straight_corridor_guard_active={self.startup_straight_corridor_guard_active_state} '
                f'startup_straight_corridor_guard_reason={self.startup_straight_corridor_guard_reason} '
                f'commit_age={max(0.0, (now_ns - self.pass_commit_started_ns) / 1e9) if self.pass_commit_started_ns > 0 else 0.0:.2f} '
                f'progress_delta={self.commit_watchdog_progress_delta:.3f} '
                f'tracked_local_x_delta={self.commit_watchdog_tracked_local_x_delta:.3f} '
                f'odom_delta_since_commit={max(0.0, self.odom_path_length_m - self.pass_commit_start_path_m) if self.have_odom and self.pass_commit_started_ns > 0 else 0.0:.3f} '
                f'stale_commit_detected={self.commit_stale_detected} '
                f'stale_obstacle_memory_detected={self.stale_obstacle_memory_detected} '
                f'commit_source={self.pass_commit_source} '
                f'commit_remaining={self.commit_remaining_sec(now_ns):.2f} '
                f'commit_remaining_distance={self.commit_remaining_distance(now_ns):.2f} '
                f'progress={self.compute_pass_progress(now_ns):.2f} '
                f'pass_enter_reason={self.commit_session_start_reason} '
                f'pass_exit_reason={self.pass_commit_exit_reason} '
                f'tracked_memory_expire_reason={self.tracked_memory_expire_reason} '
                f'authoritative_selected_gap={self.depth_selected_gap_label} '
                f'lane_hard_constraint_active={self.lane_hard_constraint_active} '
                f'center_corridor_exists={self.center_corridor_exists} '
                f'center_corridor_preferred={self.center_corridor_preferred} '
                f'center_preferred_reason={self.center_preferred_reason} '
                f'false_critical_override_detected={self.false_critical_override_detected} '
                f'critical_override_blocked_by_center_corridor={self.critical_override_blocked_by_center_corridor} '
                f'critical_trigger_consistent_with_tracked_geometry={self.critical_trigger_consistent_with_tracked_geometry} '
                f'center_corridor_override_priority_applied={self.center_corridor_override_priority_applied} '
                f'critical_commit_rejected_reason={self.critical_commit_rejected_reason} '
                f'zombie_commit_state_detected={self.zombie_commit_state_detected} '
                f'atomic_commit_state_clear_applied={self.atomic_commit_state_clear_applied} '
                f'critical_reject_forced_state_clear={self.critical_reject_forced_state_clear} '
                f'pass_state_validity_ok={self.pass_state_validity_ok} '
                f'false_emergency_demoted={self.false_emergency_demoted} '
                f'emergency_latch_rejected_due_to_low_persistence={self.emergency_latch_rejected_due_to_low_persistence} '
                f'emergency_latch_rejected_due_to_center_corridor={self.emergency_latch_rejected_due_to_center_corridor} '
                f'center_corridor_stabilizer_active={self.center_corridor_stabilizer_active} '
                f'lane_only_fallback_blocked={self.lane_only_fallback_blocked} '
                f'critical_intrusion_persistence_cycles_used={self.critical_intrusion_persistence_cycles_used} '
                f'emergency_latch_kept_reason={self.emergency_latch_kept_reason} '
                f'side_commit_cancelled_due_to_valid_center_corridor={self.side_commit_cancelled_due_to_valid_center_corridor} '
                f'center_reject_reason={self.center_reject_reason} '
                f'center_reject_strength={self.center_reject_strength:.2f} '
                f'center_reject_persistence={self.center_reject_persistence} '
                f'stop_reason={self.stop_reason}'
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
                f'pre_avoid_active={self.pre_avoid_active} preempted_by_lane={self.obstacle_preempted_by_lane} '
                f'trigger_source={self.avoid_trigger_source} speed_scale_obstacle={self.speed_scale_obstacle:.2f} '
                f'center_gap_penalty={self.center_gap_penalty:.2f} transition_reason={self.authority_transition_reason} '
                f'obstacle_latch_state={self.obstacle_latch_state} obstacle_release_reason={self.obstacle_release_reason} '
                f'tracked_local_x={self.tracked_obstacle_local_x:.2f} tracked_local_y={self.tracked_obstacle_local_y:+.2f} '
                f'pass_latch_active={self.pass_latch_active} pass_latch_source={self.pass_latch_source} '
                f'pass_latch_travel={self.pass_latch_travel_m:.2f} '
                f'post_avoid_hold={self.post_avoid_hold_active(now_ns)} post_avoid_travel={self.post_avoid_travel_m:.2f} '
                f'duba_dist={self.duba_mesafe:.2f} critical_dist={self.critical_roi_min_x:.2f} '
                f'critical_points={self.critical_roi_point_count} footprint_intrusion={self.critical_roi_intrusion_m:.2f} '
                f'critical_intrusion_persistence_cycles={self.critical_intrusion_persistence_cycles} '
                f'avoid_latched={self.summary_avoid_latched} blocked={self.blocked_persistent} '
                f'blocked_center={self.blocked_center_now} blocked_side={self.blocked_selected_side_now} '
                f'authoritative_pass_owner={self.authoritative_pass_owner} requested_pass_side={self.requested_pass_side} '
                f'published_pass_side={self.published_pass_side} consumed_pass_side={self.selected_pass_side} '
                f'pass_side_none_reason={self.pass_side_none_reason} fallback_side_triggered={self.fallback_side_triggered} '
                f'forced_side_selection={self.fallback_side_triggered} center_reject_reason={self.center_reject_reason} '
                f'lane_hard_constraint_active={self.lane_hard_constraint_active} '
                f'center_corridor_exists={self.center_corridor_exists} '
                f'center_corridor_preferred={self.center_corridor_preferred} '
                f'center_preferred_reason={self.center_preferred_reason} '
                f'false_critical_override_detected={self.false_critical_override_detected} '
                f'critical_override_blocked_by_center_corridor={self.critical_override_blocked_by_center_corridor} '
                f'critical_trigger_consistent_with_tracked_geometry={self.critical_trigger_consistent_with_tracked_geometry} '
                f'center_corridor_override_priority_applied={self.center_corridor_override_priority_applied} '
                f'critical_commit_rejected_reason={self.critical_commit_rejected_reason} '
                f'zombie_commit_state_detected={self.zombie_commit_state_detected} '
                f'atomic_commit_state_clear_applied={self.atomic_commit_state_clear_applied} '
                f'critical_reject_forced_state_clear={self.critical_reject_forced_state_clear} '
                f'pass_state_validity_ok={self.pass_state_validity_ok} '
                f'false_emergency_demoted={self.false_emergency_demoted} '
                f'emergency_latch_rejected_due_to_low_persistence={self.emergency_latch_rejected_due_to_low_persistence} '
                f'emergency_latch_rejected_due_to_center_corridor={self.emergency_latch_rejected_due_to_center_corridor} '
                f'center_corridor_stabilizer_active={self.center_corridor_stabilizer_active} '
                f'lane_only_fallback_blocked={self.lane_only_fallback_blocked} '
                f'critical_intrusion_persistence_cycles_used={self.critical_intrusion_persistence_cycles_used} '
                f'emergency_latch_kept_reason={self.emergency_latch_kept_reason} '
                f'lane_term_preserved_in_critical={self.lane_term_preserved_in_critical} '
                f'corridor_term_preserved_in_critical={self.corridor_term_preserved_in_critical} '
                f'side_commit_cancelled_due_to_valid_center_corridor={self.side_commit_cancelled_due_to_valid_center_corridor} '
                f'center_reject_strength={self.center_reject_strength:.2f} '
                f'center_reject_persistence={self.center_reject_persistence} '
                f'advisory_side_gap_strength={self.advisory_side_gap_strength:.2f} '
                f'side_gap_suppressed_due_to_no_commit={self.side_gap_suppressed_due_to_no_commit} '
                f'side_target_suppressed_reason={self.side_target_suppressed_reason} '
                f'target_clipped_to_lane_bounds={self.target_clipped_to_lane_bounds} '
                f'target_clip_reason={self.target_clip_reason} '
                f'final_controller_mode={self.final_controller_mode} '
                f'commit_session_id={self.commit_session_id} side_lock_active={self.side_lock_active} '
                f'locked_pass_side={self.locked_pass_side} side_flip_blocked={self.side_flip_blocked} '
                f'side_switch_reject_reason={self.side_switch_reject_reason} '
                f'filtered_obstacle_local_y={self.filtered_obstacle_local_y:+.3f} '
                f'obstacle_local_y_deadband_active={self.obstacle_local_y_deadband_active} '
                f'startup_straight_corridor_guard_active={self.startup_straight_corridor_guard_active_state} '
                f'startup_straight_corridor_guard_reason={self.startup_straight_corridor_guard_reason} '
                f'commit_age={max(0.0, (now_ns - self.pass_commit_started_ns) / 1e9) if self.pass_commit_started_ns > 0 else 0.0:.2f} '
                f'progress_delta={self.commit_watchdog_progress_delta:.3f} '
                f'tracked_local_x_delta={self.commit_watchdog_tracked_local_x_delta:.3f} '
                f'odom_delta_since_commit={max(0.0, self.odom_path_length_m - self.pass_commit_start_path_m) if self.have_odom and self.pass_commit_started_ns > 0 else 0.0:.3f} '
                f'stale_commit_detected={self.commit_stale_detected} '
                f'stale_obstacle_memory_detected={self.stale_obstacle_memory_detected} '
                f'commit_active={self.commit_active(now_ns)} commit_source={self.pass_commit_source} '
                f'commit_remaining={self.commit_remaining_sec(now_ns):.2f} '
                f'commit_remaining_distance={self.commit_remaining_distance(now_ns):.2f} '
                f'progress={self.compute_pass_progress(now_ns):.2f} commit_exit_reason={self.pass_commit_exit_reason} '
                f'reason_new_commit_session={self.commit_session_start_reason} '
                f'tracked_memory_expire_reason={self.tracked_memory_expire_reason} '
                f'stop_reason={self.stop_reason}'
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
