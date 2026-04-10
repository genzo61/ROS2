#!/usr/bin/env python3

import json
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class CmdFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('cmd_fusion_node')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('lane_error_topic', '/lane/error')
        self.declare_parameter('lane_heading_error_topic', '/lane/heading_error')
        self.declare_parameter('lane_confidence_topic', '/lane/confidence')
        self.declare_parameter('lane_available_topic', '/lane/available')
        self.declare_parameter('lane_valid_topic', '/lane/valid')
        self.declare_parameter('lane_predicted_topic', '/lane/predicted')
        self.declare_parameter('lane_left_topic', '/lane/left')
        self.declare_parameter('lane_right_topic', '/lane/right')
        self.declare_parameter('obstacle_bias_topic', '/obstacle/bias')
        self.declare_parameter('obstacle_speed_scale_topic', '/obstacle/speed_scale')
        self.declare_parameter('emergency_stop_topic', '/obstacle/emergency_stop')
        self.declare_parameter('obstacle_active_topic', '/obstacle/active')
        self.declare_parameter('obstacle_unknown_topic', '/obstacle/unknown')
        self.declare_parameter('obstacle_pass_state_topic', '/obstacle/pass_state')
        self.declare_parameter('obstacle_selected_gap_topic', '/obstacle/debug/selected_gap')
        self.declare_parameter('obstacle_selected_pass_side_topic', '/obstacle/debug/selected_pass_side')
        self.declare_parameter('obstacle_commit_active_topic', '/obstacle/debug/commit_active')
        self.declare_parameter('obstacle_commit_remaining_topic', '/obstacle/debug/commit_remaining_sec')
        self.declare_parameter('obstacle_commit_source_topic', '/obstacle/debug/commit_source')
        self.declare_parameter('obstacle_pass_owner_topic', '/obstacle/debug/authoritative_pass_owner')
        self.declare_parameter('obstacle_progress_topic', '/obstacle/debug/progress')
        self.declare_parameter('obstacle_blocked_center_topic', '/obstacle/debug/blocked_center')
        self.declare_parameter('obstacle_blocked_selected_side_topic', '/obstacle/debug/blocked_selected_side')
        self.declare_parameter('waypoint_heading_hint_topic', '/guidance/heading_hint')
        self.declare_parameter('waypoint_progress_topic', '/guidance/progress')

        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('base_speed', 0.22)
        self.declare_parameter('degraded_speed', 0.14)
        self.declare_parameter('lane_lost_speed', 0.08)
        self.declare_parameter('min_conf_speed_scale', 0.35)
        self.declare_parameter('lane_weight_normal', 1.00)
        self.declare_parameter('lane_weight_pre_avoid', 0.72)
        self.declare_parameter('lane_weight_commit', 0.48)
        self.declare_parameter('corridor_weight_pre_avoid', 0.42)
        self.declare_parameter('corridor_weight_commit', 0.62)
        self.declare_parameter('lane_kp', 0.95)
        self.declare_parameter('lane_heading_kp', 0.60)
        self.declare_parameter('obstacle_weight', 0.35)
        self.declare_parameter('obstacle_weight_during_avoid', 1.10)
        self.declare_parameter('lane_weight_during_avoid', 0.55)
        self.declare_parameter('obstacle_weight_during_commit', 1.35)
        self.declare_parameter('lane_weight_during_commit', 0.30)
        self.declare_parameter('avoid_lane_weight_scale', 0.40)
        self.declare_parameter('single_line_avoid_lane_weight_scale', 0.30)
        self.declare_parameter('avoid_obstacle_weight_scale', 1.08)
        self.declare_parameter('single_line_avoid_obstacle_weight_scale', 1.12)
        self.declare_parameter('waypoint_weight_with_lane', 0.0)
        self.declare_parameter('waypoint_weight_no_lane', 0.0)
        self.declare_parameter('max_angular_z', 0.55)
        self.declare_parameter('recovery_max_angular_z', 0.35)
        self.declare_parameter('avoid_max_angular_z', 0.72)
        self.declare_parameter('no_lane_crawl_speed', 0.06)
        self.declare_parameter('steering_smoothing_alpha', 0.35)
        self.declare_parameter('degraded_steering_smoothing_alpha', 0.18)
        self.declare_parameter('avoid_steering_smoothing_alpha', 0.30)
        self.declare_parameter('nominal_confidence_threshold', 0.60)
        self.declare_parameter('recovery_confidence_threshold', 0.30)
        self.declare_parameter('low_conf_threshold', 0.42)
        self.declare_parameter('offlane_error_threshold', 0.16)
        self.declare_parameter('offlane_recovery_gain', 0.35)
        self.declare_parameter('recent_lane_hold_sec', 1.0)
        self.declare_parameter('no_lane_memory_sec', 2.2)
        self.declare_parameter('low_conf_degraded_speed', 0.08)
        self.declare_parameter('unknown_obstacle_speed', 0.06)
        self.declare_parameter('max_steer_low_conf', 0.24)
        self.declare_parameter('single_line_conf_threshold', 0.55)
        self.declare_parameter('single_line_low_conf_speed', 0.10)
        self.declare_parameter('max_steer_single_line', 0.20)
        self.declare_parameter('no_lane_memory_speed', 0.08)
        self.declare_parameter('no_lane_memory_steer_gain', 0.65)
        self.declare_parameter('curve_speed_reduction_max', 0.0)
        self.declare_parameter('curve_heading_full_scale', 0.30)
        self.declare_parameter('curve_error_full_scale', 0.25)
        self.declare_parameter('invert_lane_error', False)
        self.declare_parameter('invert_lane_heading_error', False)
        self.declare_parameter('obstacle_lane_guard_error_threshold', 0.08)
        self.declare_parameter('obstacle_lane_guard_full_error', 0.22)
        self.declare_parameter('obstacle_lane_guard_min_scale', 0.20)
        self.declare_parameter('obstacle_lane_priority_gain', 0.35)
        self.declare_parameter('obstacle_lane_opposite_weight_drop', 0.45)
        self.declare_parameter('obstacle_bias_speed_reduction_gain', 0.22)
        self.declare_parameter('obstacle_lane_guard_speed_reduction_gain', 0.30)
        self.declare_parameter('obstacle_speed_min_scale', 0.45)
        self.declare_parameter('obstacle_speed_full_bias_abs', 0.25)
        self.declare_parameter('obstacle_preempt_speed_scale_threshold', 0.88)
        self.declare_parameter('obstacle_preempt_bias_abs', 0.10)
        self.declare_parameter('obstacle_preempt_lane_weight_min', 0.18)
        self.declare_parameter('obstacle_preempt_obstacle_weight_gain', 0.32)
        self.declare_parameter('obstacle_preempt_opposite_drop_relief', 0.80)
        self.declare_parameter('lane_timeout_sec', 0.6)
        self.declare_parameter('lane_boundary_timeout_sec', 0.9)
        self.declare_parameter('obstacle_timeout_sec', 0.6)
        self.declare_parameter('waypoint_timeout_sec', 1.5)
        self.declare_parameter('single_lane_transition_frames', 3)
        self.declare_parameter('no_lane_transition_frames', 6)
        self.declare_parameter('avoidance_commit_duration_sec', 0.9)
        self.declare_parameter('minimum_single_lane_forward_speed', 0.18)
        self.declare_parameter('minimum_obstacle_pass_forward_speed', 0.24)
        self.declare_parameter('single_lane_memory_blend', 0.35)
        self.declare_parameter('commit_max_angular', 0.38)
        self.declare_parameter('steer_slew_rate_limit', 1.6)
        self.declare_parameter('commit_speed_scale', 0.68)
        self.declare_parameter('no_lane_commit_scale', 0.35)
        self.declare_parameter('stale_commit_max_steer', 0.18)
        self.declare_parameter('stale_commit_steer_decay_rate', 4.0)
        self.declare_parameter('lane_reacquisition_min_confidence', 0.55)
        self.declare_parameter('lane_reacquisition_persistence_cycles', 8)
        self.declare_parameter('commit_bias_min_abs', 0.05)
        self.declare_parameter('commit_obstacle_bias_floor', 0.08)
        self.declare_parameter('commit_lane_guard_scale', 0.15)
        self.declare_parameter('advisory_gap_lane_weight_min', 0.58)
        self.declare_parameter('advisory_gap_corridor_scale', 0.42)
        self.declare_parameter('advisory_gap_bias_cap', 0.18)
        self.declare_parameter('advisory_gap_center_bias_cap', 0.12)
        self.declare_parameter('advisory_gap_corridor_term_cap', 0.18)
        self.declare_parameter('advisory_gap_center_corridor_term_cap', 0.12)
        self.declare_parameter('advisory_side_gap_max_weight', 0.22)
        self.declare_parameter('precommit_speed_scale', 0.58)
        self.declare_parameter('no_commit_side_bias_cap', 0.10)
        self.declare_parameter('center_corridor_priority_weight', 1.15)
        self.declare_parameter('center_corridor_override_priority_weight', 1.35)
        self.declare_parameter('critical_intrusion_persistence_min_cycles', 3)
        self.declare_parameter('center_corridor_stabilizer_weight', 0.28)
        self.declare_parameter('no_commit_center_stabilizer_weight', 0.22)
        self.declare_parameter('critical_lane_term_min_weight', 0.35)
        self.declare_parameter('critical_corridor_term_min_weight', 0.22)
        self.declare_parameter('lane_hard_constraint_margin', 0.05)

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        lane_error_topic = str(self.get_parameter('lane_error_topic').value)
        lane_heading_error_topic = str(self.get_parameter('lane_heading_error_topic').value)
        lane_confidence_topic = str(self.get_parameter('lane_confidence_topic').value)
        lane_available_topic = str(self.get_parameter('lane_available_topic').value)
        lane_valid_topic = str(self.get_parameter('lane_valid_topic').value)
        lane_predicted_topic = str(self.get_parameter('lane_predicted_topic').value)
        lane_left_topic = str(self.get_parameter('lane_left_topic').value)
        lane_right_topic = str(self.get_parameter('lane_right_topic').value)
        obstacle_bias_topic = str(self.get_parameter('obstacle_bias_topic').value)
        obstacle_speed_scale_topic = str(self.get_parameter('obstacle_speed_scale_topic').value)
        emergency_stop_topic = str(self.get_parameter('emergency_stop_topic').value)
        obstacle_active_topic = str(self.get_parameter('obstacle_active_topic').value)
        obstacle_unknown_topic = str(self.get_parameter('obstacle_unknown_topic').value)
        obstacle_pass_state_topic = str(self.get_parameter('obstacle_pass_state_topic').value)
        obstacle_selected_gap_topic = str(self.get_parameter('obstacle_selected_gap_topic').value)
        obstacle_selected_pass_side_topic = str(self.get_parameter('obstacle_selected_pass_side_topic').value)
        obstacle_commit_active_topic = str(self.get_parameter('obstacle_commit_active_topic').value)
        obstacle_commit_remaining_topic = str(self.get_parameter('obstacle_commit_remaining_topic').value)
        obstacle_commit_source_topic = str(self.get_parameter('obstacle_commit_source_topic').value)
        obstacle_pass_owner_topic = str(self.get_parameter('obstacle_pass_owner_topic').value)
        obstacle_progress_topic = str(self.get_parameter('obstacle_progress_topic').value)
        obstacle_blocked_center_topic = str(self.get_parameter('obstacle_blocked_center_topic').value)
        obstacle_blocked_selected_side_topic = str(
            self.get_parameter('obstacle_blocked_selected_side_topic').value
        )
        waypoint_heading_hint_topic = str(self.get_parameter('waypoint_heading_hint_topic').value)
        waypoint_progress_topic = str(self.get_parameter('waypoint_progress_topic').value)

        self.base_speed = float(self.get_parameter('base_speed').value)
        self.degraded_speed = float(self.get_parameter('degraded_speed').value)
        self.lane_lost_speed = float(self.get_parameter('lane_lost_speed').value)
        self.min_conf_speed_scale = float(self.get_parameter('min_conf_speed_scale').value)
        self.lane_weight_normal = max(0.0, float(self.get_parameter('lane_weight_normal').value))
        self.lane_weight_pre_avoid = max(0.0, float(self.get_parameter('lane_weight_pre_avoid').value))
        self.lane_weight_commit = max(0.0, float(self.get_parameter('lane_weight_commit').value))
        self.corridor_weight_pre_avoid = max(0.0, float(self.get_parameter('corridor_weight_pre_avoid').value))
        self.corridor_weight_commit = max(0.0, float(self.get_parameter('corridor_weight_commit').value))
        self.lane_kp = float(self.get_parameter('lane_kp').value)
        self.lane_heading_kp = float(self.get_parameter('lane_heading_kp').value)
        self.obstacle_weight = float(self.get_parameter('obstacle_weight').value)
        self.obstacle_weight_during_avoid = float(self.get_parameter('obstacle_weight_during_avoid').value)
        self.lane_weight_during_avoid = float(self.get_parameter('lane_weight_during_avoid').value)
        self.obstacle_weight_during_commit = float(self.get_parameter('obstacle_weight_during_commit').value)
        self.lane_weight_during_commit = float(self.get_parameter('lane_weight_during_commit').value)
        self.avoid_lane_weight_scale = clamp(
            float(self.get_parameter('avoid_lane_weight_scale').value),
            0.05,
            1.0,
        )
        self.single_line_avoid_lane_weight_scale = clamp(
            float(self.get_parameter('single_line_avoid_lane_weight_scale').value),
            0.05,
            1.0,
        )
        self.avoid_obstacle_weight_scale = max(
            0.5,
            float(self.get_parameter('avoid_obstacle_weight_scale').value),
        )
        self.single_line_avoid_obstacle_weight_scale = max(
            0.5,
            float(self.get_parameter('single_line_avoid_obstacle_weight_scale').value),
        )
        self.waypoint_weight_with_lane = float(self.get_parameter('waypoint_weight_with_lane').value)
        self.waypoint_weight_no_lane = float(self.get_parameter('waypoint_weight_no_lane').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.recovery_max_angular_z = float(self.get_parameter('recovery_max_angular_z').value)
        self.avoid_max_angular_z = float(self.get_parameter('avoid_max_angular_z').value)
        self.no_lane_crawl_speed = float(self.get_parameter('no_lane_crawl_speed').value)
        self.steering_smoothing_alpha = float(self.get_parameter('steering_smoothing_alpha').value)
        self.degraded_steering_smoothing_alpha = float(
            self.get_parameter('degraded_steering_smoothing_alpha').value
        )
        self.avoid_steering_smoothing_alpha = float(
            self.get_parameter('avoid_steering_smoothing_alpha').value
        )
        self.nominal_confidence_threshold = float(self.get_parameter('nominal_confidence_threshold').value)
        self.recovery_confidence_threshold = float(self.get_parameter('recovery_confidence_threshold').value)
        self.low_conf_threshold = float(self.get_parameter('low_conf_threshold').value)
        self.offlane_error_threshold = float(self.get_parameter('offlane_error_threshold').value)
        self.offlane_recovery_gain = float(self.get_parameter('offlane_recovery_gain').value)
        self.recent_lane_hold_ns = int(float(self.get_parameter('recent_lane_hold_sec').value) * 1e9)
        self.no_lane_memory_ns = int(float(self.get_parameter('no_lane_memory_sec').value) * 1e9)
        self.low_conf_degraded_speed = float(self.get_parameter('low_conf_degraded_speed').value)
        self.unknown_obstacle_speed = float(self.get_parameter('unknown_obstacle_speed').value)
        self.max_steer_low_conf = float(self.get_parameter('max_steer_low_conf').value)
        self.single_line_conf_threshold = float(self.get_parameter('single_line_conf_threshold').value)
        self.single_line_low_conf_speed = float(self.get_parameter('single_line_low_conf_speed').value)
        self.max_steer_single_line = float(self.get_parameter('max_steer_single_line').value)
        self.no_lane_memory_speed = float(self.get_parameter('no_lane_memory_speed').value)
        self.no_lane_memory_steer_gain = float(self.get_parameter('no_lane_memory_steer_gain').value)
        self.curve_speed_reduction_max = float(self.get_parameter('curve_speed_reduction_max').value)
        self.curve_heading_full_scale = max(
            1e-3,
            float(self.get_parameter('curve_heading_full_scale').value),
        )
        self.curve_error_full_scale = max(
            1e-3,
            float(self.get_parameter('curve_error_full_scale').value),
        )
        self.invert_lane_error = bool(self.get_parameter('invert_lane_error').value)
        self.invert_lane_heading_error = bool(self.get_parameter('invert_lane_heading_error').value)
        self.obstacle_lane_guard_error_threshold = max(
            0.0,
            float(self.get_parameter('obstacle_lane_guard_error_threshold').value),
        )
        self.obstacle_lane_guard_full_error = max(
            self.obstacle_lane_guard_error_threshold + 1e-3,
            float(self.get_parameter('obstacle_lane_guard_full_error').value),
        )
        self.obstacle_lane_guard_min_scale = clamp(
            float(self.get_parameter('obstacle_lane_guard_min_scale').value),
            0.0,
            1.0,
        )
        self.obstacle_lane_priority_gain = max(
            0.0,
            float(self.get_parameter('obstacle_lane_priority_gain').value),
        )
        self.obstacle_lane_opposite_weight_drop = clamp(
            float(self.get_parameter('obstacle_lane_opposite_weight_drop').value),
            0.0,
            1.0,
        )
        self.obstacle_bias_speed_reduction_gain = clamp(
            float(self.get_parameter('obstacle_bias_speed_reduction_gain').value),
            0.0,
            1.0,
        )
        self.obstacle_lane_guard_speed_reduction_gain = clamp(
            float(self.get_parameter('obstacle_lane_guard_speed_reduction_gain').value),
            0.0,
            1.0,
        )
        self.obstacle_speed_min_scale = clamp(
            float(self.get_parameter('obstacle_speed_min_scale').value),
            0.05,
            1.0,
        )
        self.obstacle_speed_full_bias_abs = max(
            1e-3,
            float(self.get_parameter('obstacle_speed_full_bias_abs').value),
        )
        self.obstacle_preempt_speed_scale_threshold = clamp(
            float(self.get_parameter('obstacle_preempt_speed_scale_threshold').value),
            self.obstacle_speed_min_scale,
            1.0,
        )
        self.obstacle_preempt_bias_abs = clamp(
            float(self.get_parameter('obstacle_preempt_bias_abs').value),
            0.0,
            self.obstacle_speed_full_bias_abs,
        )
        self.obstacle_preempt_lane_weight_min = clamp(
            float(self.get_parameter('obstacle_preempt_lane_weight_min').value),
            0.0,
            self.lane_weight_during_avoid,
        )
        self.obstacle_preempt_obstacle_weight_gain = max(
            0.0,
            float(self.get_parameter('obstacle_preempt_obstacle_weight_gain').value),
        )
        self.obstacle_preempt_opposite_drop_relief = clamp(
            float(self.get_parameter('obstacle_preempt_opposite_drop_relief').value),
            0.0,
            1.0,
        )
        self.lane_timeout_ns = int(float(self.get_parameter('lane_timeout_sec').value) * 1e9)
        self.lane_boundary_timeout_ns = int(
            float(self.get_parameter('lane_boundary_timeout_sec').value) * 1e9
        )
        self.obstacle_timeout_ns = int(float(self.get_parameter('obstacle_timeout_sec').value) * 1e9)
        self.waypoint_timeout_ns = int(float(self.get_parameter('waypoint_timeout_sec').value) * 1e9)
        self.single_lane_transition_frames = max(
            1,
            int(self.get_parameter('single_lane_transition_frames').value),
        )
        self.no_lane_transition_frames = max(
            1,
            int(self.get_parameter('no_lane_transition_frames').value),
        )
        self.avoidance_commit_duration_ns = int(
            max(0.0, float(self.get_parameter('avoidance_commit_duration_sec').value)) * 1e9
        )
        self.minimum_single_lane_forward_speed = max(
            0.0,
            float(self.get_parameter('minimum_single_lane_forward_speed').value),
        )
        self.minimum_obstacle_pass_forward_speed = max(
            self.minimum_single_lane_forward_speed,
            float(self.get_parameter('minimum_obstacle_pass_forward_speed').value),
        )
        self.single_lane_memory_blend = clamp(
            float(self.get_parameter('single_lane_memory_blend').value),
            0.0,
            0.95,
        )
        self.commit_max_angular = clamp(
            float(self.get_parameter('commit_max_angular').value),
            0.12,
            self.avoid_max_angular_z,
        )
        self.steer_slew_rate_limit = max(
            0.1,
            float(self.get_parameter('steer_slew_rate_limit').value),
        )
        self.commit_speed_scale = clamp(
            float(self.get_parameter('commit_speed_scale').value),
            0.20,
            1.0,
        )
        self.no_lane_commit_scale = clamp(
            float(self.get_parameter('no_lane_commit_scale').value),
            0.05,
            1.0,
        )
        self.stale_commit_max_steer = clamp(
            float(self.get_parameter('stale_commit_max_steer').value),
            0.05,
            self.commit_max_angular,
        )
        self.stale_commit_steer_decay_rate = max(
            0.1,
            float(self.get_parameter('stale_commit_steer_decay_rate').value),
        )
        self.lane_reacquisition_min_confidence = clamp(
            float(self.get_parameter('lane_reacquisition_min_confidence').value),
            0.05,
            1.0,
        )
        self.lane_reacquisition_persistence_cycles = max(
            1,
            int(self.get_parameter('lane_reacquisition_persistence_cycles').value),
        )
        self.commit_bias_min_abs = clamp(
            float(self.get_parameter('commit_bias_min_abs').value),
            0.0,
            1.0,
        )
        self.commit_obstacle_bias_floor = clamp(
            float(self.get_parameter('commit_obstacle_bias_floor').value),
            0.0,
            0.4,
        )
        self.commit_lane_guard_scale = clamp(
            float(self.get_parameter('commit_lane_guard_scale').value),
            0.0,
            1.0,
        )
        self.advisory_gap_lane_weight_min = clamp(
            float(self.get_parameter('advisory_gap_lane_weight_min').value),
            0.0,
            2.0,
        )
        self.advisory_gap_corridor_scale = clamp(
            float(self.get_parameter('advisory_gap_corridor_scale').value),
            0.05,
            1.0,
        )
        self.advisory_gap_bias_cap = clamp(
            float(self.get_parameter('advisory_gap_bias_cap').value),
            0.05,
            0.5,
        )
        self.advisory_gap_center_bias_cap = clamp(
            float(self.get_parameter('advisory_gap_center_bias_cap').value),
            0.03,
            self.advisory_gap_bias_cap,
        )
        self.advisory_gap_corridor_term_cap = clamp(
            float(self.get_parameter('advisory_gap_corridor_term_cap').value),
            0.05,
            self.avoid_max_angular_z,
        )
        self.advisory_gap_center_corridor_term_cap = clamp(
            float(self.get_parameter('advisory_gap_center_corridor_term_cap').value),
            0.03,
            self.advisory_gap_corridor_term_cap,
        )
        self.advisory_side_gap_max_weight = clamp(
            float(self.get_parameter('advisory_side_gap_max_weight').value),
            0.05,
            0.50,
        )
        self.precommit_speed_scale = clamp(
            float(self.get_parameter('precommit_speed_scale').value),
            0.10,
            1.0,
        )
        self.no_commit_side_bias_cap = clamp(
            float(self.get_parameter('no_commit_side_bias_cap').value),
            0.02,
            0.40,
        )
        self.center_corridor_priority_weight = max(
            1.0,
            float(self.get_parameter('center_corridor_priority_weight').value),
        )
        self.center_corridor_override_priority_weight = max(
            self.center_corridor_priority_weight,
            float(self.get_parameter('center_corridor_override_priority_weight').value),
        )
        self.critical_intrusion_persistence_min_cycles = max(
            1,
            int(self.get_parameter('critical_intrusion_persistence_min_cycles').value),
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
        self.lane_hard_constraint_margin = clamp(
            float(self.get_parameter('lane_hard_constraint_margin').value),
            0.0,
            0.20,
        )

        self.lane_error = 0.0
        self.lane_heading_error = 0.0
        self.lane_confidence = 0.0
        self.lane_available = False
        self.lane_valid = False
        self.lane_predicted = False
        self.left_lane_visible = False
        self.right_lane_visible = False
        self.obstacle_bias = 0.0
        self.obstacle_speed_scale = 1.0
        self.emergency_stop = False
        self.obstacle_active = False
        self.obstacle_unknown = False
        self.obstacle_selected_gap = 'CENTER'
        self.obstacle_selected_pass_side = 'NONE'
        self.obstacle_commit_active = False
        self.obstacle_commit_remaining_sec = 0.0
        self.obstacle_commit_remaining_distance_m = 0.0
        self.obstacle_commit_source = 'none'
        self.authoritative_pass_owner = 'unknown'
        self.obstacle_progress = 0.0
        self.authoritative_pre_avoid_active = False
        self.authoritative_obstacle_active = False
        self.authoritative_obstacle_latch_state = 'idle'
        self.authoritative_corridor_target = 0.0
        self.authoritative_corridor_enabled = False
        self.authoritative_corridor_reason = 'none'
        self.authoritative_exit_reason = 'none'
        self.center_reject_reason = 'none'
        self.authoritative_side_lock_active = False
        self.authoritative_locked_pass_side = 'NONE'
        self.obstacle_commit_session_id = 0
        self.commit_session_start_reason = 'none'
        self.lane_hard_constraint_active = False
        self.center_corridor_exists = False
        self.center_corridor_preferred = False
        self.center_preferred_reason = 'none'
        self.center_reject_strength = 0.0
        self.center_reject_persistence = 0
        self.advisory_side_gap_strength = 0.0
        self.side_gap_suppressed_due_to_no_commit = False
        self.side_target_suppressed_reason = 'none'
        self.target_clipped_to_lane_bounds = False
        self.target_clip_reason = 'none'
        self.final_controller_mode = 'lane_center'
        self.lane_corridor_min_offset = 0.0
        self.lane_corridor_max_offset = 0.0
        self.filtered_obstacle_local_y = 0.0
        self.tracked_local_x = 99.0
        self.tracked_local_y = 0.0
        self.critical_dist = 99.0
        self.critical_points = 0
        self.footprint_intrusion = 0.0
        self.obstacle_local_y_deadband_active = False
        self.side_flip_blocked = False
        self.side_switch_reject_reason = 'none'
        self.false_critical_override_detected = False
        self.critical_override_blocked_by_center_corridor = False
        self.critical_trigger_consistent_with_tracked_geometry = True
        self.center_corridor_override_priority_applied = False
        self.critical_commit_rejected_reason = 'none'
        self.false_emergency_demoted = False
        self.emergency_latch_rejected_due_to_low_persistence = False
        self.emergency_latch_rejected_due_to_center_corridor = False
        self.center_corridor_stabilizer_active = False
        self.lane_only_fallback_blocked = False
        self.critical_intrusion_persistence_cycles_used = 0
        self.emergency_latch_kept_reason = 'none'
        self.lane_term_preserved_in_critical = False
        self.corridor_term_preserved_in_critical = False
        self.side_commit_cancelled_due_to_valid_center_corridor = False
        self.zombie_commit_state_detected = False
        self.atomic_commit_state_clear_applied = False
        self.critical_reject_forced_state_clear = False
        self.pass_state_validity_ok = True
        self.stale_commit_active = False
        self.stale_commit_detected = False
        self.stale_obstacle_memory_detected = False
        self.commit_age = 0.0
        self.progress_delta = 0.0
        self.tracked_local_x_delta = 0.0
        self.odom_delta_since_commit = 0.0
        self.ignored_transient_none_due_to_commit_lock = False
        self.current_side_source_reason = 'none'
        self.pass_state_valid = False
        self.local_pass_inference_disabled = True
        self.lane_reacquisition_guard_active = True
        self.lane_reacquisition_valid_cycles = 0
        self.blocked_center = False
        self.blocked_selected_side = False
        self.heading_hint = 0.0
        self.waypoint_progress = 0.0

        self.lane_error_stamp_ns = 0
        self.lane_heading_stamp_ns = 0
        self.lane_conf_stamp_ns = 0
        self.lane_available_stamp_ns = 0
        self.lane_valid_stamp_ns = 0
        self.lane_predicted_stamp_ns = 0
        self.left_lane_stamp_ns = 0
        self.right_lane_stamp_ns = 0
        self.obstacle_bias_stamp_ns = 0
        self.obstacle_scale_stamp_ns = 0
        self.emergency_stamp_ns = 0
        self.obstacle_active_stamp_ns = 0
        self.obstacle_unknown_stamp_ns = 0
        self.obstacle_selected_gap_stamp_ns = 0
        self.obstacle_pass_state_stamp_ns = 0
        self.obstacle_selected_pass_side_stamp_ns = 0
        self.obstacle_commit_active_stamp_ns = 0
        self.obstacle_commit_remaining_stamp_ns = 0
        self.obstacle_commit_source_stamp_ns = 0
        self.authoritative_pass_owner_stamp_ns = 0
        self.obstacle_progress_stamp_ns = 0
        self.blocked_center_stamp_ns = 0
        self.blocked_selected_side_stamp_ns = 0
        self.heading_hint_stamp_ns = 0
        self.waypoint_progress_stamp_ns = 0
        self.last_log_ns = 0
        self.last_reliable_lane_error = 0.0
        self.last_reliable_lane_heading_error = 0.0
        self.last_reliable_lane_stamp_ns = 0
        self.last_steering_command = 0.0
        self.lane_state = 'NO_LANE'
        self.pending_single_lane_frames = 0
        self.pending_no_lane_frames = 0
        self.avoidance_commit_until_ns = 0
        self.committed_pass_side = 'NONE'
        self.committed_commit_session_id = 0
        self.last_stop_reason = 'startup'

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_lane_state_pub = self.create_publisher(String, '/fusion/debug/lane_state', 10)
        self.debug_pass_side_pub = self.create_publisher(String, '/fusion/debug/pass_side', 10)
        self.debug_commit_active_pub = self.create_publisher(Bool, '/fusion/debug/avoid_commit_active', 10)
        self.debug_commit_remaining_pub = self.create_publisher(Float32, '/fusion/debug/avoid_commit_remaining_sec', 10)
        self.debug_stop_reason_pub = self.create_publisher(String, '/fusion/debug/stop_reason', 10)
        self.debug_cmd_speed_pub = self.create_publisher(Float32, '/fusion/debug/final_linear_x', 10)
        self.debug_cmd_steer_pub = self.create_publisher(Float32, '/fusion/debug/final_angular_z', 10)
        self.create_subscription(Float32, lane_error_topic, self.lane_error_cb, 10)
        self.create_subscription(Float32, lane_heading_error_topic, self.lane_heading_error_cb, 10)
        self.create_subscription(Float32, lane_confidence_topic, self.lane_confidence_cb, 10)
        self.create_subscription(Bool, lane_available_topic, self.lane_available_cb, 10)
        self.create_subscription(Bool, lane_valid_topic, self.lane_valid_cb, 10)
        self.create_subscription(Bool, lane_predicted_topic, self.lane_predicted_cb, 10)
        self.create_subscription(Float32MultiArray, lane_left_topic, self.left_lane_cb, 10)
        self.create_subscription(Float32MultiArray, lane_right_topic, self.right_lane_cb, 10)
        self.create_subscription(Float32, obstacle_bias_topic, self.obstacle_bias_cb, 10)
        self.create_subscription(Float32, obstacle_speed_scale_topic, self.obstacle_speed_scale_cb, 10)
        self.create_subscription(Bool, emergency_stop_topic, self.emergency_stop_cb, 10)
        self.create_subscription(Bool, obstacle_active_topic, self.obstacle_active_cb, 10)
        self.create_subscription(Bool, obstacle_unknown_topic, self.obstacle_unknown_cb, 10)
        self.create_subscription(String, obstacle_pass_state_topic, self.obstacle_pass_state_cb, 10)
        self.create_subscription(String, obstacle_selected_gap_topic, self.obstacle_selected_gap_cb, 10)
        self.create_subscription(String, obstacle_selected_pass_side_topic, self.obstacle_selected_pass_side_cb, 10)
        self.create_subscription(Bool, obstacle_commit_active_topic, self.obstacle_commit_active_cb, 10)
        self.create_subscription(Float32, obstacle_commit_remaining_topic, self.obstacle_commit_remaining_cb, 10)
        self.create_subscription(String, obstacle_commit_source_topic, self.obstacle_commit_source_cb, 10)
        self.create_subscription(String, obstacle_pass_owner_topic, self.authoritative_pass_owner_cb, 10)
        self.create_subscription(Float32, obstacle_progress_topic, self.obstacle_progress_cb, 10)
        self.create_subscription(Bool, obstacle_blocked_center_topic, self.blocked_center_cb, 10)
        self.create_subscription(Bool, obstacle_blocked_selected_side_topic, self.blocked_selected_side_cb, 10)
        self.create_subscription(Float32, waypoint_heading_hint_topic, self.heading_hint_cb, 10)
        self.create_subscription(Float32, waypoint_progress_topic, self.progress_cb, 10)

        control_hz = max(5.0, float(self.get_parameter('control_hz').value))
        self.control_period = 1.0 / control_hz
        self.create_timer(self.control_period, self.timer_cb)

    def lane_error_cb(self, msg: Float32) -> None:
        self.lane_error = float(msg.data)
        self.lane_error_stamp_ns = self.get_clock().now().nanoseconds

    def lane_heading_error_cb(self, msg: Float32) -> None:
        self.lane_heading_error = float(msg.data)
        self.lane_heading_stamp_ns = self.get_clock().now().nanoseconds

    def lane_confidence_cb(self, msg: Float32) -> None:
        self.lane_confidence = clamp(float(msg.data), 0.0, 1.0)
        self.lane_conf_stamp_ns = self.get_clock().now().nanoseconds

    def lane_available_cb(self, msg: Bool) -> None:
        self.lane_available = bool(msg.data)
        self.lane_available_stamp_ns = self.get_clock().now().nanoseconds

    def lane_valid_cb(self, msg: Bool) -> None:
        self.lane_valid = bool(msg.data)
        self.lane_valid_stamp_ns = self.get_clock().now().nanoseconds

    def lane_predicted_cb(self, msg: Bool) -> None:
        self.lane_predicted = bool(msg.data)
        self.lane_predicted_stamp_ns = self.get_clock().now().nanoseconds

    def left_lane_cb(self, msg: Float32MultiArray) -> None:
        self.left_lane_visible = len(msg.data) >= 5
        self.left_lane_stamp_ns = self.get_clock().now().nanoseconds

    def right_lane_cb(self, msg: Float32MultiArray) -> None:
        self.right_lane_visible = len(msg.data) >= 5
        self.right_lane_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_bias_cb(self, msg: Float32) -> None:
        self.obstacle_bias = float(msg.data)
        self.obstacle_bias_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_speed_scale_cb(self, msg: Float32) -> None:
        self.obstacle_speed_scale = clamp(float(msg.data), 0.0, 1.0)
        self.obstacle_scale_stamp_ns = self.get_clock().now().nanoseconds

    def emergency_stop_cb(self, msg: Bool) -> None:
        self.emergency_stop = bool(msg.data)
        self.emergency_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_active_cb(self, msg: Bool) -> None:
        self.obstacle_active = bool(msg.data)
        self.obstacle_active_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_unknown_cb(self, msg: Bool) -> None:
        self.obstacle_unknown = bool(msg.data)
        self.obstacle_unknown_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_pass_state_cb(self, msg: String) -> None:
        try:
            state = json.loads(str(msg.data))
        except json.JSONDecodeError:
            return
        self.authoritative_pass_owner = str(state.get('source_node', 'unknown')) or 'unknown'
        self.authoritative_pass_owner_stamp_ns = self.get_clock().now().nanoseconds
        self.authoritative_obstacle_active = bool(state.get('obstacle_active', False))
        self.authoritative_pre_avoid_active = bool(state.get('pre_avoid_active', False))
        self.authoritative_obstacle_latch_state = (
            str(state.get('obstacle_latch_state', 'idle')) or 'idle'
        )
        self.obstacle_selected_pass_side = str(state.get('pass_side', 'NONE')).strip().upper() or 'NONE'
        self.obstacle_selected_gap = str(state.get('selected_gap', 'CENTER')).strip().upper() or 'CENTER'
        self.authoritative_corridor_target = float(state.get('corridor_target', 0.0))
        self.authoritative_corridor_enabled = bool(state.get('corridor_enabled', False))
        self.authoritative_corridor_reason = str(state.get('corridor_gating_reason', 'none')) or 'none'
        self.obstacle_commit_active = bool(state.get('commit_active', False))
        self.obstacle_commit_session_id = max(0, int(state.get('commit_session_id', 0)))
        self.authoritative_side_lock_active = bool(state.get('side_lock_active', False))
        self.authoritative_locked_pass_side = (
            str(state.get('locked_pass_side', 'NONE')).strip().upper() or 'NONE'
        )
        self.stale_commit_active = bool(state.get('stale_commit_active', False))
        self.stale_commit_detected = bool(state.get('stale_commit_detected', False))
        self.stale_obstacle_memory_detected = bool(
            state.get('stale_obstacle_memory_detected', False)
        )
        self.obstacle_commit_remaining_sec = max(0.0, float(state.get('commit_remaining_time', 0.0)))
        self.obstacle_commit_remaining_distance_m = max(
            0.0,
            float(state.get('commit_remaining_distance', 0.0)),
        )
        self.obstacle_progress = clamp(float(state.get('progress', 0.0)), 0.0, 1.0)
        self.commit_age = max(0.0, float(state.get('commit_age', 0.0)))
        self.progress_delta = max(0.0, float(state.get('progress_delta', 0.0)))
        self.tracked_local_x_delta = max(0.0, float(state.get('tracked_local_x_delta', 0.0)))
        self.odom_delta_since_commit = max(0.0, float(state.get('odom_delta_since_commit', 0.0)))
        self.blocked_center = bool(state.get('blocked_center', False))
        self.blocked_selected_side = bool(state.get('blocked_selected_side', False))
        self.obstacle_commit_source = str(state.get('enter_reason', 'none')) or 'none'
        self.authoritative_exit_reason = str(state.get('exit_reason', 'none')) or 'none'
        self.lane_hard_constraint_active = bool(state.get('lane_hard_constraint_active', False))
        self.center_corridor_exists = bool(state.get('center_corridor_exists', False))
        self.center_corridor_preferred = bool(state.get('center_corridor_preferred', False))
        self.center_preferred_reason = str(state.get('center_preferred_reason', 'none')) or 'none'
        self.center_reject_reason = str(state.get('center_reject_reason', 'none')) or 'none'
        self.center_reject_strength = max(0.0, float(state.get('center_reject_strength', 0.0)))
        self.center_reject_persistence = max(0, int(state.get('center_reject_persistence', 0)))
        self.advisory_side_gap_strength = max(0.0, float(state.get('advisory_side_gap_strength', 0.0)))
        self.side_gap_suppressed_due_to_no_commit = bool(
            state.get('side_gap_suppressed_due_to_no_commit', False)
        )
        self.side_target_suppressed_reason = (
            str(state.get('side_target_suppressed_reason', 'none')) or 'none'
        )
        self.target_clipped_to_lane_bounds = bool(state.get('target_clipped_to_lane_bounds', False))
        self.target_clip_reason = str(state.get('target_clip_reason', 'none')) or 'none'
        self.final_controller_mode = str(state.get('final_controller_mode', 'lane_center')) or 'lane_center'
        self.lane_corridor_min_offset = float(state.get('lane_corridor_min_offset', 0.0))
        self.lane_corridor_max_offset = float(state.get('lane_corridor_max_offset', 0.0))
        self.commit_session_start_reason = (
            str(state.get('commit_session_start_reason', 'none')) or 'none'
        )
        self.filtered_obstacle_local_y = float(state.get('filtered_obstacle_local_y', 0.0))
        self.tracked_local_x = float(state.get('tracked_local_x', 99.0))
        self.tracked_local_y = float(state.get('tracked_local_y', 0.0))
        self.critical_dist = float(state.get('critical_dist', 99.0))
        self.critical_points = max(0, int(state.get('critical_points', 0)))
        self.footprint_intrusion = max(0.0, float(state.get('footprint_intrusion', 0.0)))
        self.critical_intrusion_persistence_cycles_used = max(
            0,
            int(state.get('critical_intrusion_persistence_cycles_used', 0)),
        )
        self.obstacle_local_y_deadband_active = bool(
            state.get('obstacle_local_y_deadband_active', False)
        )
        self.side_flip_blocked = bool(state.get('side_flip_blocked', False))
        self.side_switch_reject_reason = (
            str(state.get('side_switch_reject_reason', 'none')) or 'none'
        )
        self.false_critical_override_detected = bool(
            state.get('false_critical_override_detected', False)
        )
        self.critical_override_blocked_by_center_corridor = bool(
            state.get('critical_override_blocked_by_center_corridor', False)
        )
        self.critical_trigger_consistent_with_tracked_geometry = bool(
            state.get('critical_trigger_consistent_with_tracked_geometry', True)
        )
        self.center_corridor_override_priority_applied = bool(
            state.get('center_corridor_override_priority_applied', False)
        )
        self.critical_commit_rejected_reason = (
            str(state.get('critical_commit_rejected_reason', 'none')) or 'none'
        )
        self.zombie_commit_state_detected = bool(state.get('zombie_commit_state_detected', False))
        self.atomic_commit_state_clear_applied = bool(
            state.get('atomic_commit_state_clear_applied', False)
        )
        self.critical_reject_forced_state_clear = bool(
            state.get('critical_reject_forced_state_clear', False)
        )
        self.pass_state_validity_ok = bool(state.get('pass_state_validity_ok', True))
        self.false_emergency_demoted = bool(state.get('false_emergency_demoted', False))
        self.emergency_latch_rejected_due_to_low_persistence = bool(
            state.get('emergency_latch_rejected_due_to_low_persistence', False)
        )
        self.emergency_latch_rejected_due_to_center_corridor = bool(
            state.get('emergency_latch_rejected_due_to_center_corridor', False)
        )
        self.center_corridor_stabilizer_active = bool(
            state.get('center_corridor_stabilizer_active', False)
        )
        self.lane_only_fallback_blocked = bool(state.get('lane_only_fallback_blocked', False))
        self.emergency_latch_kept_reason = (
            str(state.get('emergency_latch_kept_reason', 'none')) or 'none'
        )
        self.lane_term_preserved_in_critical = bool(
            state.get('lane_term_preserved_in_critical', False)
        )
        self.corridor_term_preserved_in_critical = bool(
            state.get('corridor_term_preserved_in_critical', False)
        )
        self.side_commit_cancelled_due_to_valid_center_corridor = bool(
            state.get('side_commit_cancelled_due_to_valid_center_corridor', False)
        )
        now_ns = self.get_clock().now().nanoseconds
        self.obstacle_pass_state_stamp_ns = now_ns
        self.obstacle_selected_gap_stamp_ns = now_ns
        self.obstacle_selected_pass_side_stamp_ns = now_ns
        self.obstacle_commit_active_stamp_ns = now_ns
        self.obstacle_commit_remaining_stamp_ns = now_ns
        self.obstacle_commit_source_stamp_ns = now_ns
        self.obstacle_progress_stamp_ns = now_ns
        self.blocked_center_stamp_ns = now_ns
        self.blocked_selected_side_stamp_ns = now_ns
        self.pass_state_valid = self.authoritative_pass_owner == 'yaris_pilotu'
        self.ignored_transient_none_due_to_commit_lock = False
        raw_commit_valid = (
            self.pass_state_validity_ok
            and not self.zombie_commit_state_detected
            and not self.critical_reject_forced_state_clear
            and self.obstacle_commit_active
            and self.obstacle_commit_session_id > 0
        )
        if (
            raw_commit_valid
            and self.authoritative_side_lock_active
            and self.authoritative_locked_pass_side in ('LEFT', 'RIGHT')
        ):
            self.committed_pass_side = self.authoritative_locked_pass_side
            self.committed_commit_session_id = self.obstacle_commit_session_id
            self.current_side_source_reason = 'authoritative_commit_lock'
        elif raw_commit_valid and self.obstacle_selected_pass_side in ('LEFT', 'RIGHT'):
            self.committed_pass_side = self.obstacle_selected_pass_side
            self.committed_commit_session_id = self.obstacle_commit_session_id
            self.current_side_source_reason = 'authoritative_commit_pass_side'
        elif (
            raw_commit_valid
            and self.committed_pass_side in ('LEFT', 'RIGHT')
        ):
            self.ignored_transient_none_due_to_commit_lock = True
            self.current_side_source_reason = 'latched_commit_carry'
        else:
            self.committed_pass_side = 'NONE'
            self.committed_commit_session_id = 0
            self.current_side_source_reason = 'no_commit_side'

    def obstacle_selected_gap_cb(self, msg: String) -> None:
        label = str(msg.data).strip().upper()
        self.obstacle_selected_gap = label if label else 'CENTER'
        self.obstacle_selected_gap_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_selected_pass_side_cb(self, msg: String) -> None:
        label = str(msg.data).strip().upper()
        self.obstacle_selected_pass_side = label if label else 'NONE'
        self.obstacle_selected_pass_side_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_commit_active_cb(self, msg: Bool) -> None:
        self.obstacle_commit_active = bool(msg.data)
        self.obstacle_commit_active_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_commit_remaining_cb(self, msg: Float32) -> None:
        self.obstacle_commit_remaining_sec = max(0.0, float(msg.data))
        self.obstacle_commit_remaining_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_commit_source_cb(self, msg: String) -> None:
        self.obstacle_commit_source = str(msg.data).strip() or 'none'
        self.obstacle_commit_source_stamp_ns = self.get_clock().now().nanoseconds

    def authoritative_pass_owner_cb(self, msg: String) -> None:
        self.authoritative_pass_owner = str(msg.data).strip() or 'unknown'
        self.authoritative_pass_owner_stamp_ns = self.get_clock().now().nanoseconds

    def obstacle_progress_cb(self, msg: Float32) -> None:
        self.obstacle_progress = clamp(float(msg.data), 0.0, 1.0)
        self.obstacle_progress_stamp_ns = self.get_clock().now().nanoseconds

    def blocked_center_cb(self, msg: Bool) -> None:
        self.blocked_center = bool(msg.data)
        self.blocked_center_stamp_ns = self.get_clock().now().nanoseconds

    def blocked_selected_side_cb(self, msg: Bool) -> None:
        self.blocked_selected_side = bool(msg.data)
        self.blocked_selected_side_stamp_ns = self.get_clock().now().nanoseconds

    def heading_hint_cb(self, msg: Float32) -> None:
        self.heading_hint = float(msg.data)
        self.heading_hint_stamp_ns = self.get_clock().now().nanoseconds

    def progress_cb(self, msg: Float32) -> None:
        self.waypoint_progress = clamp(float(msg.data), 0.0, 1.0)
        self.waypoint_progress_stamp_ns = self.get_clock().now().nanoseconds

    def is_recent(self, stamp_ns: int, timeout_ns: int, now_ns: int) -> bool:
        return stamp_ns > 0 and (now_ns - stamp_ns) <= timeout_ns

    def authoritative_commit_valid(self, now_ns: int) -> bool:
        return (
            self.pass_state_valid
            and self.is_recent(self.obstacle_pass_state_stamp_ns, self.obstacle_timeout_ns, now_ns)
            and self.pass_state_validity_ok
            and not self.zombie_commit_state_detected
            and not self.critical_reject_forced_state_clear
            and self.obstacle_commit_active
            and self.obstacle_commit_session_id > 0
        )

    def lane_boundary_recent(self, stamp_ns: int, visible: bool, now_ns: int) -> bool:
        return visible and self.is_recent(stamp_ns, self.lane_boundary_timeout_ns, now_ns)

    def pass_side_bias_sign(self, pass_side: str) -> float:
        if pass_side == 'LEFT':
            return 1.0
        if pass_side == 'RIGHT':
            return -1.0
        return 0.0

    def update_lane_reacquisition_guard(
        self,
        lane_available: bool,
        lane_recent: bool,
        lane_confidence: float,
        raw_lane_state: str,
    ) -> None:
        if not self.lane_reacquisition_guard_active:
            return
        lane_ready = (
            lane_available
            and lane_recent
            and raw_lane_state != 'NO_LANE'
            and lane_confidence >= self.lane_reacquisition_min_confidence
        )
        if lane_ready:
            self.lane_reacquisition_valid_cycles += 1
        else:
            self.lane_reacquisition_valid_cycles = 0
        if self.lane_reacquisition_valid_cycles >= self.lane_reacquisition_persistence_cycles:
            self.lane_reacquisition_guard_active = False

    def compute_raw_lane_state(
        self,
        now_ns: int,
        lane_available: bool,
        lane_recent: bool,
        predicted_active: bool,
    ) -> str:
        left_recent = self.lane_boundary_recent(self.left_lane_stamp_ns, self.left_lane_visible, now_ns)
        right_recent = self.lane_boundary_recent(self.right_lane_stamp_ns, self.right_lane_visible, now_ns)
        if left_recent and right_recent:
            return 'NORMAL_LANE'
        if (
            left_recent
            or right_recent
            or predicted_active
            or (lane_available and lane_recent)
            or self.is_recent(self.last_reliable_lane_stamp_ns, self.recent_lane_hold_ns, now_ns)
        ):
            return 'SINGLE_LANE'
        return 'NO_LANE'

    def update_lane_state(self, raw_lane_state: str) -> None:
        if raw_lane_state == 'NORMAL_LANE':
            self.pending_single_lane_frames = 0
            self.pending_no_lane_frames = 0
            self.lane_state = 'NORMAL_LANE'
            return

        if raw_lane_state == 'SINGLE_LANE':
            self.pending_no_lane_frames = 0
            if self.lane_state == 'NORMAL_LANE':
                self.pending_single_lane_frames += 1
                if self.pending_single_lane_frames < self.single_lane_transition_frames:
                    return
            self.pending_single_lane_frames = 0
            self.lane_state = 'SINGLE_LANE'
            return

        if self.lane_state == 'NORMAL_LANE':
            self.pending_single_lane_frames += 1
            if self.pending_single_lane_frames < self.single_lane_transition_frames:
                return
            self.pending_single_lane_frames = 0
            self.lane_state = 'SINGLE_LANE'

        if self.lane_state == 'SINGLE_LANE':
            self.pending_no_lane_frames += 1
            if self.pending_no_lane_frames < self.no_lane_transition_frames:
                return
        self.pending_no_lane_frames = 0
        self.lane_state = 'NO_LANE'

    def update_avoidance_commit(
        self,
        now_ns: int,
        obstacle_active: bool,
        obstacle_bias: float,
    ) -> None:
        del obstacle_active
        del obstacle_bias
        state_recent = self.pass_state_valid and self.is_recent(
            self.obstacle_pass_state_stamp_ns,
            self.obstacle_timeout_ns,
            now_ns,
        )
        authoritative_commit_valid = self.authoritative_commit_valid(now_ns)
        self.ignored_transient_none_due_to_commit_lock = False
        if (
            authoritative_commit_valid
            and self.authoritative_side_lock_active
            and self.authoritative_locked_pass_side in ('LEFT', 'RIGHT')
        ):
            self.committed_pass_side = self.authoritative_locked_pass_side
            self.committed_commit_session_id = self.obstacle_commit_session_id
            self.current_side_source_reason = 'authoritative_locked_side'
        elif authoritative_commit_valid and self.obstacle_selected_pass_side in ('LEFT', 'RIGHT'):
            self.committed_pass_side = self.obstacle_selected_pass_side
            self.committed_commit_session_id = self.obstacle_commit_session_id
            self.current_side_source_reason = 'authoritative_commit_pass_side'
        elif authoritative_commit_valid and self.committed_pass_side in ('LEFT', 'RIGHT'):
            self.ignored_transient_none_due_to_commit_lock = True
            self.current_side_source_reason = 'ignored_transient_none_during_commit'
        else:
            self.committed_pass_side = 'NONE'
            self.committed_commit_session_id = 0
            self.current_side_source_reason = 'no_authoritative_commit'
        remaining_sec = self.obstacle_commit_remaining_sec if state_recent else 0.0
        if authoritative_commit_valid:
            self.avoidance_commit_until_ns = now_ns + int(max(remaining_sec, 0.0) * 1e9)
        else:
            self.avoidance_commit_until_ns = now_ns

    def publish_debug_state(
        self,
        now_ns: int,
        lane_state: str,
        final_speed: float,
        final_steering: float,
        stop_reason: str,
    ) -> None:
        msg_text = String()
        msg_text.data = lane_state
        self.debug_lane_state_pub.publish(msg_text)

        msg_text = String()
        pass_side = (
            self.committed_pass_side
            if self.committed_pass_side in ('LEFT', 'RIGHT')
            else (
                'CENTER'
                if self.center_corridor_preferred
                else self.obstacle_selected_gap
            )
        )
        msg_text.data = pass_side
        self.debug_pass_side_pub.publish(msg_text)

        commit_active = self.authoritative_commit_valid(now_ns)
        msg_bool = Bool()
        msg_bool.data = bool(commit_active)
        self.debug_commit_active_pub.publish(msg_bool)

        msg_float = Float32()
        if commit_active and self.is_recent(
            self.obstacle_commit_remaining_stamp_ns,
            self.obstacle_timeout_ns,
            now_ns,
        ):
            msg_float.data = float(self.obstacle_commit_remaining_sec)
        else:
            msg_float.data = 0.0
        self.debug_commit_remaining_pub.publish(msg_float)

        msg_text = String()
        msg_text.data = stop_reason
        self.debug_stop_reason_pub.publish(msg_text)

        msg_float = Float32()
        msg_float.data = float(final_speed)
        self.debug_cmd_speed_pub.publish(msg_float)

        msg_float = Float32()
        msg_float.data = float(final_steering)
        self.debug_cmd_steer_pub.publish(msg_float)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def timer_cb(self) -> None:
        now_ns = self.get_clock().now().nanoseconds

        emergency_active = self.emergency_stop and self.is_recent(
            self.emergency_stamp_ns,
            self.obstacle_timeout_ns,
            now_ns,
        )
        if emergency_active:
            self.last_stop_reason = 'emergency_stop'
            self.publish_stop()
            self.publish_debug_state(
                now_ns,
                self.lane_state,
                0.0,
                0.0,
                self.last_stop_reason,
            )
            return

        lane_available = False
        if self.is_recent(self.lane_available_stamp_ns, self.lane_timeout_ns, now_ns):
            lane_available = self.lane_available
        elif self.is_recent(self.lane_valid_stamp_ns, self.lane_timeout_ns, now_ns):
            lane_available = self.lane_valid

        lane_error = self.lane_error if self.is_recent(self.lane_error_stamp_ns, self.lane_timeout_ns, now_ns) else 0.0
        lane_heading_error = (
            self.lane_heading_error
            if self.is_recent(self.lane_heading_stamp_ns, self.lane_timeout_ns, now_ns)
            else 0.0
        )
        lane_confidence = (
            self.lane_confidence
            if self.is_recent(self.lane_conf_stamp_ns, self.lane_timeout_ns, now_ns)
            else (0.85 if lane_available else 0.0)
        )

        obstacle_bias = (
            self.obstacle_bias
            if self.is_recent(self.obstacle_bias_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else 0.0
        )
        obstacle_speed_scale = (
            self.obstacle_speed_scale
            if self.is_recent(self.obstacle_scale_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else 1.0
        )
        obstacle_speed_scale = clamp(obstacle_speed_scale, 0.0, 1.0)
        raw_obstacle_active = (
            self.obstacle_active
            if self.is_recent(self.obstacle_active_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )
        obstacle_unknown = (
            self.obstacle_unknown
            if self.is_recent(self.obstacle_unknown_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )
        if raw_obstacle_active:
            obstacle_unknown = False
        blocked_center = (
            self.blocked_center
            if self.is_recent(self.blocked_center_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )
        blocked_selected_side = (
            self.blocked_selected_side
            if self.is_recent(self.blocked_selected_side_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else False
        )

        heading_hint = (
            self.heading_hint
            if self.is_recent(self.heading_hint_stamp_ns, self.waypoint_timeout_ns, now_ns)
            else 0.0
        )
        obstacle_progress = (
            self.obstacle_progress
            if self.is_recent(self.obstacle_progress_stamp_ns, self.obstacle_timeout_ns, now_ns)
            else 0.0
        )
        waypoint_progress = (
            self.waypoint_progress
            if self.is_recent(self.waypoint_progress_stamp_ns, self.waypoint_timeout_ns, now_ns)
            else 0.0
        )
        authoritative_state_recent = self.pass_state_valid and self.is_recent(
            self.obstacle_pass_state_stamp_ns,
            self.obstacle_timeout_ns,
            now_ns,
        )
        stale_commit_active = bool(authoritative_state_recent and self.stale_commit_active)
        obstacle_active = bool(
            authoritative_state_recent
            and raw_obstacle_active
            and self.authoritative_obstacle_active
        )
        pre_avoid_active = bool(authoritative_state_recent and self.authoritative_pre_avoid_active)
        if authoritative_state_recent:
            blocked_center = self.blocked_center
            blocked_selected_side = self.blocked_selected_side

        if self.invert_lane_error:
            lane_error *= -1.0
        if self.invert_lane_heading_error:
            lane_heading_error *= -1.0

        lane_recent = (
            self.is_recent(self.lane_error_stamp_ns, self.lane_timeout_ns, now_ns)
            or self.is_recent(self.lane_valid_stamp_ns, self.lane_timeout_ns, now_ns)
        )
        predicted_recent = self.is_recent(self.lane_predicted_stamp_ns, self.lane_timeout_ns, now_ns)
        predicted_active = predicted_recent and self.lane_predicted
        if lane_available and lane_recent and lane_confidence >= self.nominal_confidence_threshold:
            self.last_reliable_lane_error = lane_error
            self.last_reliable_lane_heading_error = lane_heading_error
            self.last_reliable_lane_stamp_ns = now_ns

        raw_lane_state = self.compute_raw_lane_state(now_ns, lane_available, lane_recent, predicted_active)
        self.update_lane_reacquisition_guard(
            lane_available,
            lane_recent,
            lane_confidence,
            raw_lane_state,
        )
        self.update_lane_state(raw_lane_state)
        self.update_avoidance_commit(now_ns, obstacle_active, obstacle_bias)
        raw_commit_active = bool(authoritative_state_recent and self.obstacle_commit_active)
        pass_state_validity_ok_runtime = bool(
            authoritative_state_recent
            and self.pass_state_validity_ok
            and not self.zombie_commit_state_detected
            and not self.critical_reject_forced_state_clear
        )
        authoritative_commit_valid = bool(
            pass_state_validity_ok_runtime
            and raw_commit_active
            and self.obstacle_commit_session_id > 0
        )
        stale_published_side_ignored = bool(
            authoritative_state_recent
            and not authoritative_commit_valid
            and self.obstacle_selected_pass_side in ('LEFT', 'RIGHT')
        )
        stale_side_lock_ignored = bool(
            authoritative_state_recent
            and not authoritative_commit_valid
            and (
                self.authoritative_side_lock_active
                or self.authoritative_locked_pass_side in ('LEFT', 'RIGHT')
            )
        )
        low_persistence_emergency = (
            authoritative_state_recent
            and self.authoritative_obstacle_latch_state == 'emergency'
            and self.critical_intrusion_persistence_cycles_used < self.critical_intrusion_persistence_min_cycles
        )
        emergency_without_commit = (
            authoritative_state_recent
            and self.authoritative_obstacle_latch_state == 'emergency'
            and self.obstacle_selected_pass_side == 'NONE'
            and not authoritative_commit_valid
        )
        false_emergency_state = (
            authoritative_state_recent
            and self.authoritative_obstacle_latch_state == 'emergency'
            and self.center_corridor_exists
            and not blocked_center
            and (
                self.false_emergency_demoted
                or self.emergency_latch_rejected_due_to_low_persistence
                or self.emergency_latch_rejected_due_to_center_corridor
                or low_persistence_emergency
                or emergency_without_commit
            )
        )
        critical_commit_active = authoritative_commit_valid and self.obstacle_commit_source == 'critical_escape'
        questionable_critical_commit = (
            critical_commit_active
            and self.center_corridor_exists
            and not blocked_center
            and (
                self.false_critical_override_detected
                or self.critical_override_blocked_by_center_corridor
                or not self.critical_trigger_consistent_with_tracked_geometry
                or self.side_commit_cancelled_due_to_valid_center_corridor
            )
        )
        commit_active = authoritative_commit_valid and not questionable_critical_commit
        if false_emergency_state:
            obstacle_active = True
            pre_avoid_active = False
        if commit_active and self.committed_pass_side in ('LEFT', 'RIGHT'):
            authoritative_pass_side = self.committed_pass_side
            self.current_side_source_reason = (
                'commit_lock'
                if self.authoritative_side_lock_active
                else self.current_side_source_reason
            )
        else:
            authoritative_pass_side = 'NONE'
        authoritative_selected_gap = self.obstacle_selected_gap if authoritative_state_recent else 'CENTER'
        if commit_active and authoritative_pass_side in ('LEFT', 'RIGHT'):
            authoritative_selected_gap = authoritative_pass_side
        elif questionable_critical_commit:
            authoritative_selected_gap = 'CENTER'
            authoritative_pass_side = 'NONE'
            self.current_side_source_reason = 'false_critical_demoted'
        elif (
            authoritative_state_recent
            and self.center_corridor_preferred
            and not authoritative_commit_valid
        ):
            authoritative_selected_gap = 'CENTER'
        side_target_blocked_due_to_no_active_commit = bool(
            authoritative_state_recent
            and self.center_corridor_preferred
            and not authoritative_commit_valid
            and (
                stale_published_side_ignored
                or stale_side_lock_ignored
                or self.obstacle_selected_gap in ('LEFT', 'RIGHT', 'CENTER_LEFT', 'CENTER_RIGHT')
            )
        )
        if side_target_blocked_due_to_no_active_commit:
            authoritative_pass_side = 'NONE'
            authoritative_selected_gap = 'CENTER'
            self.current_side_source_reason = 'side_target_blocked_due_to_no_active_commit'
        no_authoritative_side = authoritative_pass_side == 'NONE' and not commit_active
        center_corridor_should_dominate = (
            (no_authoritative_side or questionable_critical_commit or false_emergency_state)
            and not blocked_center
            and self.center_corridor_exists
        )
        avoid_context_active = pre_avoid_active or obstacle_active or commit_active or false_emergency_state
        if (
            self.lane_state == 'NO_LANE'
            and commit_active
            and self.is_recent(self.last_reliable_lane_stamp_ns, self.no_lane_memory_ns, now_ns)
        ):
            effective_lane_state = 'SINGLE_LANE'
        else:
            effective_lane_state = self.lane_state
        advisory_gap_active = (
            avoid_context_active
            and not commit_active
            and authoritative_pass_side == 'NONE'
            and authoritative_selected_gap in ('LEFT', 'RIGHT', 'CENTER_LEFT', 'CENTER_RIGHT')
            and effective_lane_state != 'NO_LANE'
        )
        advisory_gap_center_variant = authoritative_selected_gap in ('CENTER_LEFT', 'CENTER_RIGHT')
        advisory_gap_reason = 'none'
        advisory_gap_corridor_term_cap = self.advisory_gap_corridor_term_cap
        final_controller_mode = 'lane_center'

        lane_error_for_control = lane_error
        lane_heading_for_control = lane_heading_error
        if (
            effective_lane_state == 'SINGLE_LANE'
            and self.is_recent(self.last_reliable_lane_stamp_ns, self.no_lane_memory_ns, now_ns)
        ):
            blend = self.single_lane_memory_blend
            current_error = lane_error if lane_recent or predicted_active else self.last_reliable_lane_error
            current_heading = (
                lane_heading_error
                if self.is_recent(self.lane_heading_stamp_ns, self.lane_timeout_ns, now_ns)
                else self.last_reliable_lane_heading_error
            )
            lane_error_for_control = (
                (1.0 - blend) * current_error + blend * self.last_reliable_lane_error
            )
            lane_heading_for_control = (
                (1.0 - blend) * current_heading + blend * self.last_reliable_lane_heading_error
            )

        waypoint_weight = (
            self.waypoint_weight_with_lane
            if lane_available and lane_confidence >= 0.35
            else self.waypoint_weight_no_lane
        )
        if avoid_context_active:
            waypoint_weight = 0.0
        if obstacle_unknown:
            waypoint_weight = 0.0

        runtime_lane_hard_constraint_active = (
            self.lane_hard_constraint_active
            or (
                effective_lane_state != 'NO_LANE'
                and (lane_recent or predicted_active or self.is_recent(self.last_reliable_lane_stamp_ns, self.no_lane_memory_ns, now_ns))
            )
        )
        runtime_center_corridor_stabilizer_active = False
        runtime_lane_only_fallback_blocked = self.lane_only_fallback_blocked

        avoid_mode_hint = 'none'
        if avoid_context_active:
            if effective_lane_state == 'NORMAL_LANE':
                avoid_mode_hint = 'avoid'
            elif effective_lane_state == 'SINGLE_LANE':
                avoid_mode_hint = 'single_line_avoid'
            else:
                avoid_mode_hint = 'blind_avoid'

        lane_guard_ratio = 0.0
        effective_lane_guard_ratio = 0.0
        obstacle_bias_opposes_lane = False
        obstacle_speed_guard_scale = 1.0
        obstacle_bias_raw = obstacle_bias
        obstacle_preempt_ratio = 0.0
        if avoid_context_active:
            speed_preempt_ratio = clamp(
                (self.obstacle_preempt_speed_scale_threshold - obstacle_speed_scale)
                / max(
                    self.obstacle_preempt_speed_scale_threshold - self.obstacle_speed_min_scale,
                    1e-3,
                ),
                0.0,
                1.0,
            )
            bias_preempt_ratio = clamp(
                (abs(obstacle_bias_raw) - self.obstacle_preempt_bias_abs)
                / max(self.obstacle_speed_full_bias_abs - self.obstacle_preempt_bias_abs, 1e-3),
                0.0,
                1.0,
            )
            obstacle_preempt_ratio = max(speed_preempt_ratio, bias_preempt_ratio)
        if effective_lane_state != 'NO_LANE' and lane_recent and avoid_context_active:
            lane_guard_signal = lane_error_for_control + (0.35 * lane_heading_for_control)
            if abs(lane_guard_signal) < 1e-3:
                lane_guard_signal = lane_error_for_control
            lane_guard_ratio = clamp(
                (abs(lane_guard_signal) - self.obstacle_lane_guard_error_threshold)
                / max(
                    self.obstacle_lane_guard_full_error - self.obstacle_lane_guard_error_threshold,
                    1e-3,
                ),
                0.0,
                1.0,
            )
            obstacle_bias_opposes_lane = (
                abs(lane_guard_signal) >= self.obstacle_lane_guard_error_threshold
                and (obstacle_bias * lane_guard_signal) < 0.0
            )
            effective_lane_guard_ratio = lane_guard_ratio
            if commit_active and authoritative_pass_side in ('LEFT', 'RIGHT'):
                effective_lane_guard_ratio *= self.commit_lane_guard_scale
            if obstacle_bias_opposes_lane and effective_lane_guard_ratio > 0.0:
                guard_suppression = 1.0 - (
                    self.obstacle_preempt_opposite_drop_relief * obstacle_preempt_ratio
                )
                obstacle_bias *= (
                    1.0 - (
                        (1.0 - self.obstacle_lane_guard_min_scale)
                        * effective_lane_guard_ratio
                        * guard_suppression
                    )
                )

            bias_ratio = clamp(
                abs(obstacle_bias_raw) / self.obstacle_speed_full_bias_abs,
                0.0,
                1.0,
            )
            obstacle_speed_guard_scale = 1.0 - (
                self.obstacle_bias_speed_reduction_gain * bias_ratio
                + self.obstacle_lane_guard_speed_reduction_gain * effective_lane_guard_ratio
            )
            obstacle_speed_guard_scale = clamp(
                obstacle_speed_guard_scale,
                self.obstacle_speed_min_scale,
                1.0,
            )

        lane_weight_used = self.lane_weight_normal
        corridor_weight_used = self.obstacle_weight
        if commit_active:
            lane_weight_used = self.lane_weight_commit
            corridor_weight_used = self.corridor_weight_commit
        elif pre_avoid_active or obstacle_active:
            lane_weight_used = self.lane_weight_pre_avoid
            corridor_weight_used = self.corridor_weight_pre_avoid
        if avoid_mode_hint in ('single_line_avoid', 'blind_avoid'):
            lane_weight_used *= self.single_line_avoid_lane_weight_scale
            corridor_weight_used *= self.single_line_avoid_obstacle_weight_scale
        elif pre_avoid_active or obstacle_active:
            lane_weight_used *= self.avoid_lane_weight_scale
            corridor_weight_used *= self.avoid_obstacle_weight_scale
        if critical_commit_active:
            lane_weight_used = max(lane_weight_used, self.critical_lane_term_min_weight)
            corridor_weight_used = max(corridor_weight_used, self.critical_corridor_term_min_weight)
        if questionable_critical_commit:
            lane_weight_used = max(
                lane_weight_used,
                self.center_corridor_override_priority_weight,
            )
            corridor_weight_used = max(
                corridor_weight_used,
                self.critical_corridor_term_min_weight,
            )
            obstacle_bias = clamp(
                obstacle_bias,
                -self.advisory_gap_center_bias_cap,
                self.advisory_gap_center_bias_cap,
            )
        if avoid_context_active and not commit_active:
            lane_weight_used = (
                (1.0 - obstacle_preempt_ratio) * lane_weight_used
                + obstacle_preempt_ratio * max(self.obstacle_preempt_lane_weight_min, 0.20)
            )
            corridor_weight_used += self.obstacle_preempt_obstacle_weight_gain * obstacle_preempt_ratio
        if avoid_context_active and effective_lane_guard_ratio > 0.0:
            lane_weight_used += (
                self.obstacle_lane_priority_gain
                * effective_lane_guard_ratio
                * (1.0 - obstacle_preempt_ratio)
            )
            if obstacle_bias_opposes_lane and not (commit_active and authoritative_pass_side in ('LEFT', 'RIGHT')):
                corridor_weight_used *= 1.0 - (
                    self.obstacle_lane_opposite_weight_drop
                    * effective_lane_guard_ratio
                    * (1.0 - self.obstacle_preempt_opposite_drop_relief * obstacle_preempt_ratio)
                )
        aggressive_steering_suppressed_reason = 'none'
        steering_decay_due_to_stale_commit = 1.0
        final_steer_after_stale_decay = 0.0
        no_lane_commit_guard = commit_active and (
            effective_lane_state == 'NO_LANE'
            or lane_confidence < self.lane_reacquisition_min_confidence
        )
        if self.lane_reacquisition_guard_active and avoid_context_active:
            lane_weight_used = max(lane_weight_used, 0.35)
            corridor_weight_used *= self.no_lane_commit_scale
            obstacle_bias *= self.no_lane_commit_scale
            aggressive_steering_suppressed_reason = 'startup_lane_reacquisition_guard'
        elif no_lane_commit_guard:
            lane_weight_used = max(lane_weight_used, 0.28)
            corridor_weight_used *= self.no_lane_commit_scale
            obstacle_bias *= self.no_lane_commit_scale
            aggressive_steering_suppressed_reason = 'no_lane_commit_scale'
        if questionable_critical_commit and aggressive_steering_suppressed_reason == 'none':
            aggressive_steering_suppressed_reason = 'questionable_critical_commit'
        if commit_active and authoritative_pass_side in ('LEFT', 'RIGHT') and abs(obstacle_bias) < self.commit_obstacle_bias_floor:
            floor_scale = self.no_lane_commit_scale if (
                self.lane_reacquisition_guard_active or no_lane_commit_guard
            ) else 1.0
            obstacle_bias = self.pass_side_bias_sign(authoritative_pass_side) * (
                self.commit_obstacle_bias_floor * floor_scale
            )
        advisory_side_gap_strength = clamp(self.advisory_side_gap_strength, 0.0, 1.0)
        if no_authoritative_side:
            obstacle_bias = clamp(obstacle_bias, -self.no_commit_side_bias_cap, self.no_commit_side_bias_cap)
        if center_corridor_should_dominate:
            priority_weight = (
                self.center_corridor_override_priority_weight
                if (questionable_critical_commit or self.center_corridor_override_priority_applied)
                else self.center_corridor_priority_weight
            )
            lane_weight_used = max(lane_weight_used, priority_weight)
            corridor_weight_used = max(
                self.critical_corridor_term_min_weight,
                corridor_weight_used * self.advisory_side_gap_max_weight,
            )
            obstacle_bias = clamp(
                obstacle_bias,
                -self.no_commit_side_bias_cap,
                self.no_commit_side_bias_cap,
            )
            if runtime_lane_hard_constraint_active:
                obstacle_bias = clamp(
                    obstacle_bias,
                    -self.lane_hard_constraint_margin,
                    self.lane_hard_constraint_margin,
                )
            advisory_gap_reason = (
                'center_corridor_false_critical_override'
                if (questionable_critical_commit or false_emergency_state)
                else (
                    'center_corridor_preferred'
                    if self.center_corridor_preferred
                    else 'lane_bounded_center_corridor'
                )
            )
        elif advisory_gap_active:
            corridor_weight_used *= max(
                0.05,
                self.advisory_side_gap_max_weight * max(advisory_side_gap_strength, 0.35),
            )
        if advisory_gap_active:
            lane_weight_used = max(lane_weight_used, self.advisory_gap_lane_weight_min)
            corridor_weight_used *= self.advisory_gap_corridor_scale
            bias_cap = (
                self.advisory_gap_center_bias_cap
                if advisory_gap_center_variant
                else self.advisory_gap_bias_cap
            )
            obstacle_bias = clamp(obstacle_bias, -bias_cap, bias_cap)
            advisory_gap_corridor_term_cap = (
                self.advisory_gap_center_corridor_term_cap
                if advisory_gap_center_variant
                else self.advisory_gap_corridor_term_cap
            )
            advisory_gap_reason = (
                'soft_center_gap_guidance'
                if advisory_gap_center_variant
                else 'soft_side_gap_guidance'
            )
            if center_corridor_should_dominate:
                advisory_gap_reason = 'suppressed_due_to_center_corridor'
        lane_weight_used = max(0.0, lane_weight_used)
        corridor_weight_used = max(0.05, corridor_weight_used)
        lane_term = lane_weight_used * (
            (self.lane_kp * lane_error_for_control)
            + (self.lane_heading_kp * lane_heading_for_control)
        )
        corridor_term = corridor_weight_used * obstacle_bias
        center_corridor_stabilizer_term = 0.0
        if center_corridor_should_dominate and authoritative_selected_gap == 'CENTER' and no_authoritative_side:
            stabilizer_weight = (
                self.no_commit_center_stabilizer_weight
                if no_authoritative_side
                else self.center_corridor_stabilizer_weight
            )
            lateral_hint = self.filtered_obstacle_local_y
            if abs(lateral_hint) < 1e-3 and abs(self.tracked_local_y) < 90.0:
                lateral_hint = self.tracked_local_y
            if abs(lateral_hint) > 1e-3:
                center_corridor_stabilizer_term = -stabilizer_weight * lateral_hint
            elif runtime_lane_hard_constraint_active:
                center_corridor_stabilizer_term = -stabilizer_weight * (
                    0.65 * lane_heading_for_control + 0.35 * lane_error_for_control
                )
            center_corridor_stabilizer_term = clamp(
                center_corridor_stabilizer_term,
                -self.lane_hard_constraint_margin if runtime_lane_hard_constraint_active else -self.advisory_gap_center_bias_cap,
                self.lane_hard_constraint_margin if runtime_lane_hard_constraint_active else self.advisory_gap_center_bias_cap,
            )
            if abs(center_corridor_stabilizer_term) > 1e-3 or self.center_corridor_stabilizer_active:
                runtime_center_corridor_stabilizer_active = True
                runtime_lane_only_fallback_blocked = True
                corridor_term += center_corridor_stabilizer_term
        if advisory_gap_active:
            corridor_term = clamp(
                corridor_term,
                -advisory_gap_corridor_term_cap,
                advisory_gap_corridor_term_cap,
            )
        if center_corridor_should_dominate:
            corridor_term = clamp(
                corridor_term,
                -self.advisory_gap_center_corridor_term_cap,
                self.advisory_gap_center_corridor_term_cap,
            )
        avoid_term = corridor_term
        steering = lane_term + corridor_term + (waypoint_weight * heading_hint)
        speed_limit = self.base_speed
        mode = 'no_lane'
        commit_steer_clamp_used = self.commit_max_angular if commit_active else self.avoid_max_angular_z
        curve_heading_ratio = clamp(abs(lane_heading_for_control) / self.curve_heading_full_scale, 0.0, 1.0)
        curve_error_ratio = clamp(abs(lane_error_for_control) / self.curve_error_full_scale, 0.0, 1.0)
        curve_ratio = max(curve_heading_ratio, 0.65 * curve_error_ratio)
        curve_speed_scale = 1.0 - (self.curve_speed_reduction_max * curve_ratio)
        curve_speed_scale = clamp(curve_speed_scale, 0.15, 1.0)

        if (
            effective_lane_state == 'NORMAL_LANE'
            and lane_available
            and lane_recent
            and lane_confidence >= self.nominal_confidence_threshold
        ):
            mode = 'avoid_commit' if commit_active else ('pre_avoid' if pre_avoid_active else ('avoid' if obstacle_active else 'nominal'))
            if abs(lane_error_for_control) >= self.offlane_error_threshold:
                steering += self.offlane_recovery_gain * lane_error_for_control
            steer_limit = self.commit_max_angular if commit_active else (self.avoid_max_angular_z if avoid_context_active else self.max_angular_z)
            steering = clamp(steering, -steer_limit, steer_limit)
            commit_steer_clamp_used = steer_limit
            confidence_scale = clamp(max(self.min_conf_speed_scale, lane_confidence), 0.0, 1.0)
            speed = self.base_speed * confidence_scale * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif effective_lane_state == 'SINGLE_LANE':
            mode = 'single_lane_commit' if commit_active else ('single_line_pre_avoid' if pre_avoid_active else ('single_line_avoid' if obstacle_active else 'single_lane'))
            steering += self.offlane_recovery_gain * lane_error_for_control
            steer_limit = self.commit_max_angular if commit_active else (self.avoid_max_angular_z if avoid_context_active else self.recovery_max_angular_z)
            steering = clamp(steering, -steer_limit, steer_limit)
            commit_steer_clamp_used = steer_limit
            confidence_scale = clamp(
                max(
                    0.45,
                    lane_confidence if lane_recent or predicted_active else self.recovery_confidence_threshold,
                ),
                0.0,
                1.0,
            )
            speed_limit = self.degraded_speed if lane_recent or predicted_active else self.lane_lost_speed
            speed = speed_limit * confidence_scale * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif self.is_recent(self.last_reliable_lane_stamp_ns, self.recent_lane_hold_ns, now_ns):
            mode = 'recovery_commit' if commit_active else ('recovery_pre_avoid' if pre_avoid_active else ('recovery_avoid' if obstacle_active else 'recovery'))
            steering = (
                self.lane_kp * self.last_reliable_lane_error
                + self.lane_heading_kp * self.last_reliable_lane_heading_error
                + 0.25 * self.offlane_recovery_gain * self.last_reliable_lane_error
                + corridor_weight_used * obstacle_bias
            )
            steer_limit = self.commit_max_angular if commit_active else (self.avoid_max_angular_z if avoid_context_active else self.recovery_max_angular_z)
            steering = clamp(steering, -steer_limit, steer_limit)
            commit_steer_clamp_used = steer_limit
            speed_limit = self.lane_lost_speed
            speed = self.lane_lost_speed * obstacle_speed_scale
            speed_limit *= curve_speed_scale
            speed *= curve_speed_scale
        elif self.is_recent(self.last_reliable_lane_stamp_ns, self.no_lane_memory_ns, now_ns) and not obstacle_unknown:
            mode = 'no_lane_commit' if commit_active else 'no_lane_memory'
            steering = (
                self.no_lane_memory_steer_gain
                * (
                    self.lane_kp * self.last_reliable_lane_error
                    + self.lane_heading_kp * self.last_reliable_lane_heading_error
                )
                + corridor_weight_used * obstacle_bias
            )
            steer_limit = min(self.commit_max_angular, self.max_steer_single_line) if commit_active else self.max_steer_single_line
            steering = clamp(steering, -steer_limit, steer_limit)
            commit_steer_clamp_used = steer_limit
            speed_limit = self.no_lane_memory_speed
            speed = self.no_lane_memory_speed * max(0.6, obstacle_speed_scale)
        else:
            neutral_limit = 0.10 if (self.lane_reacquisition_guard_active and avoid_context_active) else 0.20
            steering = corridor_weight_used * obstacle_bias
            steering = clamp(steering, -neutral_limit, neutral_limit)
            commit_steer_clamp_used = min(commit_steer_clamp_used, neutral_limit)
            speed_limit = self.no_lane_crawl_speed
            speed = self.no_lane_crawl_speed * obstacle_speed_scale
            if abs(obstacle_bias) < 0.05:
                speed = min(self.no_lane_crawl_speed, 0.04)

        if avoid_context_active:
            speed_limit *= obstacle_speed_guard_scale
            speed *= obstacle_speed_guard_scale
        if no_authoritative_side and authoritative_selected_gap in ('LEFT', 'RIGHT', 'CENTER_LEFT', 'CENTER_RIGHT'):
            speed_limit = min(speed_limit, self.base_speed * self.precommit_speed_scale)
            speed = min(speed, self.base_speed * self.precommit_speed_scale)

        single_line_low_conf = (
            effective_lane_state == 'SINGLE_LANE'
            and lane_confidence < self.single_line_conf_threshold
        )
        if single_line_low_conf:
            floor_speed = self.single_line_low_conf_speed
            if avoid_context_active:
                floor_speed = max(floor_speed, self.minimum_obstacle_pass_forward_speed)
            else:
                floor_speed = max(floor_speed, self.minimum_single_lane_forward_speed)
            speed_limit = max(floor_speed, min(speed_limit, floor_speed))
            speed = min(speed, speed_limit)
            steer_cap = self.max_steer_single_line if not avoid_context_active else max(
                self.max_steer_single_line,
                0.26,
            )
            steering = clamp(steering, -steer_cap, steer_cap)
            if obstacle_unknown:
                mode = 'single_line_unknown'
            elif commit_active:
                mode = 'single_lane_commit_low_conf'
            elif pre_avoid_active:
                mode = 'single_line_pre_avoid_low_conf'
            elif obstacle_active:
                mode = 'single_line_avoid_low_conf'
            else:
                mode = 'single_line_low_conf'

        low_conf_guard = (
            lane_recent
            and lane_confidence < self.low_conf_threshold
            and (predicted_active or lane_available)
        )
        if obstacle_unknown and low_conf_guard and not avoid_context_active:
            mode = 'unknown_low_conf'
            speed_limit = min(speed_limit, self.low_conf_degraded_speed)
            speed = min(speed, self.low_conf_degraded_speed)
            steering = clamp(steering, -self.max_steer_low_conf, self.max_steer_low_conf)
        elif obstacle_unknown and not avoid_context_active and not lane_available:
            mode = 'unknown_crawl'
            speed_limit = min(speed_limit, self.unknown_obstacle_speed)
            speed = min(speed, self.unknown_obstacle_speed)
            steering = clamp(steering, -self.max_steer_low_conf, self.max_steer_low_conf)

        minimum_forward_speed = 0.0
        if effective_lane_state == 'SINGLE_LANE':
            minimum_forward_speed = max(minimum_forward_speed, self.minimum_single_lane_forward_speed)
        if commit_active and not (blocked_center and blocked_selected_side):
            commit_floor_scale = self.no_lane_commit_scale if (
                self.lane_reacquisition_guard_active or no_lane_commit_guard
            ) else 1.0
            minimum_forward_speed = max(
                minimum_forward_speed,
                self.minimum_obstacle_pass_forward_speed * commit_floor_scale,
            )
        if minimum_forward_speed > 0.0 and not obstacle_unknown:
            speed_limit = max(speed_limit, minimum_forward_speed)
            speed = max(speed, minimum_forward_speed)

        if self.lane_reacquisition_guard_active:
            speed_limit = min(speed_limit, self.no_lane_crawl_speed)
            speed = min(speed, speed_limit)
            steering = clamp(steering, -0.08, 0.08)
            commit_steer_clamp_used = min(commit_steer_clamp_used, 0.08)
            if aggressive_steering_suppressed_reason == 'none':
                aggressive_steering_suppressed_reason = 'startup_lane_reacquisition_guard'
        elif no_lane_commit_guard:
            speed_limit = min(speed_limit, self.minimum_obstacle_pass_forward_speed)
            speed = min(speed, speed_limit)
            steering = clamp(steering, -0.16, 0.16)
            commit_steer_clamp_used = min(commit_steer_clamp_used, 0.16)

        lane_dominant_steer = lane_term + (waypoint_weight * heading_hint)
        if stale_commit_active:
            decay_keep = clamp(
                1.0 - (self.stale_commit_steer_decay_rate * self.control_period),
                0.0,
                1.0,
            )
            steering = lane_dominant_steer + decay_keep * (steering - lane_dominant_steer)
            corridor_term *= decay_keep
            avoid_term = corridor_term
            steering = clamp(steering, -self.stale_commit_max_steer, self.stale_commit_max_steer)
            commit_steer_clamp_used = min(commit_steer_clamp_used, self.stale_commit_max_steer)
            steering_decay_due_to_stale_commit = decay_keep
            final_steer_after_stale_decay = steering
            speed_limit = min(speed_limit, self.minimum_single_lane_forward_speed)
            speed = min(speed, speed_limit)
            aggressive_steering_suppressed_reason = 'stale_commit_decay'
        else:
            final_steer_after_stale_decay = steering

        smoothing_alpha = (
            self.avoid_steering_smoothing_alpha
            if avoid_context_active
            else (
                self.steering_smoothing_alpha
                if mode == 'nominal'
                else self.degraded_steering_smoothing_alpha
            )
        )
        smoothing_alpha = clamp(smoothing_alpha, 0.0, 1.0)
        previous_steering = self.last_steering_command
        steering = (
            smoothing_alpha * steering
            + (1.0 - smoothing_alpha) * previous_steering
        )
        steer_before_slew_limit = steering
        max_steer_delta = self.steer_slew_rate_limit * self.control_period
        steering = previous_steering + clamp(
            steer_before_slew_limit - previous_steering,
            -max_steer_delta,
            max_steer_delta,
        )
        steer_after_slew_limit = steering
        speed_after_curvature_scaling = speed
        if commit_active:
            steer_ratio = clamp(abs(steer_after_slew_limit) / max(self.commit_max_angular, 1e-3), 0.0, 1.0)
            commit_curvature_scale = 1.0 - (1.0 - self.commit_speed_scale) * steer_ratio
            speed_limit *= commit_curvature_scale
            speed *= commit_curvature_scale
            speed_after_curvature_scaling = speed
        elif pre_avoid_active:
            steer_ratio = clamp(abs(steer_after_slew_limit) / max(self.avoid_max_angular_z, 1e-3), 0.0, 1.0)
            pre_avoid_speed_scale = 1.0 - 0.20 * steer_ratio
            speed_limit *= pre_avoid_speed_scale
            speed *= pre_avoid_speed_scale
            speed_after_curvature_scaling = speed
        if commit_active and authoritative_pass_side in ('LEFT', 'RIGHT'):
            final_controller_mode = 'committed_side_pass'
        elif center_corridor_should_dominate:
            final_controller_mode = 'center_corridor'
        elif advisory_gap_active:
            final_controller_mode = 'advisory_side_gap'
        else:
            final_controller_mode = 'lane_center'
        self.last_steering_command = steering
        speed = clamp(speed, 0.0, speed_limit)

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = steering
        self.cmd_pub.publish(cmd)

        if speed <= 1e-3:
            if blocked_center and blocked_selected_side:
                self.last_stop_reason = 'blocked_stop'
            elif obstacle_unknown:
                self.last_stop_reason = 'unknown_guard'
            elif effective_lane_state == 'NO_LANE':
                self.last_stop_reason = 'no_lane_crawl'
            else:
                self.last_stop_reason = mode
        else:
            self.last_stop_reason = 'none'
        self.publish_debug_state(
            now_ns,
            effective_lane_state,
            speed,
            steering,
            self.last_stop_reason,
        )

        if now_ns - self.last_log_ns >= int(1e9):
            self.last_log_ns = now_ns
            self.get_logger().info(
                f'[FUSION] mode={mode} lane_state={effective_lane_state} lane_available={lane_available} lane_conf={lane_confidence:.2f} '
                f'lane_term={lane_term:+.3f} corridor_term={corridor_term:+.3f} avoid_term={avoid_term:+.3f} obstacle_bias={obstacle_bias:+.3f} '
                f'lane_weight_used={lane_weight_used:.2f} corridor_weight_used={corridor_weight_used:.2f} '
                f'obstacle_active={obstacle_active} obstacle_unknown={obstacle_unknown} '
                f'authoritative_pass_owner={self.authoritative_pass_owner} '
                f'authoritative_obstacle_latch_state={self.authoritative_obstacle_latch_state} '
                f'published_pass_side={self.obstacle_selected_pass_side} consumed_pass_side={authoritative_pass_side} '
                f'consumed_selected_gap={authoritative_selected_gap} consumed_commit_active={commit_active} '
                f'raw_commit_active={raw_commit_active} critical_commit_active={critical_commit_active} '
                f'questionable_critical_commit={questionable_critical_commit} false_emergency_state={false_emergency_state} '
                f'pass_state_validity_ok={pass_state_validity_ok_runtime} '
                f'zombie_commit_state_detected={self.zombie_commit_state_detected} '
                f'atomic_commit_state_clear_applied={self.atomic_commit_state_clear_applied} '
                f'critical_reject_forced_state_clear={self.critical_reject_forced_state_clear} '
                f'stale_published_side_ignored={stale_published_side_ignored} '
                f'stale_side_lock_ignored={stale_side_lock_ignored} '
                f'side_target_blocked_due_to_no_active_commit={side_target_blocked_due_to_no_active_commit} '
                f'consumed_commit_session_id={self.committed_commit_session_id} '
                f'consumed_locked_pass_side={self.committed_pass_side} '
                f'commit_source={self.obstacle_commit_source} '
                f'commit_remaining={self.obstacle_commit_remaining_sec:.2f} '
                f'commit_remaining_distance={self.obstacle_commit_remaining_distance_m:.2f} '
                f'commit_steer_clamp_used={commit_steer_clamp_used:.2f} '
                f'stale_commit_active={stale_commit_active} '
                f'steering_decay_due_to_stale_commit={steering_decay_due_to_stale_commit:.2f} '
                f'blocked_center={blocked_center} blocked_selected_side={blocked_selected_side} '
                f'lane_hard_constraint_active={self.lane_hard_constraint_active} '
                f'center_corridor_exists={self.center_corridor_exists} '
                f'center_corridor_preferred={self.center_corridor_preferred} '
                f'center_preferred_reason={self.center_preferred_reason} '
                f'false_critical_override_detected={self.false_critical_override_detected} '
                f'false_emergency_demoted={self.false_emergency_demoted} '
                f'emergency_latch_rejected_due_to_low_persistence={self.emergency_latch_rejected_due_to_low_persistence} '
                f'emergency_latch_rejected_due_to_center_corridor={self.emergency_latch_rejected_due_to_center_corridor} '
                f'critical_override_blocked_by_center_corridor={self.critical_override_blocked_by_center_corridor} '
                f'critical_trigger_consistent_with_tracked_geometry={self.critical_trigger_consistent_with_tracked_geometry} '
                f'center_corridor_override_priority_applied={self.center_corridor_override_priority_applied} '
                f'center_corridor_stabilizer_active={runtime_center_corridor_stabilizer_active} '
                f'lane_only_fallback_blocked={runtime_lane_only_fallback_blocked} '
                f'critical_commit_rejected_reason={self.critical_commit_rejected_reason} '
                f'critical_intrusion_persistence_cycles_used={self.critical_intrusion_persistence_cycles_used} '
                f'emergency_latch_kept_reason={self.emergency_latch_kept_reason} '
                f'lane_term_preserved_in_critical={self.lane_term_preserved_in_critical} '
                f'corridor_term_preserved_in_critical={self.corridor_term_preserved_in_critical} '
                f'side_commit_cancelled_due_to_valid_center_corridor={self.side_commit_cancelled_due_to_valid_center_corridor} '
                f'center_reject_strength={self.center_reject_strength:.2f} '
                f'center_reject_persistence={self.center_reject_persistence} '
                f'critical_dist={self.critical_dist:.2f} critical_points={self.critical_points} '
                f'footprint_intrusion={self.footprint_intrusion:.2f} '
                f'tracked_local_x={self.tracked_local_x:.2f} tracked_local_y={self.tracked_local_y:+.2f} '
                f'center_corridor_stabilizer_term={center_corridor_stabilizer_term:+.3f} '
                f'lane_guard={lane_guard_ratio:.2f} obstacle_guard={obstacle_speed_guard_scale:.2f} '
                f'preempt_ratio={obstacle_preempt_ratio:.2f} '
                f'obstacle_opposes_lane={obstacle_bias_opposes_lane} '
                f'local_pass_inference_disabled={self.local_pass_inference_disabled} '
                f'corridor_fusion_reason={self.authoritative_corridor_reason if avoid_context_active else "lane_only"} '
                f'advisory_gap_active={advisory_gap_active} advisory_gap_reason={advisory_gap_reason} '
                f'advisory_side_gap_strength={advisory_side_gap_strength:.2f} '
                f'side_gap_suppressed_due_to_no_commit={self.side_gap_suppressed_due_to_no_commit} '
                f'side_target_suppressed_reason={self.side_target_suppressed_reason} '
                f'target_clipped_to_lane_bounds={self.target_clipped_to_lane_bounds} '
                f'target_clip_reason={self.target_clip_reason} '
                f'final_controller_mode={final_controller_mode} '
                f'ignored_transient_none_due_to_commit_lock={self.ignored_transient_none_due_to_commit_lock} '
                f'final_steer_source_side={authoritative_pass_side} '
                f'final_steer_after_stale_decay={final_steer_after_stale_decay:+.3f} '
                f'side_source_reason={self.current_side_source_reason} '
                f'lane_reacquisition_guard_active={self.lane_reacquisition_guard_active} '
                f'lane_hard_constraint_active_runtime={runtime_lane_hard_constraint_active} '
                f'aggressive_steering_suppressed_reason={aggressive_steering_suppressed_reason} '
                f'steer_before_slew_limit={steer_before_slew_limit:+.3f} '
                f'steer_after_slew_limit={steer_after_slew_limit:+.3f} '
                f'heading_hint={heading_hint:+.3f} obstacle_progress={obstacle_progress:.2f} waypoint_progress={waypoint_progress:.2f} '
                f'speed_after_curvature_scaling={speed_after_curvature_scaling:.2f} '
                f'curve_scale={curve_speed_scale:.2f} '
                f'speed={speed:.2f} steer={steering:+.3f} stop_reason={self.last_stop_reason}'
            )


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = CmdFusionNode()
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
