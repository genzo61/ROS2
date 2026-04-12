from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    lane_config = os.path.join(pkg_share, 'config', 'lane_tracker.yaml')
    lane_yolo_config = os.path.join(pkg_share, 'config', 'lane_yolo.yaml')
    igvc_config = os.path.join(pkg_share, 'config', 'igvc_mission.yaml')

    enable_lane_yolo = LaunchConfiguration('enable_lane_yolo')
    enable_waypoint_hint = LaunchConfiguration('enable_waypoint_hint')
    lane_model_path = LaunchConfiguration('lane_model_path')
    allow_lane_yolo_fallback = LaunchConfiguration('allow_lane_yolo_fallback')
    base_speed = LaunchConfiguration('base_speed')
    waypoint_source = LaunchConfiguration('waypoint_source')
    waypoint_weight_with_lane = LaunchConfiguration('waypoint_weight_with_lane')
    waypoint_weight_no_lane = LaunchConfiguration('waypoint_weight_no_lane')
    route_weight_normal_lane = LaunchConfiguration('route_weight_normal_lane')
    route_weight_single_lane = LaunchConfiguration('route_weight_single_lane')
    route_weight_no_lane = LaunchConfiguration('route_weight_no_lane')
    route_weight_pre_avoid = LaunchConfiguration('route_weight_pre_avoid')
    route_weight_committed_pass = LaunchConfiguration('route_weight_committed_pass')
    route_weight_blocked = LaunchConfiguration('route_weight_blocked')
    center_corridor_route_bias_cap = LaunchConfiguration('center_corridor_route_bias_cap')
    route_term_lane_clip_margin = LaunchConfiguration('route_term_lane_clip_margin')
    heading_hint_lowpass_alpha = LaunchConfiguration('heading_hint_lowpass_alpha')
    waypoint_arrival_distance = LaunchConfiguration('waypoint_arrival_distance')
    route_suppression_opposition_threshold = LaunchConfiguration('route_suppression_opposition_threshold')

    lane_yolo_enabled_condition = IfCondition(
        PythonExpression(["'", enable_lane_yolo, "'.lower() == 'true'"])
    )
    classic_lane_condition = UnlessCondition(
        PythonExpression(["'", enable_lane_yolo, "'.lower() == 'true'"])
    )
    waypoint_hint_condition = IfCondition(
        PythonExpression(["'", enable_waypoint_hint, "'.lower() == 'true'"])
    )

    lane_tracker_node = TimerAction(
        period=1.0,
        condition=classic_lane_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='lane_tracker',
                name='lane_tracker',
                output='screen',
                parameters=[lane_config, {'use_sim_time': True}],
            )
        ],
    )

    lane_camera_subscriber_node = TimerAction(
        period=1.0,
        condition=lane_yolo_enabled_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='lane_camera_subscriber',
                name='lane_camera_subscriber',
                output='screen',
                parameters=[lane_yolo_config, {'use_sim_time': True}],
            )
        ],
    )

    lane_yolo_inference_node = TimerAction(
        period=2.0,
        condition=lane_yolo_enabled_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='lane_yolo_inference',
                name='lane_yolo_inference',
                output='screen',
                parameters=[
                    lane_yolo_config,
                    {
                        'use_sim_time': True,
                        'model_path': ParameterValue(lane_model_path, value_type=str),
                        'allow_fallback': ParameterValue(allow_lane_yolo_fallback, value_type=bool),
                    },
                ],
            )
        ],
    )

    lane_detection_parser_node = TimerAction(
        period=3.0,
        condition=lane_yolo_enabled_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='lane_detection_parser',
                name='lane_detection_parser',
                output='screen',
                parameters=[lane_yolo_config, {'use_sim_time': True}],
            )
        ],
    )

    obstacle_provider_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='yaris_pilotu',
                name='yaris_pilotu',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'publish_cmd_vel': False,
                        'route_enabled': False,
                        'point_cloud_topic': '/points',
                        'point_cloud_alt_topic': '',
                        'pointcloud_self_filter_forward_m': 0.28,
                        'duba_center_trigger_m': 1.45,
                        'critical_roi_forward_min_m': 0.30,
                        'critical_center_ratio_min': 0.36,
                        'critical_center_dominance_min': 0.92,
                        'critical_roi_forward_max_m': 2.60,
                        'critical_roi_half_width_m': 0.92,
                        'critical_roi_min_points': 5,
                        'duba_algilama_mesafesi': 2.80,
                        'duba_y_sinir': 0.62,
                        'duba_min_z': -0.35,
                        'duba_max_z': 0.70,
                        'duba_min_nokta': 8,
                        'obstacle_corridor_weight': 0.38,
                        'avoid_bias_gain': 1.20,
                        'avoid_bias_limit': 0.36,
                        'obstacle_hold_time_sec': 1.00,
                        'critical_avoid_target_limit': 0.34,
                        'critical_avoid_min_turn': 0.22,
                        'critical_avoid_ramp_alpha': 0.24,
                        'return_to_center_sec': 1.35,
                        'return_to_center_decay': 0.58,
                        'post_avoid_straight_distance_m': 1.35,
                        'post_avoid_hold_sec': 2.80,
                        'post_avoid_lane_weight': 0.18,
                        'post_avoid_corridor_weight': 0.62,
                        'duba_pass_freeze_distance_m': 0.60,
                        'duba_pass_freeze_lateral_m': 0.08,
                        'duba_pass_hold_sec': 0.50,
                        'pre_avoid_trigger_m': 2.20,
                        'near_avoid_trigger_m': 1.25,
                        'emergency_avoid_trigger_m': 0.72,
                        'obstacle_release_distance_m': 2.10,
                        'obstacle_latch_hold_sec': 0.75,
                        'obstacle_preempt_intrusion_m': 0.04,
                        'obstacle_preempt_center_ratio': 0.18,
                        'pre_avoid_min_offset_m': 0.08,
                        'pre_avoid_max_offset_m': 0.16,
                        'pre_avoid_lane_weight': 0.92,
                        'pre_avoid_corridor_blend': 0.58,
                        'pre_avoid_speed_scale_far': 0.94,
                        'pre_avoid_speed_scale_near': 0.78,
                        'pre_avoid_speed_scale_emergency': 0.42,
                        'center_gap_penalty_gain': 1.55,
                        'center_gap_penalty_max': 2.10,
                        'duba_preempt_max_age_sec': 0.22,
                        'stale_obstacle_release_sec': 0.16,
                        'avoid_bias_lane_attenuation': 0.68,
                        'avoid_corridor_limit_degraded': 0.20,
                        'tracked_obstacle_persist_sec': 1.30,
                        'tracked_obstacle_match_distance_m': 0.90,
                        'tracked_obstacle_lateral_gate_m': 0.90,
                        'avoid_pass_longitudinal_margin_m': 0.60,
                        'avoid_pass_lateral_clearance_m': 0.16,
                        'force_odom_pass_latch': True,
                        'avoid_pass_min_progress_m': 1.50,
                        'avoid_pass_max_hold_sec': 6.00,
                        'tracked_memory_require_strong_source': True,
                        'single_lane_transition_frames': 3,
                        'no_lane_transition_frames': 6,
                        'blocked_persistence_sec': 0.45,
                    }
                ],
            )
        ],
    )

    waypoint_hint_node = TimerAction(
        period=2.5,
        condition=waypoint_hint_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='igvc_waypoint_navigator',
                name='igvc_waypoint_navigator',
                output='screen',
                parameters=[
                    igvc_config,
                    {
                        'use_sim_time': True,
                        'waypoint_source': ParameterValue(waypoint_source, value_type=str),
                        'waypoint_arrival_distance': ParameterValue(
                            waypoint_arrival_distance,
                            value_type=float,
                        ),
                    },
                ],
            )
        ],
    )

    fusion_node = TimerAction(
        period=3.5,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='cmd_fusion_node',
                name='cmd_fusion_node',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'base_speed': ParameterValue(base_speed, value_type=float),
                        'degraded_speed': 0.20,
                        'lane_lost_speed': 0.12,
                        'no_lane_crawl_speed': 0.08,
                        'lane_kp': 0.95,
                        'lane_heading_kp': 0.60,
                        'obstacle_weight': 0.25,
                        'obstacle_weight_during_avoid': 1.08,
                        'lane_weight_during_avoid': 0.48,
                        'avoid_lane_weight_scale': 0.40,
                        'single_line_avoid_lane_weight_scale': 0.28,
                        'avoid_obstacle_weight_scale': 1.08,
                        'single_line_avoid_obstacle_weight_scale': 1.12,
                        'waypoint_weight_with_lane': ParameterValue(
                            waypoint_weight_with_lane,
                            value_type=float,
                        ),
                        'waypoint_weight_no_lane': ParameterValue(
                            waypoint_weight_no_lane,
                            value_type=float,
                        ),
                        'route_weight_normal_lane': ParameterValue(
                            route_weight_normal_lane,
                            value_type=float,
                        ),
                        'route_weight_single_lane': ParameterValue(
                            route_weight_single_lane,
                            value_type=float,
                        ),
                        'route_weight_no_lane': ParameterValue(
                            route_weight_no_lane,
                            value_type=float,
                        ),
                        'route_weight_pre_avoid': ParameterValue(
                            route_weight_pre_avoid,
                            value_type=float,
                        ),
                        'route_weight_committed_pass': ParameterValue(
                            route_weight_committed_pass,
                            value_type=float,
                        ),
                        'route_weight_blocked': ParameterValue(
                            route_weight_blocked,
                            value_type=float,
                        ),
                        'center_corridor_route_bias_cap': ParameterValue(
                            center_corridor_route_bias_cap,
                            value_type=float,
                        ),
                        'route_term_lane_clip_margin': ParameterValue(
                            route_term_lane_clip_margin,
                            value_type=float,
                        ),
                        'heading_hint_lowpass_alpha': ParameterValue(
                            heading_hint_lowpass_alpha,
                            value_type=float,
                        ),
                        'waypoint_arrival_distance': ParameterValue(
                            waypoint_arrival_distance,
                            value_type=float,
                        ),
                        'route_suppression_opposition_threshold': ParameterValue(
                            route_suppression_opposition_threshold,
                            value_type=float,
                        ),
                        'max_angular_z': 0.55,
                        'recovery_max_angular_z': 0.32,
                        'avoid_max_angular_z': 0.66,
                        'steering_smoothing_alpha': 0.32,
                        'degraded_steering_smoothing_alpha': 0.22,
                        'avoid_steering_smoothing_alpha': 0.36,
                        'nominal_confidence_threshold': 0.60,
                        'recovery_confidence_threshold': 0.26,
                        'low_conf_threshold': 0.42,
                        'low_conf_degraded_speed': 0.08,
                        'unknown_obstacle_speed': 0.08,
                        'max_steer_low_conf': 0.24,
                        'single_line_conf_threshold': 0.55,
                        'single_line_low_conf_speed': 0.05,
                        'max_steer_single_line': 0.20,
                        'offlane_error_threshold': 0.16,
                        'offlane_recovery_gain': 0.35,
                        'recent_lane_hold_sec': 1.0,
                        'no_lane_memory_sec': 2.2,
                        'no_lane_memory_speed': 0.08,
                        'no_lane_memory_steer_gain': 0.65,
                        'obstacle_lane_guard_error_threshold': 0.10,
                        'obstacle_lane_guard_full_error': 0.24,
                        'obstacle_lane_guard_min_scale': 0.22,
                        'obstacle_lane_priority_gain': 0.18,
                        'obstacle_lane_opposite_weight_drop': 0.22,
                        'obstacle_bias_speed_reduction_gain': 0.12,
                        'obstacle_lane_guard_speed_reduction_gain': 0.10,
                        'obstacle_speed_min_scale': 0.62,
                        'obstacle_speed_full_bias_abs': 0.22,
                        'obstacle_preempt_speed_scale_threshold': 0.82,
                        'obstacle_preempt_bias_abs': 0.08,
                        'obstacle_preempt_lane_weight_min': 0.12,
                        'obstacle_preempt_obstacle_weight_gain': 0.28,
                        'obstacle_preempt_opposite_drop_relief': 0.85,
                        'lane_boundary_timeout_sec': 0.90,
                        'single_lane_transition_frames': 3,
                        'no_lane_transition_frames': 6,
                        'avoidance_commit_duration_sec': 0.95,
                        'minimum_single_lane_forward_speed': 0.18,
                        'minimum_obstacle_pass_forward_speed': 0.24,
                        'single_lane_memory_blend': 0.35,
                    }
                ],
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_lane_yolo', default_value='false'),
        DeclareLaunchArgument('enable_waypoint_hint', default_value='true'),
        DeclareLaunchArgument('lane_model_path', default_value='auto'),
        DeclareLaunchArgument('allow_lane_yolo_fallback', default_value='false'),
        DeclareLaunchArgument('base_speed', default_value='0.35'),
        DeclareLaunchArgument(
            'waypoint_source',
            default_value='auto',
            description='auto | map | gps waypoint source for RTK heading hints.',
        ),
        DeclareLaunchArgument(
            'waypoint_weight_with_lane',
            default_value='0.04',
            description='Legacy alias kept for older launch commands; route_weight_normal_lane is authoritative.',
        ),
        DeclareLaunchArgument(
            'waypoint_weight_no_lane',
            default_value='0.22',
            description='Legacy alias kept for older launch commands; route_weight_no_lane is authoritative.',
        ),
        DeclareLaunchArgument(
            'route_weight_normal_lane',
            default_value='0.03',
            description='Tiny advisory route bias during strong lane following.',
        ),
        DeclareLaunchArgument(
            'route_weight_single_lane',
            default_value='0.10',
            description='Moderate-low route bias while lane geometry is degraded.',
        ),
        DeclareLaunchArgument(
            'route_weight_no_lane',
            default_value='0.22',
            description='Route bias during no-lane recovery when local geometry is safe.',
        ),
        DeclareLaunchArgument(
            'route_weight_pre_avoid',
            default_value='0.02',
            description='Near-zero route bias while obstacle pre-avoidance is active.',
        ),
        DeclareLaunchArgument(
            'route_weight_committed_pass',
            default_value='0.0',
            description='Route disabled during committed side passes.',
        ),
        DeclareLaunchArgument(
            'route_weight_blocked',
            default_value='0.0',
            description='Route disabled during blocked or critical safety states.',
        ),
        DeclareLaunchArgument(
            'center_corridor_route_bias_cap',
            default_value='0.015',
            description='Maximum route steering bias allowed inside a preferred center corridor.',
        ),
        DeclareLaunchArgument(
            'route_term_lane_clip_margin',
            default_value='0.02',
            description='Lane-bound hard clip for route-derived steering contribution.',
        ),
        DeclareLaunchArgument(
            'heading_hint_lowpass_alpha',
            default_value='0.35',
            description='Low-pass alpha for advisory route heading hints.',
        ),
        DeclareLaunchArgument(
            'waypoint_arrival_distance',
            default_value='0.8',
            description='Distance threshold for treating the current waypoint as reached.',
        ),
        DeclareLaunchArgument(
            'route_suppression_opposition_threshold',
            default_value='0.03',
            description='Minimum local term magnitude before opposing route bias is suppressed.',
        ),
        lane_tracker_node,
        lane_camera_subscriber_node,
        lane_yolo_inference_node,
        lane_detection_parser_node,
        obstacle_provider_node,
        waypoint_hint_node,
        fusion_node,
    ])
