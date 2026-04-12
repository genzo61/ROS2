from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    lane_config = os.path.join(pkg_share, 'config', 'lane_tracker_phase1_curve.yaml')
    world_file = os.path.join(pkg_share, 'worlds', 'lane_curve_obstacles.world')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    cleanup_stale_gazebo = LaunchConfiguration('cleanup_stale_gazebo')
    base_speed = LaunchConfiguration('base_speed')
    enable_waypoint_hint = LaunchConfiguration('enable_waypoint_hint')

    waypoint_hint_condition = IfCondition(enable_waypoint_hint)

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'stable_sim.launch.py')),
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'world': world_file,
            'use_local_ekf': use_local_ekf,
            'cleanup_stale_gazebo': cleanup_stale_gazebo,
        }.items(),
    )

    lane_tracker_node = TimerAction(
        period=1.0,
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

    obstacle_node = TimerAction(
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
                        'point_cloud_alt_topic': '/front_depth_camera/points',
                        'recover_duration': 0.85,
                        'gap_switch_margin': 0.12,
                        'continuity_bonus': 0.20,
                        'corridor_follow_gain': 0.90,
                        'pointcloud_self_filter_forward_m': 0.28,
                        'vehicle_half_width_m': 0.25,
                        'avoidance_clearance_margin_m': 0.18,
                        'lane_corridor_cap': 0.45,
                        'duba_center_trigger_m': 1.70,
                        'duba_center_escape_y': 0.12,
                        'critical_roi_forward_min_m': 0.30,
                        'critical_roi_forward_max_m': 2.80,
                        'critical_roi_half_width_m': 0.98,
                        'critical_roi_min_points': 5,
                        'critical_center_ratio_min': 0.18,
                        'critical_center_dominance_min': 0.60,
                        'critical_commit_sec': 0.78,
                        'critical_release_forward_margin_m': 0.32,
                        'critical_release_lateral_margin_m': 0.16,
                        'critical_escape_offset_m': 0.60,
                        'critical_avoid_gain': 2.05,
                        'critical_avoid_target_limit': 0.68,
                        'critical_avoid_min_turn': 0.34,
                        'critical_avoid_ramp_alpha': 0.70,
                        'duba_algilama_mesafesi': 3.00,
                        'duba_y_sinir': 0.64,
                        'duba_min_z': -0.35,
                        'duba_max_z': 0.70,
                        'duba_min_nokta': 7,
                        'obstacle_corridor_weight': 0.60,
                        'lane_weight_during_avoid': 0.28,
                        'avoid_bias_gain': 1.65,
                        'avoid_bias_limit': 0.48,
                        'obstacle_hold_time_sec': 0.55,
                        'return_to_center_sec': 0.80,
                        'return_to_center_decay': 0.52,
                        'post_avoid_straight_distance_m': 1.45,
                        'post_avoid_hold_sec': 3.00,
                        'post_avoid_lane_weight': 0.20,
                        'post_avoid_corridor_weight': 0.66,
                        'duba_pass_freeze_distance_m': 0.72,
                        'duba_pass_freeze_lateral_m': 0.08,
                        'duba_pass_hold_sec': 0.70,
                        'close_side_avoid_distance_m': 1.85,
                        'close_side_avoid_full_distance_m': 0.88,
                        'close_side_avoid_lateral_m': 0.06,
                        'close_side_avoid_min_offset_m': 0.16,
                        'close_side_avoid_offset_m': 0.30,
                        'close_side_avoid_speed_mps': 0.32,
                        'close_side_avoid_lane_weight_min': 0.18,
                        'pre_avoid_trigger_m': 2.40,
                        'near_avoid_trigger_m': 1.30,
                        'emergency_avoid_trigger_m': 0.74,
                        'obstacle_release_distance_m': 2.25,
                        'obstacle_latch_hold_sec': 0.75,
                        'obstacle_preempt_intrusion_m': 0.04,
                        'obstacle_preempt_center_ratio': 0.16,
                        'pre_avoid_min_offset_m': 0.10,
                        'pre_avoid_max_offset_m': 0.18,
                        'pre_avoid_lane_weight': 0.88,
                        'pre_avoid_corridor_blend': 0.70,
                        'pre_avoid_speed_scale_far': 0.92,
                        'pre_avoid_speed_scale_near': 0.76,
                        'pre_avoid_speed_scale_emergency': 0.40,
                        'center_gap_penalty_gain': 1.70,
                        'center_gap_penalty_max': 2.20,
                        'duba_preempt_max_age_sec': 0.22,
                        'stale_obstacle_release_sec': 0.16,
                        'avoid_bias_lane_attenuation': 0.62,
                        'avoid_corridor_limit_degraded': 0.20,
                        'tracked_obstacle_persist_sec': 1.40,
                        'tracked_obstacle_match_distance_m': 0.95,
                        'tracked_obstacle_lateral_gate_m': 0.95,
                        'avoid_pass_longitudinal_margin_m': 0.65,
                        'avoid_pass_lateral_clearance_m': 0.16,
                        'force_odom_pass_latch': True,
                        'avoid_pass_min_progress_m': 1.55,
                        'avoid_pass_max_hold_sec': 6.50,
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
        period=2.6,
        condition=waypoint_hint_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='igvc_waypoint_navigator',
                name='igvc_waypoint_navigator',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'waypoint_source': 'map',
                        'global_frame': 'map',
                        'base_frame': 'base_footprint',
                        'publish_hz': 10.0,
                        'skip_waypoint_distance_m': 0.9,
                        'waypoint_arrival_distance': 0.8,
                        'max_heading_error_rad': 1.2,
                        # Centerline of lane_curve_obstacles.world, derived from lane marker geometry.
                        'map_waypoints': [
                            -6.5, 0.0,
                            -4.0, 0.0,
                            -2.0, 0.0,
                            -0.8, 0.15,
                             0.3, 0.55,
                             1.2, 1.25,
                             2.0, 2.10,
                             2.55, 3.20,
                             2.90, 4.50,
                             3.0, 6.80,
                             3.0, 9.40,
                             3.0, 12.00,
                             3.0, 14.50,
                             3.0, 16.50,
                        ],
                    }
                ],
            )
        ],
    )

    fusion_node = TimerAction(
        period=3.2,
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
                        'degraded_speed': 0.58,
                        'lane_lost_speed': 0.24,
                        'no_lane_crawl_speed': 0.14,
                        'min_conf_speed_scale': 0.82,
                        'lane_kp': 0.64,
                        'lane_heading_kp': 0.40,
                        'obstacle_weight': 0.34,
                        'obstacle_weight_during_avoid': 1.16,
                        'lane_weight_during_avoid': 0.30,
                        'avoid_lane_weight_scale': 0.62,
                        'single_line_avoid_lane_weight_scale': 0.48,
                        'avoid_obstacle_weight_scale': 1.08,
                        'single_line_avoid_obstacle_weight_scale': 1.12,
                        'waypoint_weight_with_lane': 0.0,
                        'waypoint_weight_no_lane': 0.0,
                        'route_weight_normal_lane': 0.03,
                        'route_weight_center_corridor': 0.015,
                        'route_weight_single_lane': 0.10,
                        'route_weight_no_lane': 0.18,
                        'route_weight_pre_avoid': 0.02,
                        'route_weight_committed_pass': 0.0,
                        'route_weight_blocked': 0.0,
                        'center_corridor_route_bias_cap': 0.015,
                        'route_term_lane_clip_margin': 0.02,
                        'heading_hint_lowpass_alpha': 0.35,
                        'waypoint_arrival_distance': 0.8,
                        'route_suppression_opposition_threshold': 0.03,
                        'max_angular_z': 0.50,
                        'recovery_max_angular_z': 0.34,
                        'avoid_max_angular_z': 0.88,
                        'steering_smoothing_alpha': 0.28,
                        'degraded_steering_smoothing_alpha': 0.22,
                        'avoid_steering_smoothing_alpha': 0.40,
                        'nominal_confidence_threshold': 0.68,
                        'recovery_confidence_threshold': 0.32,
                        'low_conf_threshold': 0.28,
                        'low_conf_degraded_speed': 0.24,
                        'unknown_obstacle_speed': 0.18,
                        'max_steer_low_conf': 0.18,
                        'single_line_conf_threshold': 0.60,
                        'single_line_low_conf_speed': 0.22,
                        'max_steer_single_line': 0.16,
                        'offlane_error_threshold': 0.07,
                        'offlane_recovery_gain': 0.22,
                        'recent_lane_hold_sec': 1.1,
                        'no_lane_memory_sec': 1.0,
                        'no_lane_memory_speed': 0.16,
                        'no_lane_memory_steer_gain': 0.32,
                        'curve_speed_reduction_max': 0.36,
                        'curve_heading_full_scale': 0.22,
                        'curve_error_full_scale': 0.24,
                        'obstacle_lane_guard_error_threshold': 0.09,
                        'obstacle_lane_guard_full_error': 0.22,
                        'obstacle_lane_guard_min_scale': 0.24,
                        'obstacle_lane_priority_gain': 0.22,
                        'obstacle_lane_opposite_weight_drop': 0.24,
                        'obstacle_bias_speed_reduction_gain': 0.12,
                        'obstacle_lane_guard_speed_reduction_gain': 0.10,
                        'obstacle_speed_min_scale': 0.60,
                        'obstacle_speed_full_bias_abs': 0.24,
                        'obstacle_preempt_speed_scale_threshold': 0.80,
                        'obstacle_preempt_bias_abs': 0.08,
                        'obstacle_preempt_lane_weight_min': 0.10,
                        'obstacle_preempt_obstacle_weight_gain': 0.28,
                        'obstacle_preempt_opposite_drop_relief': 0.88,
                        'lane_boundary_timeout_sec': 0.90,
                        'single_lane_transition_frames': 3,
                        'no_lane_transition_frames': 6,
                        'avoidance_commit_duration_sec': 0.95,
                        'minimum_single_lane_forward_speed': 0.22,
                        'minimum_obstacle_pass_forward_speed': 0.30,
                        'single_lane_memory_blend': 0.35,
                    }
                ],
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-8.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.02'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_local_ekf', default_value='false'),
        DeclareLaunchArgument('cleanup_stale_gazebo', default_value='true'),
        DeclareLaunchArgument('base_speed', default_value='1.05'),
        DeclareLaunchArgument('enable_waypoint_hint', default_value='true'),
        sim_launch,
        lane_tracker_node,
        obstacle_node,
        waypoint_hint_node,
        fusion_node,
    ])
