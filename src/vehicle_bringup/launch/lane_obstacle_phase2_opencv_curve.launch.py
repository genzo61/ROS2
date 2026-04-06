from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
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
                        'corridor_follow_gain': 0.82,
                        'pointcloud_self_filter_forward_m': 0.28,
                        'vehicle_half_width_m': 0.25,
                        'avoidance_clearance_margin_m': 0.18,
                        'lane_corridor_cap': 0.45,
                        'duba_center_trigger_m': 1.70,
                        'duba_center_escape_y': 0.12,
                        'critical_roi_forward_min_m': 0.30,
                        'critical_roi_forward_max_m': 1.50,
                        'critical_roi_half_width_m': 0.78,
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
                        'duba_algilama_mesafesi': 1.60,
                        'duba_y_sinir': 0.46,
                        'duba_min_z': -0.35,
                        'duba_max_z': 0.70,
                        'duba_min_nokta': 7,
                        'obstacle_corridor_weight': 0.60,
                        'lane_weight_during_avoid': 0.22,
                        'avoid_bias_gain': 1.65,
                        'avoid_bias_limit': 0.56,
                        'obstacle_hold_time_sec': 0.55,
                        'return_to_center_sec': 0.65,
                        'return_to_center_decay': 0.60,
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
                        'obstacle_weight_during_avoid': 1.24,
                        'lane_weight_during_avoid': 0.22,
                        'waypoint_weight_with_lane': 0.0,
                        'waypoint_weight_no_lane': 0.0,
                        'max_angular_z': 0.50,
                        'recovery_max_angular_z': 0.34,
                        'avoid_max_angular_z': 1.05,
                        'steering_smoothing_alpha': 0.28,
                        'degraded_steering_smoothing_alpha': 0.18,
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
        sim_launch,
        lane_tracker_node,
        obstacle_node,
        fusion_node,
    ])
