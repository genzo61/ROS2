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
    world_file = os.path.join(pkg_share, 'worlds', 'lane_curve_short.world')

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

    fusion_node = TimerAction(
        period=3.0,
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
                        'lane_lost_speed': 0.28,
                        'no_lane_crawl_speed': 0.14,
                        'min_conf_speed_scale': 0.88,
                        'lane_kp': 0.72,
                        'lane_heading_kp': 0.48,
                        'obstacle_weight': 0.0,
                        'obstacle_weight_during_avoid': 0.0,
                        'lane_weight_during_avoid': 1.0,
                        'waypoint_weight_with_lane': 0.0,
                        'waypoint_weight_no_lane': 0.0,
                        'max_angular_z': 0.56,
                        'recovery_max_angular_z': 0.40,
                        'avoid_max_angular_z': 0.56,
                        'steering_smoothing_alpha': 0.30,
                        'degraded_steering_smoothing_alpha': 0.22,
                        'nominal_confidence_threshold': 0.66,
                        'recovery_confidence_threshold': 0.30,
                        'low_conf_threshold': 0.28,
                        'low_conf_degraded_speed': 0.24,
                        'unknown_obstacle_speed': 0.14,
                        'max_steer_low_conf': 0.18,
                        'single_line_conf_threshold': 0.58,
                        'single_line_low_conf_speed': 0.22,
                        'max_steer_single_line': 0.18,
                        'offlane_error_threshold': 0.07,
                        'offlane_recovery_gain': 0.30,
                        'recent_lane_hold_sec': 1.0,
                        'no_lane_memory_sec': 1.4,
                        'no_lane_memory_speed': 0.18,
                        'no_lane_memory_steer_gain': 0.40,
                        'curve_speed_reduction_max': 0.32,
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
        DeclareLaunchArgument('base_speed', default_value='1.20'),
        sim_launch,
        lane_tracker_node,
        fusion_node,
    ])
