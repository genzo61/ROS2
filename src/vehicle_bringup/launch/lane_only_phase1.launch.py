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
    lane_yolo_config = os.path.join(pkg_share, 'config', 'lane_yolo_phase1.yaml')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    cleanup_stale_gazebo = LaunchConfiguration('cleanup_stale_gazebo')
    lane_model_path = LaunchConfiguration('lane_model_path')
    allow_lane_yolo_fallback = LaunchConfiguration('allow_lane_yolo_fallback')
    base_speed = LaunchConfiguration('base_speed')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'lane_only_sim.launch.py')),
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'use_local_ekf': use_local_ekf,
            'cleanup_stale_gazebo': cleanup_stale_gazebo,
        }.items(),
    )

    lane_camera_subscriber_node = TimerAction(
        period=1.0,
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
                        'degraded_speed': 0.12,
                        'lane_lost_speed': 0.08,
                        'no_lane_crawl_speed': 0.04,
                        'min_conf_speed_scale': 0.85,
                        'lane_kp': 0.62,
                        'lane_heading_kp': 0.26,
                        'obstacle_weight': 0.0,
                        'obstacle_weight_during_avoid': 0.0,
                        'lane_weight_during_avoid': 1.0,
                        'waypoint_weight_with_lane': 0.0,
                        'waypoint_weight_no_lane': 0.0,
                        'max_angular_z': 0.30,
                        'recovery_max_angular_z': 0.18,
                        'avoid_max_angular_z': 0.30,
                        'steering_smoothing_alpha': 0.16,
                        'degraded_steering_smoothing_alpha': 0.10,
                        'nominal_confidence_threshold': 0.52,
                        'recovery_confidence_threshold': 0.30,
                        'low_conf_threshold': 0.44,
                        'low_conf_degraded_speed': 0.05,
                        'unknown_obstacle_speed': 0.04,
                        'max_steer_low_conf': 0.12,
                        'single_line_conf_threshold': 0.52,
                        'single_line_low_conf_speed': 0.03,
                        'max_steer_single_line': 0.10,
                        'offlane_error_threshold': 0.10,
                        'offlane_recovery_gain': 0.18,
                        'recent_lane_hold_sec': 0.8,
                        'no_lane_memory_sec': 1.0,
                        'no_lane_memory_speed': 0.05,
                        'no_lane_memory_steer_gain': 0.35,
                    }
                ],
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-6.5'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.02'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_local_ekf', default_value='false'),
        DeclareLaunchArgument('cleanup_stale_gazebo', default_value='true'),
        DeclareLaunchArgument('lane_model_path', default_value='auto'),
        DeclareLaunchArgument(
            'allow_lane_yolo_fallback',
            default_value='false',
            description='Keep false in phase-1 so the run fails if YOLO is unavailable.',
        ),
        DeclareLaunchArgument('base_speed', default_value='0.22'),
        sim_launch,
        lane_camera_subscriber_node,
        lane_yolo_inference_node,
        lane_detection_parser_node,
        fusion_node,
    ])
