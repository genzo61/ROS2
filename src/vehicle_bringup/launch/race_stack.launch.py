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
                        'critical_roi_forward_min_m': 0.30,
                        'critical_center_ratio_min': 0.36,
                        'critical_center_dominance_min': 0.92,
                        'critical_roi_forward_max_m': 1.80,
                        'critical_roi_half_width_m': 0.70,
                        'critical_roi_min_points': 5,
                        'duba_algilama_mesafesi': 1.80,
                        'duba_y_sinir': 0.45,
                        'duba_min_z': -0.35,
                        'duba_max_z': 0.70,
                        'duba_min_nokta': 8,
                        'obstacle_corridor_weight': 0.32,
                        'avoid_bias_gain': 1.20,
                        'avoid_bias_limit': 0.40,
                        'obstacle_hold_time_sec': 0.90,
                        'critical_avoid_target_limit': 0.34,
                        'critical_avoid_min_turn': 0.22,
                        'critical_avoid_ramp_alpha': 0.32,
                        'return_to_center_sec': 1.10,
                        'return_to_center_decay': 0.72,
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
                parameters=[igvc_config, {'use_sim_time': True}],
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
                        'obstacle_weight_during_avoid': 1.15,
                        'lane_weight_during_avoid': 0.50,
                        'waypoint_weight_with_lane': 0.0,
                        'waypoint_weight_no_lane': 0.0,
                        'max_angular_z': 0.55,
                        'recovery_max_angular_z': 0.32,
                        'avoid_max_angular_z': 0.72,
                        'steering_smoothing_alpha': 0.32,
                        'degraded_steering_smoothing_alpha': 0.18,
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
        lane_tracker_node,
        lane_camera_subscriber_node,
        lane_yolo_inference_node,
        lane_detection_parser_node,
        obstacle_provider_node,
        waypoint_hint_node,
        fusion_node,
    ])
