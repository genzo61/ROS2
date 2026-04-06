from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('vehicle_bringup')
    nav2_params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    odom_topic = LaunchConfiguration('odom_topic')

    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        param_rewrites={
            'odom_topic': odom_topic,
        },
        convert_types=True
    )

    lifecycle_nodes = [
        'controller_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odometry/local',
            description='Nav2 odometry topic (/odometry/local for EKF, /odom for diff_drive-only).',
        ),

        # Controller Server
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[configured_params, {'use_sim_time': True}],
        ),

        # Planner Server
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[configured_params, {'use_sim_time': True}],
        ),

        # Behavior Server (recoveries)
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[configured_params, {'use_sim_time': True}],
        ),

        # BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[configured_params, {'use_sim_time': True}],
        ),

        # Waypoint Follower
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[configured_params, {'use_sim_time': True}],
        ),

        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': lifecycle_nodes,
                'bond_timeout': 15.0,
                'service_timeout': 30000,
                'attempt_respawn_reconnection': False,
                'bond_respawn_max_duration': 10.0,
            }],
        ),
    ])
