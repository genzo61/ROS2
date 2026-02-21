from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_share = get_package_share_directory('vehicle_bringup')
    ekf_config = os.path.join(get_package_share_directory('vehicle_localization'), 'config', 'ekf.yaml')
    world_file = os.path.join(pkg_share, 'worlds', 'amerika_parkur')
    urdf_file = os.path.expanduser(
        '~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/urdf/my_robot_amerika.urdf'
    )

    robot_description = ParameterValue(
        Command(['cat ', urdf_file]),
        value_type=str
    )

    return LaunchDescription([

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True,
            }],
            output='screen'
        ),

        # Gazebo
        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                world_file,
                '-s',
                'libgazebo_ros_init.so',
                '-s',
                'libgazebo_ros_factory.so',
            ],
            output='screen'
        ),

        # Spawn Robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'teknofest_araci'
            ],
            output='screen'
        ),

        # Local EKF (Odom -> Base Footprint)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            parameters=[
                ekf_config,
                {'use_sim_time': True},
            ],
            remappings=[('odometry/filtered', '/odometry/local')]
        ),

        # Global EKF (Map -> Odom)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            parameters=[
                ekf_config,
                {'use_sim_time': True},
            ],
            remappings=[('odometry/filtered', '/odometry/filtered')]
        ),

        # NavSat Transform Node (GPS -> Odometry)
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            parameters=[
                ekf_config,
                {'use_sim_time': True},
            ],
            remappings=[
                ('imu', '/imu'),
                ('imu/data', '/imu'),
                ('gps/fix', '/gps/fix'),
                # NavSat must use local odometry to avoid a startup loop with global EKF.
                ('odometry/filtered', '/odometry/local'),
                ('odometry/gps', '/odometry/gps'),
            ]
        ),

        # Nav2 (delayed to let localization settle)
        TimerAction(
            period=15.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg_share, 'launch', 'nav2_bringup.launch.py')
                    )
                )
            ]
        ),
    ])
