#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # URDF dosyasının yolu
    pkg_share = get_package_share_directory('my_robot_description')
    urdf_file = os.path.join(pkg_share, 'urdf', 'my_robot.urdf')

    # Gazebo başlat
    start_gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Robot spawn, Gazebo açıldıktan 5 saniye sonra
    spawn_robot = TimerAction(
        period=5.0,  # Gazebo'nun açılması için bekle
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', urdf_file,
                '-entity', 'ika_arac',
                '-x', '0', '-y', '0', '-z', '0.01'
            ],
            output='screen'
        )]
    )

    return LaunchDescription([
        start_gazebo,
        spawn_robot
    ])
