from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    default_params = os.path.join(pkg_share, 'config', 'drivable_area_local_planner.yaml')

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'params_file',
                default_value=default_params,
                description='Parameters file for drivable area local planner.',
            ),
            Node(
                package='vehicle_bringup',
                executable='drivable_area_local_planner',
                name='drivable_area_local_planner',
                output='screen',
                parameters=[params_file],
            ),
        ]
    )
