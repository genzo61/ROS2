from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    world_file = os.path.join(pkg_share, 'worlds', 'lane_only_short.world')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    cleanup_stale_gazebo = LaunchConfiguration('cleanup_stale_gazebo')

    stable_sim_launch = IncludeLaunchDescription(
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

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-6.5'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='0.02'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_local_ekf', default_value='false'),
        DeclareLaunchArgument('cleanup_stale_gazebo', default_value='true'),
        stable_sim_launch,
    ])
