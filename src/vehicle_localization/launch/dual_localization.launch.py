import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'vehicle_localization'
    config_file = os.path.join(get_package_share_directory(pkg_name), 'config', 'dual_ekf.yaml')

    return LaunchDescription([
        # 1. LOCAL EKF (Odom Frame)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[config_file, {'use_sim_time': True}],
            remappings=[('odometry/filtered', 'odometry/local')] # Çıktı ismi değişti
        ),

        # 2. GLOBAL EKF (Map Frame)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_file, {'use_sim_time': True}],
            remappings=[('odometry/filtered', 'odometry/global')] # Çıktı ismi değişti
        ),

        # 3. NAVSAT TRANSFORM (GPS İşleyici)
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=[config_file, {'use_sim_time': True}],
            remappings=[
                ('imu/data', '/vehicle/imu/data'),
                ('gps/fix', '/vehicle/gps/fix'), # Fake RTK'dan gelen veri
                ('odometry/filtered', 'odometry/global'), # Global konum bilgisi lazım
                ('odometry/gps', '/odometry/gps') # Çıktı
            ]
        )
    ])
