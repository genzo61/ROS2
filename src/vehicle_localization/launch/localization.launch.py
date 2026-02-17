import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'vehicle_localization'
    
    # Config dosyasının yolunu bul
    config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'ekf.yaml'
    )

    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[config_file], # Config dosyasını yükle
            # use_sim_time parametresi config dosyasında zaten true
        ),
    ])
