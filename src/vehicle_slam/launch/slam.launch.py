import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Gazebo simülasyonunda olduğumuz için simülasyon saatini kullanmalıyız
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Paketimizin (vehicle_slam) sistemdeki yerini buluyoruz
    pkg_dir = get_package_share_directory('vehicle_slam')
    
    # Bir önceki adımda yazdığımız config dosyasının tam yolunu gösteriyoruz
    config_file = os.path.join(pkg_dir, 'config', 'mapper_params_online_async.yaml')

    return LaunchDescription([
        # Dışarıdan 'use_sim_time' parametresi alabilmek için tanımlama
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        # SLAM Toolbox düğümünü (node) başlatma komutu
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                config_file,
                {'use_sim_time': use_sim_time}
            ]
        )
    ])