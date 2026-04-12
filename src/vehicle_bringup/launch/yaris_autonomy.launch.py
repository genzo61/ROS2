from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    default_world = os.path.join(pkg_share, 'worlds', 'amerika_parkur')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    world = LaunchConfiguration('world')
    enable_lane_yolo = LaunchConfiguration('enable_lane_yolo')
    enable_waypoint_hint = LaunchConfiguration('enable_waypoint_hint')
    lane_model_path = LaunchConfiguration('lane_model_path')
    base_speed = LaunchConfiguration('base_speed')
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    enable_gps_stack = LaunchConfiguration('enable_gps_stack')
    enable_fake_rtk = LaunchConfiguration('enable_fake_rtk')
    fake_rtk_input_topic = LaunchConfiguration('fake_rtk_input_topic')
    gps_fix_topic = LaunchConfiguration('gps_fix_topic')
    rtk_status = LaunchConfiguration('rtk_status')
    waypoint_source = LaunchConfiguration('waypoint_source')
    waypoint_weight_with_lane = LaunchConfiguration('waypoint_weight_with_lane')
    waypoint_weight_no_lane = LaunchConfiguration('waypoint_weight_no_lane')

    stable_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'stable_sim.launch.py')),
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'world': world,
            'use_local_ekf': use_local_ekf,
            'enable_gps_stack': enable_gps_stack,
            'enable_fake_rtk': enable_fake_rtk,
            'fake_rtk_input_topic': fake_rtk_input_topic,
            'gps_fix_topic': gps_fix_topic,
            'rtk_status': rtk_status,
        }.items(),
    )

    race_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'race_stack.launch.py')),
        launch_arguments={
            'enable_lane_yolo': enable_lane_yolo,
            'enable_waypoint_hint': enable_waypoint_hint,
            'lane_model_path': lane_model_path,
            'base_speed': base_speed,
            'waypoint_source': waypoint_source,
            'waypoint_weight_with_lane': waypoint_weight_with_lane,
            'waypoint_weight_no_lane': waypoint_weight_no_lane,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-16.239442'),
        DeclareLaunchArgument('spawn_y', default_value='-4.701300'),
        DeclareLaunchArgument('spawn_z', default_value='0.018912'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.618679'),
        DeclareLaunchArgument('world', default_value=default_world),
        DeclareLaunchArgument('use_local_ekf', default_value='false'),
        DeclareLaunchArgument('enable_lane_yolo', default_value='false'),
        DeclareLaunchArgument('enable_waypoint_hint', default_value='true'),
        DeclareLaunchArgument('lane_model_path', default_value='auto'),
        DeclareLaunchArgument('base_speed', default_value='0.75'),
        DeclareLaunchArgument('enable_gps_stack', default_value='false'),
        DeclareLaunchArgument('enable_fake_rtk', default_value='false'),
        DeclareLaunchArgument('fake_rtk_input_topic', default_value='/gps/fix'),
        DeclareLaunchArgument('gps_fix_topic', default_value='/vehicle/gps/fix'),
        DeclareLaunchArgument('rtk_status', default_value='FIX'),
        DeclareLaunchArgument('waypoint_source', default_value='auto'),
        DeclareLaunchArgument('waypoint_weight_with_lane', default_value='0.04'),
        DeclareLaunchArgument('waypoint_weight_no_lane', default_value='0.22'),
        stable_sim_launch,
        race_stack_launch,
    ])
