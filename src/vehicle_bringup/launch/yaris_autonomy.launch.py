from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    lane_config = os.path.join(pkg_share, 'config', 'lane_tracker.yaml')
    lane_yolo_config = os.path.join(pkg_share, 'config', 'lane_yolo.yaml')
    igvc_config = os.path.join(pkg_share, 'config', 'igvc_mission.yaml')

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    mode = LaunchConfiguration('mode')
    enable_igvc_waypoints = LaunchConfiguration('enable_igvc_waypoints')
    enable_lane_yolo = LaunchConfiguration('enable_lane_yolo')
    lane_model_path = LaunchConfiguration('lane_model_path')
    route_enabled = LaunchConfiguration('route_enabled')
    lane_only_speed = LaunchConfiguration('lane_only_speed')

    race_mode_condition = IfCondition(PythonExpression(["'", mode, "' == 'race_mode'"]))
    race_with_classic_lane_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "' == 'race_mode' and '",
                enable_lane_yolo,
                "'.lower() != 'true'",
            ]
        )
    )
    race_with_yolo_lane_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "' == 'race_mode' and '",
                enable_lane_yolo,
                "'.lower() == 'true'",
            ]
        )
    )
    local_nav2_condition = IfCondition(PythonExpression(["'", mode, "' == 'local_nav2_mode'"]))
    gps_nav2_condition = IfCondition(PythonExpression(["'", mode, "' == 'gps_nav2_mode'"]))
    igvc_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "' == 'gps_nav2_mode' and '",
                enable_igvc_waypoints,
                "'.lower() == 'true'",
            ]
        )
    )

    sim_race_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        condition=race_mode_condition,
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'mode': 'race_mode',
            'enable_nav2': 'false',
            'enable_gps_stack': 'false',
            'use_local_ekf': 'false',
            'nav2_odom_topic': '/odom',
            'cleanup_stale_gazebo': 'true',
        }.items(),
    )

    sim_local_nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        condition=local_nav2_condition,
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'mode': 'local_nav2_mode',
            'enable_nav2': 'true',
            'enable_gps_stack': 'false',
            'use_local_ekf': 'false',
            'nav2_odom_topic': '/odom',
            'cleanup_stale_gazebo': 'true',
        }.items(),
    )

    sim_gps_nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'sim.launch.py')),
        condition=gps_nav2_condition,
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
            'mode': 'gps_nav2_mode',
            'enable_nav2': 'true',
            'enable_gps_stack': 'true',
            'use_local_ekf': 'true',
            'nav2_odom_topic': '/odometry/local',
            'cleanup_stale_gazebo': 'true',
        }.items(),
    )

    pilot_node = TimerAction(
        period=3.0,
        condition=race_mode_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='yaris_pilotu',
                name='yaris_pilotu',
                output='screen',
                parameters=[
                    {
                        'use_sim_time': True,
                        'route_enabled': route_enabled,
                        'lane_only_speed': lane_only_speed,
                    }
                ],
            )
        ],
    )

    lane_tracker_node = TimerAction(
        period=4.0,
        condition=race_with_classic_lane_condition,
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
        period=4.0,
        condition=race_with_yolo_lane_condition,
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
        period=5.0,
        condition=race_with_yolo_lane_condition,
        actions=[
            Node(
                package='vehicle_bringup',
                executable='lane_yolo_inference',
                name='lane_yolo_inference',
                output='screen',
                parameters=[lane_yolo_config, {'use_sim_time': True, 'model_path': lane_model_path}],
            )
        ],
    )

    lane_detection_parser_node = TimerAction(
        period=6.0,
        condition=race_with_yolo_lane_condition,
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

    igvc_waypoint_navigator_node = TimerAction(
        period=25.0,
        condition=igvc_condition,
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

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-16.239442'),
        DeclareLaunchArgument('spawn_y', default_value='-4.701300'),
        DeclareLaunchArgument('spawn_z', default_value='0.018912'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.618679'),
        DeclareLaunchArgument(
            'mode',
            default_value='race_mode',
            description='race_mode | local_nav2_mode | gps_nav2_mode',
        ),
        DeclareLaunchArgument(
            'enable_igvc_waypoints',
            default_value='true',
            description='Only used in gps_nav2_mode',
        ),
        DeclareLaunchArgument(
            'enable_lane_yolo',
            default_value='false',
            description='Enable YOLO-based lane pipeline in race_mode.',
        ),
        DeclareLaunchArgument(
            'lane_model_path',
            default_value='auto',
            description='YOLO lane model path (.pt/.onnx) or "auto" for packaged best.pt.',
        ),
        DeclareLaunchArgument(
            'route_enabled',
            default_value='true',
            description='Enable/disable ROTA guidance in race mode.',
        ),
        DeclareLaunchArgument(
            'lane_only_speed',
            default_value='0.40',
            description='Cruise speed used when route_enabled=false.',
        ),
        sim_race_launch,
        sim_local_nav2_launch,
        sim_gps_nav2_launch,
        pilot_node,
        lane_tracker_node,
        lane_camera_subscriber_node,
        lane_yolo_inference_node,
        lane_detection_parser_node,
        igvc_waypoint_navigator_node,
    ])
