from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vehicle_bringup')
    ekf_config = os.path.join(get_package_share_directory('vehicle_localization'), 'config', 'ekf.yaml')
    default_world = os.path.join(pkg_share, 'worlds', 'amerika_parkur')
    urdf_file = os.path.expanduser(
        '~/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/urdf/my_robot_amerika.urdf'
    )

    with open(urdf_file, 'r', encoding='utf-8') as urdf_in:
        robot_description = urdf_in.read()
    urdf_for_diffdrive_tf = urdf_file
    if '<publish_odom_tf>false</publish_odom_tf>' in robot_description:
        robot_description_diffdrive_tf = robot_description.replace(
            '<publish_odom_tf>false</publish_odom_tf>',
            '<publish_odom_tf>true</publish_odom_tf>',
            1,
        )
        urdf_for_diffdrive_tf = f'/tmp/my_robot_amerika_diffdrive_tf_{os.getpid()}.urdf'
        with open(urdf_for_diffdrive_tf, 'w', encoding='utf-8') as urdf_out:
            urdf_out.write(robot_description_diffdrive_tf)

    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')
    mode = LaunchConfiguration('mode')
    enable_nav2 = LaunchConfiguration('enable_nav2')
    enable_gps_stack = LaunchConfiguration('enable_gps_stack')
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    nav2_odom_topic = LaunchConfiguration('nav2_odom_topic')
    cleanup_stale_gazebo = LaunchConfiguration('cleanup_stale_gazebo')
    world_file = LaunchConfiguration('world')

    gps_stack_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "' == 'gps_nav2_mode' and '",
                enable_gps_stack,
                "'.lower() == 'true'",
            ]
        )
    )
    nav2_without_gps_condition = IfCondition(
        PythonExpression(
            [
                "('",
                mode,
                "' == 'gps_nav2_mode' or '",
                mode,
                "' == 'local_nav2_mode') and '",
                enable_nav2,
                "'.lower() == 'true' and '",
                enable_gps_stack,
                "'.lower() != 'true'",
            ]
        )
    )
    nav2_with_gps_condition = IfCondition(
        PythonExpression(
            [
                "'",
                mode,
                "' == 'gps_nav2_mode' and '",
                enable_nav2,
                "'.lower() == 'true' and '",
                enable_gps_stack,
                "'.lower() == 'true'",
            ]
        )
    )
    local_ekf_condition = IfCondition(use_local_ekf)

    gazebo_process = ExecuteProcess(
        condition=UnlessCondition(cleanup_stale_gazebo),
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
    )
    gazebo_process_with_cleanup = ExecuteProcess(
        condition=IfCondition(cleanup_stale_gazebo),
        cmd=[
            'bash',
            '-lc',
            [
                'pkill -TERM gzserver || true; '
                'pkill -TERM gzclient || true; '
                'pkill -TERM gazebo || true; '
                'sleep 1; '
                'pkill -KILL gzserver || true; '
                'pkill -KILL gzclient || true; '
                'pkill -KILL gazebo || true; '
                'exec gazebo --verbose ',
                world_file,
                ' -s libgazebo_ros_init.so -s libgazebo_ros_factory.so',
            ],
        ],
        output='screen',
    )

    gazebo_shutdown_cleanup = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=[
                        'bash',
                        '-lc',
                        'pkill -TERM gzserver || true; '
                        'pkill -TERM gzclient || true; '
                        'pkill -TERM gazebo || true; '
                        'sleep 1; '
                        'pkill -KILL gzserver || true; '
                        'pkill -KILL gzclient || true; '
                        'pkill -KILL gazebo || true'
                    ],
                    output='screen',
                )
            ]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-16.239442'),
        DeclareLaunchArgument('spawn_y', default_value='-4.701300'),
        DeclareLaunchArgument('spawn_z', default_value='0.018912'),
        DeclareLaunchArgument('spawn_yaw', default_value='1.618679'),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Absolute path to the Gazebo world file.',
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='local_nav2_mode',
            description='race_mode | local_nav2_mode | gps_nav2_mode',
        ),
        DeclareLaunchArgument('enable_nav2', default_value='true'),
        DeclareLaunchArgument(
            'enable_gps_stack',
            default_value='false',
            description='Enable navsat_transform + map EKF (GPS stack).',
        ),
        DeclareLaunchArgument(
            'use_local_ekf',
            default_value='false',
            description='Start local EKF (owner of odom->base_footprint).',
        ),
        DeclareLaunchArgument(
            'nav2_odom_topic',
            default_value='/odom',
            description='Nav2 odom topic: /odometry/local (EKF) or /odom (diff_drive-only).',
        ),
        DeclareLaunchArgument(
            'cleanup_stale_gazebo',
            default_value='true',
            description='Kill stale gazebo/gzserver/gzclient before start and on shutdown.',
        ),

        gazebo_shutdown_cleanup,

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen'
        ),

        gazebo_process,
        gazebo_process_with_cleanup,

        # Remove stale robot entity if an old gazebo server survived.
        TimerAction(
            period=6.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'bash',
                        '-lc',
                        "timeout 4 ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "
                        "\"{name: 'teknofest_araci'}\" || true"
                    ],
                    output='screen',
                )
            ],
        ),

        # Spawn after stale-entity cleanup.
        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    condition=local_ekf_condition,
                    arguments=[
                        '-file', urdf_file,
                        '-entity', 'teknofest_araci',
                        '-x', spawn_x,
                        '-y', spawn_y,
                        '-z', spawn_z,
                        '-Y', spawn_yaw,
                    ],
                    output='screen'
                ),
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    condition=UnlessCondition(use_local_ekf),
                    arguments=[
                        '-file', urdf_for_diffdrive_tf,
                        '-entity', 'teknofest_araci',
                        '-x', spawn_x,
                        '-y', spawn_y,
                        '-z', spawn_z,
                        '-Y', spawn_yaw,
                    ],
                    output='screen'
                ),
            ],
        ),

        # Stage 1 (both modes): local EKF owner of odom->base_footprint.
        TimerAction(
            period=12.0,
            condition=local_ekf_condition,
            actions=[
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node_odom',
                    parameters=[ekf_config, {'use_sim_time': True}],
                    remappings=[('odometry/filtered', '/odometry/local')],
                    output='screen',
                )
            ],
        ),

        # Stage 1b (gps_nav2_mode without GPS stack): provide map->odom identity for Nav2 global frame.
        TimerAction(
            period=13.0,
            condition=nav2_without_gps_condition,
            actions=[
                Node(
                    package='tf2_ros',
                    executable='static_transform_publisher',
                    name='map_to_odom_identity_tf',
                    arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
                    parameters=[{'use_sim_time': True}],
                    output='screen',
                )
            ],
        ),

        # Stage 2 (gps_nav2_mode): global EKF owns map->odom as soon as /odometry/local exists.
        TimerAction(
            period=16.0,
            condition=gps_stack_condition,
            actions=[
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node_map',
                    parameters=[ekf_config, {'use_sim_time': True}],
                    remappings=[('odometry/filtered', '/odometry/global')],
                    output='screen',
                )
            ],
        ),

        # Stage 3 (gps_nav2_mode): navsat consumes /odometry/local, /imu, /gps/fix and outputs /odometry/gps.
        TimerAction(
            period=18.0,
            condition=gps_stack_condition,
            actions=[
                Node(
                    package='robot_localization',
                    executable='navsat_transform_node',
                    name='navsat_transform',
                    parameters=[ekf_config, {'use_sim_time': True}],
                    remappings=[
                        ('imu', '/imu'),
                        ('imu/data', '/imu'),
                        ('gps/fix', '/gps/fix'),
                        ('odometry/filtered', '/odometry/local'),
                        ('odometry/gps', '/odometry/gps'),
                    ],
                    output='screen',
                )
            ],
        ),

        # Stage 4a (local EKF mode): start Nav2 after odom TF and map->odom(static) are up.
        TimerAction(
            period=28.0,
            condition=nav2_without_gps_condition,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg_share, 'launch', 'nav2_bringup.launch.py')
                    ),
                    launch_arguments={
                        'odom_topic': nav2_odom_topic,
                    }.items(),
                )
            ]
        ),

        # Stage 4b (GPS stack mode): start Nav2 after GPS stack settles, but
        # without the long delay that makes the vehicle appear idle in simulation.
        TimerAction(
            period=30.0,
            condition=nav2_with_gps_condition,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(pkg_share, 'launch', 'nav2_bringup.launch.py')
                    ),
                    launch_arguments={
                        'odom_topic': nav2_odom_topic,
                    }.items(),
                )
            ]
        ),
    ])
