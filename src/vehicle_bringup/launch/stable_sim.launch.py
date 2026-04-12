from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnShutdown
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
    use_local_ekf = LaunchConfiguration('use_local_ekf')
    enable_gps_stack = LaunchConfiguration('enable_gps_stack')
    enable_fake_rtk = LaunchConfiguration('enable_fake_rtk')
    fake_rtk_input_topic = LaunchConfiguration('fake_rtk_input_topic')
    gps_fix_topic = LaunchConfiguration('gps_fix_topic')
    rtk_status = LaunchConfiguration('rtk_status')
    cleanup_stale_gazebo = LaunchConfiguration('cleanup_stale_gazebo')
    world_file = LaunchConfiguration('world')

    gps_stack_condition = IfCondition(
        PythonExpression(
            [
                "'",
                enable_gps_stack,
                "'.lower() == 'true' and '",
                use_local_ekf,
                "'.lower() == 'true'",
            ]
        )
    )
    fake_rtk_condition = IfCondition(
        PythonExpression(
            [
                "'",
                enable_gps_stack,
                "'.lower() == 'true' and '",
                enable_fake_rtk,
                "'.lower() == 'true'",
            ]
        )
    )
    static_map_condition = UnlessCondition(
        PythonExpression(
            [
                "'",
                enable_gps_stack,
                "'.lower() == 'true' and '",
                use_local_ekf,
                "'.lower() == 'true'",
            ]
        )
    )

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
        output='screen',
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
        DeclareLaunchArgument('use_local_ekf', default_value='false'),
        DeclareLaunchArgument(
            'enable_gps_stack',
            default_value='false',
            description='Enable RTK/GPS navsat_transform and global EKF.',
        ),
        DeclareLaunchArgument(
            'enable_fake_rtk',
            default_value='false',
            description='Use the Gazebo GPS fix as a simulated RTK source.',
        ),
        DeclareLaunchArgument(
            'fake_rtk_input_topic',
            default_value='/gps/fix',
            description='Input NavSatFix topic for the simulated RTK adapter.',
        ),
        DeclareLaunchArgument(
            'gps_fix_topic',
            default_value='/vehicle/gps/fix',
            description='RTK NavSatFix topic consumed by navsat_transform.',
        ),
        DeclareLaunchArgument(
            'rtk_status',
            default_value='FIX',
            description='Simulated RTK quality: FIX | FLOAT | NO_FIX.',
        ),
        DeclareLaunchArgument('cleanup_stale_gazebo', default_value='true'),

        gazebo_shutdown_cleanup,

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        ),

        gazebo_process,
        gazebo_process_with_cleanup,

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

        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    condition=IfCondition(use_local_ekf),
                    arguments=[
                        '-file', urdf_file,
                        '-entity', 'teknofest_araci',
                        '-x', spawn_x,
                        '-y', spawn_y,
                        '-z', spawn_z,
                        '-Y', spawn_yaw,
                    ],
                    output='screen',
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
                    output='screen',
                ),
            ],
        ),

        TimerAction(
            period=18.0,
            condition=IfCondition(use_local_ekf),
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

        TimerAction(
            period=16.0,
            condition=static_map_condition,
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

        TimerAction(
            period=18.0,
            condition=fake_rtk_condition,
            actions=[
                Node(
                    package='vehicle_sensor_adapters',
                    executable='fake_rtk_node',
                    name='fake_rtk_node',
                    parameters=[
                        {
                            'use_sim_time': True,
                            'input_fix_topic': fake_rtk_input_topic,
                            'output_fix_topic': gps_fix_topic,
                            'rtk_status': rtk_status,
                        }
                    ],
                    output='screen',
                )
            ],
        ),

        TimerAction(
            period=20.0,
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

        TimerAction(
            period=22.0,
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
                        ('gps/fix', gps_fix_topic),
                        ('odometry/filtered', '/odometry/local'),
                        ('odometry/gps', '/odometry/gps'),
                    ],
                    output='screen',
                )
            ],
        ),
    ])
