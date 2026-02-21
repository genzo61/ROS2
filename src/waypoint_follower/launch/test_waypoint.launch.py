from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # -------------------
        # EKF Node
        # -------------------
        Node(
            package='vehicle_localization',   # senin EKF paketin
            executable='ekf_filter_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                '/home/ali/turtlebot3_ws/src/vehicle_localization/config/ekf.yaml'  # EKF config dosyan
            ]
        ),

        # -------------------
        # Fake Odometry
        # -------------------
        Node(
            package='waypoint_follower',
            executable='fake_odom_pub',
            name='fake_odom_pub',
            output='screen'
        ),

        # -------------------
        # Fake GPS
        # -------------------
        Node(
            package='waypoint_follower',
            executable='fake_gps_pub',
            name='fake_gps_pub',
            output='screen'
        ),

        # -------------------
        # Waypoint Follower
        # -------------------
        Node(
            package='waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen'
        )
    ])

