#!/usr/bin/env python3
"""
TEKNOFEST Waypoint Follower Node.

Converts hardcoded GPS waypoints (lat/lon) to local map-frame coordinates
using robot_localization's /fromLL service, then sends them sequentially
to Nav2 via the nav2_simple_commander BasicNavigator API.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
from robot_localization.srv import FromLL
from geographic_msgs.msg import GeoPoint
import math
import time


class TeknofestWaypointFollower(Node):
    def __init__(self):
        super().__init__('teknofest_waypoint_follower')

        # ---- MOCK TEKNOFEST GPS WAYPOINTS (lat, lon) ----
        # Replace these with actual competition coordinates
        self.gps_waypoints = [
            (40.80690,  29.35790),   # Waypoint 1
            (40.80700,  29.35810),   # Waypoint 2
            (40.80710,  29.35830),   # Waypoint 3
            (40.80720,  29.35850),   # Waypoint 4
        ]

        self.get_logger().info(
            f'TEKNOFEST Waypoint Follower started with {len(self.gps_waypoints)} GPS waypoints.'
        )

        # Service client for GPS -> Map coordinate conversion
        self.fromll_client = self.create_client(FromLL, '/fromLL')

    def convert_gps_to_map(self, lat: float, lon: float) -> tuple:
        """Convert GPS (lat, lon) to map frame (x, y) using robot_localization's /fromLL service."""
        self.get_logger().info(f'Waiting for /fromLL service...')
        if not self.fromll_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/fromLL service not available! Is navsat_transform_node running?')
            return None

        request = FromLL.Request()
        request.ll_point = GeoPoint()
        request.ll_point.latitude = lat
        request.ll_point.longitude = lon
        request.ll_point.altitude = 0.0

        future = self.fromll_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None:
            result = future.result()
            x = result.map_point.x
            y = result.map_point.y
            self.get_logger().info(
                f'GPS ({lat:.6f}, {lon:.6f}) -> Map ({x:.2f}, {y:.2f})'
            )
            return (x, y)
        else:
            self.get_logger().error(f'Failed to convert GPS ({lat}, {lon}) to map frame.')
            return None

    def create_pose(self, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
        """Create a PoseStamped message in the map frame."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        # Simple yaw -> quaternion
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestWaypointFollower()

    # Wait for the system to come up
    node.get_logger().info('Waiting 5 seconds for localization & Nav2 to initialize...')
    time.sleep(5.0)

    # Initialize BasicNavigator
    navigator = BasicNavigator()

    # Wait for Nav2 to be active
    node.get_logger().info('Waiting for Nav2 to become active...')
    navigator.waitUntilNav2Active()
    node.get_logger().info('Nav2 is active!')

    # Convert all GPS waypoints to map coordinates
    map_waypoints = []
    for i, (lat, lon) in enumerate(node.gps_waypoints):
        result = node.convert_gps_to_map(lat, lon)
        if result is None:
            node.get_logger().error(f'Skipping waypoint {i+1} - GPS conversion failed.')
            continue
        map_waypoints.append(result)

    if not map_waypoints:
        node.get_logger().error('No valid waypoints! Exiting.')
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info(
        f'Successfully converted {len(map_waypoints)}/{len(node.gps_waypoints)} waypoints.'
    )

    # Navigate to each waypoint sequentially
    for i, (x, y) in enumerate(map_waypoints):
        node.get_logger().info(f'=== Navigating to Waypoint {i+1}/{len(map_waypoints)}: ({x:.2f}, {y:.2f}) ===')

        goal_pose = node.create_pose(x, y)
        navigator.goToPose(goal_pose)

        # Wait for navigation to complete
        while not navigator.isTaskComplete():
            feedback = navigator.getFeedback()
            if feedback:
                distance_remaining = feedback.distance_remaining
                if distance_remaining is not None:
                    node.get_logger().info(
                        f'  Distance remaining to WP {i+1}: {distance_remaining:.2f} m',
                        throttle_duration_sec=2.0,
                    )
            time.sleep(0.5)

        # Check result
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            node.get_logger().info(f'✓ Waypoint {i+1} reached successfully!')
        elif result == TaskResult.CANCELED:
            node.get_logger().warn(f'✗ Waypoint {i+1} was canceled.')
        elif result == TaskResult.FAILED:
            node.get_logger().error(f'✗ Waypoint {i+1} failed! Continuing to next...')
        else:
            node.get_logger().warn(f'? Waypoint {i+1}: Unknown result: {result}')

    node.get_logger().info('=== ALL WAYPOINTS COMPLETED ===')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
