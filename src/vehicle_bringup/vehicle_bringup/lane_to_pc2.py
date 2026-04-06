#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as point_cloud2
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header


class LaneToPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_to_pc2')

        self.declare_parameter('camera_height', 0.5)
        self.declare_parameter('camera_pitch', 0.0)
        self.declare_parameter('point_step', 5)
        self.declare_parameter('corridor_wall_width', 0.0)
        self.declare_parameter('corridor_wall_points', 0)

        self.camera_height = 0.5
        self.camera_pitch_deg = 0.0
        self.camera_pitch_rad = math.radians(0.0)
        self.point_step = 5
        self.corridor_wall_width = 0.0
        self.corridor_wall_points = 0
        self._load_runtime_parameters()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.bridge = CvBridge()
        self.camera_frame_id = ''
        self.camera_matrix: Optional[np.ndarray] = None
        self.camera_matrix_inv: Optional[np.ndarray] = None
        self.camera_info_ready = False
        self.warned_missing_camera_info = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            CameraInfo,
            '/front_camera/camera_info',
            self.camera_info_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            '/lane/mask_image',
            self.mask_callback,
            sensor_qos,
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/lane_points', sensor_qos)

        self.get_logger().info(
            'lane_to_pc2 ready. '
            f'camera_height={self.camera_height:.3f} m, '
            f'camera_pitch={self.camera_pitch_deg:.3f} deg, '
            f'point_step={self.point_step}'
        )

    def _load_runtime_parameters(self) -> None:
        self.camera_height = float(self.get_parameter('camera_height').value)
        self.camera_pitch_deg = float(self.get_parameter('camera_pitch').value)
        self.camera_pitch_rad = math.radians(self.camera_pitch_deg)
        self.point_step = int(self.get_parameter('point_step').value)
        self.corridor_wall_width = float(self.get_parameter('corridor_wall_width').value)
        self.corridor_wall_points = int(self.get_parameter('corridor_wall_points').value)

    def _on_set_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        next_height = self.camera_height
        next_pitch_deg = self.camera_pitch_deg
        next_point_step = self.point_step

        for parameter in parameters:
            if parameter.name == 'camera_height':
                next_height = float(parameter.value)
                if next_height <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason='camera_height must be > 0.0',
                    )
            elif parameter.name == 'camera_pitch':
                next_pitch_deg = float(parameter.value)
            elif parameter.name == 'point_step':
                next_point_step = int(parameter.value)
                if next_point_step <= 0:
                    return SetParametersResult(
                        successful=False,
                        reason='point_step must be >= 1',
                    )

        self.camera_height = next_height
        self.camera_pitch_deg = next_pitch_deg
        self.camera_pitch_rad = math.radians(next_pitch_deg)
        self.point_step = next_point_step
        # Re-read corridor params on live update
        self.corridor_wall_width = float(self.get_parameter('corridor_wall_width').value)
        self.corridor_wall_points = int(self.get_parameter('corridor_wall_points').value)

        self.get_logger().info(
            'Updated runtime params. '
            f'camera_height={self.camera_height:.3f} m, '
            f'camera_pitch={self.camera_pitch_deg:.3f} deg, '
            f'point_step={self.point_step}'
        )
        return SetParametersResult(successful=True)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        if fx <= 1e-9 or fy <= 1e-9:
            self.get_logger().warning('Ignoring CameraInfo with invalid focal length values.')
            return

        try:
            camera_matrix_inv = np.linalg.inv(camera_matrix)
        except np.linalg.LinAlgError:
            self.get_logger().warning('Ignoring CameraInfo because K matrix is singular.')
            return

        self.camera_matrix = camera_matrix
        self.camera_matrix_inv = camera_matrix_inv
        self.camera_frame_id = msg.header.frame_id

        if not self.camera_info_ready:
            self.get_logger().info(
                'CameraInfo received. '
                f'frame_id={self.camera_frame_id}, '
                f'fx={fx:.3f}, fy={fy:.3f}, '
                f'cx={camera_matrix[0, 2]:.3f}, cy={camera_matrix[1, 2]:.3f}'
            )
        self.camera_info_ready = True
        self.warned_missing_camera_info = False

    def mask_callback(self, msg: Image) -> None:
        if self.camera_matrix_inv is None:
            if not self.warned_missing_camera_info:
                self.get_logger().warning(
                    'Waiting for /front_camera/camera_info before projecting lane mask.'
                )
                self.warned_missing_camera_info = True
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as exc:
            self.get_logger().warning(f'cv_bridge failed to convert /lane/mask_image: {exc}')
            return

        sampled_mask = np.asarray(mask)[:: self.point_step, :: self.point_step] > 127
        sampled_rows, sampled_cols = np.nonzero(sampled_mask)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.camera_frame_id or msg.header.frame_id

        if sampled_rows.size == 0:
            self.publish_pointcloud(header, np.empty((0, 3), dtype=np.float32))
            return

        pixel_u = (sampled_cols * self.point_step).astype(np.float64)
        pixel_v = (sampled_rows * self.point_step).astype(np.float64)

        # Project sampled pixels with K^-1 to obtain camera-frame rays.
        homogeneous_pixels = np.vstack(
            (
                pixel_u,
                pixel_v,
                np.ones_like(pixel_u, dtype=np.float64),
            )
        )
        rays = (self.camera_matrix_inv @ homogeneous_pixels).T

        # Optical-frame convention: +x right, +y down, +z forward.
        # A positive downward camera pitch rotates the world-down axis toward +z.
        ground_down_axis = np.array(
            [0.0, math.cos(self.camera_pitch_rad), math.sin(self.camera_pitch_rad)],
            dtype=np.float64,
        )
        ray_plane_denominator = rays @ ground_down_axis
        valid_intersections = ray_plane_denominator > 1e-6

        if not np.any(valid_intersections):
            self.publish_pointcloud(header, np.empty((0, 3), dtype=np.float32))
            return

        valid_rays = rays[valid_intersections]
        valid_denominator = ray_plane_denominator[valid_intersections]
        ray_scales = self.camera_height / valid_denominator
        points_camera = valid_rays * ray_scales[:, None]

        finite_points = np.isfinite(points_camera).all(axis=1)
        forward_points = points_camera[:, 2] > 0.0
        points_camera = points_camera[finite_points & forward_points]

        # FIX 5: Add corridor wall virtual points before publishing
        if points_camera.shape[0] > 0:
            points_camera = self._add_corridor_walls(points_camera)

        self.publish_pointcloud(header, points_camera.astype(np.float32, copy=False))

    def _add_corridor_walls(self, points_camera: np.ndarray) -> np.ndarray:
        """FIX 5: For each projected lane point, add outward virtual wall points.

        The robot camera frame uses +x right, +y down, +z forward.
        'Outward' means bilaterally in the x-direction (left AND right) because
        the node does not know which side a pixel belongs to — widening the
        obstacle footprint on both sides thickens lane boundaries symmetrically,
        which nudges Nav2 DWB to stay centred.

        Each original point spawns `corridor_wall_points` additional points on
        each side, spaced equally up to `corridor_wall_width` metres.
        """
        if self.corridor_wall_points <= 0 or self.corridor_wall_width <= 0.0:
            return points_camera

        n_pts = self.corridor_wall_points
        step = self.corridor_wall_width / float(n_pts)
        offsets = np.arange(1, n_pts + 1, dtype=np.float32) * step  # [step, 2*step, ...]

        extra_chunks = []
        for sign in (+1.0, -1.0):
            for off in offsets:
                shifted = points_camera.copy()
                shifted[:, 0] += sign * off  # shift laterally (camera x-axis)
                extra_chunks.append(shifted)

        return np.vstack([points_camera] + extra_chunks)

    def publish_pointcloud(self, header: Header, points: np.ndarray) -> None:
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.pointcloud_pub.publish(cloud_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneToPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
