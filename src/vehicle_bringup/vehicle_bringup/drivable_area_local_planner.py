#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


PointArray = Sequence[float]


@dataclass
class LookaheadTarget:
    x: float
    y: int
    corridor_left_px: int
    corridor_right_px: int
    corridor_width_px: int


@dataclass
class CandidateSolution:
    label: str
    target: LookaheadTarget
    drivable_mask: np.ndarray
    obstacle_mask: np.ndarray
    score: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def points_from_parameter(flat_points: PointArray, name: str) -> np.ndarray:
    points = np.asarray(flat_points, dtype=np.float32).reshape(-1)
    if points.size != 8:
        raise ValueError(f'{name} must contain exactly 8 numbers, got {points.size}')
    return points.reshape(4, 2)


def warp_to_bev(
    image: np.ndarray,
    src_points: np.ndarray,
    output_size: Tuple[int, int],
    dst_points: Optional[np.ndarray] = None,
    interpolation: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bev_width, bev_height = output_size
    if dst_points is None:
        dst_points = np.array(
            [
                [0.0, 0.0],
                [float(bev_width - 1), 0.0],
                [float(bev_width - 1), float(bev_height - 1)],
                [0.0, float(bev_height - 1)],
            ],
            dtype=np.float32,
        )

    homography = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(
        image,
        homography,
        (bev_width, bev_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, src_points.astype(np.int32), 255)
    valid_mask = cv2.warpPerspective(
        roi_mask,
        homography,
        (bev_width, bev_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return bev, homography, valid_mask


def build_obstacle_mask_from_bgr(
    frame_bgr: np.ndarray,
    white_value_min: int,
    white_sat_max: int,
    yellow_h_min: int,
    yellow_h_max: int,
    yellow_sat_min: int,
    yellow_val_min: int,
    orange_h_min: int,
    orange_h_max: int,
    orange_sat_min: int,
    orange_val_min: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        (0, 0, white_value_min),
        (180, white_sat_max, 255),
    )
    yellow_mask = cv2.inRange(
        hsv,
        (yellow_h_min, yellow_sat_min, yellow_val_min),
        (yellow_h_max, 255, 255),
    )
    orange_mask = cv2.inRange(
        hsv,
        (orange_h_min, orange_sat_min, orange_val_min),
        (orange_h_max, 255, 255),
    )
    obstacle_mask = cv2.bitwise_or(white_mask, yellow_mask)
    obstacle_mask = cv2.bitwise_or(obstacle_mask, orange_mask)
    return obstacle_mask


def build_obstacle_mask(
    image: np.ndarray,
    input_semantics: str,
    mono_threshold: int,
    white_value_min: int,
    white_sat_max: int,
    yellow_h_min: int,
    yellow_h_max: int,
    yellow_sat_min: int,
    yellow_val_min: int,
    orange_h_min: int,
    orange_h_max: int,
    orange_sat_min: int,
    orange_val_min: int,
) -> np.ndarray:
    normalized_semantics = input_semantics.strip().lower()

    if normalized_semantics == 'lanes_and_cones_bgr':
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return build_obstacle_mask_from_bgr(
            frame_bgr=image,
            white_value_min=white_value_min,
            white_sat_max=white_sat_max,
            yellow_h_min=yellow_h_min,
            yellow_h_max=yellow_h_max,
            yellow_sat_min=yellow_sat_min,
            yellow_val_min=yellow_val_min,
            orange_h_min=orange_h_min,
            orange_h_max=orange_h_max,
            orange_sat_min=orange_sat_min,
            orange_val_min=orange_val_min,
        )

    if image.ndim == 3:
        mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        mono = image.copy()

    if normalized_semantics == 'obstacles_white':
        return np.where(mono >= mono_threshold, 255, 0).astype(np.uint8)
    if normalized_semantics == 'drivable_white':
        return np.where(mono >= mono_threshold, 0, 255).astype(np.uint8)

    raise ValueError(
        'input_semantics must be one of: '
        'lanes_and_cones_bgr, obstacles_white, drivable_white'
    )


def build_drivable_area_mask(
    bev_obstacle_mask: np.ndarray,
    valid_mask: np.ndarray,
    inflation_radius_px: int,
    close_kernel_px: int,
    open_kernel_px: int,
    connected_seed_band_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    obstacle_mask = np.where(bev_obstacle_mask > 0, 255, 0).astype(np.uint8)

    if inflation_radius_px > 0:
        diameter = (2 * inflation_radius_px) + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (diameter, diameter),
        )
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    drivable_mask = np.where(
        (valid_mask > 0) & (obstacle_mask == 0),
        255,
        0,
    ).astype(np.uint8)

    if close_kernel_px > 0:
        size = close_kernel_px if close_kernel_px % 2 == 1 else close_kernel_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)

    if open_kernel_px > 0:
        size = open_kernel_px if open_kernel_px % 2 == 1 else open_kernel_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, kernel)

    if connected_seed_band_px <= 0:
        return drivable_mask, obstacle_mask

    num_labels, labels = cv2.connectedComponents(drivable_mask)
    if num_labels <= 1:
        return drivable_mask, obstacle_mask

    seed_band_px = max(1, connected_seed_band_px)
    seed_region = labels[max(0, labels.shape[0] - seed_band_px):, :]
    free_region = drivable_mask[max(0, drivable_mask.shape[0] - seed_band_px):, :]
    seed_labels = np.unique(seed_region[free_region > 0])
    seed_labels = seed_labels[seed_labels != 0]

    if seed_labels.size == 0:
        counts = np.bincount(labels.reshape(-1))
        if counts.size <= 1:
            return drivable_mask, obstacle_mask
        counts[0] = 0
        seed_labels = np.array([int(np.argmax(counts))], dtype=np.int32)

    connected_mask = np.where(np.isin(labels, seed_labels), 255, 0).astype(np.uint8)
    return connected_mask, obstacle_mask


def find_free_runs(row_mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None

    for idx, pixel in enumerate(row_mask):
        is_free = pixel > 0
        if is_free and start_idx is None:
            start_idx = idx
        if not is_free and start_idx is not None:
            runs.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        runs.append((start_idx, len(row_mask) - 1))
    return runs


def find_lookahead_target(
    drivable_mask: np.ndarray,
    lookahead_y: int,
    min_corridor_width_px: int,
    search_window_px: int,
    center_weight: float,
    memory_weight: float,
    preferred_x: float,
    previous_target_x: Optional[float] = None,
) -> Optional[LookaheadTarget]:
    image_height, _ = drivable_mask.shape
    candidate_rows: List[int] = [lookahead_y]

    for delta in range(1, max(0, search_window_px) + 1):
        upper = lookahead_y - delta
        lower = lookahead_y + delta
        if upper >= 0:
            candidate_rows.append(upper)
        if lower < image_height:
            candidate_rows.append(lower)

    best_target: Optional[LookaheadTarget] = None
    best_score = float('-inf')
    preferred_previous_x = previous_target_x if previous_target_x is not None else preferred_x

    for row_idx in candidate_rows:
        runs = find_free_runs(drivable_mask[row_idx, :])
        for left_px, right_px in runs:
            corridor_width_px = right_px - left_px + 1
            if corridor_width_px < min_corridor_width_px:
                continue

            center_x = 0.5 * (left_px + right_px)
            score = float(corridor_width_px)
            score -= center_weight * abs(center_x - preferred_x)
            score -= memory_weight * abs(center_x - preferred_previous_x)

            if score > best_score:
                best_score = score
                best_target = LookaheadTarget(
                    x=center_x,
                    y=row_idx,
                    corridor_left_px=left_px,
                    corridor_right_px=right_px,
                    corridor_width_px=corridor_width_px,
                )

    return best_target


def compute_cmd_vel(
    target_x: float,
    image_width: int,
    dt: float,
    previous_error: float,
    steering_kp: float,
    steering_kd: float,
    max_angular_z: float,
    min_linear_x: float,
    max_linear_x: float,
    speed_reduction_gain: float,
    image_center_bias_px: float = 0.0,
) -> Tuple[float, float, float]:
    image_center_x = (0.5 * image_width) + image_center_bias_px
    normalized_error = (target_x - image_center_x) / max(1.0, 0.5 * image_width)
    derivative = 0.0
    if dt > 1e-4:
        derivative = (normalized_error - previous_error) / dt

    angular_z = -(steering_kp * normalized_error) - (steering_kd * derivative)
    angular_z = clamp(angular_z, -max_angular_z, max_angular_z)

    turn_severity = clamp(abs(normalized_error), 0.0, 1.0)
    speed_scale = clamp(1.0 - (speed_reduction_gain * turn_severity), 0.0, 1.0)
    linear_x = min_linear_x + ((max_linear_x - min_linear_x) * speed_scale)
    return linear_x, angular_z, normalized_error


class DrivableAreaLocalPlanner(Node):
    def __init__(self) -> None:
        super().__init__('drivable_area_local_planner')

        self.declare_parameter('image_topic', '/front_camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('target_topic', '/drivable_area/target_x')
        self.declare_parameter('target_valid_topic', '/drivable_area/target_valid')
        self.declare_parameter('debug_image_topic', '/drivable_area/debug_bev')
        self.declare_parameter('publish_debug_image', True)

        self.declare_parameter('input_semantics', 'lanes_and_cones_bgr')
        self.declare_parameter('mono_threshold', 127)
        self.declare_parameter('white_value_min', 170)
        self.declare_parameter('white_sat_max', 90)
        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 42)
        self.declare_parameter('yellow_sat_min', 40)
        self.declare_parameter('yellow_val_min', 80)
        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 22)
        self.declare_parameter('orange_sat_min', 90)
        self.declare_parameter('orange_val_min', 80)

        self.declare_parameter('bev_width', 400)
        self.declare_parameter('bev_height', 480)
        self.declare_parameter(
            'bev_source_points',
            [190.0, 250.0, 450.0, 250.0, 630.0, 470.0, 10.0, 470.0],
        )
        self.declare_parameter('bev_destination_points', [])
        self.declare_parameter('inflation_radius_px', 11)
        self.declare_parameter('close_kernel_px', 7)
        self.declare_parameter('open_kernel_px', 3)
        self.declare_parameter('connected_seed_band_px', 40)

        self.declare_parameter('lookahead_offset_px', 140)
        self.declare_parameter('lookahead_search_window_px', 14)
        self.declare_parameter('lookahead_min_corridor_width_px', 45)
        self.declare_parameter('corridor_score_center_weight', 0.35)
        self.declare_parameter('corridor_score_memory_weight', 0.55)
        self.declare_parameter('target_smoothing_alpha', 0.35)
        self.declare_parameter('image_center_bias_px', 0.0)
        self.declare_parameter('enable_auto_mask_recovery', True)

        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('target_timeout_sec', 0.35)
        self.declare_parameter('steering_kp', 1.35)
        self.declare_parameter('steering_kd', 0.18)
        self.declare_parameter('max_angular_z', 1.60)
        self.declare_parameter('min_linear_x', 0.25)
        self.declare_parameter('max_linear_x', 0.85)
        self.declare_parameter('speed_reduction_gain', 0.85)
        self.declare_parameter('narrow_corridor_threshold_px', 85)
        self.declare_parameter('narrow_corridor_speed', 0.22)

        self.image_topic = str(self.get_parameter('image_topic').value)
        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        target_topic = str(self.get_parameter('target_topic').value)
        target_valid_topic = str(self.get_parameter('target_valid_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        self.input_semantics = str(self.get_parameter('input_semantics').value)
        self.mono_threshold = int(self.get_parameter('mono_threshold').value)
        self.white_value_min = int(self.get_parameter('white_value_min').value)
        self.white_sat_max = int(self.get_parameter('white_sat_max').value)
        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_sat_min = int(self.get_parameter('yellow_sat_min').value)
        self.yellow_val_min = int(self.get_parameter('yellow_val_min').value)
        self.orange_h_min = int(self.get_parameter('orange_h_min').value)
        self.orange_h_max = int(self.get_parameter('orange_h_max').value)
        self.orange_sat_min = int(self.get_parameter('orange_sat_min').value)
        self.orange_val_min = int(self.get_parameter('orange_val_min').value)

        self.bev_width = int(self.get_parameter('bev_width').value)
        self.bev_height = int(self.get_parameter('bev_height').value)
        self.bev_source_points = points_from_parameter(
            self.get_parameter('bev_source_points').value,
            'bev_source_points',
        )
        dst_points_param = self.get_parameter_or(
            'bev_destination_points',
            Parameter(
                'bev_destination_points',
                Parameter.Type.DOUBLE_ARRAY,
                [],
            ),
        ).value
        self.bev_destination_points = None
        if hasattr(dst_points_param, '__len__') and len(dst_points_param) == 8:
            self.bev_destination_points = points_from_parameter(
                dst_points_param,
                'bev_destination_points',
            )

        self.inflation_radius_px = int(self.get_parameter('inflation_radius_px').value)
        self.close_kernel_px = int(self.get_parameter('close_kernel_px').value)
        self.open_kernel_px = int(self.get_parameter('open_kernel_px').value)
        self.connected_seed_band_px = int(self.get_parameter('connected_seed_band_px').value)

        self.lookahead_offset_px = int(self.get_parameter('lookahead_offset_px').value)
        self.lookahead_search_window_px = int(
            self.get_parameter('lookahead_search_window_px').value
        )
        self.lookahead_min_corridor_width_px = int(
            self.get_parameter('lookahead_min_corridor_width_px').value
        )
        self.corridor_score_center_weight = float(
            self.get_parameter('corridor_score_center_weight').value
        )
        self.corridor_score_memory_weight = float(
            self.get_parameter('corridor_score_memory_weight').value
        )
        self.target_smoothing_alpha = float(self.get_parameter('target_smoothing_alpha').value)
        self.image_center_bias_px = float(self.get_parameter('image_center_bias_px').value)
        self.enable_auto_mask_recovery = bool(self.get_parameter('enable_auto_mask_recovery').value)

        self.target_timeout_sec = float(self.get_parameter('target_timeout_sec').value)
        self.steering_kp = float(self.get_parameter('steering_kp').value)
        self.steering_kd = float(self.get_parameter('steering_kd').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.min_linear_x = float(self.get_parameter('min_linear_x').value)
        self.max_linear_x = float(self.get_parameter('max_linear_x').value)
        self.speed_reduction_gain = float(self.get_parameter('speed_reduction_gain').value)
        self.narrow_corridor_threshold_px = int(
            self.get_parameter('narrow_corridor_threshold_px').value
        )
        self.narrow_corridor_speed = float(self.get_parameter('narrow_corridor_speed').value)

        control_rate_hz = max(1.0, float(self.get_parameter('control_rate_hz').value))

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.sensor_qos,
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.target_pub = self.create_publisher(Float32, target_topic, 10)
        self.target_valid_pub = self.create_publisher(Bool, target_valid_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, self.sensor_qos)
        self.control_timer = self.create_timer(1.0 / control_rate_hz, self.control_loop)

        self.latest_target: Optional[LookaheadTarget] = None
        self.last_target_time_ns: Optional[int] = None
        self.previous_target_x: Optional[float] = None
        self.previous_steering_error = 0.0
        self.last_control_time_ns: Optional[int] = None
        self.latest_valid_mask: Optional[np.ndarray] = None
        self.latest_obstacle_mask: Optional[np.ndarray] = None
        self.latest_drivable_mask: Optional[np.ndarray] = None
        self.latest_debug_header = None
        self.last_warn_time_ns = 0
        self.last_solution_label = 'uninitialized'

        self.publish_target_valid(False)
        self.get_logger().info(
            'Drivable area planner ready. '
            f'image_topic={self.image_topic}, input_semantics={self.input_semantics}, '
            f'cmd_vel_topic={cmd_vel_topic}'
        )

    @staticmethod
    def image_to_cv(msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.lower()
        if encoding in ('mono8', '8uc1'):
            channels = 1
        elif encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif encoding in ('bgra8', 'rgba8'):
            channels = 4
        else:
            return None

        expected_step = msg.width * channels
        if msg.step < expected_step:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))
        pixels = rows[:, :expected_step]

        if channels == 1:
            return pixels.reshape((msg.height, msg.width))

        image = pixels.reshape((msg.height, msg.width, channels))
        if encoding == 'bgr8':
            return image
        if encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return None

    @staticmethod
    def bgr_to_image_msg(frame: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        return msg

    def publish_target_valid(self, is_valid: bool) -> None:
        valid_msg = Bool()
        valid_msg.data = bool(is_valid)
        self.target_valid_pub.publish(valid_msg)

    def publish_target(self, target_x: float) -> None:
        target_msg = Float32()
        target_msg.data = float(target_x)
        self.target_pub.publish(target_msg)

    def zero_twist(self) -> Twist:
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        return twist

    def build_debug_image(
        self,
        valid_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        drivable_mask: np.ndarray,
        target: Optional[LookaheadTarget],
        linear_x: Optional[float] = None,
        angular_z: Optional[float] = None,
    ) -> np.ndarray:
        debug = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        debug[valid_mask > 0] = (40, 40, 40)
        debug[drivable_mask > 0] = (255, 255, 255)
        debug[obstacle_mask > 0] = (0, 0, 0)

        image_center_x = int((0.5 * self.bev_width) + self.image_center_bias_px)
        cv2.line(
            debug,
            (image_center_x, 0),
            (image_center_x, self.bev_height - 1),
            (255, 255, 0),
            1,
        )

        lookahead_y = int(clamp(
            self.bev_height - 1 - self.lookahead_offset_px,
            0,
            self.bev_height - 1,
        ))
        cv2.line(
            debug,
            (0, lookahead_y),
            (self.bev_width - 1, lookahead_y),
            (0, 255, 255),
            1,
        )

        if target is not None:
            cv2.line(
                debug,
                (target.corridor_left_px, target.y),
                (target.corridor_right_px, target.y),
                (255, 0, 255),
                2,
            )
            cv2.circle(
                debug,
                (int(round(target.x)), target.y),
                6,
                (0, 200, 0),
                -1,
            )

        if linear_x is not None and angular_z is not None:
            cv2.putText(
                debug,
                f'v={linear_x:.2f} w={angular_z:.2f}',
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return debug

    def warn_throttled(self, message: str, throttle_sec: float = 1.5) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self.last_warn_time_ns == 0 or (now_ns - self.last_warn_time_ns) / 1e9 >= throttle_sec:
            self.get_logger().warn(message)
            self.last_warn_time_ns = now_ns

    def select_candidate_solution(
        self,
        bev_obstacle_mask: np.ndarray,
        valid_mask: np.ndarray,
        lookahead_y: int,
        preferred_x: float,
    ) -> Optional[CandidateSolution]:
        obstacle_candidates: List[Tuple[str, np.ndarray]] = [('base', bev_obstacle_mask)]
        if self.enable_auto_mask_recovery and self.input_semantics in ('obstacles_white', 'drivable_white'):
            obstacle_candidates.append(('inverted', cv2.bitwise_not(bev_obstacle_mask)))

        inflation_candidates = [max(0, self.inflation_radius_px)]
        min_width_candidates = [max(1, self.lookahead_min_corridor_width_px)]
        seed_band_candidates = [self.connected_seed_band_px]

        if self.enable_auto_mask_recovery and self.input_semantics in ('obstacles_white', 'drivable_white'):
            inflation_candidates.extend([max(0, self.inflation_radius_px // 2), 0])
            min_width_candidates.append(max(12, self.lookahead_min_corridor_width_px // 2))
            seed_band_candidates.append(0)

        inflation_candidates = list(dict.fromkeys(inflation_candidates))
        min_width_candidates = list(dict.fromkeys(min_width_candidates))
        seed_band_candidates = list(dict.fromkeys(seed_band_candidates))

        best_solution: Optional[CandidateSolution] = None

        for obstacle_label, candidate_obstacle_mask in obstacle_candidates:
            for inflation_radius_px in inflation_candidates:
                for connected_seed_band_px in seed_band_candidates:
                    drivable_mask, inflated_obstacle_mask = build_drivable_area_mask(
                        bev_obstacle_mask=candidate_obstacle_mask,
                        valid_mask=valid_mask,
                        inflation_radius_px=inflation_radius_px,
                        close_kernel_px=self.close_kernel_px,
                        open_kernel_px=self.open_kernel_px,
                        connected_seed_band_px=connected_seed_band_px,
                    )
                    for min_corridor_width_px in min_width_candidates:
                        target = find_lookahead_target(
                            drivable_mask=drivable_mask,
                            lookahead_y=lookahead_y,
                            min_corridor_width_px=min_corridor_width_px,
                            search_window_px=self.lookahead_search_window_px,
                            center_weight=self.corridor_score_center_weight,
                            memory_weight=self.corridor_score_memory_weight,
                            preferred_x=preferred_x,
                            previous_target_x=self.previous_target_x,
                        )
                        if target is None:
                            continue

                        score = float(target.corridor_width_px)
                        score -= 0.18 * abs(target.x - preferred_x)
                        score -= 2.0 * abs(inflation_radius_px - self.inflation_radius_px)
                        score -= 0.35 * abs(min_corridor_width_px - self.lookahead_min_corridor_width_px)
                        if obstacle_label != 'base':
                            score -= 8.0
                        if connected_seed_band_px <= 0:
                            score -= 6.0

                        solution = CandidateSolution(
                            label=(
                                f'{obstacle_label}'
                                f'|infl={inflation_radius_px}'
                                f'|seed={connected_seed_band_px}'
                                f'|minw={min_corridor_width_px}'
                            ),
                            target=target,
                            drivable_mask=drivable_mask,
                            obstacle_mask=inflated_obstacle_mask,
                            score=score,
                        )
                        if best_solution is None or solution.score > best_solution.score:
                            best_solution = solution

        return best_solution

    def image_callback(self, msg: Image) -> None:
        frame = self.image_to_cv(msg)
        if frame is None:
            self.warn_throttled(f'Unsupported image encoding: {msg.encoding}')
            return

        try:
            obstacle_mask = build_obstacle_mask(
                image=frame,
                input_semantics=self.input_semantics,
                mono_threshold=self.mono_threshold,
                white_value_min=self.white_value_min,
                white_sat_max=self.white_sat_max,
                yellow_h_min=self.yellow_h_min,
                yellow_h_max=self.yellow_h_max,
                yellow_sat_min=self.yellow_sat_min,
                yellow_val_min=self.yellow_val_min,
                orange_h_min=self.orange_h_min,
                orange_h_max=self.orange_h_max,
                orange_sat_min=self.orange_sat_min,
                orange_val_min=self.orange_val_min,
            )
        except ValueError as exc:
            self.warn_throttled(str(exc))
            return

        bev_obstacle_mask, _, valid_mask = warp_to_bev(
            obstacle_mask,
            src_points=self.bev_source_points,
            output_size=(self.bev_width, self.bev_height),
            dst_points=self.bev_destination_points,
            interpolation=cv2.INTER_NEAREST,
        )

        drivable_mask, inflated_obstacle_mask = build_drivable_area_mask(
            bev_obstacle_mask=bev_obstacle_mask,
            valid_mask=valid_mask,
            inflation_radius_px=self.inflation_radius_px,
            close_kernel_px=self.close_kernel_px,
            open_kernel_px=self.open_kernel_px,
            connected_seed_band_px=self.connected_seed_band_px,
        )

        lookahead_y = int(
            clamp(
                self.bev_height - 1 - self.lookahead_offset_px,
                0,
                self.bev_height - 1,
            )
        )
        preferred_x = (0.5 * self.bev_width) + self.image_center_bias_px
        solution = self.select_candidate_solution(
            bev_obstacle_mask=bev_obstacle_mask,
            valid_mask=valid_mask,
            lookahead_y=lookahead_y,
            preferred_x=preferred_x,
        )

        if solution is not None:
            target = solution.target
            drivable_mask = solution.drivable_mask
            inflated_obstacle_mask = solution.obstacle_mask
            if solution.label != self.last_solution_label:
                self.get_logger().info(f'Drivable mask solution selected: {solution.label}')
                self.last_solution_label = solution.label
        else:
            target = None

        if target is not None:
            if self.previous_target_x is not None and self.target_smoothing_alpha > 0.0:
                smoothed_x = (
                    self.target_smoothing_alpha * target.x
                    + (1.0 - self.target_smoothing_alpha) * self.previous_target_x
                )
                target = LookaheadTarget(
                    x=smoothed_x,
                    y=target.y,
                    corridor_left_px=target.corridor_left_px,
                    corridor_right_px=target.corridor_right_px,
                    corridor_width_px=target.corridor_width_px,
                )
            self.latest_target = target
            self.previous_target_x = target.x
            self.last_target_time_ns = self.get_clock().now().nanoseconds
            self.publish_target(target.x)
            self.publish_target_valid(True)
        else:
            self.latest_target = None
            self.publish_target_valid(False)

        self.latest_valid_mask = valid_mask
        self.latest_obstacle_mask = inflated_obstacle_mask
        self.latest_drivable_mask = drivable_mask
        self.latest_debug_header = msg.header

    def control_loop(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        dt = 0.0
        if self.last_control_time_ns is not None:
            dt = (now_ns - self.last_control_time_ns) / 1e9
        self.last_control_time_ns = now_ns

        target_is_stale = True
        if self.last_target_time_ns is not None:
            target_is_stale = ((now_ns - self.last_target_time_ns) / 1e9) > self.target_timeout_sec

        if self.latest_target is None or target_is_stale:
            self.cmd_pub.publish(self.zero_twist())
            self.previous_steering_error = 0.0
            self.publish_target_valid(False)
            if self.latest_target is None:
                self.warn_throttled('No valid drivable corridor on lookahead row; stopping.')
            else:
                self.warn_throttled('Lookahead target timed out; stopping.')
            if (
                self.publish_debug_image
                and self.latest_valid_mask is not None
                and self.latest_obstacle_mask is not None
                and self.latest_drivable_mask is not None
                and self.latest_debug_header is not None
            ):
                debug = self.build_debug_image(
                    valid_mask=self.latest_valid_mask,
                    obstacle_mask=self.latest_obstacle_mask,
                    drivable_mask=self.latest_drivable_mask,
                    target=None if target_is_stale else self.latest_target,
                    linear_x=0.0,
                    angular_z=0.0,
                )
                self.debug_pub.publish(self.bgr_to_image_msg(debug, self.latest_debug_header))
            return

        linear_x, angular_z, steering_error = compute_cmd_vel(
            target_x=self.latest_target.x,
            image_width=self.bev_width,
            dt=dt,
            previous_error=self.previous_steering_error,
            steering_kp=self.steering_kp,
            steering_kd=self.steering_kd,
            max_angular_z=self.max_angular_z,
            min_linear_x=self.min_linear_x,
            max_linear_x=self.max_linear_x,
            speed_reduction_gain=self.speed_reduction_gain,
            image_center_bias_px=self.image_center_bias_px,
        )

        if self.latest_target.corridor_width_px < self.narrow_corridor_threshold_px:
            linear_x = min(linear_x, self.narrow_corridor_speed)

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)
        self.previous_steering_error = steering_error

        if (
            self.publish_debug_image
            and self.latest_valid_mask is not None
            and self.latest_obstacle_mask is not None
            and self.latest_drivable_mask is not None
            and self.latest_debug_header is not None
        ):
            debug = self.build_debug_image(
                valid_mask=self.latest_valid_mask,
                obstacle_mask=self.latest_obstacle_mask,
                drivable_mask=self.latest_drivable_mask,
                target=self.latest_target,
                linear_x=linear_x,
                angular_z=angular_z,
            )
            self.debug_pub.publish(self.bgr_to_image_msg(debug, self.latest_debug_header))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DrivableAreaLocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
