#!/usr/bin/env python3

import collections
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


LaneBox = Tuple[float, float, float, float, float]


class LaneDetectionParser(Node):
    def __init__(self) -> None:
        super().__init__('lane_detection_parser')

        self.declare_parameter('detections_topic', '/lane/detections')
        self.declare_parameter('input_image_topic', '/lane/camera/image_raw')
        self.declare_parameter('mask_image_topic', '/lane/mask_image')
        self.declare_parameter('lane_error_topic', '/lane/error')
        self.declare_parameter('lane_heading_error_topic', '/lane/heading_error')
        self.declare_parameter('lane_available_topic', '/lane/available')
        self.declare_parameter('lane_valid_topic', '/lane/valid')
        self.declare_parameter('left_lane_topic', '/lane/left')
        self.declare_parameter('right_lane_topic', '/lane/right')
        self.declare_parameter('lane_deviation_px_topic', '/lane/deviation_px')
        self.declare_parameter('near_error_topic', '/lane/near_error')
        self.declare_parameter('far_error_topic', '/lane/far_error')
        self.declare_parameter('curve_indicator_topic', '/lane/curve_indicator')
        self.declare_parameter('lane_confidence_topic', '/lane/confidence')
        self.declare_parameter('lane_available_confidence_min', 0.30)
        self.declare_parameter('debug_image_topic', '/lane/debug')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('expected_lane_width_px', 260.0)
        self.declare_parameter('smoothing_alpha', 0.35)
        self.declare_parameter('fused_error_smoothing_alpha', -1.0)
        self.declare_parameter('single_lane_error_scale', 1.0)
        self.declare_parameter('adaptive_lane_width_alpha', 0.12)
        self.declare_parameter('single_lane_bias_px', 14.0)
        self.declare_parameter('single_lane_left_bias_px', 0.0)
        self.declare_parameter('single_lane_right_bias_px', 0.0)
        self.declare_parameter('single_lane_center_smoothing_alpha', 0.28)
        self.declare_parameter('single_lane_confidence_floor', 0.38)
        self.declare_parameter('single_lane_projected_confidence_floor', 0.46)
        self.declare_parameter('min_lane_width_px', 120.0)
        self.declare_parameter('max_lane_width_px', 360.0)
        self.declare_parameter('single_lane_max_abs_error', 0.22)
        self.declare_parameter('single_lane_valid_abs_error', 0.18)
        self.declare_parameter('single_lane_low_conf_threshold', 0.55)
        self.declare_parameter('single_lane_low_conf_max_abs_error', 0.14)
        self.declare_parameter('single_lane_low_conf_max_heading_error', 0.10)
        self.declare_parameter('two_lane_max_abs_error', 0.45)
        self.declare_parameter('near_roi_y_start', 0.76)
        self.declare_parameter('near_roi_y_end', 0.94)
        self.declare_parameter('far_roi_y_start', 0.34)
        self.declare_parameter('far_roi_y_end', 0.54)
        self.declare_parameter('w_near', 0.60)
        self.declare_parameter('w_far', 0.40)
        self.declare_parameter('near_search_half_width_ratio', 0.12)
        self.declare_parameter('far_search_half_width_ratio', 0.22)
        self.declare_parameter('side_hist_margin_px', 28)
        self.declare_parameter('sample_bbox_margin_px', 18)
        self.declare_parameter('sample_min_pixels', 24)
        self.declare_parameter('sample_min_peak_pixels', 5)
        self.declare_parameter('sample_min_rows', 4)
        self.declare_parameter('side_hist_min_pixels', 28)
        self.declare_parameter('side_hist_min_peak_pixels', 6)
        self.declare_parameter('far_sample_min_pixels', 10)
        self.declare_parameter('far_sample_min_peak_pixels', 2)
        self.declare_parameter('far_sample_min_rows', 2)
        self.declare_parameter('far_side_hist_min_pixels', 12)
        self.declare_parameter('far_side_hist_min_peak_pixels', 2)
        self.declare_parameter('mask_open_kernel_px', 0)
        self.declare_parameter('mask_dilate_kernel_px', 3)
        self.declare_parameter('prefer_model_mask', True)
        self.declare_parameter('model_mask_min_pixels', 800)
        self.declare_parameter('white_value_min', 170)
        self.declare_parameter('white_sat_max', 80)
        self.declare_parameter('enable_yellow_lane_mask', False)
        self.declare_parameter('yellow_h_min', 15)
        self.declare_parameter('yellow_h_max', 40)
        self.declare_parameter('publish_debug_logs', True)

        # FIX 1: Curve memory & prediction params
        self.declare_parameter('curve_memory_frames', 10)
        self.declare_parameter('max_predict_duration', 4.0)
        self.declare_parameter('decay_start', 2.0)
        self.declare_parameter('max_predicted_error', 0.15)  # FIX F: cap overcorrection
        # FIX 3: Curve warning params
        self.declare_parameter('curve_warning_threshold', 0.05)
        self.declare_parameter('curve_warning_consecutive', 3)

        detections_topic = str(self.get_parameter('detections_topic').value)
        input_image_topic = str(self.get_parameter('input_image_topic').value)
        mask_image_topic = str(self.get_parameter('mask_image_topic').value)
        lane_error_topic = str(self.get_parameter('lane_error_topic').value)
        lane_heading_error_topic = str(self.get_parameter('lane_heading_error_topic').value)
        lane_available_topic = str(self.get_parameter('lane_available_topic').value)
        lane_valid_topic = str(self.get_parameter('lane_valid_topic').value)
        left_lane_topic = str(self.get_parameter('left_lane_topic').value)
        right_lane_topic = str(self.get_parameter('right_lane_topic').value)
        deviation_topic = str(self.get_parameter('lane_deviation_px_topic').value)
        near_error_topic = str(self.get_parameter('near_error_topic').value)
        far_error_topic = str(self.get_parameter('far_error_topic').value)
        curve_indicator_topic = str(self.get_parameter('curve_indicator_topic').value)
        lane_confidence_topic = str(self.get_parameter('lane_confidence_topic').value)
        self.lane_available_confidence_min = float(
            self.get_parameter('lane_available_confidence_min').value
        )
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        self.expected_lane_width_px = float(self.get_parameter('expected_lane_width_px').value)
        legacy_smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        fused_smoothing_alpha = float(self.get_parameter('fused_error_smoothing_alpha').value)
        self.parser_smoothing_alpha = (
            fused_smoothing_alpha if fused_smoothing_alpha >= 0.0 else legacy_smoothing_alpha
        )
        self.single_lane_error_scale = float(self.get_parameter('single_lane_error_scale').value)
        self.adaptive_lane_width_alpha = float(self.get_parameter('adaptive_lane_width_alpha').value)
        self.single_lane_bias_px = float(self.get_parameter('single_lane_bias_px').value)
        self.single_lane_left_bias_px = float(self.get_parameter('single_lane_left_bias_px').value)
        self.single_lane_right_bias_px = float(self.get_parameter('single_lane_right_bias_px').value)
        self.single_lane_center_smoothing_alpha = float(
            self.get_parameter('single_lane_center_smoothing_alpha').value
        )
        self.single_lane_confidence_floor = float(
            self.get_parameter('single_lane_confidence_floor').value
        )
        self.single_lane_projected_confidence_floor = float(
            self.get_parameter('single_lane_projected_confidence_floor').value
        )
        self.min_lane_width_px = float(self.get_parameter('min_lane_width_px').value)
        self.max_lane_width_px = float(self.get_parameter('max_lane_width_px').value)
        self.single_lane_max_abs_error = float(self.get_parameter('single_lane_max_abs_error').value)
        self.single_lane_valid_abs_error = float(self.get_parameter('single_lane_valid_abs_error').value)
        self.single_lane_low_conf_threshold = float(
            self.get_parameter('single_lane_low_conf_threshold').value
        )
        self.single_lane_low_conf_max_abs_error = float(
            self.get_parameter('single_lane_low_conf_max_abs_error').value
        )
        self.single_lane_low_conf_max_heading_error = float(
            self.get_parameter('single_lane_low_conf_max_heading_error').value
        )
        self.two_lane_max_abs_error = float(self.get_parameter('two_lane_max_abs_error').value)
        self.near_roi_y_start = float(self.get_parameter('near_roi_y_start').value)
        self.near_roi_y_end = float(self.get_parameter('near_roi_y_end').value)
        self.far_roi_y_start = float(self.get_parameter('far_roi_y_start').value)
        self.far_roi_y_end = float(self.get_parameter('far_roi_y_end').value)
        self.w_near = float(self.get_parameter('w_near').value)
        self.w_far = float(self.get_parameter('w_far').value)
        self.near_search_half_width_ratio = float(self.get_parameter('near_search_half_width_ratio').value)
        self.far_search_half_width_ratio = float(self.get_parameter('far_search_half_width_ratio').value)
        self.side_hist_margin_px = int(self.get_parameter('side_hist_margin_px').value)
        self.sample_bbox_margin_px = int(self.get_parameter('sample_bbox_margin_px').value)
        self.sample_min_pixels = int(self.get_parameter('sample_min_pixels').value)
        self.sample_min_peak_pixels = int(self.get_parameter('sample_min_peak_pixels').value)
        self.sample_min_rows = int(self.get_parameter('sample_min_rows').value)
        self.side_hist_min_pixels = int(self.get_parameter('side_hist_min_pixels').value)
        self.side_hist_min_peak_pixels = int(self.get_parameter('side_hist_min_peak_pixels').value)
        self.far_sample_min_pixels = int(self.get_parameter('far_sample_min_pixels').value)
        self.far_sample_min_peak_pixels = int(self.get_parameter('far_sample_min_peak_pixels').value)
        self.far_sample_min_rows = int(self.get_parameter('far_sample_min_rows').value)
        self.far_side_hist_min_pixels = int(self.get_parameter('far_side_hist_min_pixels').value)
        self.far_side_hist_min_peak_pixels = int(self.get_parameter('far_side_hist_min_peak_pixels').value)
        self.mask_open_kernel_px = int(self.get_parameter('mask_open_kernel_px').value)
        self.mask_dilate_kernel_px = int(self.get_parameter('mask_dilate_kernel_px').value)
        self.prefer_model_mask = bool(self.get_parameter('prefer_model_mask').value)
        self.model_mask_min_pixels = int(self.get_parameter('model_mask_min_pixels').value)
        self.white_value_min = int(self.get_parameter('white_value_min').value)
        self.white_sat_max = int(self.get_parameter('white_sat_max').value)
        self.enable_yellow_lane_mask = bool(self.get_parameter('enable_yellow_lane_mask').value)
        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.publish_debug_logs = bool(self.get_parameter('publish_debug_logs').value)
        self.adaptive_lane_width_px = self.expected_lane_width_px

        # FIX 1 params
        self.curve_memory_frames = int(self.get_parameter('curve_memory_frames').value)
        self.max_predict_duration = float(self.get_parameter('max_predict_duration').value)
        self.decay_start = float(self.get_parameter('decay_start').value)
        self.max_predicted_error = float(self.get_parameter('max_predicted_error').value)
        # FIX 3 params
        self.curve_warning_threshold = float(self.get_parameter('curve_warning_threshold').value)
        self.curve_warning_consecutive = int(self.get_parameter('curve_warning_consecutive').value)

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(String, detections_topic, self.detections_callback, 10)
        self.image_sub = self.create_subscription(Image, input_image_topic, self.image_callback, self.sensor_qos)
        self.mask_sub = self.create_subscription(Image, mask_image_topic, self.mask_callback, self.sensor_qos)
        self.error_pub = self.create_publisher(Float32, lane_error_topic, 10)
        self.heading_error_pub = self.create_publisher(Float32, lane_heading_error_topic, 10)
        self.available_pub = self.create_publisher(Bool, lane_available_topic, 10)
        self.valid_pub = self.create_publisher(Bool, lane_valid_topic, 10)
        self.left_pub = self.create_publisher(Float32MultiArray, left_lane_topic, 10)
        self.right_pub = self.create_publisher(Float32MultiArray, right_lane_topic, 10)
        self.deviation_pub = self.create_publisher(Float32, deviation_topic, 10)
        self.near_error_pub = self.create_publisher(Float32, near_error_topic, 10)
        self.far_error_pub = self.create_publisher(Float32, far_error_topic, 10)
        self.curve_indicator_pub = self.create_publisher(Float32, curve_indicator_topic, 10)
        self.confidence_pub = self.create_publisher(Float32, lane_confidence_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, self.sensor_qos)
        # FIX 1+3: New publishers
        self.predicted_pub = self.create_publisher(Bool, '/lane/predicted', 10)
        self.curve_warning_pub = self.create_publisher(Bool, '/lane/curve_warning', 10)

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_header = None
        self.latest_model_mask: Optional[np.ndarray] = None
        self.smoothed_error: Optional[float] = None
        self.smoothed_near_center_px: Optional[float] = None
        self.smoothed_far_center_px: Optional[float] = None
        self.last_log_time_ns = 0
        self.last_reliable_near_center_px: Optional[float] = None
        self.last_reliable_far_center_px: Optional[float] = None

        # FIX 1: Ring buffer storing (near_center, curve_indicator, lane_error, wall_time)
        self._valid_frame_buffer: collections.deque = collections.deque(
            maxlen=self.curve_memory_frames
        )
        # Wall time of the last valid detection (for extrapolation timing)
        self._last_valid_time: Optional[float] = None
        # Smoothed predicted error (to avoid jumps when re-entering prediction)
        self._predicted_error: Optional[float] = None

        # FIX 3: Curve warning state — recent |curve| values for trend detection
        self._recent_curve_abs: collections.deque = collections.deque(
            maxlen=self.curve_warning_consecutive
        )

        self.get_logger().info(
            'Lane parser ready. '
            f'detections={detections_topic}, image={input_image_topic}, mask={mask_image_topic}, '
            f'lane_error={lane_error_topic}, '
            f'parser_smoothing_alpha={self.parser_smoothing_alpha:.2f} '
            f'single_lane_biases=({self.single_lane_left_bias_px:+.1f},{self.single_lane_right_bias_px:+.1f}) '
            f'enable_yellow_lane_mask={self.enable_yellow_lane_mask}'
        )
        self.run_single_lane_sign_self_test()

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def parse_detections(payload: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
        image_width = int(payload.get('image_width', 0))
        image_height = int(payload.get('image_height', 0))
        detections = payload.get('detections', [])
        if not isinstance(detections, list):
            detections = []
        return image_width, image_height, detections

    @staticmethod
    def extract_lane_box(det: Dict[str, Any]) -> Optional[LaneBox]:
        bbox = det.get('bbox_xywh')
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            cx = float(bbox[0])
            cy = float(bbox[1])
            w = float(bbox[2])
            h = float(bbox[3])
            conf = float(det.get('confidence', 0.0))
            return cx, cy, w, h, conf
        except (TypeError, ValueError):
            return None

    @staticmethod
    def image_to_bgr(msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.lower()
        if encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif encoding in ('bgra8', 'rgba8'):
            channels = 4
        elif encoding == 'mono8':
            channels = 1
        else:
            return None

        expected_step = msg.width * channels
        if msg.step < expected_step:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))
        pixels = rows[:, :expected_step]

        if channels == 1:
            gray = pixels.reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = pixels.reshape((msg.height, msg.width, channels))
        if encoding == 'bgr8':
            return img
        if encoding == 'rgb8':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return None

    @staticmethod
    def image_to_mask(msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.lower()
        if encoding not in ('mono8', '8uc1'):
            return None

        expected_step = msg.width
        if msg.step < expected_step:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        rows = data.reshape((msg.height, msg.step))
        return rows[:, :expected_step].copy()

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

    @staticmethod
    def point_from_candidate(candidate: Optional[Dict[str, Any]], region_key: str) -> Optional[Dict[str, float]]:
        if candidate is None:
            return None
        point = candidate.get(region_key)
        if point is None:
            return None
        return point

    def image_callback(self, msg: Image) -> None:
        frame = self.image_to_bgr(msg)
        if frame is None:
            return
        self.latest_frame = frame
        self.latest_frame_header = msg.header

    def mask_callback(self, msg: Image) -> None:
        mask = self.image_to_mask(msg)
        if mask is None:
            return
        self.latest_model_mask = mask

    def select_lane_candidates(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[Optional[LaneBox], Optional[LaneBox]]:
        left_best = None
        right_best = None
        left_score = -1.0
        right_score = -1.0

        for det in detections:
            side = str(det.get('side', '')).lower()
            lane_box = self.extract_lane_box(det)
            if lane_box is None:
                continue
            _, cy, _, h, conf = lane_box
            bottom_y = cy + 0.5 * h
            score = conf + 0.002 * bottom_y + 0.0002 * h
            if side == 'left' and score > left_score:
                left_best = lane_box
                left_score = score
            elif side == 'right' and score > right_score:
                right_best = lane_box
                right_score = score

        return left_best, right_best

    def publish_lane_box(self, pub, lane_box: Optional[LaneBox]) -> None:
        msg = Float32MultiArray()
        if lane_box is None:
            msg.data = []
        else:
            cx, cy, w, h, conf = lane_box
            msg.data = [float(cx), float(cy), float(w), float(h), float(conf)]
        pub.publish(msg)

    def publish_scalar(self, pub, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def region_bounds(self, image_height: int, start_ratio: float, end_ratio: float) -> Tuple[int, int]:
        y0 = int(self.clamp(start_ratio, 0.0, 1.0) * image_height)
        y1 = int(self.clamp(end_ratio, 0.0, 1.0) * image_height)
        if y1 <= y0:
            y1 = min(image_height, y0 + 1)
        return y0, y1

    def postprocess_lane_mask(self, mask: np.ndarray) -> np.ndarray:
        processed = mask.copy()
        if self.mask_open_kernel_px > 0:
            kernel = np.ones((self.mask_open_kernel_px, self.mask_open_kernel_px), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        if self.mask_dilate_kernel_px > 0:
            kernel = np.ones((self.mask_dilate_kernel_px, self.mask_dilate_kernel_px), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
        return processed

    def build_lane_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.white_value_min),
            (180, self.white_sat_max, 255),
        )
        mask = white_mask
        if self.enable_yellow_lane_mask:
            yellow_mask = cv2.inRange(
                hsv,
                (self.yellow_h_min, 40, 80),
                (self.yellow_h_max, 255, 255),
            )
            mask = cv2.bitwise_or(mask, yellow_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return self.postprocess_lane_mask(mask)

    def resolve_lane_mask(
        self,
        frame: Optional[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> Optional[np.ndarray]:
        model_mask = None
        if self.latest_model_mask is not None:
            if (
                self.latest_model_mask.shape[1] == image_width
                and self.latest_model_mask.shape[0] == image_height
            ):
                model_mask = self.postprocess_lane_mask(self.latest_model_mask)

        if (
            self.prefer_model_mask
            and model_mask is not None
            and int(np.count_nonzero(model_mask)) >= self.model_mask_min_pixels
        ):
            return model_mask

        if frame is None:
            return model_mask

        return self.build_lane_mask(frame)

    def extract_mask_point(
        self,
        lane_mask: np.ndarray,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        row_bias: str,
        source: str,
        min_pixels: int,
        min_peak_pixels: int,
    ) -> Optional[Dict[str, float]]:
        roi = lane_mask[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        binary = roi > 0
        pixel_count = int(np.count_nonzero(binary))
        if pixel_count < min_pixels:
            return None

        row_counts = np.count_nonzero(binary, axis=1).astype(np.float32)
        active_rows = np.flatnonzero(row_counts > 0.0)
        if active_rows.size < max(1, self.sample_min_rows):
            return None

        column_counts = np.count_nonzero(binary, axis=0).astype(np.float32)
        if column_counts.size == 0:
            return None
        peak = float(np.max(column_counts))
        if peak < float(min_peak_pixels):
            return None

        row_centers: List[float] = []
        row_weights: List[float] = []
        denom = max(1.0, float(binary.shape[0] - 1))
        for row_idx in active_rows:
            cols = np.flatnonzero(binary[row_idx])
            if cols.size == 0:
                continue
            row_center = float(np.mean(cols))
            row_weight = float(cols.size)
            row_pos = float(row_idx) / denom
            if row_bias == 'bottom':
                row_weight *= 0.7 + row_pos
            elif row_bias == 'top':
                row_weight *= 0.7 + (1.0 - row_pos)
            row_centers.append(row_center)
            row_weights.append(row_weight)

        if not row_centers:
            return None

        x_local = float(np.average(np.asarray(row_centers), weights=np.asarray(row_weights)))
        y_local = float(np.average(active_rows.astype(np.float32), weights=np.asarray(row_weights)))
        strength = self.clamp(peak / max(1.0, float(binary.shape[0])), 0.0, 1.0)
        support = self.clamp(pixel_count / max(1.0, float(binary.size)), 0.0, 1.0)
        return {
            'x': float(x0 + x_local),
            'y': float(y0 + y_local),
            'count': float(pixel_count),
            'peak': peak,
            'strength': strength,
            'support': support,
            'source': source,
        }

    def lane_box_vertical_bounds(self, lane_box: LaneBox) -> Tuple[float, float]:
        _, cy, _, h, _ = lane_box
        return cy - 0.5 * h, cy + 0.5 * h

    def select_region_lane_box(
        self,
        candidates: List[Dict[str, Any]],
        side: str,
        y0: int,
        y1: int,
        image_height: int,
    ) -> Optional[Dict[str, Any]]:
        best_candidate = None
        best_score = -1e9
        region_center_y = 0.5 * float(y0 + y1)
        region_height = max(1.0, float(y1 - y0))

        for candidate in candidates:
            if candidate['side'] != side:
                continue
            lane_box = candidate['lane_box']
            top_y, bottom_y = self.lane_box_vertical_bounds(lane_box)
            overlap = max(0.0, min(bottom_y, float(y1)) - max(top_y, float(y0)))
            overlap_ratio = overlap / region_height
            vertical_fit = 1.0 - min(1.0, abs(float(lane_box[1]) - region_center_y) / max(1.0, float(image_height)))
            score = float(candidate['confidence'])
            score += 2.40 * overlap_ratio
            score += 0.55 * vertical_fit
            score += 0.0002 * float(lane_box[3])
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def extract_region_point_from_box(
        self,
        lane_mask: np.ndarray,
        lane_box: LaneBox,
        side: str,
        y0: int,
        y1: int,
        image_width: int,
        region_name: str,
    ) -> Optional[Dict[str, float]]:
        width = image_width
        half = width // 2
        cx, _, w, _, _ = lane_box
        margin = max(0, int(self.sample_bbox_margin_px))
        side_margin = max(0, int(self.side_hist_margin_px))
        width_ratio = self.near_search_half_width_ratio if region_name == 'near' else self.far_search_half_width_ratio
        half_width = max(
            int(np.ceil(0.5 * w)) + margin,
            int(np.ceil(width * max(0.05, width_ratio))),
        )

        x0 = max(0, int(np.floor(cx)) - half_width)
        x1 = min(width, int(np.ceil(cx)) + half_width)
        if side == 'left':
            x1 = min(x1, half + side_margin)
        elif side == 'right':
            x0 = max(x0, half - side_margin)
        if x1 <= x0:
            return None

        row_bias = 'bottom' if region_name == 'near' else 'top'
        return self.extract_mask_point(
            lane_mask=lane_mask,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            row_bias=row_bias,
            source='window',
            min_pixels=self.sample_min_pixels,
            min_peak_pixels=self.sample_min_peak_pixels,
        )

    def extract_region_side_hist_point(
        self,
        lane_mask: np.ndarray,
        side: str,
        y0: int,
        y1: int,
        image_width: int,
        region_name: str,
    ) -> Optional[Dict[str, float]]:
        width = image_width
        half = width // 2
        margin = max(0, int(self.side_hist_margin_px))
        if side == 'left':
            x0 = 0
            x1 = min(width, half + margin)
        else:
            x0 = max(0, half - margin)
            x1 = width

        row_bias = 'bottom' if region_name == 'near' else 'top'
        return self.extract_mask_point(
            lane_mask=lane_mask,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            row_bias=row_bias,
            source='side_hist',
            min_pixels=self.side_hist_min_pixels,
            min_peak_pixels=self.side_hist_min_peak_pixels,
        )

    def annotate_region_candidates(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        for det in detections:
            side = str(det.get('side', '')).lower()
            lane_box = self.extract_lane_box(det)
            if lane_box is None:
                continue
            annotated.append(
                {
                    'side': side,
                    'lane_box': lane_box,
                    'confidence': lane_box[4],
                }
            )
        return annotated

    def resolve_region_side_point(
        self,
        candidates: List[Dict[str, Any]],
        lane_mask: Optional[np.ndarray],
        side: str,
        y0: int,
        y1: int,
        image_width: int,
        image_height: int,
        region_name: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, float]], str]:
        candidate = self.select_region_lane_box(candidates, side, y0, y1, image_height)
        if lane_mask is None:
            return candidate, None, 'no_frame'

        point = None
        if candidate is not None:
            point = self.extract_region_point_from_box(
                lane_mask=lane_mask,
                lane_box=candidate['lane_box'],
                side=side,
                y0=y0,
                y1=y1,
                image_width=image_width,
                region_name=region_name,
            )
            if point is not None:
                return candidate, point, 'window'

        point = self.extract_region_side_hist_point(
            lane_mask=lane_mask,
            side=side,
            y0=y0,
            y1=y1,
            image_width=image_width,
            region_name=region_name,
        )
        if point is not None:
            return candidate, point, 'side_hist'

        return candidate, None, 'box_fallback' if candidate is not None else 'missing'

    def maybe_update_lane_width(
        self,
        left_point: Optional[Dict[str, float]],
        right_point: Optional[Dict[str, float]],
    ) -> None:
        if left_point is None or right_point is None:
            return

        measured_lane_width = abs(float(right_point['x']) - float(left_point['x']))
        if self.min_lane_width_px <= measured_lane_width <= self.max_lane_width_px:
            alpha_w = self.clamp(self.adaptive_lane_width_alpha, 0.0, 1.0)
            self.adaptive_lane_width_px = (
                (1.0 - alpha_w) * self.adaptive_lane_width_px + alpha_w * measured_lane_width
            )

    def smooth_center_value(
        self,
        current_value: Optional[float],
        previous_value: Optional[float],
        alpha: float,
    ) -> Optional[float]:
        if current_value is None:
            return previous_value
        if previous_value is None:
            return current_value
        alpha = self.clamp(alpha, 0.0, 1.0)
        return (alpha * current_value) + ((1.0 - alpha) * previous_value)

    def virtual_center_from_left(self, left_x: float, image_width: int) -> float:
        center = float(left_x) + 0.5 * self.adaptive_lane_width_px + self.single_lane_left_bias_px
        return self.clamp(center, 0.0, float(image_width))

    def virtual_center_from_right(self, right_x: float, image_width: int) -> float:
        center = float(right_x) - 0.5 * self.adaptive_lane_width_px + self.single_lane_right_bias_px
        return self.clamp(center, 0.0, float(image_width))

    @staticmethod
    def center_to_error(center_x: float, image_width: int) -> float:
        image_center = 0.5 * float(image_width)
        return (image_center - float(center_x)) / max(1.0, image_center)

    def run_single_lane_sign_self_test(self) -> None:
        image_width = 640
        self.adaptive_lane_width_px = self.clamp(
            self.adaptive_lane_width_px,
            self.min_lane_width_px,
            self.max_lane_width_px,
        )
        left_center = self.virtual_center_from_left(110.0, image_width)
        right_center = self.virtual_center_from_right(530.0, image_width)
        left_error = self.center_to_error(left_center, image_width)
        right_error = self.center_to_error(right_center, image_width)
        if not (left_error > 0.0 and right_error < 0.0):
            raise RuntimeError(
                f'Single-line sign self-test failed: left_error={left_error:+.3f} '
                f'right_error={right_error:+.3f}'
            )
        self.get_logger().info(
            f'[SINGLE_LINE_SELF_TEST] left_error={left_error:+.3f} right_error={right_error:+.3f}'
        )

    def project_missing_centers(
        self,
        near_center: Optional[float],
        far_center: Optional[float],
        image_width: int,
    ) -> Tuple[Optional[float], Optional[float], bool, bool]:
        near_projected = False
        far_projected = False

        if self.last_reliable_near_center_px is None or self.last_reliable_far_center_px is None:
            return near_center, far_center, near_projected, far_projected

        historical_curve_px = self.last_reliable_far_center_px - self.last_reliable_near_center_px
        if near_center is None and far_center is not None:
            near_center = self.clamp(far_center - historical_curve_px, 0.0, float(image_width))
            near_projected = True
        elif far_center is None and near_center is not None:
            far_center = self.clamp(near_center + historical_curve_px, 0.0, float(image_width))
            far_projected = True

        return near_center, far_center, near_projected, far_projected

    def compute_point_center(
        self,
        left_point: Optional[Dict[str, float]],
        right_point: Optional[Dict[str, float]],
        image_width: int,
    ) -> Optional[float]:
        if left_point is not None and right_point is not None:
            center = 0.5 * (float(left_point['x']) + float(right_point['x']))
            return self.clamp(center, 0.0, float(image_width))
        if left_point is not None:
            return self.virtual_center_from_left(float(left_point['x']), image_width)
        if right_point is not None:
            return self.virtual_center_from_right(float(right_point['x']), image_width)
        return None

    def compute_box_lane_center(
        self,
        left_lane: Optional[LaneBox],
        right_lane: Optional[LaneBox],
        image_width: int,
    ) -> Optional[float]:
        lane_center = None
        if left_lane is not None and right_lane is not None:
            self.maybe_update_lane_width(
                {'x': float(left_lane[0])},
                {'x': float(right_lane[0])},
            )
            lane_center = 0.5 * (left_lane[0] + right_lane[0])
        elif left_lane is not None:
            lane_center = self.virtual_center_from_left(float(left_lane[0]), image_width)
        elif right_lane is not None:
            lane_center = self.virtual_center_from_right(float(right_lane[0]), image_width)

        if lane_center is None:
            return None
        return self.clamp(lane_center, 0.0, float(image_width))

    @staticmethod
    def summarize_region_source(
        left_point: Optional[Dict[str, float]],
        right_point: Optional[Dict[str, float]],
        left_source: str,
        right_source: str,
    ) -> str:
        point_sources = [
            source
            for source, point in ((left_source, left_point), (right_source, right_point))
            if point is not None
        ]
        if not point_sources:
            if left_source == 'no_frame' or right_source == 'no_frame':
                return 'no_frame'
            if left_source == 'missing' and right_source == 'missing':
                return 'missing'
            return 'box_fallback'
        if any(source == 'window' for source in point_sources):
            return 'window'
        if any(source == 'side_hist' for source in point_sources):
            return 'side_hist'
        return point_sources[0]

    @staticmethod
    def format_point_x(point: Optional[Dict[str, float]]) -> str:
        if point is None:
            return 'None'
        return f"{float(point['x']):.1f}"

    @staticmethod
    def format_lane_box_x(candidate: Optional[Dict[str, Any]]) -> str:
        if candidate is None:
            return 'None'
        lane_box = candidate['lane_box']
        return f"{float(lane_box[0]):.1f}"

    def compute_lane_confidence(
        self,
        lane_valid: bool,
        left_visible: bool,
        right_visible: bool,
        near_preview_valid: bool,
        far_preview_valid: bool,
        selected_points: List[Optional[Dict[str, float]]],
        selected_candidates: List[Optional[Dict[str, Any]]],
    ) -> float:
        point_visibility = sum(point is not None for point in selected_points) / 4.0
        boundary_visibility = 0.5 * float(left_visible) + 0.5 * float(right_visible)
        region_visibility = 0.5 * float(near_preview_valid) + 0.5 * float(far_preview_valid)

        det_conf_vals = [
            float(candidate['confidence'])
            for candidate in selected_candidates
            if candidate is not None
        ]
        point_strength_vals = [
            float(point['strength'])
            for point in selected_points
            if point is not None
        ]
        point_source_vals = [
            {
                'window': 1.0,
                'side_hist': 0.82,
                'box_fallback': 0.25,
                'missing': 0.0,
                'no_frame': 0.0,
            }.get(str(point.get('source', 'missing')), 0.0)
            for point in selected_points
            if point is not None
        ]

        det_conf = float(np.mean(det_conf_vals)) if det_conf_vals else 0.0
        point_strength = float(np.mean(point_strength_vals)) if point_strength_vals else 0.0
        point_source_quality = float(np.mean(point_source_vals)) if point_source_vals else 0.0

        confidence = (
            0.22 * point_visibility
            + 0.20 * boundary_visibility
            + 0.18 * region_visibility
            + 0.14 * det_conf
            + 0.12 * point_strength
            + 0.14 * point_source_quality
        )
        confidence = self.clamp(confidence, 0.0, 1.0)

        if not near_preview_valid or not far_preview_valid:
            confidence *= 0.85
        if not (left_visible and right_visible):
            confidence *= 0.70
        if not (near_preview_valid and far_preview_valid):
            confidence *= 0.75
        if not lane_valid:
            confidence = min(confidence, 0.35)
        return confidence

    def publish_debug_overlay(
        self,
        frame: Optional[np.ndarray],
        header,
        image_width: int,
        image_height: int,
        near_left: Optional[Dict[str, float]],
        near_right: Optional[Dict[str, float]],
        far_left: Optional[Dict[str, float]],
        far_right: Optional[Dict[str, float]],
        near_center: Optional[float],
        far_center: Optional[float],
        near_preview_valid: bool,
        far_preview_valid: bool,
        near_source: str,
        far_source: str,
        near_error: float,
        far_error: float,
        curve_indicator: float,
        lane_confidence: float,
    ) -> None:
        if not self.publish_debug_image or frame is None or header is None:
            return
        if self.debug_pub.get_subscription_count() == 0:
            return

        debug = frame.copy()
        near_y0, near_y1 = self.region_bounds(image_height, self.near_roi_y_start, self.near_roi_y_end)
        far_y0, far_y1 = self.region_bounds(image_height, self.far_roi_y_start, self.far_roi_y_end)

        overlay = debug.copy()
        cv2.rectangle(overlay, (0, far_y0), (image_width - 1, far_y1), (255, 170, 0), -1)
        cv2.rectangle(overlay, (0, near_y0), (image_width - 1, near_y1), (0, 180, 255), -1)
        cv2.addWeighted(overlay, 0.12, debug, 0.88, 0.0, debug)

        image_center_x = int(0.5 * image_width)
        cv2.line(debug, (image_center_x, 0), (image_center_x, image_height - 1), (190, 190, 190), 1)
        cv2.line(debug, (0, far_y0), (image_width - 1, far_y0), (255, 170, 0), 1)
        cv2.line(debug, (0, far_y1), (image_width - 1, far_y1), (255, 170, 0), 1)
        cv2.line(debug, (0, near_y0), (image_width - 1, near_y0), (0, 180, 255), 1)
        cv2.line(debug, (0, near_y1), (image_width - 1, near_y1), (0, 180, 255), 1)

        def draw_point(point: Optional[Dict[str, float]], color: Tuple[int, int, int], label: str) -> None:
            if point is None:
                return
            px = int(point['x'])
            py = int(point['y'])
            cv2.circle(debug, (px, py), 6, color, -1)
            cv2.putText(debug, label, (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        draw_point(far_left, (0, 220, 0), 'far_L')
        draw_point(far_right, (0, 140, 255), 'far_R')
        draw_point(near_left, (0, 255, 120), 'near_L')
        draw_point(near_right, (0, 220, 255), 'near_R')

        if far_center is not None:
            far_center_pt = (int(far_center), int(0.5 * (far_y0 + far_y1)))
            cv2.circle(debug, far_center_pt, 7, (255, 0, 255), -1)
        if near_center is not None:
            near_center_pt = (int(near_center), int(0.5 * (near_y0 + near_y1)))
            cv2.circle(debug, near_center_pt, 7, (255, 255, 0), -1)

        text_lines = [
            f'near_err={near_error:+.3f} src={near_source} {"PREVIEW" if near_preview_valid else "FALLBACK"}',
            f'far_err={far_error:+.3f} src={far_source} {"PREVIEW" if far_preview_valid else "FALLBACK"}',
            f'curve={curve_indicator:+.3f} conf={lane_confidence:.2f}',
        ]
        for idx, text in enumerate(text_lines):
            cv2.putText(
                debug,
                text,
                (12, 28 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        self.debug_pub.publish(self.bgr_to_image_msg(debug, header))

    def detections_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        image_width, image_height, detections = self.parse_detections(payload)
        if image_width <= 0 or image_height <= 0:
            return

        left_lane, right_lane = self.select_lane_candidates(detections)
        fallback_center = self.compute_box_lane_center(left_lane, right_lane, image_width)

        frame = None
        header = None
        if (
            self.latest_frame is not None
            and self.latest_frame.shape[1] == image_width
            and self.latest_frame.shape[0] == image_height
        ):
            frame = self.latest_frame
            header = self.latest_frame_header

        lane_mask = self.resolve_lane_mask(frame, image_width, image_height)
        candidates = self.annotate_region_candidates(detections)
        near_y0, near_y1 = self.region_bounds(image_height, self.near_roi_y_start, self.near_roi_y_end)
        far_y0, far_y1 = self.region_bounds(image_height, self.far_roi_y_start, self.far_roi_y_end)

        near_left_candidate, near_left_point, near_left_source = self.resolve_region_side_point(
            candidates=candidates,
            lane_mask=lane_mask,
            side='left',
            y0=near_y0,
            y1=near_y1,
            image_width=image_width,
            image_height=image_height,
            region_name='near',
        )
        near_right_candidate, near_right_point, near_right_source = self.resolve_region_side_point(
            candidates=candidates,
            lane_mask=lane_mask,
            side='right',
            y0=near_y0,
            y1=near_y1,
            image_width=image_width,
            image_height=image_height,
            region_name='near',
        )
        far_left_candidate, far_left_point, far_left_source = self.resolve_region_side_point(
            candidates=candidates,
            lane_mask=lane_mask,
            side='left',
            y0=far_y0,
            y1=far_y1,
            image_width=image_width,
            image_height=image_height,
            region_name='far',
        )
        far_right_candidate, far_right_point, far_right_source = self.resolve_region_side_point(
            candidates=candidates,
            lane_mask=lane_mask,
            side='right',
            y0=far_y0,
            y1=far_y1,
            image_width=image_width,
            image_height=image_height,
            region_name='far',
        )

        if near_left_point is not None and near_right_point is not None:
            self.maybe_update_lane_width(near_left_point, near_right_point)
        elif far_left_point is not None and far_right_point is not None:
            self.maybe_update_lane_width(far_left_point, far_right_point)

        near_center_region = self.compute_point_center(near_left_point, near_right_point, image_width)
        far_center_region = self.compute_point_center(far_left_point, far_right_point, image_width)
        raw_near_preview_valid = near_center_region is not None
        raw_far_preview_valid = far_center_region is not None
        near_center_region, far_center_region, near_projected, far_projected = self.project_missing_centers(
            near_center_region,
            far_center_region,
            image_width,
        )
        near_preview_valid = near_center_region is not None
        far_preview_valid = far_center_region is not None
        near_source = self.summarize_region_source(
            near_left_point,
            near_right_point,
            near_left_source,
            near_right_source,
        )
        far_source = self.summarize_region_source(
            far_left_point,
            far_right_point,
            far_left_source,
            far_right_source,
        )
        if near_projected:
            near_source = 'history_projected'
        if far_projected:
            far_source = 'history_projected'

        near_center = near_center_region if near_center_region is not None else fallback_center
        far_center = far_center_region if far_center_region is not None else fallback_center
        center_alpha = self.single_lane_center_smoothing_alpha
        near_center = self.smooth_center_value(near_center, self.smoothed_near_center_px, center_alpha)
        far_center = self.smooth_center_value(far_center, self.smoothed_far_center_px, center_alpha)
        self.smoothed_near_center_px = near_center
        self.smoothed_far_center_px = far_center

        image_center = 0.5 * image_width
        fallback_error = (
            (image_center - fallback_center) / max(1.0, image_center)
            if fallback_center is not None
            else 0.0
        )
        near_error = (
            (image_center - near_center) / max(1.0, image_center)
            if near_center is not None
            else fallback_error
        )
        far_error = (
            (image_center - far_center) / max(1.0, image_center)
            if far_center is not None
            else fallback_error
        )
        curve_indicator = (
            (far_center - near_center) / max(1.0, image_center)
            if near_center is not None and far_center is not None
            else 0.0
        )
        heading_error = (
            (far_error - near_error)
            if near_center is not None and far_center is not None
            else 0.0
        )

        fusion_terms: List[Tuple[float, float]] = []
        if near_center_region is not None:
            fusion_terms.append((self.w_near, near_error))
        if far_center_region is not None:
            fusion_terms.append((self.w_far, far_error))

        if fusion_terms:
            total_weight = sum(weight for weight, _ in fusion_terms)
            lane_error = sum(weight * value for weight, value in fusion_terms) / max(total_weight, 1e-6)
        else:
            lane_error = fallback_error

        left_visible = left_lane is not None
        right_visible = right_lane is not None
        lane_valid = (left_visible or right_visible) and (near_center is not None or far_center is not None)
        both_lanes = left_visible and right_visible
        single_lane_pair_available = (
            (near_left_point is not None and far_left_point is not None)
            or (near_right_point is not None and far_right_point is not None)
        )
        single_lane_supported = single_lane_pair_available or near_projected or far_projected
        deviation_px = image_center - (
            near_center_region
            if near_center_region is not None
            else far_center_region
            if far_center_region is not None
            else fallback_center
            if fallback_center is not None
            else image_center
        )

        if not both_lanes:
            lane_error *= self.single_lane_error_scale
            lane_error = self.clamp(lane_error, -self.single_lane_max_abs_error, self.single_lane_max_abs_error)
            if abs(lane_error) > self.single_lane_valid_abs_error:
                lane_valid = False
            if not (single_lane_supported or (near_preview_valid and far_preview_valid)):
                lane_valid = False
        else:
            lane_error = self.clamp(lane_error, -self.two_lane_max_abs_error, self.two_lane_max_abs_error)
        lane_error = self.clamp(lane_error, -1.0, 1.0)
        heading_error = self.clamp(heading_error, -1.0, 1.0)

        if self.smoothed_error is None:
            self.smoothed_error = lane_error
        else:
            alpha = self.clamp(self.parser_smoothing_alpha, 0.0, 1.0)
            self.smoothed_error = alpha * lane_error + (1.0 - alpha) * self.smoothed_error

        lane_confidence = self.compute_lane_confidence(
            lane_valid=lane_valid,
            left_visible=left_visible,
            right_visible=right_visible,
            near_preview_valid=near_preview_valid,
            far_preview_valid=far_preview_valid,
            selected_points=[
                near_left_point,
                near_right_point,
                far_left_point,
                far_right_point,
            ],
            selected_candidates=[
                near_left_candidate,
                near_right_candidate,
                far_left_candidate,
                far_right_candidate,
            ],
        )
        if not both_lanes:
            if single_lane_supported:
                lane_confidence = max(lane_confidence, self.single_lane_projected_confidence_floor)
            elif raw_near_preview_valid or raw_far_preview_valid:
                lane_confidence = max(lane_confidence, self.single_lane_confidence_floor)
            if (near_projected or far_projected) and not single_lane_pair_available:
                lane_confidence = min(lane_confidence, self.single_lane_low_conf_threshold)
        lane_confidence = self.clamp(lane_confidence, 0.0, 1.0)
        if not both_lanes and lane_confidence < self.single_lane_low_conf_threshold:
            lane_error = self.clamp(
                lane_error,
                -self.single_lane_low_conf_max_abs_error,
                self.single_lane_low_conf_max_abs_error,
            )
            heading_error = self.clamp(
                heading_error,
                -self.single_lane_low_conf_max_heading_error,
                self.single_lane_low_conf_max_heading_error,
            )

        # ------------------------------------------------------------------
        # FIX 1: Curve memory & prediction
        # ------------------------------------------------------------------
        now_wall = time.monotonic()
        is_predicted = False
        published_error = self.smoothed_error if self.smoothed_error is not None else lane_error

        if lane_valid:
            if near_center is not None:
                self.last_reliable_near_center_px = near_center
            if far_center is not None:
                self.last_reliable_far_center_px = far_center
            # Store this valid frame in ring buffer
            self._valid_frame_buffer.append(
                (near_center if near_center is not None else float(0.5 * image_width),
                 float(curve_indicator),
                 float(lane_error),
                 now_wall)
            )
            self._last_valid_time = now_wall
            self._predicted_error = None  # reset prediction state on valid frame
        else:
            # Lane invalid — attempt prediction from ring buffer
            if self._last_valid_time is not None and len(self._valid_frame_buffer) >= 2:
                dt = now_wall - self._last_valid_time
                if dt <= self.max_predict_duration:
                    # Compute average curve_rate from buffer
                    frames = list(self._valid_frame_buffer)
                    last_near_center = float(frames[-1][0])
                    curve_values = [f[1] for f in frames]
                    avg_curve_rate = float(np.mean(curve_values))

                    # Extrapolate lane center
                    image_center = 0.5 * float(image_width)
                    predicted_center = last_near_center + avg_curve_rate * dt * float(image_width)
                    predicted_center = self.clamp(predicted_center, 0.0, float(image_width))
                    raw_predicted_err = (image_center - predicted_center) / max(1.0, image_center)

                    # Apply decay toward 0 after decay_start seconds
                    if dt > self.decay_start:
                        decay_window = max(1e-6, self.max_predict_duration - self.decay_start)
                        decay_factor = max(0.0, 1.0 - (dt - self.decay_start) / decay_window)
                        raw_predicted_err *= decay_factor

                    # FIX F: cap predicted error to prevent overcorrection (was hitting ±0.48)
                    raw_predicted_err = self.clamp(
                        raw_predicted_err,
                        -self.max_predicted_error,
                        self.max_predicted_error,
                    )

                    # Smooth the predicted error to avoid jumps
                    if self._predicted_error is None:
                        self._predicted_error = raw_predicted_err
                    else:
                        self._predicted_error = 0.25 * raw_predicted_err + 0.75 * self._predicted_error
                    self._predicted_error = self.clamp(
                        self._predicted_error,
                        -self.max_predicted_error,
                        self.max_predicted_error,
                    )

                    published_error = self._predicted_error
                    is_predicted = True

        # ------------------------------------------------------------------
        # FIX 3: Curve warning — detect imminent curve BEFORE lane_valid drops
        # ------------------------------------------------------------------
        self._recent_curve_abs.append(abs(float(curve_indicator)))
        curve_warning = False
        if len(self._recent_curve_abs) == self.curve_warning_consecutive:
            curve_vals = list(self._recent_curve_abs)
            # All values above threshold AND strictly increasing
            if (all(v > self.curve_warning_threshold for v in curve_vals)
                    and all(curve_vals[i] < curve_vals[i + 1] for i in range(len(curve_vals) - 1))):
                curve_warning = True

        # ------------------------------------------------------------------
        # Publish error (real or predicted) and all status topics
        # ------------------------------------------------------------------
        self.publish_scalar(self.error_pub, float(published_error))
        self.publish_scalar(self.heading_error_pub, float(heading_error))

        valid_msg = Bool()
        valid_msg.data = bool(lane_valid)
        self.valid_pub.publish(valid_msg)

        available_msg = Bool()
        available_msg.data = bool(lane_valid and lane_confidence >= self.lane_available_confidence_min)
        self.available_pub.publish(available_msg)

        predicted_msg = Bool()
        predicted_msg.data = bool(is_predicted)
        self.predicted_pub.publish(predicted_msg)

        curve_warning_msg = Bool()
        curve_warning_msg.data = bool(curve_warning)
        self.curve_warning_pub.publish(curve_warning_msg)

        self.publish_scalar(self.deviation_pub, float(deviation_px))
        self.publish_scalar(self.near_error_pub, float(near_error))
        self.publish_scalar(self.far_error_pub, float(far_error))
        self.publish_scalar(self.curve_indicator_pub, float(curve_indicator))
        self.publish_scalar(self.confidence_pub, float(lane_confidence))

        self.publish_lane_box(self.left_pub, left_lane)
        self.publish_lane_box(self.right_pub, right_lane)
        self.publish_debug_overlay(
            frame=frame,
            header=header,
            image_width=image_width,
            image_height=image_height,
            near_left=near_left_point,
            near_right=near_right_point,
            far_left=far_left_point,
            far_right=far_right_point,
            near_center=near_center,
            far_center=far_center,
            near_preview_valid=near_preview_valid,
            far_preview_valid=far_preview_valid,
            near_source=near_source,
            far_source=far_source,
            near_error=near_error,
            far_error=far_error,
            curve_indicator=curve_indicator,
            lane_confidence=lane_confidence,
        )

        if self.publish_debug_logs:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_log_time_ns > int(1e9):
                self.last_log_time_ns = now_ns
                self.get_logger().info(
                    f'lane_valid={lane_valid} predicted={is_predicted} curve_warn={curve_warning} '
                    f'err={published_error:+.3f} near={near_error:+.3f} '
                    f'far={far_error:+.3f} curve={curve_indicator:+.3f} conf={lane_confidence:.2f} '
                    f'left={left_visible} right={right_visible} near_center={near_center if near_center is not None else float("nan"):.1f} '
                    f'far_center={far_center if far_center is not None else float("nan"):.1f} '
                    f'near_src={near_source} far_src={far_source} single_supported={single_lane_supported} '
                    f'near_sel=L:{self.format_lane_box_x(near_left_candidate)}/{self.format_point_x(near_left_point)}({near_left_source}) '
                    f'R:{self.format_lane_box_x(near_right_candidate)}/{self.format_point_x(near_right_point)}({near_right_source}) '
                    f'far_sel=L:{self.format_lane_box_x(far_left_candidate)}/{self.format_point_x(far_left_point)}({far_left_source}) '
                    f'R:{self.format_lane_box_x(far_right_candidate)}/{self.format_point_x(far_right_point)}({far_right_source})'
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneDetectionParser()
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
