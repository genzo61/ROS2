#!/usr/bin/env python3

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


class LaneYoloInference(Node):
    def __init__(self) -> None:
        super().__init__('lane_yolo_inference')

        self.declare_parameter('image_topic', '/lane/camera/image_raw')
        self.declare_parameter('detections_topic', '/lane/detections')
        self.declare_parameter('debug_image_topic', '/lane/yolo/debug')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('left_labels', ['left_lane', 'lane_left', 'left'])
        self.declare_parameter('right_labels', ['right_lane', 'lane_right', 'right'])
        self.declare_parameter('enable_fallback_lane_extraction', True)
        self.declare_parameter('fallback_roi_top_ratio', 0.55)
        self.declare_parameter('fallback_white_value_min', 170)
        self.declare_parameter('fallback_white_sat_max', 80)
        self.declare_parameter('fallback_yellow_h_min', 15)
        self.declare_parameter('fallback_yellow_h_max', 40)
        self.declare_parameter('fallback_min_peak_pixels', 120)

        self.image_topic = str(self.get_parameter('image_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        requested_model_path = str(self.get_parameter('model_path').value).strip()
        self.model_path = self.resolve_model_path(requested_model_path)
        self.conf_th = float(self.get_parameter('confidence_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.left_labels = {str(label).lower() for label in self.get_parameter('left_labels').value}
        self.right_labels = {str(label).lower() for label in self.get_parameter('right_labels').value}
        self.enable_fallback = bool(self.get_parameter('enable_fallback_lane_extraction').value)
        self.fallback_roi_top_ratio = float(self.get_parameter('fallback_roi_top_ratio').value)
        self.fallback_white_value_min = int(self.get_parameter('fallback_white_value_min').value)
        self.fallback_white_sat_max = int(self.get_parameter('fallback_white_sat_max').value)
        self.fallback_yellow_h_min = int(self.get_parameter('fallback_yellow_h_min').value)
        self.fallback_yellow_h_max = int(self.get_parameter('fallback_yellow_h_max').value)
        self.fallback_min_peak_pixels = int(self.get_parameter('fallback_min_peak_pixels').value)

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, self.sensor_qos)
        self.detections_pub = self.create_publisher(String, detections_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, self.sensor_qos)

        self.model = self.load_model()
        mode = 'YOLO' if self.model is not None else 'fallback'
        self.get_logger().info(f'Lane inference ready. mode={mode} image_topic={self.image_topic}')

    def resolve_model_path(self, requested_path: str) -> str:
        requested = requested_path.strip()
        if requested and requested.lower() != 'auto':
            return requested

        candidate_paths: List[str] = []
        env_model_path = os.getenv('LANE_MODEL_PATH', '').strip()
        if env_model_path:
            candidate_paths.append(env_model_path)

        try:
            pkg_share = get_package_share_directory('vehicle_bringup')
            candidate_paths.append(os.path.join(pkg_share, 'models', 'best.pt'))
        except Exception:
            pass

        candidate_paths.extend(
            [
                os.path.expanduser('~/turtlebot3_ws/src/vehicle_bringup/models/best.pt'),
                os.path.expanduser('~/turtlebot3_ws/src/best.pt'),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')),
            ]
        )

        for path in candidate_paths:
            if path and os.path.isfile(path):
                self.get_logger().info(f'Lane model auto-resolved: {path}')
                return path

        return ''

    def load_model(self):
        if not self.model_path:
            self.get_logger().warn(
                'YOLO model_path is empty/unresolved. Fallback lane extraction will be used.'
            )
            return None
        if not os.path.isfile(self.model_path):
            self.get_logger().warn(
                f'YOLO model file does not exist: {self.model_path}. Fallback lane extraction will be used.'
            )
            return None
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f'ultralytics import failed: {exc}. Fallback lane extraction will be used.')
            return None
        try:
            model = YOLO(self.model_path)
            self.get_logger().info(f'YOLO model loaded: {self.model_path}')
            return model
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f'YOLO model load failed: {exc}. Fallback lane extraction will be used.')
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

    def classify_lane_side(self, label: str, cx: float, image_width: int) -> str:
        normalized = label.lower()
        if normalized in self.left_labels:
            return 'left'
        if normalized in self.right_labels:
            return 'right'
        return 'left' if cx < (image_width * 0.5) else 'right'

    def infer_with_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        try:
            results = self.model.predict(frame, conf=self.conf_th, iou=self.iou_th, verbose=False)
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f'YOLO inference failed: {exc}')
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return []

        names = getattr(self.model, 'names', {})
        detections: List[Dict[str, Any]] = []

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, f'class_{cls_id}'))

            x1, y1, x2, y2 = xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h
            detections.append(
                {
                    'label': label,
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox_xywh': [cx, cy, w, h],
                }
            )

        return detections

    def fallback_lane_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.enable_fallback:
            return []

        height, width, _ = frame.shape
        roi_top = int(max(0.1, min(0.9, self.fallback_roi_top_ratio)) * height)
        roi = frame[roi_top:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.fallback_white_value_min),
            (180, self.fallback_white_sat_max, 255),
        )
        yellow_mask = cv2.inRange(
            hsv,
            (self.fallback_yellow_h_min, 40, 80),
            (self.fallback_yellow_h_max, 255, 255),
        )
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        hist = np.sum(mask[mask.shape[0] // 2 :, :] > 0, axis=0)
        half = width // 2

        detections: List[Dict[str, Any]] = []
        left_idx = int(np.argmax(hist[:half]))
        right_idx = int(np.argmax(hist[half:])) + half
        left_ok = int(hist[left_idx]) >= self.fallback_min_peak_pixels
        right_ok = int(hist[right_idx]) >= self.fallback_min_peak_pixels
        bbox_h = float(height - roi_top)

        if left_ok:
            detections.append(
                {
                    'label': 'left_lane',
                    'class_id': -1,
                    'confidence': 0.5,
                    'bbox_xywh': [float(left_idx), float(roi_top + bbox_h * 0.5), 12.0, bbox_h],
                }
            )
        if right_ok:
            detections.append(
                {
                    'label': 'right_lane',
                    'class_id': -1,
                    'confidence': 0.5,
                    'bbox_xywh': [float(right_idx), float(roi_top + bbox_h * 0.5), 12.0, bbox_h],
                }
            )
        return detections

    def publish_debug_overlay(self, frame: np.ndarray, detections: List[Dict[str, Any]], header) -> None:
        if not self.publish_debug_image or self.debug_pub.get_subscription_count() == 0:
            return
        debug = frame.copy()
        for det in detections:
            cx, cy, w, h = det['bbox_xywh']
            x1 = int(max(0, cx - w * 0.5))
            y1 = int(max(0, cy - h * 0.5))
            x2 = int(min(frame.shape[1] - 1, cx + w * 0.5))
            y2 = int(min(frame.shape[0] - 1, cy + h * 0.5))
            side = self.classify_lane_side(str(det['label']), float(cx), frame.shape[1])
            color = (0, 255, 0) if side == 'left' else (0, 200, 255)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            text = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(debug, text, (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        self.debug_pub.publish(self.bgr_to_image_msg(debug, header))

    def image_callback(self, msg: Image) -> None:
        frame = self.image_to_bgr(msg)
        if frame is None:
            return

        detections = self.infer_with_yolo(frame)
        if not detections:
            detections = self.fallback_lane_detections(frame)

        image_h, image_w = frame.shape[:2]
        normalized_detections: List[Dict[str, Any]] = []
        for det in detections:
            cx, cy, w, h = [float(v) for v in det['bbox_xywh']]
            side = self.classify_lane_side(str(det['label']), cx, image_w)
            normalized_detections.append(
                {
                    'label': str(det['label']),
                    'side': side,
                    'confidence': float(det['confidence']),
                    'bbox_xywh': [cx, cy, w, h],
                }
            )

        payload = {
            'stamp': {'sec': int(msg.header.stamp.sec), 'nanosec': int(msg.header.stamp.nanosec)},
            'frame_id': msg.header.frame_id,
            'image_width': int(image_w),
            'image_height': int(image_h),
            'detections': normalized_detections,
        }
        out = String()
        out.data = json.dumps(payload, separators=(',', ':'))
        self.detections_pub.publish(out)
        self.publish_debug_overlay(frame, normalized_detections, msg.header)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneYoloInference()
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
