#!/usr/bin/env python3

import inspect
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def normalize_label_set(values: Sequence[str]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def normalize_id_set(values: Sequence[int]) -> set[int]:
    normalized: set[int] = set()
    for value in values:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


class SegmentationPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('segmentation_perception_node')

        self.declare_parameter('image_topic', '/front_camera/image_raw')
        self.declare_parameter('drivable_mask_topic', '/perception/drivable_mask')
        self.declare_parameter('obstacle_mask_topic', '/perception/obstacle_mask')
        self.declare_parameter('debug_image_topic', '/perception/segmentation_debug')
        self.declare_parameter('publish_debug_image', True)

        self.declare_parameter('model_path', 'auto')
        self.declare_parameter('device', '')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('mask_threshold', 0.50)
        self.declare_parameter('retina_masks', True)

        self.declare_parameter('primary_mask_semantics', 'drivable')
        self.declare_parameter('unknown_class_mode', 'primary')
        self.declare_parameter('derive_missing_mask', True)
        self.declare_parameter('drivable_class_labels', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('obstacle_class_labels', Parameter.Type.STRING_ARRAY)
        self.declare_parameter('drivable_class_ids', Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter('obstacle_class_ids', Parameter.Type.INTEGER_ARRAY)

        self.image_topic = str(self.get_parameter('image_topic').value)
        drivable_mask_topic = str(self.get_parameter('drivable_mask_topic').value)
        obstacle_mask_topic = str(self.get_parameter('obstacle_mask_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        requested_model_path = str(self.get_parameter('model_path').value).strip()
        self.model_path = self.resolve_model_path(requested_model_path)
        self.device = str(self.get_parameter('device').value).strip()
        self.conf_th = float(self.get_parameter('confidence_threshold').value)
        self.iou_th = float(self.get_parameter('iou_threshold').value)
        self.mask_threshold = float(self.get_parameter('mask_threshold').value)
        self.retina_masks = bool(self.get_parameter('retina_masks').value)

        self.primary_mask_semantics = str(
            self.get_parameter('primary_mask_semantics').value
        ).strip().lower()
        self.unknown_class_mode = str(self.get_parameter('unknown_class_mode').value).strip().lower()
        self.derive_missing_mask = bool(self.get_parameter('derive_missing_mask').value)
        self.drivable_class_labels = normalize_label_set(
            self.get_parameter_or(
                'drivable_class_labels',
                Parameter('drivable_class_labels', Parameter.Type.STRING_ARRAY, []),
            ).value
        )
        self.obstacle_class_labels = normalize_label_set(
            self.get_parameter_or(
                'obstacle_class_labels',
                Parameter('obstacle_class_labels', Parameter.Type.STRING_ARRAY, []),
            ).value
        )
        self.drivable_class_ids = normalize_id_set(
            self.get_parameter_or(
                'drivable_class_ids',
                Parameter('drivable_class_ids', Parameter.Type.INTEGER_ARRAY, []),
            ).value
        )
        self.obstacle_class_ids = normalize_id_set(
            self.get_parameter_or(
                'obstacle_class_ids',
                Parameter('obstacle_class_ids', Parameter.Type.INTEGER_ARRAY, []),
            ).value
        )

        if self.primary_mask_semantics not in ('drivable', 'obstacle'):
            raise RuntimeError("primary_mask_semantics must be 'drivable' or 'obstacle'")
        if self.unknown_class_mode not in ('ignore', 'primary', 'drivable', 'obstacle'):
            raise RuntimeError(
                "unknown_class_mode must be one of: ignore, primary, drivable, obstacle"
            )

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.model = self.load_model()
        self.class_names = self.extract_class_names()

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.sensor_qos,
        )
        self.drivable_pub = self.create_publisher(Image, drivable_mask_topic, self.sensor_qos)
        self.obstacle_pub = self.create_publisher(Image, obstacle_mask_topic, self.sensor_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, self.sensor_qos)

        self.get_logger().info(
            'Segmentation perception ready. '
            f'image_topic={self.image_topic}, model_path={self.model_path}, '
            f'primary_mask_semantics={self.primary_mask_semantics}'
        )

    def resolve_model_path(self, requested_path: str) -> str:
        requested = requested_path.strip()
        if requested and requested.lower() != 'auto':
            return requested

        candidate_paths: List[str] = []
        env_model_path = os.getenv('SEGMENTATION_MODEL_PATH', '').strip()
        if env_model_path:
            candidate_paths.append(env_model_path)

        try:
            pkg_share = get_package_share_directory('vehicle_bringup')
            candidate_paths.append(os.path.join(pkg_share, 'models', 'segment.pt'))
        except Exception:
            pass

        candidate_paths.extend(
            [
                os.path.expanduser('~/turtlebot3_ws/src/vehicle_bringup/models/segment.pt'),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'segment.pt')),
            ]
        )

        for path in candidate_paths:
            if path and os.path.isfile(path):
                return path

        raise RuntimeError(
            'Segmentation model could not be resolved. '
            'Set model_path or SEGMENTATION_MODEL_PATH.'
        )

    def load_model(self):
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f'Segmentation model file does not exist: {self.model_path}')

        try:
            self.install_ultralytics_compat_aliases()
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(f'ultralytics import failed: {exc}') from exc

        try:
            return YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to load segmentation model: {exc}') from exc

    @staticmethod
    def install_ultralytics_compat_aliases() -> None:
        # Older checkpoints can pickle dynamically suffixed class names like
        # Segment26 / Proto26. Current ultralytics exposes only the base names.
        from ultralytics.nn.modules import block, conv, head, transformer  # type: ignore

        for module in (head, block, conv, transformer):
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if getattr(obj, '__module__', '') != module.__name__:
                    continue
                for idx in range(100):
                    alias = f'{name}{idx}'
                    if not hasattr(module, alias):
                        setattr(module, alias, obj)

    def extract_class_names(self) -> Dict[int, str]:
        raw_names = getattr(self.model, 'names', {})
        if isinstance(raw_names, dict):
            return {int(key): str(value).strip().lower() for key, value in raw_names.items()}
        if isinstance(raw_names, list):
            return {idx: str(value).strip().lower() for idx, value in enumerate(raw_names)}
        return {}

    @staticmethod
    def image_to_bgr(msg: Image) -> Optional[np.ndarray]:
        encoding = msg.encoding.lower()

        if encoding in ('bgr8', 'rgb8'):
            channels = 3
        elif encoding in ('bgra8', 'rgba8'):
            channels = 4
        elif encoding in ('mono8', '8uc1'):
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
    def mono_to_image_msg(mask: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(mask.shape[0])
        msg.width = int(mask.shape[1])
        msg.encoding = 'mono8'
        msg.is_bigendian = False
        msg.step = int(mask.shape[1])
        msg.data = mask.tobytes()
        return msg

    @staticmethod
    def bgr_to_image_msg(frame: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = int(frame.shape[1] * 3)
        msg.data = frame.tobytes()
        return msg

    def classify_mask_semantics(self, class_id: int, label: str) -> Optional[str]:
        normalized_label = label.strip().lower()

        if class_id in self.drivable_class_ids or normalized_label in self.drivable_class_labels:
            return 'drivable'
        if class_id in self.obstacle_class_ids or normalized_label in self.obstacle_class_labels:
            return 'obstacle'

        if self.unknown_class_mode == 'ignore':
            return None
        if self.unknown_class_mode == 'primary':
            return self.primary_mask_semantics
        return self.unknown_class_mode

    def infer_masks(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        height, width = frame.shape[:2]
        direct_drivable = np.zeros((height, width), dtype=np.uint8)
        direct_obstacle = np.zeros((height, width), dtype=np.uint8)
        class_counts: Dict[str, int] = {}

        predict_kwargs = {
            'source': frame,
            'conf': self.conf_th,
            'iou': self.iou_th,
            'verbose': False,
            'retina_masks': self.retina_masks,
        }
        if self.device:
            predict_kwargs['device'] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return direct_drivable, direct_obstacle, direct_drivable.copy(), direct_obstacle.copy(), class_counts

        result = results[0]
        masks_obj = getattr(result, 'masks', None)
        boxes = getattr(result, 'boxes', None)
        if masks_obj is None or boxes is None or len(boxes) == 0:
            return direct_drivable, direct_obstacle, direct_drivable.copy(), direct_obstacle.copy(), class_counts

        raw_masks = masks_obj.data.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for idx, raw_mask in enumerate(raw_masks):
            class_id = int(class_ids[idx]) if idx < len(class_ids) else -1
            label = self.class_names.get(class_id, f'class_{class_id}')
            semantics = self.classify_mask_semantics(class_id, label)
            if semantics is None:
                continue

            mask = np.where(raw_mask >= self.mask_threshold, 255, 0).astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            class_counts[label] = class_counts.get(label, 0) + 1
            if semantics == 'drivable':
                direct_drivable = cv2.bitwise_or(direct_drivable, mask)
            elif semantics == 'obstacle':
                direct_obstacle = cv2.bitwise_or(direct_obstacle, mask)

        published_drivable = direct_drivable.copy()
        published_obstacle = direct_obstacle.copy()

        if np.any(published_obstacle):
            published_drivable[published_obstacle > 0] = 0

        if self.derive_missing_mask:
            if not np.any(published_obstacle) and np.any(published_drivable):
                published_obstacle = cv2.bitwise_not(published_drivable)
            elif not np.any(published_drivable) and np.any(published_obstacle):
                published_drivable = cv2.bitwise_not(published_obstacle)

        if np.any(published_obstacle):
            published_drivable[published_obstacle > 0] = 0

        return (
            direct_drivable,
            direct_obstacle,
            published_drivable,
            published_obstacle,
            class_counts,
        )

    def build_debug_overlay(
        self,
        frame: np.ndarray,
        direct_drivable: np.ndarray,
        direct_obstacle: np.ndarray,
        class_counts: Dict[str, int],
    ) -> np.ndarray:
        debug = frame.copy()

        if np.any(direct_drivable):
            overlay = debug.copy()
            overlay[direct_drivable > 0] = (0, 220, 0)
            cv2.addWeighted(overlay, 0.35, debug, 0.65, 0.0, debug)

        if np.any(direct_obstacle):
            overlay = debug.copy()
            overlay[direct_obstacle > 0] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.35, debug, 0.65, 0.0, debug)

        if class_counts:
            label_text = ', '.join(f'{label}:{count}' for label, count in sorted(class_counts.items()))
        else:
            label_text = 'no masks'

        cv2.putText(
            debug,
            label_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    def image_callback(self, msg: Image) -> None:
        frame = self.image_to_bgr(msg)
        if frame is None:
            self.get_logger().warn(f'Unsupported image encoding: {msg.encoding}')
            return

        try:
            direct_drivable, direct_obstacle, drivable_mask, obstacle_mask, class_counts = self.infer_masks(frame)
        except Exception as exc:
            self.get_logger().error(f'Segmentation inference failed: {exc}')
            return

        self.drivable_pub.publish(self.mono_to_image_msg(drivable_mask, msg.header))
        self.obstacle_pub.publish(self.mono_to_image_msg(obstacle_mask, msg.header))

        if self.publish_debug_image:
            debug = self.build_debug_overlay(
                frame=frame,
                direct_drivable=direct_drivable,
                direct_obstacle=direct_obstacle,
                class_counts=class_counts,
            )
            self.debug_pub.publish(self.bgr_to_image_msg(debug, msg.header))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
