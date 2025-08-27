"""
Object Detection Pipeline - Object Detection Module
YOLOv8m-based object detection for war-related objects
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from config import (
    MODEL_PATHS, OBJECT_DETECTION_CONFIG, DEVICE,
    PERFORMANCE_CONFIG, FRONTEND_CONFIG
)
from utils import (
    VideoProcessor, TimestampManager, DetectionUtils,
    PerformanceMonitor, logger, perf_monitor
)

class WarObjectDetector:
    """War object detection using fine-tuned YOLOv8m"""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the war object detector

        Args:
            model_path: Path to fine-tuned YOLOv8 model
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
        """
        self.model_path = model_path or MODEL_PATHS["yolov8"]
        self.device = device or DEVICE
        self.config = OBJECT_DETECTION_CONFIG
        self.performance_config = PERFORMANCE_CONFIG

        # Initialize model
        self.model = None
        self.is_loaded = False

        # Performance tracking
        self.inference_times = []
        self.detection_stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "detections_by_class": {},
        }

        logger.info(f"Initialized WarObjectDetector with model: {self.model_path}")
        logger.info(f"Target device: {self.device}")

    def load_model(self) -> bool:
        """Load and initialize the YOLOv8 model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            logger.info(f"Loading YOLOv8 model from: {self.model_path}")

            # Load the model
            self.model = YOLO(str(self.model_path))

            # Move to specified device
            if hasattr(self.model.model, 'to'):
                self.model.model.to(self.device)

            # Test inference with dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_image, verbose=False)

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model classes: {self.config['classes']}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False

    def detect_objects_in_image(self, image: np.ndarray, timestamp: float = 0.0) -> List[Dict]:
        """
        Detect objects in a single image

        Args:
            image: Input image as numpy array (BGR format)
            timestamp: Timestamp for the detection

        Returns:
            List of detection dictionaries
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        start_time = time.time()

        try:
            # Run inference
            results = self.model(
                image,
                conf=self.config["confidence_threshold"],
                iou=self.config["iou_threshold"],
                max_det=self.config["max_det"],
                verbose=False
            )

            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First (and only) result

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

                    for i in range(len(boxes)):
                        class_id = class_ids[i]
                        class_name = self.config["classes"].get(class_id, f"class_{class_id}")

                        detection = {
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": float(confidences[i]),
                            "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
                            "timestamp": timestamp,
                            "center": [
                                float((boxes[i][0] + boxes[i][2]) / 2),
                                float((boxes[i][1] + boxes[i][3]) / 2)
                            ],
                            "area": float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])),
                        }

                        detections.append(detection)

            # Update performance stats
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detection_stats["total_frames_processed"] += 1
            self.detection_stats["total_detections"] += len(detections)

            # Update class statistics
            for det in detections:
                class_name = det["class"]
                if class_name not in self.detection_stats["detections_by_class"]:
                    self.detection_stats["detections_by_class"][class_name] = 0
                self.detection_stats["detections_by_class"][class_name] += 1

            logger.debug(f"Detected {len(detections)} objects in {inference_time:.3f}s")
            return detections

        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

    def detect_objects_in_video(self, video_path: Path, fps: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in video file

        Args:
            video_path: Path to video file
            fps: Target FPS for processing (None for original FPS)

        Returns:
            List of all detections with timestamps
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        logger.info(f"Processing video: {video_path}")
        all_detections = []

        try:
            # Process video frame by frame
            for timestamp, frame in VideoProcessor.extract_frames(video_path, fps):
                frame_detections = self.detect_objects_in_image(frame, timestamp)
                all_detections.extend(frame_detections)

                # Log progress periodically
                if len(all_detections) % 100 == 0:
                    logger.info(f"Processed {self.detection_stats['total_frames_processed']} frames, "
                              f"found {len(all_detections)} detections")

            # Apply post-processing
            all_detections = self._post_process_detections(all_detections)

            logger.info(f"Video processing complete: {len(all_detections)} final detections")
            return all_detections

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return []

    def detect_objects_batch(self, images: List[np.ndarray], timestamps: List[float]) -> List[List[Dict]]:
        """
        Process multiple images in batch for efficiency

        Args:
            images: List of images as numpy arrays
            timestamps: Corresponding timestamps

        Returns:
            List of detection lists (one per image)
        """
        if not self.is_loaded:
            if not self.load_model():
                return [[] for _ in images]

        batch_size = self.performance_config["batch_size"]["object_detection"]
        all_results = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_timestamps = timestamps[i:i + batch_size]

            # Process batch
            batch_results = []
            for img, ts in zip(batch_images, batch_timestamps):
                detections = self.detect_objects_in_image(img, ts)
                batch_results.append(detections)

            all_results.extend(batch_results)

            # Log progress
            progress = min(i + batch_size, len(images))
            logger.info(f"Batch processed: {progress}/{len(images)} images")

        return all_results

    async def detect_objects_async(self, image: np.ndarray, timestamp: float = 0.0) -> List[Dict]:
        """
        Async wrapper for object detection

        Args:
            image: Input image
            timestamp: Timestamp for detection

        Returns:
            List of detections
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.detect_objects_in_image,
                image,
                timestamp
            )
        return result

    def _post_process_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply post-processing to detections

        Args:
            detections: Raw detection results

        Returns:
            Post-processed detections
        """
        if not detections:
            return detections

        # Filter by confidence
        filtered_detections = DetectionUtils.filter_detections_by_confidence(
            detections,
            self.config["confidence_threshold"]
        )

        # Apply temporal merging to reduce noise
        merged_detections = DetectionUtils.merge_temporal_detections(
            filtered_detections,
            time_window=1.0
        )

        # Apply Non-Maximum Suppression within time windows
        final_detections = []
        current_window = []
        window_duration = 2.0  # 2-second windows

        for detection in merged_detections:
            if not current_window or detection["timestamp"] - current_window[0]["timestamp"] <= window_duration:
                current_window.append(detection)
            else:
                # Process current window
                nms_results = DetectionUtils.non_max_suppression(
                    current_window,
                    self.config["iou_threshold"]
                )
                final_detections.extend(nms_results)
                current_window = [detection]

        # Process final window
        if current_window:
            nms_results = DetectionUtils.non_max_suppression(
                current_window,
                self.config["iou_threshold"]
            )
            final_detections.extend(nms_results)

        logger.info(f"Post-processing: {len(detections)} → {len(final_detections)} detections")
        return final_detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on image

        Args:
            image: Input image
            detections: List of detections

        Returns:
            Image with drawn detections
        """
        result_image = image.copy()

        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]

            # Get color for class
            color = self.config["colors"].get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(
                result_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            cv2.rectangle(
                result_image,
                (int(bbox[0]), int(bbox[1] - label_size[1] - 10)),
                (int(bbox[0] + label_size[0]), int(bbox[1])),
                color,
                -1
            )

            cv2.putText(
                result_image,
                label,
                (int(bbox[0]), int(bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        return result_image

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        avg_inference_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        stats = {
            "total_frames_processed": self.detection_stats["total_frames_processed"],
            "total_detections": self.detection_stats["total_detections"],
            "detections_by_class": self.detection_stats["detections_by_class"],
            "average_inference_time": avg_inference_time,
            "average_fps": avg_fps,
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "device": self.device,
            "model_loaded": self.is_loaded,
        }

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times = []
        self.detection_stats = {
            "total_frames_processed": 0,
            "total_detections": 0,
            "detections_by_class": {},
        }
        logger.info("Performance statistics reset")

    def __enter__(self):
        """Context manager entry"""
        if not self.is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed"""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("WarObjectDetector context closed")

# Convenience functions for easy usage
def detect_objects_in_image(image_path: Path, model_path: Optional[Path] = None) -> List[Dict]:
    """
    Convenience function to detect objects in a single image

    Args:
        image_path: Path to image file
        model_path: Path to model (uses default if None)

    Returns:
        List of detections
    """
    detector = WarObjectDetector(model_path)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return []

    return detector.detect_objects_in_image(image)

def detect_objects_in_video(video_path: Path, model_path: Optional[Path] = None, fps: Optional[float] = None) -> List[Dict]:
    """
    Convenience function to detect objects in a video

    Args:
        video_path: Path to video file
        model_path: Path to model (uses default if None)
        fps: Target processing FPS

    Returns:
        List of all detections with timestamps
    """
    detector = WarObjectDetector(model_path)
    return detector.detect_objects_in_video(video_path, fps)

# Export classes and functions
__all__ = [
    "WarObjectDetector",
    "detect_objects_in_image",
    "detect_objects_in_video"
]

# Test functionality
if __name__ == "__main__":
    print("Testing War Object Detection Module...")

    # Initialize detector
    detector = WarObjectDetector()

    # Test model loading
    if detector.load_model():
        print("Model loaded successfully")

        # Test with dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.detect_objects_in_image(dummy_image)
        print(f"Dummy detection test: {len(detections)} detections")

        # Print performance stats
        stats = detector.get_performance_stats()
        print(f"Performance stats: {stats}")

    else:
        print("Failed to load model - check model path in config.py")
        print(f"Expected model path: {MODEL_PATHS['yolov8']}")

    print("Object detection module test complete!")