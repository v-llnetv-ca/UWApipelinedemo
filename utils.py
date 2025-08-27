"""
War Detection Pipeline - Utility Functions
Comprehensive utilities for video, audio, and file processing
"""
import os
import cv2
import uuid
import hashlib
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Generator
from datetime import datetime, timedelta
import logging
from loguru import logger
import asyncio
import aiofiles
from moviepy.editor import VideoFileClip
import torch
import json

from config import (
    AUDIO_DETECTION_CONFIG, FILE_CONFIG, PROCESSED_DIR,
    UPLOAD_DIR, LOGGING_CONFIG
)

# Setup logging
logger.add(
    LOGGING_CONFIG["log_file"],
    rotation=LOGGING_CONFIG["rotation"],
    retention=LOGGING_CONFIG["retention"],
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"]
)


class VideoProcessor:
    """Professional video processing utilities"""

    @staticmethod
    def extract_frames(video_path: Path, fps: Optional[float] = None) -> Generator[
        Tuple[float, np.ndarray], None, None]:
        """
        Extract frames from video with timestamps

        Args:
            video_path: Path to video file
            fps: Target FPS for extraction (None for original FPS)

        Yields:
            Tuple of (timestamp, frame) for each frame
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        # Calculate frame step for target FPS
        if fps is None:
            frame_step = 1
            target_fps = original_fps
        else:
            frame_step = max(1, int(original_fps / fps))
            target_fps = fps

        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Original FPS: {original_fps:.2f}, Target FPS: {target_fps:.2f}")
        logger.info(f"Duration: {duration:.2f}s, Total frames: {total_frames}")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                timestamp = frame_count / original_fps
                yield timestamp, frame

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {frame_count // frame_step} frames")

    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """Get comprehensive video information"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "file_size": video_path.stat().st_size,
        }

        cap.release()
        return info

    @staticmethod
    def extract_thumbnail(video_path: Path, timestamp: float = 0.0) -> np.ndarray:
        """Extract thumbnail from video at specific timestamp"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Seek to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Cannot extract frame at {timestamp}s")

        return frame


class AudioProcessor:
    """Professional audio processing utilities"""

    @staticmethod
    def extract_audio_from_video(video_path: Path, output_path: Optional[Path] = None) -> Path:
        """Extract audio from video file"""
        if output_path is None:
            output_path = PROCESSED_DIR / f"{video_path.stem}_audio.wav"

        try:
            video = VideoFileClip(str(video_path))
            audio = video.audio

            if audio is None:
                raise RuntimeError(f"No audio track found in video: {video_path}")

            audio.write_audiofile(
                str(output_path),
                verbose=False,
                logger=None
            )

            video.close()
            audio.close()

            logger.info(f"Extracted audio: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to extract audio from {video_path}: {e}")
            raise

    @staticmethod
    def load_audio(audio_path: Path, sr: int = None) -> Tuple[np.ndarray, int]:
        """Load audio file with proper sampling rate"""
        sr = sr or AUDIO_DETECTION_CONFIG["sample_rate"]

        try:
            audio, sample_rate = librosa.load(
                str(audio_path),
                sr=sr,
                mono=True
            )

            logger.info(f"Loaded audio: {audio_path.name}, Duration: {len(audio) / sample_rate:.2f}s")
            return audio, sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise

    @staticmethod
    def generate_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
        """Generate mel spectrogram for visualization and processing"""
        config = AUDIO_DETECTION_CONFIG

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            window=config["window"],
            n_mels=config["n_mels"]
        )

        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    @staticmethod
    def chunk_audio(audio: np.ndarray, sr: int, chunk_duration: float, overlap: float) -> Generator[
        Tuple[float, np.ndarray], None, None]:
        """Split audio into overlapping chunks for processing"""
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        step_samples = chunk_samples - overlap_samples

        for start in range(0, len(audio) - chunk_samples + 1, step_samples):
            end = start + chunk_samples
            timestamp = start / sr
            chunk = audio[start:end]
            yield timestamp, chunk


class FileManager:
    """Professional file management utilities"""

    @staticmethod
    def generate_unique_filename(original_name: str, extension: str = None) -> str:
        """Generate unique filename with timestamp and UUID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        if extension:
            return f"{timestamp}_{unique_id}_{original_name}{extension}"
        else:
            name, ext = os.path.splitext(original_name)
            return f"{timestamp}_{unique_id}_{name}{ext}"

    @staticmethod
    def validate_file(file_path: Path) -> Dict[str, Union[bool, str]]:
        """Comprehensive file validation"""
        result = {"valid": True, "errors": []}

        # Check if file exists
        if not file_path.exists():
            result["valid"] = False
            result["errors"].append("File does not exist")
            return result

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > FILE_CONFIG["max_file_size"]:
            result["valid"] = False
            result["errors"].append(f"File size ({file_size} bytes) exceeds limit")

        # Check file extension
        extension = file_path.suffix.lower()
        allowed_extensions = (
                FILE_CONFIG["allowed_video_formats"] +
                FILE_CONFIG["allowed_audio_formats"] +
                FILE_CONFIG["allowed_image_formats"]
        )

        if extension not in allowed_extensions:
            result["valid"] = False
            result["errors"].append(f"File type {extension} not allowed")

        # Check if file is corrupted by trying to read it
        try:
            if extension in FILE_CONFIG["allowed_video_formats"]:
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    result["valid"] = False
                    result["errors"].append("Video file is corrupted or unreadable")
                cap.release()
            elif extension in FILE_CONFIG["allowed_audio_formats"]:
                librosa.load(str(file_path), sr=None, duration=0.1)  # Test load first 0.1s
            elif extension in FILE_CONFIG["allowed_image_formats"]:
                img = cv2.imread(str(file_path))
                if img is None:
                    result["valid"] = False
                    result["errors"].append("Image file is corrupted or unreadable")
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"File validation error: {str(e)}")

        return result

    @staticmethod
    async def save_uploaded_file(file_data: bytes, filename: str) -> Path:
        """Asynchronously save uploaded file"""
        unique_filename = FileManager.generate_unique_filename(filename)
        file_path = UPLOAD_DIR / unique_filename

        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    @staticmethod
    def cleanup_old_files(directory: Path, max_age_days: int = 7):
        """Clean up old files from directory"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        for file_path in directory.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old files from {directory}")


class TimestampManager:
    """Professional timestamp management utilities"""

    @staticmethod
    def frame_to_timestamp(frame_number: int, fps: float) -> float:
        """Convert frame number to timestamp in seconds"""
        return frame_number / fps

    @staticmethod
    def timestamp_to_frame(timestamp: float, fps: float) -> int:
        """Convert timestamp to frame number"""
        return int(timestamp * fps)

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp as MM:SS.mmm"""
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> float:
        """Parse timestamp string back to float seconds"""
        if ":" in timestamp_str:
            parts = timestamp_str.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(timestamp_str)

    @staticmethod
    def sync_timestamps(video_timestamps: List[float], audio_timestamps: List[float], tolerance: float = 0.1) -> List[
        Tuple[float, float]]:
        """Synchronize video and audio timestamps within tolerance"""
        synchronized = []

        for v_ts in video_timestamps:
            for a_ts in audio_timestamps:
                if abs(v_ts - a_ts) <= tolerance:
                    synchronized.append((v_ts, a_ts))
                    break

        return synchronized


class DetectionUtils:
    """Utilities for detection result processing"""

    @staticmethod
    def filter_detections_by_confidence(detections: List[Dict], threshold: float) -> List[Dict]:
        """Filter detections by confidence threshold"""
        return [det for det in detections if det.get('confidence', 0) >= threshold]

    @staticmethod
    def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return detections

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)

        keep = []
        for current in detections:
            current_box = current.get('bbox', [])
            if not current_box:
                continue

            # Check if current detection overlaps significantly with any kept detection
            should_keep = True
            for kept in keep:
                kept_box = kept.get('bbox', [])
                if not kept_box:
                    continue

                iou = DetectionUtils.calculate_iou(current_box, kept_box)
                if iou > iou_threshold and current.get('class') == kept.get('class'):
                    should_keep = False
                    break

            if should_keep:
                keep.append(current)

        return keep

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def merge_temporal_detections(detections: List[Dict], time_window: float = 1.0) -> List[Dict]:
        """Merge similar detections within a time window"""
        if not detections:
            return detections

        # Sort by timestamp
        detections = sorted(detections, key=lambda x: x.get('timestamp', 0))

        merged = []
        current_group = [detections[0]]

        for detection in detections[1:]:
            last_timestamp = current_group[-1].get('timestamp', 0)
            current_timestamp = detection.get('timestamp', 0)

            # If within time window and same class, add to current group
            if (current_timestamp - last_timestamp <= time_window and
                    detection.get('class') == current_group[0].get('class')):
                current_group.append(detection)
            else:
                # Merge current group and start new group
                merged.append(DetectionUtils._merge_detection_group(current_group))
                current_group = [detection]

        # Merge final group
        if current_group:
            merged.append(DetectionUtils._merge_detection_group(current_group))

        return merged

    @staticmethod
    def _merge_detection_group(group: List[Dict]) -> Dict:
        """Merge a group of similar detections"""
        if len(group) == 1:
            return group[0]

        # Use highest confidence detection as base
        base = max(group, key=lambda x: x.get('confidence', 0))
        result = base.copy()

        # Average timestamps and confidences
        result['timestamp'] = np.mean([d.get('timestamp', 0) for d in group])
        result['confidence'] = np.mean([d.get('confidence', 0) for d in group])
        result['count'] = len(group)  # Track how many detections were merged

        return result


class PerformanceMonitor:
    """Performance monitoring utilities"""

    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
        return self

    def checkpoint(self, name: str):
        """Add a checkpoint"""
        if self.start_time is None:
            self.start()

        self.checkpoints[name] = datetime.now()
        elapsed = (self.checkpoints[name] - self.start_time).total_seconds()
        logger.debug(f"Checkpoint '{name}': {elapsed:.3f}s")

    def get_elapsed(self, checkpoint_name: str = None) -> float:
        """Get elapsed time since start or checkpoint"""
        if self.start_time is None:
            return 0.0

        if checkpoint_name and checkpoint_name in self.checkpoints:
            end_time = self.checkpoints[checkpoint_name]
        else:
            end_time = datetime.now()

        return (end_time - self.start_time).total_seconds()

    def report(self) -> Dict[str, float]:
        """Get performance report"""
        if self.start_time is None:
            return {}

        report = {"total_time": self.get_elapsed()}

        prev_time = self.start_time
        for name, checkpoint_time in self.checkpoints.items():
            duration = (checkpoint_time - prev_time).total_seconds()
            report[f"{name}_duration"] = duration
            prev_time = checkpoint_time

        return report


class DataSerializer:
    """Data serialization utilities"""

    @staticmethod
    def serialize_detection_results(results: Dict) -> str:
        """Serialize detection results to JSON string"""

        # Handle numpy arrays and other non-serializable types
        def serialize_item(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif torch.is_tensor(obj):
                return obj.cpu().numpy().tolist()
            return obj

        def deep_serialize(obj):
            if isinstance(obj, dict):
                return {k: deep_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [deep_serialize(item) for item in obj]
            else:
                return serialize_item(obj)

        serialized_results = deep_serialize(results)
        return json.dumps(serialized_results, indent=2)

    @staticmethod
    def deserialize_detection_results(json_str: str) -> Dict:
        """Deserialize detection results from JSON string"""
        return json.loads(json_str)


# Async utilities for concurrent processing
class AsyncProcessor:
    """Async processing utilities"""

    @staticmethod
    async def process_in_batches(items: List, batch_size: int, processor_func, *args, **kwargs):
        """Process items in batches asynchronously"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [processor_func(item, *args, **kwargs) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Log progress
            progress = min(i + batch_size, len(items))
            logger.info(f"Processed {progress}/{len(items)} items")

        return results

    @staticmethod
    async def run_with_timeout(coro, timeout_seconds: float):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout_seconds}s")
            raise


# Initialize module-level performance monitor
perf_monitor = PerformanceMonitor()

# Export all utilities
__all__ = [
    "VideoProcessor", "AudioProcessor", "FileManager", "TimestampManager",
    "DetectionUtils", "PerformanceMonitor", "DataSerializer", "AsyncProcessor",
    "perf_monitor", "logger"
]

if __name__ == "__main__":
    # Test utilities
    print("Testing war detection utilities...")

    # Test video processor
    print("VideoProcessor ready")

    # Test audio processor
    print("AudioProcessor ready")

    # Test file manager
    print("FileManager ready")

    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start()
    monitor.checkpoint("test")
    print(f"PerformanceMonitor ready: {monitor.get_elapsed():.3f}s")

    print("All utilities initialized successfully!")