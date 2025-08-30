"""
War Detection Pipeline - Audio Detection Module
Professional PANN-based audio detection for war-related sounds
"""
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import panns_inference
    from panns_inference import AudioTagging

    PANN_AVAILABLE = True
except ImportError:
    print("PANN not available. Install with: pip install panns-inference")
    PANN_AVAILABLE = False

from config import (
    MODEL_PATHS, AUDIO_DETECTION_CONFIG, DEVICE,
    PERFORMANCE_CONFIG, FRONTEND_CONFIG
)
from utils import (
    AudioProcessor, TimestampManager, DetectionUtils,
    PerformanceMonitor, logger, perf_monitor
)


class WarAudioDetector:
    """Professional war audio detection using PANN"""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the war audio detector

        Args:
            model_path: Path to PANN model (optional - uses pre-trained)
            device: Device to run inference on ('cuda', 'cpu')
        """
        self.model_path = model_path
        self.device = device or DEVICE
        self.config = AUDIO_DETECTION_CONFIG
        self.performance_config = PERFORMANCE_CONFIG

        # Initialize model
        self.model = None
        self.is_loaded = False

        # Audio processing parameters
        self.sample_rate = self.config["sample_rate"]
        self.chunk_duration = self.config["chunk_duration"]
        self.overlap = self.config["overlap"]

        # Performance tracking
        self.inference_times = []
        self.detection_stats = {
            "total_chunks_processed": 0,
            "total_detections": 0,
            "detections_by_class": {},
            "total_audio_duration": 0.0,
        }

        # Class mappings for specific war sounds (AudioSet indices)
        self.war_classes = {
            "gunshot_gunfire": [427],  # "Gunshot, gunfire" - AudioSet class 427
            "helicopter": [339],  # "Helicopter" - AudioSet class 339
            "explosion": [426],  # "Explosion" - AudioSet class 426
        }

        logger.info(f"Initialized WarAudioDetector")
        logger.info(f"Target device: {self.device}")
        logger.info(f"Sample rate: {self.sample_rate}Hz")

    def load_model(self) -> bool:
        """Load and initialize the PANN model"""
        try:
            if not PANN_AVAILABLE:
                logger.error("PANN library not available. Install with: pip install panns-inference")
                return False

            logger.info("Loading PANN AudioTagging model...")

            # Try different PANN initialization approaches
            try:
                # First try: Standard AudioTagging
                self.model = AudioTagging(checkpoint_path=None, device=self.device)
                logger.info("Standard AudioTagging loaded successfully")
            except Exception as e1:
                logger.warning(f"Standard AudioTagging failed: {e1}")
                try:
                    # Second try: Specify checkpoint manually
                    from panns_inference import SoundEventDetection
                    self.model = SoundEventDetection(checkpoint_path=None, device=self.device)
                    logger.info("SoundEventDetection loaded successfully")
                except Exception as e2:
                    logger.warning(f"SoundEventDetection also failed: {e2}")
                    # Third try: Use a minimal wrapper
                    raise RuntimeError(f"Could not load PANN model. AudioTagging error: {e1}, SED error: {e2}")

            # Test inference with detailed debugging
            logger.info("Testing PANN inference with detailed debugging...")

            # Check PANN model details
            logger.info(f"PANN model type: {type(self.model)}")
            logger.info(f"PANN model device: {getattr(self.model, 'device', 'unknown')}")

            # Try different audio configurations
            test_configs = [
                {"duration": 10, "sample_rate": 32000},
                {"duration": 10, "sample_rate": 22050},
                {"duration": 5, "sample_rate": 32000},
                {"duration": 1, "sample_rate": 32000},
            ]

            for config in test_configs:
                try:
                    test_length = config["sample_rate"] * config["duration"]
                    test_audio = np.random.uniform(-0.5, 0.5, test_length).astype(np.float32)

                    # CRITICAL: Add batch dimension for PANN
                    test_audio = test_audio[None, :]  # Shape: (1, samples)

                    logger.info(
                        f"Testing: {config['duration']}s at {config['sample_rate']}Hz, shape: {test_audio.shape}, dtype: {test_audio.dtype}")
                    logger.info(
                        f"Audio stats: min={test_audio.min():.3f}, max={test_audio.max():.3f}, mean={test_audio.mean():.3f}")

                    # Try inference
                    result = self.model.inference(test_audio)

                    logger.info(f"✅ Success with {config['duration']}s audio!")
                    logger.info(f"Result type: {type(result)}")
                    if isinstance(result, (list, tuple)):
                        logger.info(f"Result length: {len(result)}")
                        if len(result) > 0:
                            logger.info(
                                f"First result shape: {result[0].shape if hasattr(result[0], 'shape') else 'no shape'}")

                    # Found working configuration
                    self.working_config = config
                    break

                except Exception as e:
                    logger.error(f"Failed with {config['duration']}s at {config['sample_rate']}Hz: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    continue
            else:
                raise RuntimeError("All audio configurations failed - PANN compatibility issue")

            self.is_loaded = True
            logger.info("✅ PANN model loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to load PANN model: {e}")
            self.is_loaded = False
            return False

    def detect_audio_events(self, audio: np.ndarray, timestamp: float = 0.0) -> List[Dict]:
        """
        Detect audio events in audio chunk

        Args:
            audio: Input audio chunk (32kHz, float32)
            timestamp: Timestamp for the chunk start

        Returns:
            List of detection dictionaries
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        start_time = time.time()

        try:
            # Ensure audio is correct format and length for PANN
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)  # Convert to mono

            # PANN typically expects 10-second chunks at 32kHz (320,000 samples)
            expected_length = self.sample_rate * 10
            if len(audio) < expected_length:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, expected_length - len(audio)), 'constant')
            elif len(audio) > expected_length:
                # Truncate if too long
                audio = audio[:expected_length]

            # Normalize audio
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9

            # CRITICAL: Add batch dimension for PANN
            audio = audio[None, :]  # Shape: (1, samples)

            # Run PANN inference
            clipwise_output, embedding = self.model.inference(audio)

            # Process results
            detections = []
            clipwise_output = clipwise_output[0]  # Remove batch dimension

            # Check each war-related class
            for class_name, class_indices in self.war_classes.items():
                max_confidence = 0.0
                for class_idx in class_indices:
                    if class_idx < len(clipwise_output):
                        confidence = float(clipwise_output[class_idx])
                        max_confidence = max(max_confidence, confidence)

                # Apply threshold
                print(f"[DEBUG] {class_name}: max_conf={max_confidence:.3f}, thr={self.config['confidence_threshold']}")
                if max_confidence >= self.config["confidence_threshold"]:
                    detection = {
                        "class": class_name,
                        "confidence": max_confidence,
                        "timestamp": timestamp,
                        "duration": len(audio) / self.sample_rate,
                        "chunk_start": timestamp,
                        "chunk_end": timestamp + len(audio) / self.sample_rate,
                    }
                    detections.append(detection)

            # Update performance stats
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detection_stats["total_chunks_processed"] += 1
            self.detection_stats["total_detections"] += len(detections)
            self.detection_stats["total_audio_duration"] += len(audio) / self.sample_rate

            # Update class statistics
            for det in detections:
                class_name = det["class"]
                if class_name not in self.detection_stats["detections_by_class"]:
                    self.detection_stats["detections_by_class"][class_name] = 0
                self.detection_stats["detections_by_class"][class_name] += 1

            logger.debug(f"Detected {len(detections)} audio events in {inference_time:.3f}s")
            return detections

        except Exception as e:
            logger.error(f"Error in audio detection: {e}")
            return []

    def detect_audio_in_file(self, audio_path: Path) -> List[Dict]:
        """
        Detect audio events in entire audio file

        Args:
            audio_path: Path to audio file

        Returns:
            List of all detections with timestamps
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        logger.info(f"Processing audio file: {audio_path}")
        all_detections = []

        try:
            # Load audio file
            audio, sr = AudioProcessor.load_audio(audio_path, self.sample_rate)

            # Process in chunks
            for timestamp, chunk in AudioProcessor.chunk_audio(
                    audio, sr, self.chunk_duration, self.overlap
            ):
                chunk_detections = self.detect_audio_events(chunk, timestamp)
                all_detections.extend(chunk_detections)

                # Log progress periodically
                if len(all_detections) % 20 == 0 and len(all_detections) > 0:
                    duration_processed = timestamp + len(chunk) / sr
                    logger.info(f"Processed {duration_processed:.1f}s audio, "
                                f"found {len(all_detections)} detections")

            # Apply post-processing
            all_detections = self._post_process_audio_detections(all_detections)

            logger.info(f"✅ Audio processing complete: {len(all_detections)} final detections")
            return all_detections

        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return []

    def detect_audio_in_video(self, video_path: Path) -> Tuple[List[Dict], np.ndarray]:
        """
        Extract audio from video and detect audio events

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (detections, mel_spectrogram)
        """
        if not self.is_loaded:
            if not self.load_model():
                return [], np.array([])

        logger.info(f"Extracting and processing audio from video: {video_path}")

        try:
            # Extract audio from video
            audio_path = AudioProcessor.extract_audio_from_video(video_path)

            # Load audio for spectrogram generation
            audio, sr = AudioProcessor.load_audio(audio_path, self.sample_rate)

            # Generate mel spectrogram for visualization
            mel_spectrogram = AudioProcessor.generate_mel_spectrogram(audio, sr)

            # Detect audio events
            detections = self.detect_audio_in_file(audio_path)

            # Cleanup temporary audio file
            try:
                audio_path.unlink()
            except:
                pass

            return detections, mel_spectrogram

        except Exception as e:
            logger.error(f"Error processing video audio {video_path}: {e}")
            return [], np.array([])

    def generate_spectrogram_data(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Generate spectrogram data for frontend visualization

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            Dictionary with spectrogram data for D3.js
        """
        try:
            # Generate mel spectrogram
            mel_spec_db = AudioProcessor.generate_mel_spectrogram(audio, sr)

            # Create time and frequency axes
            n_frames = mel_spec_db.shape[1]
            n_mels = mel_spec_db.shape[0]

            duration = len(audio) / sr
            time_frames = np.linspace(0, duration, n_frames)
            mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)

            # Prepare data for D3.js
            spectrogram_data = {
                "data": mel_spec_db.tolist(),  # 2D array for heatmap
                "time": time_frames.tolist(),
                "frequencies": mel_frequencies.tolist(),
                "shape": [n_mels, n_frames],
                "db_range": [float(np.min(mel_spec_db)), float(np.max(mel_spec_db))],
                "duration": duration,
                "sample_rate": sr,
            }

            return spectrogram_data

        except Exception as e:
            logger.error(f"Error generating spectrogram data: {e}")
            return {}

    async def detect_audio_async(self, audio: np.ndarray, timestamp: float = 0.0) -> List[Dict]:
        """
        Async wrapper for audio detection

        Args:
            audio: Input audio chunk
            timestamp: Timestamp for detection

        Returns:
            List of detections
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.detect_audio_events,
                audio,
                timestamp
            )
        return result

    def _post_process_audio_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply post-processing to audio detections

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

        # Merge overlapping detections of same class
        merged_detections = []
        sorted_detections = sorted(filtered_detections, key=lambda x: x["timestamp"])

        for detection in sorted_detections:
            merged = False
            for existing in merged_detections:
                # Check if same class and overlapping time
                if (detection["class"] == existing["class"] and
                        detection["chunk_start"] < existing["chunk_end"] and
                        detection["chunk_end"] > existing["chunk_start"]):

                    # Merge detections - use higher confidence and expand time range
                    if detection["confidence"] > existing["confidence"]:
                        existing["confidence"] = detection["confidence"]

                    existing["chunk_start"] = min(existing["chunk_start"], detection["chunk_start"])
                    existing["chunk_end"] = max(existing["chunk_end"], detection["chunk_end"])
                    existing["timestamp"] = existing["chunk_start"]
                    existing["duration"] = existing["chunk_end"] - existing["chunk_start"]

                    merged = True
                    break

            if not merged:
                merged_detections.append(detection)

        logger.info(f"Audio post-processing: {len(detections)} → {len(merged_detections)} detections")
        return merged_detections

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        avg_inference_time = np.mean(self.inference_times)
        total_audio_time = self.detection_stats["total_audio_duration"]

        stats = {
            "total_chunks_processed": self.detection_stats["total_chunks_processed"],
            "total_detections": self.detection_stats["total_detections"],
            "detections_by_class": self.detection_stats["detections_by_class"],
            "total_audio_duration": total_audio_time,
            "average_inference_time": avg_inference_time,
            "processing_speed_ratio": total_audio_time / sum(self.inference_times) if self.inference_times else 0,
            "device": self.device,
            "model_loaded": self.is_loaded,
        }

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times = []
        self.detection_stats = {
            "total_chunks_processed": 0,
            "total_detections": 0,
            "detections_by_class": {},
            "total_audio_duration": 0.0,
        }
        logger.info("Audio detection statistics reset")

    def __enter__(self):
        """Context manager entry"""
        if not self.is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("WarAudioDetector context closed")


# Convenience functions for easy usage
def detect_audio_in_file(audio_path: Path) -> List[Dict]:
    """
    Convenience function to detect audio events in file

    Args:
        audio_path: Path to audio file

    Returns:
        List of detections
    """
    detector = WarAudioDetector()
    return detector.detect_audio_in_file(audio_path)


def detect_audio_in_video(video_path: Path) -> Tuple[List[Dict], np.ndarray]:
    """
    Convenience function to detect audio events in video

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (detections, mel_spectrogram)
    """
    detector = WarAudioDetector()
    return detector.detect_audio_in_video(video_path)


# Export classes and functions
__all__ = [
    "WarAudioDetector",
    "detect_audio_in_file",
    "detect_audio_in_video"
]

# Test functionality
if __name__ == "__main__":
    print("Testing War Audio Detection Module...")

    if not PANN_AVAILABLE:
        print("❌ PANN not available. Install with:")
        print("   pip install panns-inference")
        exit(1)

    # Initialize detector
    detector = WarAudioDetector()

    # Test model loading
    if detector.load_model():
        print("✅ PANN model loaded successfully")

        # Test with dummy audio (10 seconds at 32kHz for PANN)
        expected_length = detector.sample_rate * 10
        dummy_audio = np.random.randn(expected_length).astype(np.float32)
        detections = detector.detect_audio_events(dummy_audio)
        print(f"✅ Dummy audio test: {len(detections)} detections")

        # Test spectrogram generation
        spec_data = detector.generate_spectrogram_data(dummy_audio, detector.sample_rate)
        print(f"✅ Spectrogram generation: {len(spec_data)} data points")

        # Print performance stats
        stats = detector.get_performance_stats()
        print(f"✅ Performance stats: {stats}")

    else:
        print("❌ Failed to load PANN model")

    print("Audio detection module test complete!")