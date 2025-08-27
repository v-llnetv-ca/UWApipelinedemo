"""
War Detection Pipeline - Main Media Processor
Main orchestrator that coordinates object detection, audio detection, deduplication, and fusion
"""
import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import json

from config import (
    UPLOAD_DIR, PROCESSED_DIR, PERFORMANCE_CONFIG,
    FILE_CONFIG, OBJECT_DETECTION_CONFIG, AUDIO_DETECTION_CONFIG
)
from utils import (
    VideoProcessor, AudioProcessor, FileManager,
    PerformanceMonitor, logger
)
from object_detection import WarObjectDetector
from audio_detection import WarAudioDetector
from deduplication import MediaDeduplicator
from fusion import SimpleFusionEngine


class ProcessingStatus:
    """Track processing status for real-time updates"""

    def __init__(self, job_id: str, file_path: Path):
        self.job_id = job_id
        self.file_path = file_path
        self.status = "queued"  # queued, processing, completed, failed
        self.progress = 0.0  # 0.0 to 1.0
        self.current_step = "Initializing..."
        self.steps_completed = 0
        self.total_steps = 7  # dedup, object, audio, fusion, save, cleanup, done
        self.results = {}
        self.error_message = None
        self.started_at = datetime.now()
        self.completed_at = None
        self.processing_time = None

        # Step details
        self.step_details = {
            1: "Checking for duplicates",
            2: "Processing object detection",
            3: "Processing audio detection",
            4: "Fusing multi-modal results",
            5: "Saving results",
            6: "Cleaning up temporary files",
            7: "Complete"
        }

    def update_progress(self, step: int, message: str = None):
        """Update processing progress"""
        self.steps_completed = step
        self.progress = step / self.total_steps
        self.current_step = message or self.step_details.get(step, f"Step {step}")

        if step >= self.total_steps:
            self.status = "completed"
            self.completed_at = datetime.now()
            self.processing_time = (self.completed_at - self.started_at).total_seconds()

        logger.info(f"Job {self.job_id}: Step {step}/{self.total_steps} - {self.current_step}")

    def set_error(self, error_message: str):
        """Set error status"""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.now()
        self.processing_time = (self.completed_at - self.started_at).total_seconds()
        logger.error(f"Job {self.job_id} failed: {error_message}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'job_id': self.job_id,
            'file_path': str(self.file_path),
            'status': self.status,
            'progress': self.progress,
            'current_step': self.current_step,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'results': self.results,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_time': self.processing_time
        }


class MediaProcessor:
    """Main pipeline processor for war detection"""

    def __init__(self):
        # Initialize all components
        self.object_detector = WarObjectDetector()
        self.audio_detector = WarAudioDetector()
        self.deduplicator = MediaDeduplicator()
        self.fusion_engine = SimpleFusionEngine()

        # Job tracking
        self.active_jobs: Dict[str, ProcessingStatus] = {}
        self.completed_jobs: Dict[str, ProcessingStatus] = {}

        # Performance tracking
        self.performance_stats = {
            'total_files_processed': 0,
            'successful_processings': 0,
            'failed_processings': 0,
            'avg_processing_time': 0.0,
            'processing_times': []
        }

        # Callbacks for real-time updates
        self.progress_callbacks: List[Callable] = []

        logger.info("MediaProcessor initialized with all components")

    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)

    def _notify_progress(self, job_status: ProcessingStatus):
        """Notify all callbacks of progress update"""
        for callback in self.progress_callbacks:
            try:
                callback(job_status.to_dict())
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def generate_job_id(self) -> str:
        """Generate unique job ID"""
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"job_{timestamp}_{unique_id}"

    async def process_file(self, file_path: Path,
                           progress_callback: Optional[Callable] = None) -> Dict:
        """
        Process a single media file through the complete pipeline

        Args:
            file_path: Path to media file
            progress_callback: Optional callback for progress updates

        Returns:
            Processing results dictionary
        """
        job_id = self.generate_job_id()
        job_status = ProcessingStatus(job_id, file_path)
        job_status.status = "processing"

        self.active_jobs[job_id] = job_status

        # Add temporary callback if provided
        if progress_callback:
            self.progress_callbacks.append(progress_callback)

        try:
            logger.info(f"Starting processing job {job_id} for {file_path}")

            # Step 1: File validation
            job_status.update_progress(0, "Validating file...")
            self._notify_progress(job_status)

            validation_result = FileManager.validate_file(file_path)
            if not validation_result['valid']:
                raise ValueError(f"Invalid file: {validation_result['errors']}")

            # Step 2: Deduplication check
            job_status.update_progress(1, "Checking for duplicates...")
            self._notify_progress(job_status)

            dedup_result = self.deduplicator.process_file(file_path)

            # Step 3: Object detection
            job_status.update_progress(2, "Processing object detection...")
            self._notify_progress(job_status)

            if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats']:
                # Process video
                object_detections = self.object_detector.detect_objects_in_video(file_path)
            else:
                # Process image
                import cv2
                image = cv2.imread(str(file_path))
                object_detections = self.object_detector.detect_objects_in_image(image)

            logger.info(f"Object detection found {len(object_detections)} detections")

            # Step 4: Audio detection (only for videos)
            job_status.update_progress(3, "Processing audio detection...")
            self._notify_progress(job_status)

            audio_detections = []
            mel_spectrogram = None

            if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats']:
                try:
                    audio_detections, mel_spectrogram = self.audio_detector.detect_audio_in_video(file_path)
                    logger.info(f"Audio detection found {len(audio_detections)} detections")
                except Exception as e:
                    logger.warning(f"Audio detection failed: {e}")
                    # Continue without audio detection
                    audio_detections = []
            else:
                logger.info("Image file - skipping audio detection")

            # Step 5: Fusion
            job_status.update_progress(4, "Fusing multi-modal results...")
            self._notify_progress(job_status)

            fusion_results = self.fusion_engine.fuse_detections(object_detections, audio_detections)

            # Step 6: Generate final results
            job_status.update_progress(5, "Generating final results...")
            self._notify_progress(job_status)

            # Prepare comprehensive results
            final_results = {
                'job_id': job_id,
                'file_info': {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'type': 'video' if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats'] else 'image'
                },
                'deduplication': {
                    'is_duplicate': dedup_result.get('is_duplicate', False),
                    'duplicate_count': dedup_result.get('duplicate_count', 0),
                    'duplicates': dedup_result.get('duplicates', [])
                },
                'detections': {
                    'object_detections': object_detections,
                    'audio_detections': audio_detections,
                    'fusion_results': fusion_results
                },
                'summary': {
                    'all_tags': fusion_results['summary']['all_tags'],
                    'object_classes': fusion_results['summary']['object_classes_detected'],
                    'audio_classes': fusion_results['summary']['audio_classes_detected'],
                    'total_detections': (
                            fusion_results['summary']['total_object_detections'] +
                            fusion_results['summary']['total_audio_detections']
                    ),
                    'timeline': fusion_results['all_detections_timeline']
                },
                'performance': {
                    'object_detector_stats': self.object_detector.get_performance_stats(),
                    'audio_detector_stats': self.audio_detector.get_performance_stats(),
                    'deduplicator_stats': self.deduplicator.get_performance_stats(),
                    'fusion_stats': self.fusion_engine.get_performance_stats()
                },
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'processing_time': None,  # Will be set when complete
                    'pipeline_version': '1.0'
                }
            }

            # Add spectrogram data if available
            if mel_spectrogram is not None:
                final_results['spectrogram'] = {
                    'data': mel_spectrogram.tolist(),
                    'shape': list(mel_spectrogram.shape)
                }

            # Step 7: Cleanup and finalize
            job_status.update_progress(6, "Cleaning up...")
            self._notify_progress(job_status)

            # Save results to file
            results_path = PROCESSED_DIR / f"{job_id}_results.json"
            with open(results_path, 'w') as f:
                # Create JSON-serializable copy
                json_results = json.loads(json.dumps(final_results, default=str))
                json.dump(json_results, f, indent=2)

            final_results['results_file'] = str(results_path)

            # Complete
            job_status.update_progress(7, "Complete!")
            job_status.results = final_results
            final_results['metadata']['processing_time'] = job_status.processing_time
            self._notify_progress(job_status)

            # Move to completed jobs
            self.completed_jobs[job_id] = job_status
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # Update performance stats
            self.performance_stats['total_files_processed'] += 1
            self.performance_stats['successful_processings'] += 1
            self.performance_stats['processing_times'].append(job_status.processing_time)

            if self.performance_stats['processing_times']:
                self.performance_stats['avg_processing_time'] = np.mean(
                    self.performance_stats['processing_times']
                )

            logger.info(f"Job {job_id} completed successfully in {job_status.processing_time:.2f}s")
            logger.info(f"Final tags: {final_results['summary']['all_tags']}")

            return final_results

        except Exception as e:
            # Handle errors
            job_status.set_error(str(e))
            self._notify_progress(job_status)

            # Move to completed jobs with error
            self.completed_jobs[job_id] = job_status
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # Update performance stats
            self.performance_stats['total_files_processed'] += 1
            self.performance_stats['failed_processings'] += 1

            logger.error(f"Job {job_id} failed: {e}")

            # Return error result
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'file_info': {
                    'path': str(file_path),
                    'name': file_path.name
                }
            }

        finally:
            # Clean up temporary callback
            if progress_callback and progress_callback in self.progress_callbacks:
                self.progress_callbacks.remove(progress_callback)

    async def process_file_fast(self, file_path: Path,
                                progress_callback: Optional[Callable] = None) -> Dict:
        """
        Fast processing for demo - skips deduplication for speed

        Args:
            file_path: Path to media file
            progress_callback: Optional callback for progress updates

        Returns:
            Processing results dictionary
        """
        job_id = self.generate_job_id()
        job_status = ProcessingStatus(job_id, file_path)
        job_status.status = "processing"
        job_status.total_steps = 5  # Reduced steps (no deduplication)

        # Update step details for fast processing
        job_status.step_details = {
            1: "Validating file",
            2: "Processing object detection",
            3: "Processing audio detection",
            4: "Fusing multi-modal results",
            5: "Complete"
        }

        self.active_jobs[job_id] = job_status

        # Add temporary callback if provided
        if progress_callback:
            self.progress_callbacks.append(progress_callback)

        try:
            logger.info(f"Starting FAST processing job {job_id} for {file_path}")

            # Step 1: File validation (same as before)
            job_status.update_progress(1, "Validating file...")
            self._notify_progress(job_status)

            validation_result = FileManager.validate_file(file_path)
            if not validation_result['valid']:
                raise ValueError(f"Invalid file: {validation_result['errors']}")

            # SKIP DEDUPLICATION - go straight to object detection

            # Step 2: Object detection
            job_status.update_progress(2, "Processing object detection...")
            self._notify_progress(job_status)

            if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats']:
                # Process video
                object_detections = self.object_detector.detect_objects_in_video(file_path)
            else:
                # Process image
                import cv2
                image = cv2.imread(str(file_path))
                object_detections = self.object_detector.detect_objects_in_image(image)

            logger.info(f"Fast object detection found {len(object_detections)} detections")

            # Step 3: Audio detection (only for videos)
            job_status.update_progress(3, "Processing audio detection...")
            self._notify_progress(job_status)

            audio_detections = []
            mel_spectrogram = None

            if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats']:
                try:
                    audio_detections, mel_spectrogram = self.audio_detector.detect_audio_in_video(file_path)
                    logger.info(f"Fast audio detection found {len(audio_detections)} detections")
                except Exception as e:
                    logger.warning(f"Audio detection failed: {e}")
                    # Continue without audio detection
                    audio_detections = []
            else:
                logger.info("Image file - skipping audio detection")

            # Step 4: Fusion
            job_status.update_progress(4, "Fusing multi-modal results...")
            self._notify_progress(job_status)

            fusion_results = self.fusion_engine.fuse_detections(object_detections, audio_detections)

            # Step 5: Generate final results (no file saving for speed)
            job_status.update_progress(5, "Complete!")
            self._notify_progress(job_status)

            # Prepare comprehensive results
            final_results = {
                'job_id': job_id,
                'processing_mode': 'fast',
                'file_info': {
                    'path': str(file_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'type': 'video' if file_path.suffix.lower() in FILE_CONFIG['allowed_video_formats'] else 'image'
                },
                'deduplication': {
                    'skipped': True,
                    'reason': 'Fast processing mode'
                },
                'detections': {
                    'object_detections': object_detections,
                    'audio_detections': audio_detections,
                    'fusion_results': fusion_results
                },
                'summary': {
                    'all_tags': fusion_results['summary']['all_tags'],
                    'object_classes': fusion_results['summary']['object_classes_detected'],
                    'audio_classes': fusion_results['summary']['audio_classes_detected'],
                    'total_detections': (
                            fusion_results['summary']['total_object_detections'] +
                            fusion_results['summary']['total_audio_detections']
                    ),
                    'timeline': fusion_results['all_detections_timeline']
                },
                'performance': {
                    'object_detector_stats': self.object_detector.get_performance_stats(),
                    'audio_detector_stats': self.audio_detector.get_performance_stats(),
                    'fusion_stats': self.fusion_engine.get_performance_stats()
                },
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'processing_time': job_status.processing_time,
                    'pipeline_version': '1.0_fast'
                }
            }

            # Add spectrogram data if available
            if mel_spectrogram is not None:
                final_results['spectrogram'] = {
                    'data': mel_spectrogram.tolist(),
                    'shape': list(mel_spectrogram.shape)
                }

            # Complete
            job_status.results = final_results
            final_results['metadata']['processing_time'] = job_status.processing_time

            # Move to completed jobs
            self.completed_jobs[job_id] = job_status
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # Update performance stats
            self.performance_stats['total_files_processed'] += 1
            self.performance_stats['successful_processings'] += 1
            self.performance_stats['processing_times'].append(job_status.processing_time)

            if self.performance_stats['processing_times']:
                self.performance_stats['avg_processing_time'] = np.mean(
                    self.performance_stats['processing_times']
                )

            logger.info(f"FAST job {job_id} completed successfully in {job_status.processing_time:.2f}s")
            logger.info(f"Final tags: {final_results['summary']['all_tags']}")

            return final_results

        except Exception as e:
            # Handle errors (same as regular processing)
            job_status.set_error(str(e))
            self._notify_progress(job_status)

            # Move to completed jobs with error
            self.completed_jobs[job_id] = job_status
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

            # Update performance stats
            self.performance_stats['total_files_processed'] += 1
            self.performance_stats['failed_processings'] += 1

            logger.error(f"FAST job {job_id} failed: {e}")

            # Return error result
            return {
                'job_id': job_id,
                'processing_mode': 'fast',
                'status': 'failed',
                'error': str(e),
                'file_info': {
                    'path': str(file_path),
                    'name': file_path.name
                }
            }

        finally:
            # Clean up temporary callback
            if progress_callback and progress_callback in self.progress_callbacks:
                self.progress_callbacks.remove(progress_callback)

    async def process_batch(self, file_paths: List[Path]) -> List[Dict]:
        """Process multiple files concurrently"""
        logger.info(f"Starting batch processing of {len(file_paths)} files")

        # Process files with limited concurrency
        max_concurrent = min(PERFORMANCE_CONFIG.get('max_concurrent_processing', 3), len(file_paths))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_file(file_path)

        tasks = [process_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for {file_paths[i]}: {result}")
                processed_results.append({
                    'job_id': f"failed_{i}",
                    'status': 'failed',
                    'error': str(result),
                    'file_info': {'path': str(file_paths[i])}
                })
            else:
                processed_results.append(result)

        logger.info(f"Batch processing complete: {len(processed_results)} results")
        return processed_results

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of specific job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id].to_dict()
        else:
            return None

    def get_all_jobs(self) -> Dict:
        """Get status of all jobs"""
        return {
            'active': {job_id: status.to_dict() for job_id, status in self.active_jobs.items()},
            'completed': {job_id: status.to_dict() for job_id, status in self.completed_jobs.items()},
            'performance_stats': self.performance_stats
        }

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        jobs_to_remove = []
        for job_id, status in self.completed_jobs.items():
            if status.completed_at and status.completed_at < cutoff_time:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]

            # Also clean up results file
            try:
                results_path = PROCESSED_DIR / f"{job_id}_results.json"
                if results_path.exists():
                    results_path.unlink()
            except:
                pass

        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()

        # Add component stats
        stats['components'] = {
            'object_detector': self.object_detector.get_performance_stats(),
            'audio_detector': self.audio_detector.get_performance_stats(),
            'deduplicator': self.deduplicator.get_performance_stats(),
            'fusion_engine': self.fusion_engine.get_performance_stats()
        }

        return stats


# Convenience functions
async def process_war_footage(file_path: Path) -> Dict:
    """
    Convenience function to process single file

    Args:
        file_path: Path to media file

    Returns:
        Processing results
    """
    processor = MediaProcessor()
    return await processor.process_file(file_path)


async def batch_process_war_footage(file_paths: List[Path]) -> List[Dict]:
    """
    Convenience function to process multiple files

    Args:
        file_paths: List of file paths

    Returns:
        List of processing results
    """
    processor = MediaProcessor()
    return await processor.process_batch(file_paths)


# Export classes and functions
__all__ = [
    'ProcessingStatus', 'MediaProcessor',
    'process_war_footage', 'batch_process_war_footage'
]

# Test functionality
if __name__ == "__main__":
    import asyncio

    print("Testing Media Processor...")

    # Initialize processor
    processor = MediaProcessor()
    print("✅ MediaProcessor initialized")

    # Test with dummy file (if exists)
    test_files = [
        Path("test_video.mp4"),
        Path("test_image.jpg")
    ]

    for test_file in test_files:
        if test_file.exists():
            try:
                print(f"Testing with {test_file}...")
                result = asyncio.run(processor.process_file(test_file))
                print(f"✅ Processed {test_file}: {result['summary']['all_tags']}")
            except Exception as e:
                print(f"⚠️  Could not process {test_file}: {e}")
        else:
            print(f"⚠️  Test file not found: {test_file}")

    # Print performance stats
    stats = processor.get_performance_stats()
    print(f"✅ Performance stats: {stats['total_files_processed']} files processed")

    print("Media processor test complete!")