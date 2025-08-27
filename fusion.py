"""
War Detection Pipeline - Simple Multi-Modal Fusion Engine
Simple fusion system that combines object detection and audio detection results
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import FUSION_CONFIG
from utils import logger


class SimpleFusionEngine:
    """Simple multi-modal fusion - just combine and organize detections"""

    def __init__(self):
        self.config = FUSION_CONFIG
        self.performance_stats = {
            'total_fusions': 0,
            'object_detections_processed': 0,
            'audio_detections_processed': 0,
            'processing_times': []
        }

        logger.info("Initialized SimpleFusionEngine")

    def fuse_detections(self, object_detections: List[Dict],
                        audio_detections: List[Dict]) -> Dict:
        """
        Simple fusion: combine all detections and organize by timestamp

        Args:
            object_detections: Object detection results from YOLOv8
            audio_detections: Audio detection results from PANN

        Returns:
            Dictionary with organized detection results
        """
        start_time = time.time()

        try:
            # Prepare results structure
            fused_results = {
                'object_detections': [],
                'audio_detections': [],
                'all_detections_timeline': [],
                'summary': {
                    'object_classes_detected': set(),
                    'audio_classes_detected': set(),
                    'total_object_detections': len(object_detections),
                    'total_audio_detections': len(audio_detections),
                    'time_range': None
                }
            }

            all_timestamps = []

            # Process object detections
            for detection in object_detections:
                processed_detection = {
                    'type': 'object',
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'timestamp': detection['timestamp'],
                    'source': 'yolov8',
                    'metadata': {
                        'bbox': detection.get('bbox'),
                        'center': detection.get('center'),
                        'area': detection.get('area')
                    }
                }

                fused_results['object_detections'].append(processed_detection)
                fused_results['all_detections_timeline'].append(processed_detection)
                fused_results['summary']['object_classes_detected'].add(detection['class'])
                all_timestamps.append(detection['timestamp'])

            # Process audio detections
            for detection in audio_detections:
                processed_detection = {
                    'type': 'audio',
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'timestamp': detection['timestamp'],
                    'source': 'pann',
                    'metadata': {
                        'duration': detection.get('duration'),
                        'chunk_start': detection.get('chunk_start'),
                        'chunk_end': detection.get('chunk_end')
                    }
                }

                fused_results['audio_detections'].append(processed_detection)
                fused_results['all_detections_timeline'].append(processed_detection)
                fused_results['summary']['audio_classes_detected'].add(detection['class'])
                all_timestamps.append(detection['timestamp'])

            # Sort timeline by timestamp
            fused_results['all_detections_timeline'].sort(key=lambda x: x['timestamp'])

            # Calculate time range
            if all_timestamps:
                fused_results['summary']['time_range'] = {
                    'start': min(all_timestamps),
                    'end': max(all_timestamps),
                    'duration': max(all_timestamps) - min(all_timestamps)
                }

            # Convert sets to lists for JSON serialization
            fused_results['summary']['object_classes_detected'] = list(
                fused_results['summary']['object_classes_detected'])
            fused_results['summary']['audio_classes_detected'] = list(
                fused_results['summary']['audio_classes_detected'])

            # Generate combined tags for the footage
            all_tags = fused_results['summary']['object_classes_detected'] + fused_results['summary'][
                'audio_classes_detected']
            fused_results['summary']['all_tags'] = sorted(list(set(all_tags)))

            # Performance tracking
            processing_time = time.time() - start_time
            self.performance_stats['processing_times'].append(processing_time)
            self.performance_stats['total_fusions'] += 1
            self.performance_stats['object_detections_processed'] += len(object_detections)
            self.performance_stats['audio_detections_processed'] += len(audio_detections)

            logger.info(f"Simple fusion complete: {len(object_detections)} object + "
                        f"{len(audio_detections)} audio detections combined "
                        f"(processed in {processing_time:.3f}s)")
            logger.info(f"Tags generated: {fused_results['summary']['all_tags']}")

            return fused_results

        except Exception as e:
            logger.error(f"Simple fusion failed: {e}")
            # Return empty structure on error
            return {
                'object_detections': object_detections,
                'audio_detections': audio_detections,
                'all_detections_timeline': [],
                'summary': {'error': str(e)},
                'error': True
            }

    def filter_detections_by_confidence(self, fused_results: Dict,
                                        object_threshold: float = 0.5,
                                        audio_threshold: float = 0.6) -> Dict:
        """
        Filter detections by confidence thresholds

        Args:
            fused_results: Results from fuse_detections
            object_threshold: Minimum confidence for object detections
            audio_threshold: Minimum confidence for audio detections

        Returns:
            Filtered results
        """
        filtered_results = fused_results.copy()

        # Filter object detections
        filtered_results['object_detections'] = [
            det for det in fused_results['object_detections']
            if det['confidence'] >= object_threshold
        ]

        # Filter audio detections
        filtered_results['audio_detections'] = [
            det for det in fused_results['audio_detections']
            if det['confidence'] >= audio_threshold
        ]

        # Rebuild timeline
        filtered_results['all_detections_timeline'] = (
                filtered_results['object_detections'] +
                filtered_results['audio_detections']
        )
        filtered_results['all_detections_timeline'].sort(key=lambda x: x['timestamp'])

        # Update summary
        object_classes = set(det['class'] for det in filtered_results['object_detections'])
        audio_classes = set(det['class'] for det in filtered_results['audio_detections'])

        filtered_results['summary']['object_classes_detected'] = list(object_classes)
        filtered_results['summary']['audio_classes_detected'] = list(audio_classes)
        filtered_results['summary']['all_tags'] = sorted(list(object_classes | audio_classes))
        filtered_results['summary']['total_object_detections'] = len(filtered_results['object_detections'])
        filtered_results['summary']['total_audio_detections'] = len(filtered_results['audio_detections'])

        logger.info(f"Filtered detections: {len(filtered_results['object_detections'])} object, "
                    f"{len(filtered_results['audio_detections'])} audio")

        return filtered_results

    def get_timeline_for_frontend(self, fused_results: Dict) -> List[Dict]:
        """
        Generate timeline data optimized for frontend display

        Args:
            fused_results: Results from fuse_detections

        Returns:
            List of timeline events for frontend
        """
        timeline = []

        for detection in fused_results['all_detections_timeline']:
            timeline_event = {
                'timestamp': detection['timestamp'],
                'type': detection['type'],
                'class': detection['class'],
                'confidence': detection['confidence'],
                'formatted_time': self._format_timestamp(detection['timestamp']),
                'color': self._get_class_color(detection['class'], detection['type']),
            }
            timeline.append(timeline_event)

        return timeline

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp as MM:SS"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _get_class_color(self, class_name: str, detection_type: str) -> str:
        """Get color for class visualization"""
        colors = {
            # Object detection colors
            'tank': '#FF4444',
            'helicopter': '#44FF44',
            'weapon': '#4444FF',

            # Audio detection colors
            'gunshot_gunfire': '#FF8800',
            'explosion': '#FF0088',
            'helicopter': '#44FF44',  # Same as object helicopter
        }

        return colors.get(class_name, '#888888')

    async def fuse_detections_async(self, object_detections: List[Dict],
                                    audio_detections: List[Dict]) -> Dict:
        """Async wrapper for fusion"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.fuse_detections,
                object_detections,
                audio_detections
            )
        return result

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()

        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_fusions': 0,
            'object_detections_processed': 0,
            'audio_detections_processed': 0,
            'processing_times': []
        }
        logger.info("Simple fusion engine statistics reset")


# Convenience functions
def simple_fuse_detections(object_detections: List[Dict], audio_detections: List[Dict]) -> Dict:
    """
    Simple convenience function to fuse detections

    Args:
        object_detections: Object detection results
        audio_detections: Audio detection results

    Returns:
        Fused results dictionary
    """
    engine = SimpleFusionEngine()
    return engine.fuse_detections(object_detections, audio_detections)


def get_footage_tags(object_detections: List[Dict], audio_detections: List[Dict]) -> List[str]:
    """
    Get simple list of tags for footage

    Args:
        object_detections: Object detection results
        audio_detections: Audio detection results

    Returns:
        List of unique tags found in footage
    """
    engine = SimpleFusionEngine()
    results = engine.fuse_detections(object_detections, audio_detections)
    return results['summary']['all_tags']


# Export classes and functions
__all__ = [
    'SimpleFusionEngine',
    'simple_fuse_detections',
    'get_footage_tags'
]

# Test functionality
if __name__ == "__main__":
    print("Testing Simple Multi-Modal Fusion Engine...")

    # Create sample detections
    sample_object_detections = [
        {'class': 'tank', 'confidence': 0.85, 'timestamp': 10.5, 'bbox': [100, 100, 200, 200]},
        {'class': 'helicopter', 'confidence': 0.92, 'timestamp': 25.3, 'bbox': [300, 150, 450, 300]},
        {'class': 'weapon', 'confidence': 0.78, 'timestamp': 45.1, 'bbox': [50, 250, 120, 320]}
    ]

    sample_audio_detections = [
        {'class': 'gunshot_gunfire', 'confidence': 0.88, 'timestamp': 10.8, 'duration': 2.0},
        {'class': 'helicopter', 'confidence': 0.91, 'timestamp': 25.1, 'duration': 5.0},
        {'class': 'explosion', 'confidence': 0.82, 'timestamp': 50.2, 'duration': 1.5}
    ]

    # Test simple fusion
    engine = SimpleFusionEngine()
    print("SimpleFusionEngine initialized")

    # Perform fusion
    results = engine.fuse_detections(sample_object_detections, sample_audio_detections)
    print(f"Simple fusion: {results['summary']['total_object_detections']} object + "
          f"{results['summary']['total_audio_detections']} audio detections")
    print(f"Tags generated: {results['summary']['all_tags']}")

    # Test timeline generation
    timeline = engine.get_timeline_for_frontend(results)
    print(f"Timeline events: {len(timeline)}")

    # Show timeline
    for event in timeline[:5]:  # Show first 5 events
        print(f"   {event['formatted_time']}: {event['icon']} {event['class']} ({event['confidence']:.2f})")

    # Test convenience function
    tags = get_footage_tags(sample_object_detections, sample_audio_detections)
    print(f"Footage tags: {tags}")

    # Performance stats
    stats = engine.get_performance_stats()
    print(f"Performance: {stats}")

    print("Simple fusion engine test complete!")