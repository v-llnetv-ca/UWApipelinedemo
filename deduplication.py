"""
War Detection Pipeline - Media Deduplication Module
Professional deduplication system for images and videos with watermark/filter tolerance
"""
import cv2
import numpy as np
import hashlib
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set
import sqlite3
import json
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import (
    DEDUPLICATION_CONFIG, DATABASE_CONFIG, PROCESSED_DIR,
    PERFORMANCE_CONFIG
)
from utils import (
    VideoProcessor, FileManager, PerformanceMonitor,
    logger, perf_monitor
)


class MediaFingerprint:
    """Media fingerprint for efficient similarity comparison"""

    def __init__(self, file_path: Path, media_type: str):
        self.file_path = file_path
        self.media_type = media_type  # 'image' or 'video'
        self.file_hash = None
        self.perceptual_hashes = {}
        self.temporal_fingerprint = None
        self.metadata = {}
        self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert fingerprint to dictionary for storage"""
        return {
            'file_path': str(self.file_path),
            'media_type': self.media_type,
            'file_hash': self.file_hash,
            'perceptual_hashes': self.perceptual_hashes,
            'temporal_fingerprint': self.temporal_fingerprint,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MediaFingerprint':
        """Create fingerprint from dictionary"""
        fingerprint = cls(Path(data['file_path']), data['media_type'])
        fingerprint.file_hash = data['file_hash']
        fingerprint.perceptual_hashes = data['perceptual_hashes']
        fingerprint.temporal_fingerprint = data['temporal_fingerprint']
        fingerprint.metadata = data['metadata']
        fingerprint.created_at = datetime.fromisoformat(data['created_at'])
        return fingerprint


class ImageDeduplicator:
    """Advanced image deduplication with perceptual hashing"""

    def __init__(self, hash_size: int = None):
        self.hash_size = hash_size or DEDUPLICATION_CONFIG["image_hash_size"]
        self.config = DEDUPLICATION_CONFIG

    def generate_perceptual_hashes(self, image_path: Path) -> Dict[str, str]:
        """
        Generate multiple perceptual hashes for robust comparison

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with different hash types
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Generate different types of perceptual hashes
            hashes = {
                'phash': str(imagehash.phash(image, hash_size=self.hash_size)),
                'dhash': str(imagehash.dhash(image, hash_size=self.hash_size)),
                'ahash': str(imagehash.average_hash(image, hash_size=self.hash_size)),
                'whash': str(imagehash.whash(image, hash_size=self.hash_size)),
            }

            return hashes

        except Exception as e:
            logger.error(f"Failed to generate hashes for {image_path}: {e}")
            return {}

    def calculate_similarity(self, hash1: str, hash2: str, hash_type: str = 'phash') -> float:
        """
        Calculate similarity between two perceptual hashes

        Args:
            hash1, hash2: Perceptual hash strings
            hash_type: Type of hash for threshold selection

        Returns:
            Similarity score (0-1, where 1 is identical)
        """
        try:
            # Convert string hashes back to ImageHash objects
            if hash_type == 'phash':
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
            elif hash_type == 'dhash':
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
            elif hash_type == 'ahash':
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
            elif hash_type == 'whash':
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
            else:
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)

            # Calculate Hamming distance
            hamming_distance = h1 - h2

            # Convert to similarity (lower distance = higher similarity)
            max_distance = self.hash_size ** 2  # Maximum possible distance
            similarity = 1.0 - (hamming_distance / max_distance)

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def is_duplicate(self, hashes1: Dict[str, str], hashes2: Dict[str, str]) -> Tuple[bool, float, Dict]:
        """
        Determine if two images are duplicates based on multiple hash comparisons

        Args:
            hashes1, hashes2: Perceptual hash dictionaries

        Returns:
            Tuple of (is_duplicate, max_similarity, similarity_details)
        """
        similarities = {}
        max_similarity = 0.0

        # Compare each hash type
        for hash_type in ['phash', 'dhash', 'ahash', 'whash']:
            if hash_type in hashes1 and hash_type in hashes2:
                similarity = self.calculate_similarity(
                    hashes1[hash_type],
                    hashes2[hash_type],
                    hash_type
                )
                similarities[hash_type] = similarity
                max_similarity = max(max_similarity, similarity)

        # Determine if duplicate based on thresholds
        is_duplicate = False
        for hash_type, similarity in similarities.items():
            threshold = self.config["similarity_thresholds"].get(hash_type, 0.85)
            if similarity >= threshold:
                is_duplicate = True
                break

        # Additional logic: If multiple hash types show high similarity
        high_similarity_count = sum(1 for s in similarities.values() if s > 0.8)
        if high_similarity_count >= 2:
            is_duplicate = True

        return is_duplicate, max_similarity, similarities


class VideoDeduplicator:
    """Advanced video deduplication with temporal fingerprinting"""

    def __init__(self, frame_sample_rate: float = None):
        self.frame_sample_rate = frame_sample_rate or DEDUPLICATION_CONFIG["video_frame_sample_rate"]
        self.config = DEDUPLICATION_CONFIG
        self.image_dedup = ImageDeduplicator()

    def generate_temporal_fingerprint(self, video_path: Path) -> List[Dict]:
        """
        Generate temporal fingerprint by sampling frames and creating hashes

        Args:
            video_path: Path to video file

        Returns:
            List of frame fingerprints with timestamps
        """
        try:
            fingerprint = []

            # Sample frames at specified rate
            for timestamp, frame in VideoProcessor.extract_frames(video_path, self.frame_sample_rate):
                # Save frame temporarily for hash generation
                temp_frame_path = PROCESSED_DIR / f"temp_frame_{int(timestamp * 1000)}.jpg"
                cv2.imwrite(str(temp_frame_path), frame)

                # Generate perceptual hashes for the frame
                frame_hashes = self.image_dedup.generate_perceptual_hashes(temp_frame_path)

                if frame_hashes:
                    fingerprint.append({
                        'timestamp': timestamp,
                        'hashes': frame_hashes,
                        'frame_stats': self._calculate_frame_stats(frame)
                    })

                # Cleanup temp frame
                try:
                    temp_frame_path.unlink()
                except:
                    pass

                # Limit fingerprint size for performance
                if len(fingerprint) >= 100:  # Max 100 sample points
                    break

            logger.info(f"Generated temporal fingerprint with {len(fingerprint)} frames for {video_path.name}")
            return fingerprint

        except Exception as e:
            logger.error(f"Failed to generate temporal fingerprint for {video_path}: {e}")
            return []

    def _calculate_frame_stats(self, frame: np.ndarray) -> Dict:
        """Calculate additional frame statistics for comparison"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {
                'mean_brightness': float(np.mean(gray)),
                'std_brightness': float(np.std(gray)),
                'edges': float(np.mean(cv2.Canny(gray, 50, 150))),
                'histogram': cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten().tolist()
            }
        except:
            return {}

    def calculate_temporal_similarity(self, fingerprint1: List[Dict], fingerprint2: List[Dict]) -> float:
        """
        Calculate similarity between two temporal fingerprints

        Args:
            fingerprint1, fingerprint2: Temporal fingerprints

        Returns:
            Similarity score (0-1)
        """
        if not fingerprint1 or not fingerprint2:
            return 0.0

        # Compare frame-by-frame using sliding window approach
        total_similarity = 0.0
        comparisons = 0

        for i, frame1 in enumerate(fingerprint1):
            best_similarity = 0.0

            # Find best matching frame in fingerprint2 within a time window
            target_time = frame1['timestamp']
            time_window = 5.0  # ±5 seconds

            for frame2 in fingerprint2:
                if abs(frame2['timestamp'] - target_time) <= time_window:
                    # Compare frame hashes
                    frame_similarity = self._compare_frame_hashes(
                        frame1['hashes'],
                        frame2['hashes']
                    )

                    # Add frame statistics comparison
                    stats_similarity = self._compare_frame_stats(
                        frame1.get('frame_stats', {}),
                        frame2.get('frame_stats', {})
                    )

                    # Combined similarity
                    combined_similarity = (frame_similarity * 0.8 + stats_similarity * 0.2)
                    best_similarity = max(best_similarity, combined_similarity)

            total_similarity += best_similarity
            comparisons += 1

            # Performance optimization: sample every few frames
            if i > 0 and i % 5 == 0:
                continue

        return total_similarity / comparisons if comparisons > 0 else 0.0

    def _compare_frame_hashes(self, hashes1: Dict, hashes2: Dict) -> float:
        """Compare frame-level perceptual hashes"""
        if not hashes1 or not hashes2:
            return 0.0

        similarities = []
        for hash_type in ['phash', 'dhash', 'ahash']:
            if hash_type in hashes1 and hash_type in hashes2:
                similarity = self.image_dedup.calculate_similarity(
                    hashes1[hash_type],
                    hashes2[hash_type],
                    hash_type
                )
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _compare_frame_stats(self, stats1: Dict, stats2: Dict) -> float:
        """Compare frame statistics"""
        if not stats1 or not stats2:
            return 0.0

        similarities = []

        # Compare brightness statistics
        for key in ['mean_brightness', 'std_brightness', 'edges']:
            if key in stats1 and key in stats2:
                diff = abs(stats1[key] - stats2[key])
                max_val = max(stats1[key], stats2[key], 1.0)
                similarity = 1.0 - (diff / max_val)
                similarities.append(max(0.0, similarity))

        # Compare histograms if available
        if 'histogram' in stats1 and 'histogram' in stats2:
            hist1 = np.array(stats1['histogram'])
            hist2 = np.array(stats2['histogram'])
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarities.append(max(0.0, correlation))

        return np.mean(similarities) if similarities else 0.0

    def is_duplicate(self, fingerprint1: List[Dict], fingerprint2: List[Dict]) -> Tuple[bool, float]:
        """
        Determine if two videos are duplicates

        Args:
            fingerprint1, fingerprint2: Temporal fingerprints

        Returns:
            Tuple of (is_duplicate, similarity_score)
        """
        similarity = self.calculate_temporal_similarity(fingerprint1, fingerprint2)
        threshold = self.config["similarity_thresholds"]["video"]

        is_duplicate = similarity >= threshold
        return is_duplicate, similarity


class MediaDeduplicator:
    """Main deduplication system combining image and video deduplication"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("deduplication.db")
        self.image_dedup = ImageDeduplicator()
        self.video_dedup = VideoDeduplicator()
        self.config = DEDUPLICATION_CONFIG

        # Initialize database
        self._init_database()

        # Performance tracking
        self.performance_stats = {
            'total_processed': 0,
            'duplicates_found': 0,
            'processing_times': [],
            'db_query_times': []
        }

        logger.info(f"Initialized MediaDeduplicator with database: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database for fingerprint storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create fingerprints table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fingerprints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        media_type TEXT NOT NULL,
                        file_hash TEXT,
                        perceptual_hashes TEXT,
                        temporal_fingerprint TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP,
                        last_accessed TIMESTAMP
                    )
                ''')

                # Create duplicates table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS duplicates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_file TEXT NOT NULL,
                        duplicate_file TEXT NOT NULL,
                        similarity_score REAL NOT NULL,
                        similarity_details TEXT,
                        detected_at TIMESTAMP,
                        FOREIGN KEY (original_file) REFERENCES fingerprints (file_path),
                        FOREIGN KEY (duplicate_file) REFERENCES fingerprints (file_path)
                    )
                ''')

                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON fingerprints (file_path)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_type ON fingerprints (media_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON fingerprints (file_hash)')

                conn.commit()
                logger.info("Deduplication database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def generate_fingerprint(self, file_path: Path) -> MediaFingerprint:
        """
        Generate comprehensive fingerprint for media file

        Args:
            file_path: Path to media file

        Returns:
            MediaFingerprint object
        """
        start_time = time.time()

        try:
            # Determine media type
            extension = file_path.suffix.lower()
            if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                media_type = 'image'
            elif extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                media_type = 'video'
            else:
                raise ValueError(f"Unsupported file type: {extension}")

            # Create fingerprint
            fingerprint = MediaFingerprint(file_path, media_type)

            # Generate file hash for exact duplicate detection
            fingerprint.file_hash = FileManager.calculate_file_hash(file_path)

            # Generate perceptual fingerprints based on media type
            if media_type == 'image':
                fingerprint.perceptual_hashes = self.image_dedup.generate_perceptual_hashes(file_path)
            elif media_type == 'video':
                # Generate video temporal fingerprint
                fingerprint.temporal_fingerprint = self.video_dedup.generate_temporal_fingerprint(file_path)

                # Also generate perceptual hashes for first frame
                try:
                    cap = cv2.VideoCapture(str(file_path))
                    ret, first_frame = cap.read()
                    cap.release()

                    if ret:
                        temp_frame_path = PROCESSED_DIR / f"temp_first_frame_{int(time.time() * 1000)}.jpg"
                        cv2.imwrite(str(temp_frame_path), first_frame)
                        fingerprint.perceptual_hashes = self.image_dedup.generate_perceptual_hashes(temp_frame_path)
                        temp_frame_path.unlink()
                except:
                    pass

            # Add metadata
            file_stat = file_path.stat()
            fingerprint.metadata = {
                'file_size': file_stat.st_size,
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                'extension': extension
            }

            # Performance tracking
            processing_time = time.time() - start_time
            self.performance_stats['processing_times'].append(processing_time)
            self.performance_stats['total_processed'] += 1

            logger.info(f"Generated fingerprint for {file_path.name} in {processing_time:.3f}s")
            return fingerprint

        except Exception as e:
            logger.error(f"Failed to generate fingerprint for {file_path}: {e}")
            raise

    def store_fingerprint(self, fingerprint: MediaFingerprint):
        """Store fingerprint in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO fingerprints 
                    (file_path, media_type, file_hash, perceptual_hashes, 
                     temporal_fingerprint, metadata, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(fingerprint.file_path),
                    fingerprint.media_type,
                    fingerprint.file_hash,
                    json.dumps(fingerprint.perceptual_hashes),
                    json.dumps(fingerprint.temporal_fingerprint),
                    json.dumps(fingerprint.metadata),
                    fingerprint.created_at.isoformat(),
                    datetime.now().isoformat()
                ))

                conn.commit()
                logger.debug(f"Stored fingerprint for {fingerprint.file_path}")

        except Exception as e:
            logger.error(f"Failed to store fingerprint: {e}")
            raise

    def find_duplicates(self, fingerprint: MediaFingerprint) -> List[Tuple[str, float, Dict]]:
        """
        Find duplicates for given fingerprint

        Args:
            fingerprint: MediaFingerprint to check

        Returns:
            List of (file_path, similarity_score, similarity_details) tuples
        """
        start_time = time.time()
        duplicates = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First check for exact duplicates (same file hash)
                cursor.execute('''
                    SELECT file_path FROM fingerprints 
                    WHERE file_hash = ? AND file_path != ?
                ''', (fingerprint.file_hash, str(fingerprint.file_path)))

                exact_duplicates = cursor.fetchall()
                for (file_path,) in exact_duplicates:
                    duplicates.append((file_path, 1.0, {'exact_duplicate': True}))

                # Then check for perceptual duplicates
                cursor.execute('''
                    SELECT file_path, perceptual_hashes, temporal_fingerprint 
                    FROM fingerprints 
                    WHERE media_type = ? AND file_path != ? AND file_hash != ?
                ''', (fingerprint.media_type, str(fingerprint.file_path), fingerprint.file_hash))

                candidates = cursor.fetchall()

                # Compare with each candidate
                for file_path, stored_phashes, stored_temporal in candidates:
                    if fingerprint.media_type == 'image':
                        # Image comparison
                        if fingerprint.perceptual_hashes and stored_phashes:
                            stored_hashes = json.loads(stored_phashes)
                            is_dup, similarity, details = self.image_dedup.is_duplicate(
                                fingerprint.perceptual_hashes,
                                stored_hashes
                            )

                            if is_dup:
                                duplicates.append((file_path, similarity, details))

                    elif fingerprint.media_type == 'video':
                        # Video comparison
                        if fingerprint.temporal_fingerprint and stored_temporal:
                            stored_fingerprint = json.loads(stored_temporal)
                            is_dup, similarity = self.video_dedup.is_duplicate(
                                fingerprint.temporal_fingerprint,
                                stored_fingerprint
                            )

                            if is_dup:
                                duplicates.append((file_path, similarity, {'temporal_similarity': similarity}))

            # Performance tracking
            query_time = time.time() - start_time
            self.performance_stats['db_query_times'].append(query_time)
            if duplicates:
                self.performance_stats['duplicates_found'] += 1

            logger.info(f"Found {len(duplicates)} duplicates for {fingerprint.file_path.name} in {query_time:.3f}s")
            return duplicates

        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            return []

    def record_duplicate(self, original_file: str, duplicate_file: str, similarity_score: float, details: Dict):
        """Record duplicate relationship in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO duplicates 
                    (original_file, duplicate_file, similarity_score, similarity_details, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    original_file,
                    duplicate_file,
                    similarity_score,
                    json.dumps(details),
                    datetime.now().isoformat()
                ))

                conn.commit()
                logger.debug(f"Recorded duplicate: {duplicate_file} -> {original_file}")

        except Exception as e:
            logger.error(f"Failed to record duplicate: {e}")

    def process_file(self, file_path: Path) -> Dict:
        """
        Process a file for deduplication

        Args:
            file_path: Path to media file

        Returns:
            Dictionary with processing results
        """
        try:
            # Generate fingerprint
            fingerprint = self.generate_fingerprint(file_path)

            # Find duplicates
            duplicates = self.find_duplicates(fingerprint)

            # Store fingerprint
            self.store_fingerprint(fingerprint)

            # Record duplicates
            for dup_path, similarity, details in duplicates:
                self.record_duplicate(str(file_path), dup_path, similarity, details)

            result = {
                'file_path': str(file_path),
                'is_duplicate': len(duplicates) > 0,
                'duplicate_count': len(duplicates),
                'duplicates': duplicates,
                'fingerprint_id': fingerprint.file_hash
            }

            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'is_duplicate': False,
                'error': str(e)
            }

    async def process_file_async(self, file_path: Path) -> Dict:
        """Async wrapper for file processing"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, self.process_file, file_path)
        return result

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.performance_stats.copy()

        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])

        if stats['db_query_times']:
            stats['avg_query_time'] = np.mean(stats['db_query_times'])
            stats['total_query_time'] = sum(stats['db_query_times'])

        return stats

    def cleanup_old_fingerprints(self, max_age_days: int = 30):
        """Clean up old fingerprints from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()

                cursor.execute('''
                    DELETE FROM fingerprints 
                    WHERE created_at < ? AND last_accessed < ?
                ''', (cutoff_date, cutoff_date))

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old fingerprints")

        except Exception as e:
            logger.error(f"Failed to cleanup old fingerprints: {e}")


# Convenience functions
def check_duplicate(file_path: Path) -> Dict:
    """
    Convenience function to check if file is duplicate

    Args:
        file_path: Path to media file

    Returns:
        Duplicate check result
    """
    deduplicator = MediaDeduplicator()
    return deduplicator.process_file(file_path)


def batch_deduplicate(file_paths: List[Path]) -> List[Dict]:
    """
    Process multiple files for deduplication

    Args:
        file_paths: List of file paths

    Returns:
        List of deduplication results
    """
    deduplicator = MediaDeduplicator()
    results = []

    for file_path in file_paths:
        result = deduplicator.process_file(file_path)
        results.append(result)

        # Log progress
        if len(results) % 10 == 0:
            logger.info(f"Processed {len(results)}/{len(file_paths)} files")

    return results


# Export classes and functions
__all__ = [
    'MediaFingerprint', 'ImageDeduplicator', 'VideoDeduplicator',
    'MediaDeduplicator', 'check_duplicate', 'batch_deduplicate'
]

# Test functionality
if __name__ == "__main__":
    print("Testing Media Deduplication System...")

    # Initialize deduplicator
    deduplicator = MediaDeduplicator()

    print("✅ Database initialized successfully")

    # Test with dummy files (if they exist)
    test_files = [
        Path("test_image.jpg"),
        Path("test_video.mp4")
    ]

    for test_file in test_files:
        if test_file.exists():
            try:
                result = deduplicator.process_file(test_file)
                print(f"✅ Processed {test_file}: {result['duplicate_count']} duplicates found")
            except Exception as e:
                print(f"⚠️  Could not process {test_file}: {e}")
        else:
            print(f"⚠️  Test file not found: {test_file}")

    # Print performance stats
    stats = deduplicator.get_performance_stats()
    print(f"✅ Performance stats: {stats}")

    print("Deduplication system test complete!")