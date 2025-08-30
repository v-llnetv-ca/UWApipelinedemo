"""
Object Detection Pipeline - Configuration Module
Complete configuration management for all system components
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# Base directories
PROJECT_ROOT = Path(__file__).parent.absolute()
UPLOAD_DIR = PROJECT_ROOT / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(8, os.cpu_count() or 1)

# Model paths and configurations
MODEL_PATHS = {
    "yolov8": MODELS_DIR / "best.pt",  # Fine-tuned model
    "pann": None,  # Pre-trained PANN model uses weights automatically
}

# Object detection configuration
OBJECT_DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.4,
    "max_det": 300,
    "classes": {
        0: "tank",
        1: "helicopter",
        2: "weapon",

    },
    "colors": {
        "tank": (255, 0, 0),  # Red
        "helicopter": (0, 255, 0),  # Green
        "weapon": (0, 0, 255),  # Blue
    }
}

# Audio detection configuration
AUDIO_DETECTION_CONFIG = {
    "sample_rate": 32000,
    "hop_length": 1024,
    "n_fft": 2048,
    "n_mels": 128,
    "win_length": 2048,
    "window": "hann",
    "confidence_threshold": 0.2, #very low threshold for now
    "classes": {
        "gunshot_gunfire": "Gunshot, gunfire",
        "helicopter": "Helicopter",
        "explosion": "Explosion",
    },
    "chunk_duration": 10.0,  # Process audio in 10-second chunks
    "overlap": 2.0,          # 2-second overlap between chunks
}

# Deduplication configuration
DEDUPLICATION_CONFIG = {
    "image_hash_size": 16,
    "video_frame_sample_rate": 1.0,  # Sample every second
    "similarity_thresholds": {
        "phash": 0.85,
        "dhash": 0.85,
        "ahash": 0.85,
        "video": 0.80,
    },
    "watermark_tolerance": 0.1,  # Allow 10% difference for watermarks
    "max_comparison_batches": 1000,  # Limit comparisons for performance
}

# Fusion engine configuration
FUSION_CONFIG = {
    "temporal_window": 2.0,  # ±2 seconds for correlation
    "confidence_weights": {
        "object": 0.6,
        "audio": 0.4,
    },
    "correlation_threshold": 0.7,
    "independent_threshold": 0.8,  # High confidence for independent detections
    "uncertainty_alpha": 0.05,  # 95% confidence intervals
}

# Database configuration
DATABASE_CONFIG = {
    "url": "sqlite:///./database.db",  # Change to PostgreSQL for production
    "echo": False,  # Set to True for SQL debugging
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,
}

# File processing configuration
FILE_CONFIG = {
    "max_file_size": 500 * 1024 * 1024,  # 500MB max file size
    "allowed_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
    "allowed_audio_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "chunk_size": 8192,  # File upload chunk size
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,  # Development only
    "log_level": "info",
    "cors_origins": ["*"],  # Restrict in production
    "max_concurrent_uploads": 5,
    "request_timeout": 300,  # 5 minutes
}

# WebSocket configuration
WEBSOCKET_CONFIG = {
    "ping_interval": 20,
    "ping_timeout": 10,
    "close_timeout": 10,
    "max_connections": 100,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    "rotation": "100 MB",
    "retention": "30 days",
    "log_file": LOGS_DIR / "war_detection.log",
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "batch_size": {
        "object_detection": 4,
        "audio_processing": 2,
    },
    "cache_size": 1000,  # Number of processed items to cache
    "memory_limit": 8 * 1024 * 1024 * 1024,  # 8GB memory limit
    "gpu_memory_fraction": 0.8,  # Use 80% of available GPU memory
}

# Frontend configuration
FRONTEND_CONFIG = {
    "spectrogram": {
        "width": 800,
        "height": 400,
        "color_scale": "viridis",
        "db_range": [-80, 0],  # dB range for visualization
    },
    "video_player": {
        "controls": True,
        "autoplay": False,
        "preload": "metadata",
    },
    "update_interval": 100,  # Milliseconds for real-time updates
}

# Security configuration (for production)
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "rate_limit": "100/minute",
}

# Development vs Production settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    API_CONFIG["reload"] = False
    API_CONFIG["cors_origins"] = ["https://yourdomain.com"]
    DATABASE_CONFIG["url"] = os.getenv("DATABASE_URL", DATABASE_CONFIG["url"])
    LOGGING_CONFIG["level"] = "WARNING"

# Export commonly used configurations
__all__ = [
    "PROJECT_ROOT", "UPLOAD_DIR", "PROCESSED_DIR", "MODELS_DIR", "LOGS_DIR",
    "DEVICE", "NUM_WORKERS", "MODEL_PATHS",
    "OBJECT_DETECTION_CONFIG", "AUDIO_DETECTION_CONFIG",
    "DEDUPLICATION_CONFIG", "FUSION_CONFIG", "DATABASE_CONFIG",
    "FILE_CONFIG", "API_CONFIG", "WEBSOCKET_CONFIG",
    "LOGGING_CONFIG", "PERFORMANCE_CONFIG", "FRONTEND_CONFIG",
    "SECURITY_CONFIG", "ENVIRONMENT"
]


# Validation function
def validate_config():
    """Validate configuration and model paths"""
    errors = []

    # Check model files exist
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            errors.append(f"Model file not found: {model_path}")

    # Check CUDA availability if specified
    if DEVICE == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA specified but not available")

    # Check directory permissions
    for directory in [UPLOAD_DIR, PROCESSED_DIR, LOGS_DIR]:
        if not os.access(directory, os.W_OK):
            errors.append(f"No write permission for: {directory}")

    if errors:
        raise RuntimeError("Configuration errors:\n" + "\n".join(errors))

    return True


# Initialize configuration validation
if __name__ == "__main__":
    try:
        validate_config()
        print("Configuration validation passed!")
        print(f"Device: {DEVICE}")
        print(f"Models directory: {MODELS_DIR}")
        print(f"Upload directory: {UPLOAD_DIR}")
    except RuntimeError as e:
        print(f"Configuration validation failed:\n{e}")