# War Detection Pipeline - Demo Application

A FastAPI-based web application for detecting war-related objects and audio events in crowdsourced media using computer vision (YOLOv8m) and audio analysis neural networks.

## Overview

This application provides a streamlined demo interface for the war detection pipeline, designed to analyze uploaded images and videos for:

- **Object Detection**: War-related objects using YOLOv8m
- **Audio Analysis**: Conflict-related sounds using pretrained PANN (Pretrained Audio Neural Networks)  
- **Multi-modal Fusion**: Combined analysis results with confidence scoring
- **Real-time Processing**: WebSocket-based progress updates and live video frame analysis

**Note**: This is the simplified demo version (`app.py`) without deduplication features. For the complete pipeline with media deduplication, see `media_processor.py`.

## Features

### Core Capabilities
- **Real-time Object Detection**: Live analysis of video frames with bounding box overlays
- **Audio Spectrogram Visualization**: Mel-spectrogram generation and display
- **WebSocket Updates**: Real-time progress tracking and notifications
- **RESTful API**: Comprehensive endpoints for programmatic access
- **Background Processing**: Asynchronous file processing with job tracking

### Processing Modes
1. **Fast Processing** (`/process_fast/{filename}`): Optimized for demo speed, skips deduplication
2. **Standard Processing** (`/process/{filename}`): Full pipeline analysis
3. **Audio-Only Analysis** (`/process_audio_only/{filename}`): Spectrogram generation for interface demos

## Installation

### Prerequisites
- Python 3.12+
- FFmpeg (for video processing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/v-llnetv-ca/UWApipelinedemo.git
cd UWApipelinedemo
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create required directories**
```bash
mkdir -p uploads processed static templates
```

**Note**: The deduplication system will automatically create a `deduplication.db` SQLite database file to store media fingerprints and similarity data. This file persists duplicate detection results across application restarts.

## Usage

### Starting the Application

```bash
python app.py
```

### API Endpoints

#### Core Processing
```bash
# Upload file
POST /upload

# Start fast processing (recommended for demo)
POST /process_fast/{filename}

# Start standard processing
POST /process/{filename}

# Audio-only analysis
POST /process_audio_only/{filename}
```

#### Real-time Detection
```bash
# Analyze single video frame
POST /detect/frame
```

#### Job Management
```bash
# Get job status
GET /jobs/{job_id}/status

# Get job results
GET /jobs/{job_id}/results

# Get spectrogram data
GET /jobs/{job_id}/spectrogram

# List all jobs
GET /jobs

# Delete job
DELETE /jobs/{job_id}
```

#### System Information
```bash
# Health check
GET /health

# Performance statistics
GET /stats
```

#### Media Serving
```bash
# Serve uploaded video files
GET /video/{filename}

# Serve uploaded image files  
GET /image/{filename}
```

#### WebSocket
```bash
# WebSocket endpoint for real-time updates
WS /ws
```

## Detection Classes

### Object Detection
- **tank**: Tank vehicles
- **helicopter**: Helicopter aircraft  
- **weapon**: Weapons

### Audio Detection
- **gunshot_gunfire**: Gunshot, gunfire sounds
- **helicopter**: Helicopter sounds
- **explosion**: Explosion sounds

## Architecture

### Components

1. **FastAPI Application** (`app.py`)
   - Web server and API endpoints
   - File upload and validation
   - WebSocket management
   - Background job orchestration

2. **Media Processor** (`media_processor.py`)
   - Core pipeline coordination
   - Object detection integration
   - Audio analysis pipeline
   - Multi-modal fusion
   - Progress tracking and callbacks

3. **Detection Modules**
   - **Object Detector**: YOLOv8m-based visual analysis
   - **Audio Detector**: PANN-based audio event detection
   - **Fusion Engine**: Combined confidence scoring

### Data Flow

```
Upload → Validation → Processing Queue → Analysis Pipeline → Results Storage → API Response
                                      ↓
                               WebSocket Updates
```

## Database Management

### Deduplication Database

The application uses a SQLite database (`deduplication.db`) to store media fingerprints and similarity data for duplicate detection:

- **Location**: Created automatically in the application root directory
- **Purpose**: Stores perceptual hashes and metadata for uploaded media
- **Persistence**: Data persists across application restarts
- **Growth**: Database size increases with each unique media file processed

### Database Maintenance

```bash
# Check database size
ls -lh deduplication.db

# Reset deduplication data (caution: removes all fingerprint history)
rm deduplication.db

# Backup deduplication database
cp deduplication.db deduplication_backup_$(date +%Y%m%d).db
```

**Note**: In the demo version (`app.py`), fast processing mode skips deduplication checks, but the database may still be created and maintained by the underlying `MediaProcessor` components.

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Limitations

This demo version has several limitations compared to the full pipeline:

- **No Deduplication**: Fast mode skips media similarity analysis
- **In-Memory Storage**: Job results stored in application memory
- **Single Instance**: Not designed for horizontal scaling

For production use, consider the full `media_processor.py` pipeline with deduplication capabilities.


