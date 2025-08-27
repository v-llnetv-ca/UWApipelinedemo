"""
War Detection Pipeline - FastAPI Application
Main web application for the war detection demo interface
"""
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import librosa

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    API_CONFIG, WEBSOCKET_CONFIG, FILE_CONFIG, UPLOAD_DIR,
    PROCESSED_DIR, PROJECT_ROOT, FRONTEND_CONFIG
)
from utils import FileManager, logger
from media_processor import MediaProcessor, ProcessingStatus

# Initialize FastAPI app
app = FastAPI(
    title="War Detection Pipeline API",
    description="Professional API for multi-modal war content detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
media_processor = MediaProcessor()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_audio_duration(video_file_path):
    """Get duration of audio extracted from video file"""
    try:
        logger.info(f"Video file path: {video_file_path}")

        # Get the project root directory (parent of uploads)
        project_root = video_file_path.parent.parent
        processed_dir = project_root / "processed"

        video_stem = video_file_path.stem
        audio_filename = f"{video_stem}_audio.wav"
        audio_file_path = processed_dir / audio_filename

        logger.info(f"Looking for audio file: {audio_file_path}")
        logger.info(f"Full constructed path: {audio_file_path.absolute()}")  # ADD THIS LINE

        if not audio_file_path.exists():
            logger.error(f"Audio file does not exist: {audio_file_path}")
            return 0

        duration = librosa.get_duration(filename=str(audio_file_path))
        logger.info(f"Calculated duration: {duration}")
        return duration

    except Exception as e:
        logger.error(f"Could not get audio duration: {e}")
        return 0

# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.job_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from job subscriptions
        for job_id, subscribers in self.job_subscribers.items():
            if websocket in subscribers:
                subscribers.remove(websocket)

        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_job_update(self, job_id: str, update: dict):
        """Send update to clients subscribed to specific job"""
        logger.info(f"Trying to send update for job: {job_id}")
        logger.info(f"Subscribers count: {len(self.job_subscribers.get(job_id, []))}")

        if job_id in self.job_subscribers:
            disconnected = []
            for websocket in self.job_subscribers[job_id]:
                try:
                    await websocket.send_text(json.dumps({
                        'type': 'job_update',
                        'job_id': job_id,
                        'data': update
                    }))
                except Exception:
                    disconnected.append(websocket)

            # Clean up disconnected clients
            for websocket in disconnected:
                self.job_subscribers[job_id].remove(websocket)

    def subscribe_to_job(self, job_id: str, websocket: WebSocket):
        """Subscribe websocket to job updates"""
        if job_id not in self.job_subscribers:
            self.job_subscribers[job_id] = []

        if websocket not in self.job_subscribers[job_id]:
            self.job_subscribers[job_id].append(websocket)
            logger.info(f"WebSocket subscribed to job {job_id}. Subscribers: {len(self.job_subscribers[job_id])}")


manager = ConnectionManager()


# Add progress callback to media processor
def websocket_progress_callback(job_status: dict):
    """Callback to send progress updates via WebSocket"""
    asyncio.create_task(manager.send_job_update(job_status['job_id'], job_status))


media_processor.add_progress_callback(websocket_progress_callback)


# Routes

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main demo interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": {
            "max_file_size": FILE_CONFIG["max_file_size"],
            "allowed_formats": (
                    FILE_CONFIG["allowed_video_formats"] +
                    FILE_CONFIG["allowed_image_formats"]
            ),
            "frontend_config": FRONTEND_CONFIG
        }
    })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "object_detector": media_processor.object_detector.is_loaded,
            "audio_detector": media_processor.audio_detector.is_loaded,
            "deduplicator": "ready",
            "fusion_engine": "ready"
        }
    }


@app.post("/detect/frame")
async def detect_frame_realtime(file: UploadFile = File(...)):
    """
    Real-time object detection on single video frame

    Args:
        file: Image frame from video

    Returns:
        Detection results with bounding boxes
    """
    try:
        # Read frame data
        frame_data = await file.read()

        # Convert to OpenCV format
        import numpy as np
        import cv2

        # Decode image
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image frame")

        # Run object detection only (fast, real-time)
        detections = media_processor.object_detector.detect_objects_in_image(frame, timestamp=0.0)

        # Format for frontend overlay
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                "class": detection["class"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],  # [x1, y1, x2, y2]
                "center": detection["center"],
                "color": media_processor.object_detector.config["colors"].get(
                    detection["class"], [255, 255, 255]
                )
            })

        return {
            "status": "success",
            "detections": formatted_detections,
            "frame_info": {
                "width": frame.shape[1],
                "height": frame.shape[0]
            },
            "processing_time": time.time()  # For performance monitoring
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Real-time detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and validate media file

    Args:
        file: Uploaded media file

    Returns:
        Upload confirmation with file info
    """
    try:
        # Validate file size
        if file.size and file.size > FILE_CONFIG["max_file_size"]:
            raise HTTPException(
                status_code=413,
                detail=f"File size {file.size} exceeds maximum {FILE_CONFIG['max_file_size']} bytes"
            )

        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = (
                FILE_CONFIG["allowed_video_formats"] +
                FILE_CONFIG["allowed_image_formats"]
        )

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Supported: {allowed_extensions}"
            )

        # Save uploaded file
        file_content = await file.read()
        unique_filename = FileManager.generate_unique_filename(file.filename)
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        # Additional validation
        validation_result = FileManager.validate_file(file_path)
        if not validation_result["valid"]:
            # Clean up invalid file
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file: {validation_result['errors']}"
            )

        logger.info(f"File uploaded successfully: {unique_filename}")

        return {
            "status": "uploaded",
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": str(file_path),
            "file_size": file.size,
            "file_type": "video" if file_extension in FILE_CONFIG["allowed_video_formats"] else "image",
            "upload_time": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/process_audio_only/{filename}")
async def process_audio_only(filename: str, background_tasks: BackgroundTasks):
    """
    Audio-only processing for spectrogram generation - interface only, not pipeline

    Args:
        filename: Name of uploaded video file
        background_tasks: FastAPI background tasks

    Returns:
        Job initiation confirmation
    """
    try:
        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Only process videos
        if file_path.suffix.lower() not in FILE_CONFIG['allowed_video_formats']:
            raise HTTPException(status_code=400, detail="Audio analysis only supports video files")

        # Generate job ID for tracking
        job_id = f"audio_{int(time.time() * 1000)}"

        async def audio_analysis_job():
            try:
                logger.info(f"Starting audio-only analysis for {filename}")

                # Use audio detector directly (not pipeline)
                audio_detections, mel_spectrogram = media_processor.audio_detector.detect_audio_in_video(file_path)

                # Save results for frontend
                results = {
                    'job_id': job_id,
                    'audio_detections': audio_detections,
                    'spectrogram': {
                        'data': mel_spectrogram.tolist() if mel_spectrogram is not None else [],
                        'shape': list(mel_spectrogram.shape) if mel_spectrogram is not None else [],
                        'duration': mel_spectrogram.shape[1] * 0.032 if mel_spectrogram is not None else 0
                    },
                    'status': 'completed'
                }

                # Store in completed jobs with proper structure
                class AudioJobStatus:
                    def __init__(self, job_id, results):
                        self.job_id = job_id
                        self.results = results
                        self.status = 'completed'

                    def to_dict(self):
                        return {
                            'job_id': self.job_id,
                            'status': self.status,
                            'results': self.results
                        }

                media_processor.completed_jobs[job_id] = AudioJobStatus(job_id, results)

                logger.info(f"Audio analysis complete: {len(audio_detections)} detections")

            # 100ms delay will ensure the subscription completes before the completion message is sent
                await asyncio.sleep(0.1)

                # Notify via websocket - fixed notification
                await manager.send_job_update(job_id, {
                    'job_id': job_id,
                    'status': 'completed',
                    'progress': 1.0,
                    'current_step': 'Audio analysis complete',
                    'steps_completed': 1,
                    'results': results
                })

            except Exception as e:
                logger.error(f"Audio analysis failed: {e}")

        background_tasks.add_task(audio_analysis_job)

        return {
            "status": "audio_processing_started",
            "job_id": job_id,
            "filename": filename,
            "message": "Audio analysis started (interface demo mode)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_fast/{filename}")
async def process_file_fast(filename: str, background_tasks: BackgroundTasks):
    """
    Fast processing for demo - skips deduplication for speed

    Args:
        filename: Name of uploaded file
        background_tasks: FastAPI background tasks

    Returns:
        Job initiation confirmation
    """
    try:
        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Start fast processing in background (skip deduplication)
        job_id = media_processor.generate_job_id()

        async def process_job_fast():
            try:
                await media_processor.process_file_fast(file_path)
            except Exception as e:
                logger.error(f"Fast background processing failed: {e}")

        background_tasks.add_task(process_job_fast)

        return {
            "status": "processing_started",
            "job_id": job_id,
            "filename": filename,
            "message": "Fast processing started (skipping deduplication)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fast process initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/{filename}")
async def process_file(filename: str, background_tasks: BackgroundTasks):
    """
    Start processing uploaded file

    Args:
        filename: Name of uploaded file
        background_tasks: FastAPI background tasks

    Returns:
        Job initiation confirmation
    """
    try:
        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Start processing in background
        job_id = media_processor.generate_job_id()

        async def process_job():
            try:
                await media_processor.process_file(file_path)
            except Exception as e:
                logger.error(f"Background processing failed: {e}")

        background_tasks.add_task(process_job)

        return {
            "status": "processing_started",
            "job_id": job_id,
            "filename": filename,
            "message": "Processing started in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get processing job status

    Args:
        job_id: Job identifier

    Returns:
        Job status information
    """
    status = media_processor.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return status


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get processing job results

    Args:
        job_id: Job identifier

    Returns:
        Complete job results
    """
    status = media_processor.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {status['status']}"
        )

    return status["results"]


@app.get("/jobs/{job_id}/spectrogram")
async def get_spectrogram_data(job_id: str):
    """
    Get mel spectrogram data for job

    Args:
        job_id: Job identifier

    Returns:
        Spectrogram data for visualization
    """
    status = media_processor.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    results = status.get("results", {})
    spectrogram = results.get("spectrogram")

    if not spectrogram:
        raise HTTPException(status_code=404, detail="No spectrogram data available")

    # Prepare spectrogram for D3.js visualization
    from utils import AudioProcessor

    spectrogram_data = {
        "data": spectrogram["data"],
        "shape": spectrogram["shape"],
        "time_axis": list(range(spectrogram["shape"][1])),  # Frame indices
        "frequency_axis": list(range(spectrogram["shape"][0])),  # Mel bin indices
        "db_range": [-80, 0],  # Typical dB range
        "config": {
            "width": FRONTEND_CONFIG["spectrogram"]["width"],
            "height": FRONTEND_CONFIG["spectrogram"]["height"],
            "color_scale": FRONTEND_CONFIG["spectrogram"]["color_scale"]
        }
    }

    return spectrogram_data


@app.get("/jobs")
async def get_all_jobs():
    """Get status of all jobs"""
    return media_processor.get_all_jobs()


@app.get("/stats")
async def get_system_stats():
    """Get system performance statistics"""
    return {
        "system_stats": media_processor.get_performance_stats(),
        "active_jobs": len(media_processor.active_jobs),
        "completed_jobs": len(media_processor.completed_jobs),
        "websocket_connections": len(manager.active_connections)
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and clean up associated files"""
    status = media_processor.get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Remove from job tracking
    if job_id in media_processor.active_jobs:
        del media_processor.active_jobs[job_id]
    if job_id in media_processor.completed_jobs:
        del media_processor.completed_jobs[job_id]

    # Clean up files
    try:
        results_path = PROCESSED_DIR / f"{job_id}_results.json"
        if results_path.exists():
            results_path.unlink()
    except Exception as e:
        logger.warning(f"Could not clean up results file: {e}")

    return {"status": "deleted", "job_id": job_id}


@app.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve uploaded video files"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        logger.error(f"Video file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Video not found")

    # Get file extension to determine media type
    extension = file_path.suffix.lower()
    media_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv'
    }

    media_type = media_type_map.get(extension, 'video/mp4')

    logger.info(f"Serving video: {filename} ({media_type})")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={filename}"
        }
    )


@app.get("/image/{filename}")
async def serve_image(filename: str):
    """Serve uploaded image files"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "subscribe_job":
                job_id = message.get("job_id")
                logger.info(f"Received subscription request for job: {job_id}")
                if job_id:
                    manager.subscribe_to_job(job_id, websocket)
                    logger.info(f"Subscription processed for job: {job_id}")
                    await manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "job_id": job_id
                    }, websocket)

            elif message.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    logger.info("Starting War Detection Pipeline API")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Processed directory: {PROCESSED_DIR}")

    # Ensure directories exist
    UPLOAD_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Load models
    logger.info("Loading detection models...")
    media_processor.object_detector.load_model()
    media_processor.audio_detector.load_model()

    logger.info("✅ API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down War Detection Pipeline API")

    # Cancel any running jobs
    for job_id in list(media_processor.active_jobs.keys()):
        try:
            # Note: In production, you'd want to save job state for recovery
            del media_processor.active_jobs[job_id]
        except:
            pass

    logger.info("✅ API shutdown complete")


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        log_level=API_CONFIG["log_level"]
    )