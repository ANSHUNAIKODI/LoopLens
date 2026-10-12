import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import is_api_key_configured, get_youtube_api_key
from backend.youtube_service import (
    YouTubeAPIError,
    extract_video_id,
    fetch_video_metadata,
    fetch_video_comments,
)
from backend.analyzer import analyze_comments
from backend.video_frame_analyzer import (
    analyze_video_file,
    generate_permitted_sample_video,
    generate_static_sample_video,
)

app = FastAPI(
    title="LoopLens API",
    description="Backend service for detecting YouTube video repetition and community flags",
    version="1.0.0",
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit for prototype
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or 11-character video ID")


@app.get("/api/status")
def get_system_status():
    """Returns server status, YouTube API configuration, and video frame analysis capabilities."""
    configured = is_api_key_configured()
    return {
        "status": "online",
        "api_key_configured": configured,
        "video_frame_analysis_supported": True,
        "supported_video_formats": sorted(list(ALLOWED_VIDEO_EXTENSIONS)),
        "max_video_size_mb": MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        "instructions": (
            "To analyze live YouTube videos via comments, configure YOUTUBE_API_KEY in the server's .env file. "
            "Local video frame repetition analysis is ready without an API key."
            if not configured
            else "Server is ready for both live YouTube comment analysis and local video frame repetition analysis."
        ),
    }


@app.post("/api/analyze")
def analyze_video_endpoint(payload: AnalyzeRequest):
    """Analyze a YouTube video by fetching real metadata and community comments."""
    raw_url = payload.url.strip() if payload.url else ""
    if not raw_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "EMPTY_URL",
                "message": "Please enter a YouTube video URL or ID to analyze.",
            },
        )

    video_id = extract_video_id(raw_url)
    if not video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_URL",
                "message": (
                    "Could not extract a valid YouTube video ID. "
                    "Please provide a standard YouTube URL (e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)."
                ),
            },
        )

    # Check if API key is configured on server
    if not is_api_key_configured():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error_code": "API_KEY_REQUIRED",
                "video_id": video_id,
                "message": (
                    "YouTube API key is not configured on the server. "
                    "To retrieve real video details and community comments, "
                    "please add your YOUTUBE_API_KEY to the .env file in the project root."
                ),
            },
        )

    # 1. Fetch real video metadata
    try:
        video_metadata = fetch_video_metadata(video_id)
    except YouTubeAPIError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "success": False,
                "error_code": e.error_code,
                "video_id": video_id,
                "message": e.message,
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "UNEXPECTED_ERROR",
                "video_id": video_id,
                "message": f"An unexpected error occurred while fetching video data: {str(e)}",
            },
        )

    # 2. Fetch real public comments
    comments, comments_disabled, comments_warning = fetch_video_comments(video_id)

    # 3. Analyze comments and metadata through the multi-source evidence pipeline
    analysis_result = analyze_comments(
        comments=comments,
        comments_disabled=comments_disabled,
        warning_note=comments_warning,
        video_metadata=video_metadata,
        video_url=raw_url,
        video_id=video_id,
    )

    return {
        "success": True,
        "video": video_metadata,
        "analysis": analysis_result,
    }


@app.post("/api/analyze-video")
async def analyze_video_upload_endpoint(
    file: UploadFile = File(..., description="User video file to analyze for repeated footage"),
    sample_interval_sec: float = Form(1.0, description="Sampling interval in seconds (0.2 to 5.0)"),
    min_sequence_sec: float = Form(3.0, description="Minimum sequence duration to flag as repeated (2.0 to 10.0)"),
    permission_confirmed: bool = Form(False, description="User confirmation that they have permission to analyze the video"),
):
    """Analyze a user-provided video file for visual repetition using computer-vision perceptual hashing.
    
    Validates file format, enforces size limits, and requires explicit user permission.
    Does NOT download or stream from YouTube.
    """
    if not permission_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "PERMISSION_REQUIRED",
                "message": (
                    "Permission confirmation is required. You must verify that you own or "
                    "have explicit permission to analyze this video file."
                ),
            },
        )

    filename = file.filename or "uploaded_video.mp4"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_FILE_TYPE",
                "message": (
                    f"Unsupported video format '{file_ext}'. "
                    f"Supported formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}."
                ),
            },
        )

    # Save uploaded file in chunks while enforcing size limit
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                temp_file.close()
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error_code": "FILE_TOO_LARGE",
                        "message": (
                            f"File size exceeds prototype limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB. "
                            "Please upload a shorter clip or trim the file."
                        ),
                    },
                )
            temp_file.write(chunk)
        temp_file.close()

        # Clamp sampling intervals to safe boundaries
        clamped_sample_interval = max(0.2, min(5.0, sample_interval_sec))
        clamped_min_sequence = max(2.0, min(10.0, min_sequence_sec))

        result = analyze_video_file(
            video_path=temp_path,
            original_filename=filename,
            sample_interval_sec=clamped_sample_interval,
            min_sequence_sec=clamped_min_sequence,
        )
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "VIDEO_PROCESSING_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "UNEXPECTED_ANALYSIS_ERROR",
                "message": f"An unexpected error occurred during frame repetition analysis: {str(e)}",
            },
        )
    finally:
        # Guarantee cleanup of temporary uploaded file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/api/analyze-sample-video")
def analyze_sample_video_endpoint(
    sample_type: str = Form("repeated", description="Type of sample video: 'repeated' or 'static'"),
    sample_interval_sec: float = Form(1.0, description="Sampling interval in seconds"),
    min_sequence_sec: float = Form(2.5, description="Minimum sequence duration to flag"),
):
    """Run repetition analysis on a permitted synthetic demonstration video.
    
    Generates a short, non-infringing test video (12s with repeated sequence or 8s static slide)
    to allow immediate testing without requiring users to upload private files.
    """
    temp_dir = tempfile.mkdtemp(prefix="looplens_sample_")
    try:
        if sample_type == "static":
            sample_path = Path(temp_dir) / "sample_static_slide.mp4"
            generate_static_sample_video(sample_path)
            orig_name = "demo_static_presentation.mp4"
        else:
            sample_path = Path(temp_dir) / "sample_synthetic_loop.mp4"
            generate_permitted_sample_video(sample_path)
            orig_name = "demo_permitted_repeated_footage.mp4"

        result = analyze_video_file(
            video_path=sample_path,
            original_filename=orig_name,
            sample_interval_sec=sample_interval_sec,
            min_sequence_sec=min_sequence_sec,
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Mount frontend static files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

