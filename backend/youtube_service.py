import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import get_youtube_api_key

YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(Exception):
    """Custom exception for YouTube API specific errors."""
    def __init__(self, message: str, status_code: int = 400, error_code: str = "API_ERROR"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract an 11-character YouTube video ID from various URL formats or raw ID."""
    if not url_or_id:
        return None

    cleaned = url_or_id.strip()

    # If it's already an 11-character alphanumeric/dash/underscore string
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", cleaned):
        return cleaned

    # Regular expressions for different YouTube URL formats
    patterns = [
        # Standard watch URL: youtube.com/watch?v=ID
        r"(?:https?:\/\/)?(?:www\.|m\.|music\.)?youtube\.com\/watch\?(?:.*&)?v=([A-Za-z0-9_-]{11})",
        # Short URL: youtu.be/ID
        r"(?:https?:\/\/)?youtu\.be\/([A-Za-z0-9_-]{11})",
        # Embed URL: youtube.com/embed/ID
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([A-Za-z0-9_-]{11})",
        # Shorts URL: youtube.com/shorts/ID
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([A-Za-z0-9_-]{11})",
        # Live URL: youtube.com/live/ID
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([A-Za-z0-9_-]{11})",
        # Attribution or v path: youtube.com/v/ID
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    # Fallback to query param parsing
    try:
        parsed = urllib.parse.urlparse(cleaned)
        if "youtube" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            if "v" in qs and len(qs["v"]) > 0:
                candidate = qs["v"][0]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    return candidate
    except Exception:
        pass

    return None


def parse_iso8601_duration(duration_str: str) -> Tuple[str, int]:
    """Parse ISO 8601 duration (e.g. PT1H23M45S) into human readable string and total seconds."""
    if not duration_str or not duration_str.startswith("P"):
        return ("0s", 0)

    pattern = re.compile(
        r"P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?"
    )
    match = pattern.match(duration_str)
    if not match:
        return (duration_str, 0)

    parts = match.groupdict()
    days = int(parts["days"] or 0)
    hours = int(parts["hours"] or 0)
    minutes = int(parts["minutes"] or 0)
    seconds = int(parts["seconds"] or 0)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds

    readable_parts = []
    if days > 0:
        readable_parts.append(f"{days}d")
    if hours > 0:
        readable_parts.append(f"{hours}h")
    if minutes > 0:
        readable_parts.append(f"{minutes}m")
    if seconds > 0 or not readable_parts:
        readable_parts.append(f"{seconds}s")

    return (" ".join(readable_parts), total_seconds)


def fetch_video_metadata(video_id: str) -> Dict[str, Any]:
    """Fetch real YouTube video metadata using YouTube Data API v3.
    
    Raises YouTubeAPIError if API key is missing or request fails.
    """
    api_key = get_youtube_api_key()
    if not api_key:
        raise YouTubeAPIError(
            message="Server configuration error: YOUTUBE_API_KEY environment variable is not configured.",
            status_code=503,
            error_code="API_KEY_MISSING",
        )

    endpoint = f"{YOUTUBE_API_BASE_URL}/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": api_key,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
    except requests.RequestException as e:
        raise YouTubeAPIError(
            message=f"Failed to connect to YouTube API: {str(e)}",
            status_code=502,
            error_code="NETWORK_ERROR",
        )

    if response.status_code != 200:
        try:
            error_data = response.json().get("error", {})
            error_message = error_data.get("message", "YouTube API error")
            reasons = [item.get("reason") for item in error_data.get("errors", [])]
        except Exception:
            error_message = response.text
            reasons = []

        if "quotaExceeded" in reasons or response.status_code == 403 and "quota" in error_message.lower():
            raise YouTubeAPIError(
                message="YouTube API quota exceeded for the configured key. Please check Google Cloud Console quotas.",
                status_code=429,
                error_code="QUOTA_EXCEEDED",
            )
        elif "keyInvalid" in reasons or "API key not valid" in error_message:
            raise YouTubeAPIError(
                message="The configured YOUTUBE_API_KEY is invalid. Please verify the credentials.",
                status_code=401,
                error_code="INVALID_API_KEY",
            )
        else:
            raise YouTubeAPIError(
                message=f"YouTube API returned error: {error_message}",
                status_code=response.status_code,
                error_code="UPSTREAM_ERROR",
            )

    data = response.json()
    items = data.get("items", [])
    if not items:
        raise YouTubeAPIError(
            message=f"Video with ID '{video_id}' was not found. It may be private, deleted, or the ID is invalid.",
            status_code=404,
            error_code="VIDEO_NOT_FOUND",
        )

    item = items[0]
    snippet = item.get("snippet", {})
    content_details = item.get("contentDetails", {})
    statistics = item.get("statistics", {})

    iso_duration = content_details.get("duration", "PT0S")
    duration_formatted, duration_seconds = parse_iso8601_duration(iso_duration)

    thumbnails = snippet.get("thumbnails", {})
    best_thumb = (
        thumbnails.get("maxres")
        or thumbnails.get("standard")
        or thumbnails.get("high")
        or thumbnails.get("medium")
        or thumbnails.get("default")
        or {}
    ).get("url")

    return {
        "video_id": video_id,
        "title": snippet.get("title", "Untitled Video"),
        "channel_title": snippet.get("channelTitle", "Unknown Channel"),
        "channel_id": snippet.get("channelId", ""),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
        "duration_iso": iso_duration,
        "duration_formatted": duration_formatted,
        "duration_seconds": duration_seconds,
        "thumbnail_url": best_thumb,
        "view_count": int(statistics.get("viewCount", 0)),
        "like_count": int(statistics.get("likeCount", 0)) if "likeCount" in statistics else None,
        "comment_count": int(statistics.get("commentCount", 0)) if "commentCount" in statistics else None,
    }


def fetch_video_comments(video_id: str, max_results: int = 100) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Fetch publicly available comments for a video using YouTube Data API v3.
    
    Returns:
        (comments_list, comments_disabled_flag, warning_message_if_any)
    """
    api_key = get_youtube_api_key()
    if not api_key:
        return ([], False, "YOUTUBE_API_KEY is not configured on the server.")

    endpoint = f"{YOUTUBE_API_BASE_URL}/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_results, 100),
        "order": "relevance",
        "textFormat": "plainText",
        "key": api_key,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=12)
    except requests.RequestException as e:
        return ([], False, f"Failed to retrieve comments due to network error: {str(e)}")

    if response.status_code != 200:
        try:
            err = response.json().get("error", {})
            errors = err.get("errors", [])
            for item in errors:
                reason = item.get("reason", "")
                if reason == "commentsDisabled":
                    return ([], True, "Comments are disabled by the creator for this video.")
            err_msg = err.get("message", "Unknown error")
            if "commentsDisabled" in err_msg:
                return ([], True, "Comments are disabled by the creator for this video.")
        except Exception:
            err_msg = response.text

        return ([], False, f"Could not retrieve comments: {err_msg}")

    data = response.json()
    items = data.get("items", [])
    comments = []

    for item in items:
        snippet_top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        if snippet_top:
            comments.append({
                "comment_id": item.get("id"),
                "author": snippet_top.get("authorDisplayName", "Anonymous"),
                "author_avatar": snippet_top.get("authorProfileImageUrl", ""),
                "text": snippet_top.get("textDisplay", "").strip(),
                "like_count": int(snippet_top.get("likeCount", 0)),
                "published_at": snippet_top.get("publishedAt", ""),
            })

    return (comments, False, None)
