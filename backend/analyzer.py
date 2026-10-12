import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2

from backend.gemini_service import (
    analyze_youtube_video_with_gemini,
    RESULT_REPEATED_FOOTAGE_DETECTED,
    RESULT_NO_QUALIFYING_REPETITION,
    RESULT_INCONCLUSIVE,
    RESULT_ANALYSIS_UNAVAILABLE,
)
from backend.youtube_service import extract_video_id

ROOT_DIR = Path(__file__).resolve().parent.parent
AUTHORIZED_FOOTAGE_DIRS = [
    ROOT_DIR / "backend" / "authorized_footage",
    ROOT_DIR / "authorized_footage",
]


# Backward-compatibility aliases for local CV / older tests
RESULT_REPEATED_SEQUENCE_DETECTED = RESULT_REPEATED_FOOTAGE_DETECTED
RESULT_NO_REPEATED_SEQUENCE = RESULT_NO_QUALIFYING_REPETITION
RESULT_FOOTAGE_UNAVAILABLE = RESULT_ANALYSIS_UNAVAILABLE


# Regular expression patterns for community signals
PATTERNS = {
    "repeated_footage": [
        r"\b(?:repeats?|repeating|repeated)\b",
        r"\b(?:loop(?:s|ed|ing)?)\b",
        r"\b(?:same\s+(?:footage|scene|clip|video|part|thing)\s+(?:again|twice|repeated))\b",
        r"\b(?:plays?\s+(?:twice|again|in\s+a\s+loop))\b",
        r"\b(?:starts?\s+over\s+at)\b",
        r"\b(?:replays?\s+at)\b",
        r"\b(?:duplicate\s+(?:footage|content|scenes?|parts?))\b",
        r"\b(?:second\s+half\s+is\s+the\s+same)\b",
        r"\b(?:re-?uploaded?\s+the\s+same)\b",
        r"\b(?:copy\s+and\s+paste[d]?\s+footage)\b",
    ],
    "incomplete_video": [
        r"\b(?:cuts?\s+off(?:\s+at|\s+abruptly|\s+early)?)\b",
        r"\b(?:cut\s+off(?:\s+at|\s+abruptly|\s+early)?)\b",
        r"\b(?:ends?\s+(?:abruptly|too\s+soon|in\s+the\s+middle|prematurely))\b",
        r"\b(?:incomplete\s+(?:video|version|ending))\b",
        r"\b(?:where(?:'s|\s+is)\s+the\s+rest)\b",
        r"\b(?:where(?:'s|\s+is)\s+part\s+2)\b",
        r"\b(?:missing\s+(?:the\s+ending|part\s+of\s+it|the\s+rest|content))\b",
        r"\b(?:not\s+the\s+full\s+(?:video|episode|movie|version))\b",
        r"\b(?:video\s+is\s+cut)\b",
        r"\b(?:cliffhanger\s+without\s+ending)\b",
    ],
    "filler_or_padding": [
        r"\b(?:filler\s+(?:footage|time|to\s+reach))\b",
        r"\b(?:stretched\s+to\s+10\s+min)\b",
        r"\b(?:padding\s+(?:out\s+the\s+video|for\s+ads?))\b",
        r"\b(?:black\s+screen\s+(?:for|at\s+the\s+end))\b",
        r"\b(?:blank\s+screen\s+at\s+the\s+end)\b",
        r"\b(?:silent\s+for\s+the\s+rest)\b",
    ],
    "intentional_recap": [
        r"\b(?:recap(?:s|ped|ping)?)\b",
        r"\b(?:flashback(?:s)?)\b",
        r"\b(?:montage(?:s)?)\b",
        r"\b(?:compilation(?:s)?)\b",
        r"\b(?:previously\s+on)\b",
        r"\b(?:summary\s+of\s+(?:previous|season|part))\b",
        r"\b(?:highlights?\s+from)\b",
        r"\b(?:intro(?:duction)?\s+clip)\b",
    ],
    "ambient_loop": [
        r"\b(?:1\s*hour\s+loop|10\s*hours?\s+loop|looped\s+for\s+hours?|1hr\s+loop)\b",
        r"\b(?:ambient\s+(?:sound|music|loop))\b",
        r"\b(?:lofi\s+(?:loop|music|beats))\b",
        r"\b(?:asmr\s+loop)\b",
    ],
}

TIMESTAMP_REGEX = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")
CHAPTER_REGEX = re.compile(r"(?:^|\n)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–:]?\s*([^\n\r]{2,80})")


def extract_timestamps(text: str) -> List[str]:
    """Find timestamp mentions like 02:45 or 1:15:30 in comment or description text."""
    return sorted(list(set(TIMESTAMP_REGEX.findall(text))))


def extract_chapters_from_description(description: str) -> List[Dict[str, str]]:
    """Extract creator-defined chapter markers or timestamp labels from description text."""
    if not description:
        return []
    chapters: List[Dict[str, str]] = []
    seen = set()
    for match in CHAPTER_REGEX.finditer(description):
        ts = match.group(1).strip()
        label = match.group(2).strip()
        if ts not in seen and len(label) >= 2:
            seen.add(ts)
            chapters.append({"timestamp": ts, "label": label})
    return chapters


def analyze_metadata(video_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze video metadata (Source A) to detect intentional recaps, ambient loops, and chapters."""
    if not video_metadata:
        return {
            "source_name": "Video Metadata",
            "evaluated": False,
            "title": "",
            "is_explicit_loop": False,
            "is_explicit_recap": False,
            "chapters": [],
            "summary": "Metadata not provided.",
        }

    title = video_metadata.get("title", "")
    description = video_metadata.get("description", "")
    duration_sec = video_metadata.get("duration_seconds", 0)

    # Check for title/description keywords indicating intentional loops or recaps
    is_explicit_loop = bool(re.search(
        r"\b(?:1\s*hour\s+loop|10\s*hours?\s+loop|1hr\s+loop|ambient\s+loop|lofi\s+loop|sleep\s+loop|music\s+loop)\b",
        title + " " + description,
        re.IGNORECASE,
    ))

    is_explicit_recap = bool(re.search(
        r"\b(?:recap|season\s+\d+\s+recap|previously\s+on|movie\s+recap|series\s+recap|story\s+recap|compilation)\b",
        title,
        re.IGNORECASE,
    ))

    chapters = extract_chapters_from_description(description)

    summary_notes = []
    if is_explicit_loop:
        summary_notes.append("Creator labeled this content as an ambient or extended loop.")
    if is_explicit_recap:
        summary_notes.append("Title indicates this video is an intentional recap or compilation.")
    if chapters:
        summary_notes.append(f"Found {len(chapters)} creator-defined chapter marker(s) in description.")

    return {
        "source_name": "Video Metadata",
        "evaluated": True,
        "title": title,
        "channel_title": video_metadata.get("channel_title", ""),
        "duration_formatted": video_metadata.get("duration_formatted", ""),
        "duration_seconds": duration_sec,
        "view_count": video_metadata.get("view_count", 0),
        "comment_count": video_metadata.get("comment_count"),
        "is_explicit_loop": is_explicit_loop,
        "is_explicit_recap": is_explicit_recap,
        "chapters": chapters,
        "summary": " ".join(summary_notes) if summary_notes else "Standard metadata with no explicit loop or recap tags.",
    }


def analyze_single_comment(comment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Analyze a single comment text against patterns.
    
    Returns structured evidence if matching, else None.
    Never invents or modifies the real comment text.
    """
    raw_text = comment.get("text", "")
    if not raw_text:
        return None

    matched_categories: List[str] = []
    matched_phrases: List[str] = []

    for category, regex_list in PATTERNS.items():
        for pattern in regex_list:
            if re.search(pattern, raw_text, re.IGNORECASE):
                if category not in matched_categories:
                    matched_categories.append(category)
                matched_phrases.append(pattern)

    if not matched_categories:
        return None

    timestamps = extract_timestamps(raw_text)

    return {
        "comment_id": comment.get("comment_id"),
        "author": comment.get("author", "Anonymous"),
        "author_avatar": comment.get("author_avatar", ""),
        "text": raw_text,
        "like_count": comment.get("like_count", 0),
        "published_at": comment.get("published_at", ""),
        "categories": matched_categories,
        "matched_timestamps": timestamps,
    }


def attempt_direct_footage_analysis(
    video_url: Optional[str] = None,
    video_id: Optional[str] = None,
    local_video_path: Optional[str] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attempt direct visual analysis of video footage through supported methods.
    
    Order of execution:
    1. Authorized local video source: If local_video_path is supplied or an authorized
       local video file exists in backend/authorized_footage/{video_id}.mp4, runs OpenCV CV engine.
    2. Direct YouTube URL analysis via Gemini multimodal models: For public YouTube URLs or IDs,
       invokes Google's Gemini multimodal video understanding endpoint using public YouTube fileData.
    3. Direct container stream probe: checks if non-YouTube URL points to a direct unencrypted stream.
    """
    target_path: Optional[Path] = None

    # 1. Check local_video_path if explicitly supplied
    if local_video_path:
        p = Path(local_video_path)
        if p.exists() and p.is_file():
            target_path = p

    # 2. Check authorized footage repositories for video_id or matching file
    resolved_id = video_id or (extract_video_id(video_url) if video_url else None)
    if not target_path and resolved_id:
        for adir in AUTHORIZED_FOOTAGE_DIRS:
            if not adir.exists():
                continue
            for ext in [".mp4", ".webm", ".mkv", ".mov"]:
                candidate = adir / f"{resolved_id}{ext}"
                if candidate.exists() and candidate.is_file():
                    target_path = candidate
                    break
            if target_path:
                break

    # If an authorized local file source is found, run CV analysis
    if target_path:
        from backend.video_frame_analyzer import analyze_video_file
        local_result = analyze_video_file(target_path, target_path.name)
        verdict = local_result.get("verdict", {})
        return {
            "source_name": "Direct Visual Sequence Analysis",
            "attempted": True,
            "accessible": True,
            "performed": True,
            "result_state": verdict.get("result_state", RESULT_REPEATED_FOOTAGE_DETECTED),
            "status": verdict.get("status", "REPEATED_SEQUENCES_DETECTED"),
            "sub_classification": verdict.get("sub_classification", ""),
            "local_result": local_result,
            "calculated_matches": local_result.get("matches", []),
            "summary": verdict.get("summary", "Completed computer-vision frame perceptual analysis on authorized footage source."),
        }

    # 3. Direct YouTube URL analysis via Gemini Multimodal Models
    if resolved_id or (video_url and extract_video_id(video_url)):
        gemini_target = video_url or f"https://www.youtube.com/watch?v={resolved_id}"
        gemini_result = analyze_youtube_video_with_gemini(
            video_url_or_id=gemini_target,
            video_metadata=video_metadata,
        )

        if gemini_result.get("performed"):
            # Successful footage analysis from Gemini!
            return {
                "source_name": "Gemini Multimodal Direct Analysis",
                "attempted": True,
                "accessible": True,
                "performed": True,
                "result_state": gemini_result.get("result_state", RESULT_INCONCLUSIVE),
                "status": gemini_result.get("status", "ANALYSIS_COMPLETE"),
                "sub_classification": gemini_result.get("sub_classification", ""),
                "confidence": gemini_result.get("confidence", "Moderate"),
                "model_used": gemini_result.get("model_used"),
                "summary": gemini_result.get("summary", "Gemini video sequence analysis completed."),
                "technical_reason": None,
                "calculated_matches": gemini_result.get("calculated_matches", []),
                "local_result": gemini_result.get("visual_analysis"),
                "gemini_result": gemini_result,
            }
        else:
            # Gemini analysis was attempted, but failed / unavailable
            return {
                "source_name": "Gemini Multimodal Direct Analysis",
                "attempted": True,
                "accessible": False,
                "performed": False,
                "result_state": RESULT_ANALYSIS_UNAVAILABLE,
                "status": gemini_result.get("status", "ANALYSIS_UNAVAILABLE"),
                "error_category": gemini_result.get("error_category", "Footage Analysis Unavailable"),
                "technical_reason": gemini_result.get("technical_reason"),
                "model_used": gemini_result.get("model_used"),
                "summary": gemini_result.get("summary"),
                "calculated_matches": [],
                "loop_absence_warning": (
                    "Notice: The absence of viewer comments cannot be used as evidence that a video has no loops. "
                    f"Direct footage analysis is unavailable: {gemini_result.get('error_category', 'Analysis unavailable')}."
                ),
            }

    # 4. Direct container stream probe on non-YouTube URLs
    is_opencv_streamable = False
    if video_url:
        is_youtube_domain = any(dom in video_url.lower() for dom in ["youtube.com", "youtu.be"])
        if not is_youtube_domain:
            try:
                cap = cv2.VideoCapture(video_url)
                is_opencv_streamable = cap.isOpened()
                cap.release()
                if is_opencv_streamable:
                    from backend.video_frame_analyzer import analyze_video_file
                    local_result = analyze_video_file(Path(video_url), Path(video_url).name)
                    verdict = local_result.get("verdict", {})
                    return {
                        "source_name": "Direct Visual Sequence Analysis",
                        "attempted": True,
                        "accessible": True,
                        "performed": True,
                        "result_state": verdict.get("result_state", RESULT_REPEATED_FOOTAGE_DETECTED),
                        "status": verdict.get("status", "REPEATED_SEQUENCES_DETECTED"),
                        "sub_classification": verdict.get("sub_classification", ""),
                        "local_result": local_result,
                        "calculated_matches": local_result.get("matches", []),
                        "summary": verdict.get("summary", "Completed computer-vision frame perceptual analysis on accessible direct stream."),
                    }
            except Exception:
                is_opencv_streamable = False

    return {
        "source_name": "Direct Visual Sequence Analysis",
        "attempted": True,
        "accessible": False,
        "performed": False,
        "result_state": RESULT_ANALYSIS_UNAVAILABLE,
        "status": "UNAVAILABLE_FOR_REMOTE_YOUTUBE_URL",
        "error_category": "Footage Analysis Unavailable",
        "technical_reason": (
            "Official YouTube APIs (YouTube Data API v3 and IFrame Player API) do not expose "
            "raw video stream files or frame buffers. Direct footage analysis requires an authorized footage source or Gemini API key."
        ),
        "loop_absence_warning": (
            "Notice: The absence of viewer comments cannot be used as evidence that a video has no loops. "
            "Footage analysis is unavailable."
        ),
        "calculated_matches": [],
        "summary": "Direct footage analysis was attempted, but video frames cannot be accessed.",
    }


def get_direct_footage_analysis_status(
    is_local_file: bool = False,
    local_result: Optional[Dict[str, Any]] = None,
    footage_attempt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Provide transparent status of Direct Visual Sequence Analysis (Source C).
    
    Ensures LoopLens never claims direct footage analysis occurred unless an authorized
    footage source or Gemini multimodal analysis was successfully executed.
    """
    if footage_attempt and footage_attempt.get("performed"):
        return {
            "source_name": footage_attempt.get("source_name", "Gemini Multimodal Direct Analysis"),
            "attempted": True,
            "performed": True,
            "status": footage_attempt.get("status", "COMPLETED_VIA_GEMINI"),
            "result_state": footage_attempt.get("result_state", RESULT_REPEATED_FOOTAGE_DETECTED),
            "sub_classification": footage_attempt.get("sub_classification", ""),
            "confidence": footage_attempt.get("confidence", "Moderate"),
            "model_used": footage_attempt.get("model_used"),
            "technical_reason": None,
            "calculated_matches": footage_attempt.get("calculated_matches", []),
            "summary": footage_attempt.get("summary", "Completed multimodal AI sequence analysis via Gemini."),
        }

    if is_local_file and local_result:
        verdict = local_result.get("verdict", {})
        return {
            "source_name": "Direct Visual Sequence Analysis",
            "attempted": True,
            "performed": True,
            "status": verdict.get("status", "COMPLETED_ON_LOCAL_FILE"),
            "result_state": verdict.get("result_state", RESULT_REPEATED_FOOTAGE_DETECTED),
            "sub_classification": verdict.get("sub_classification", ""),
            "technical_reason": None,
            "calculated_matches": local_result.get("matches", []),
            "summary": "Completed computer-vision frame perceptual analysis on user-uploaded file.",
        }

    reason = (footage_attempt and footage_attempt.get("technical_reason")) or (
        "Direct video frame analysis was unavailable. Video frames could not be accessed."
    )
    warning = (footage_attempt and footage_attempt.get("loop_absence_warning")) or (
        "Notice: The absence of viewer comments cannot be used as evidence that a video has no loops. "
        "Direct footage analysis is unavailable."
    )
    error_cat = (footage_attempt and footage_attempt.get("error_category")) or "Footage Analysis Unavailable"

    return {
        "source_name": (footage_attempt and footage_attempt.get("source_name")) or "Gemini Multimodal Direct Analysis",
        "attempted": True,
        "performed": False,
        "result_state": RESULT_ANALYSIS_UNAVAILABLE,
        "status": (footage_attempt and footage_attempt.get("status")) or "ANALYSIS_UNAVAILABLE",
        "error_category": error_cat,
        "technical_reason": reason,
        "model_used": footage_attempt.get("model_used") if footage_attempt else None,
        "calculated_matches": [],
        "summary": (
            (footage_attempt and footage_attempt.get("summary")) or
            f"Direct AI footage analysis is unavailable: {error_cat}. {reason}"
        ),
        "loop_absence_warning": warning,
    }


def analyze_comments(
    comments: List[Dict[str, Any]],
    comments_disabled: bool,
    warning_note: Optional[str] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
    video_url: Optional[str] = None,
    video_id: Optional[str] = None,
    local_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute evidence-based pipeline across metadata, community feedback, and footage status.
    
    Adheres strictly to the four evidence-based result states:
    1. 'Repeated footage detected'
    2. 'No qualifying repetition detected'
    3. 'Inconclusive'
    4. 'Analysis unavailable'
    
    Distinguishes community reports from algorithmic video verification, accounts for
    recaps/montages to reduce false positives, and NEVER equates absence of comments
    with absence of loops.
    """
    meta_evidence = analyze_metadata(video_metadata)

    # Automatically attempt direct visual analysis when evaluating footage
    footage_attempt = attempt_direct_footage_analysis(
        video_url=video_url,
        video_id=video_id,
        local_video_path=local_video_path,
        video_metadata=video_metadata,
    )

    footage_performed = bool(footage_attempt.get("performed"))

    footage_evidence = get_direct_footage_analysis_status(
        is_local_file=(not video_url and not video_id and bool(local_video_path)),
        local_result=footage_attempt.get("local_result"),
        footage_attempt=footage_attempt,
    )

    err_category = footage_attempt.get("error_category", "Footage Analysis Unavailable")

    loop_absence_notice = (
        "Notice: The absence of viewer comments cannot be used as evidence that a video has no loops. "
        + (
            f"Direct footage analysis is unavailable: {err_category}."
            if not footage_performed
            else "Direct footage analysis completed."
        )
    )

    is_explicit_loop = meta_evidence.get("is_explicit_loop", False)
    is_explicit_recap = meta_evidence.get("is_explicit_recap", False)

    # 1. Check for metadata explicit intentional content first
    if is_explicit_loop:
        result_state = footage_attempt.get("result_state") if footage_performed else RESULT_ANALYSIS_UNAVAILABLE
        status = "INTENTIONAL_LOOP_CONTENT"
        status_title = f"{result_state} — Intentional Looped Content"
        severity = "caution"
        summary = (
            f"Video metadata indicates this is an intentional ambient or extended loop "
            f"('{meta_evidence.get('title')}'). Any internal repetition is intentional by design. "
            + (
                f"Footage analysis completed: {footage_attempt.get('summary')}"
                if footage_performed
                else f"Direct visual footage analysis is unavailable: {err_category}."
            )
        )
        return {
            "result_state": result_state,
            "status": status,
            "status_title": status_title,
            "severity": severity,
            "summary": summary,
            "total_analyzed": len(comments),
            "evidence_count": 0,
            "repeat_reports_count": 0,
            "incomplete_reports_count": 0,
            "recap_reports_count": 0,
            "filler_reports_count": 0,
            "evidence_comments": [],
            "identified_timestamps": [ch["timestamp"] for ch in meta_evidence.get("chapters", [])],
            "community_timestamps": [],
            "creator_chapter_timestamps": [ch["timestamp"] for ch in meta_evidence.get("chapters", [])],
            "total_evidence_likes": 0,
            "direct_footage_analysis_performed": footage_performed,
            "visual_analysis": footage_attempt.get("local_result"),
            "notice": loop_absence_notice,
            "evidence_pipeline": {
                "metadata": meta_evidence,
                "community_comments": {
                    "source_name": "Public Community Comments",
                    "evaluated": bool(comments),
                    "total_analyzed": len(comments),
                    "evidence_count": 0,
                    "repeat_reports": 0,
                    "recap_mentions": 0,
                    "incomplete_reports": 0,
                    "summary": "Metadata indicates intentional ambient loop.",
                },
                "direct_footage": footage_evidence,
            },
        }

    if is_explicit_recap and not comments:
        result_state = footage_attempt.get("result_state") if footage_performed else RESULT_ANALYSIS_UNAVAILABLE
        status = "INTENTIONAL_RECAP_NOTED"
        status_title = f"{result_state} — Intentional Recap Noted"
        severity = "caution"
        summary = (
            f"Video metadata indicates this video is an intentional recap, compilation, or summary "
            f"('{meta_evidence.get('title')}'). "
            + (
                f"Footage analysis completed: {footage_attempt.get('summary')}"
                if footage_performed
                else f"Direct visual footage analysis is unavailable: {err_category}."
            )
        )
        return {
            "result_state": result_state,
            "status": status,
            "status_title": status_title,
            "severity": severity,
            "summary": summary,
            "total_analyzed": 0,
            "evidence_count": 0,
            "repeat_reports_count": 0,
            "incomplete_reports_count": 0,
            "recap_reports_count": 0,
            "filler_reports_count": 0,
            "evidence_comments": [],
            "identified_timestamps": [ch["timestamp"] for ch in meta_evidence.get("chapters", [])],
            "community_timestamps": [],
            "creator_chapter_timestamps": [ch["timestamp"] for ch in meta_evidence.get("chapters", [])],
            "total_evidence_likes": 0,
            "direct_footage_analysis_performed": footage_performed,
            "visual_analysis": footage_attempt.get("local_result"),
            "notice": loop_absence_notice,
            "evidence_pipeline": {
                "metadata": meta_evidence,
                "community_comments": {
                    "source_name": "Public Community Comments",
                    "evaluated": False,
                    "total_analyzed": 0,
                    "evidence_count": 0,
                    "summary": "Metadata indicates intentional recap.",
                },
                "direct_footage": footage_evidence,
            },
        }

    # 2. Extract community evidence comments
    evidence_list: List[Dict[str, Any]] = []
    all_timestamps: List[str] = []
    repeat_count = 0
    incomplete_count = 0
    recap_count = 0
    filler_count = 0

    for comment in comments:
        res = analyze_single_comment(comment)
        if res:
            evidence_list.append(res)
            cats = res["categories"]
            if "repeated_footage" in cats:
                repeat_count += 1
            if "incomplete_video" in cats:
                incomplete_count += 1
            if "intentional_recap" in cats or "ambient_loop" in cats:
                recap_count += 1
            if "filler_or_padding" in cats:
                filler_count += 1
            for ts in res["matched_timestamps"]:
                if ts not in all_timestamps:
                    all_timestamps.append(ts)

    evidence_list.sort(key=lambda x: x["like_count"], reverse=True)
    total_analyzed = len(comments)
    evidence_count = len(evidence_list)
    total_likes_on_evidence = sum(e["like_count"] for e in evidence_list)
    creator_chapter_timestamps = [ch["timestamp"] for ch in meta_evidence.get("chapters", [])]
    combined_timestamps = sorted(list(set(all_timestamps + creator_chapter_timestamps)))

    has_high_likes = total_likes_on_evidence >= 10
    has_multiple_reports = evidence_count >= 2

    # 3. Synthesize Findings Across Sources
    # Priority A: Direct Visual Footage Analysis succeeded
    if footage_performed:
        visual_state = footage_attempt.get("result_state", RESULT_INCONCLUSIVE)
        if visual_state in (RESULT_REPEATED_FOOTAGE_DETECTED, "Repeated sequence detected"):
            result_state = footage_attempt.get("result_state", RESULT_REPEATED_FOOTAGE_DETECTED)
            status = "REPEATED_FOOTAGE_DETECTED"
            status_title = f"{result_state} — AI Visual Analysis"
            severity = "warning"
            summary = (
                f"Visual footage analysis confirmed repeated sequence(s): {footage_attempt.get('summary')} "
                + (f"Viewer comments also reported repetition ({repeat_count} report(s))." if repeat_count > 0 else "")
            )
        elif visual_state in (RESULT_NO_QUALIFYING_REPETITION, "No qualifying repeated sequence detected in the analysed footage"):
            result_state = footage_attempt.get("result_state", RESULT_NO_QUALIFYING_REPETITION)
            severity = "safe"
            if repeat_count > 0:
                status = "NO_QUALIFYING_REPETITION_COMMENTS_NOTED"
                status_title = f"{result_state} (Viewer Reports Noted)"
                summary = (
                    f"Visual footage analysis detected no qualifying repeated sequences: {footage_attempt.get('summary')} "
                    f"Note: {repeat_count} viewer comment(s) mentioned repetition, but algorithmic footage analysis found no qualifying duplicate sequence."
                )
            else:
                status = "NO_QUALIFYING_REPETITION"
                status_title = f"{result_state} — AI Visual Analysis"
                summary = f"Visual footage analysis detected no qualifying repeated sequence: {footage_attempt.get('summary')}"
        else:
            result_state = RESULT_INCONCLUSIVE
            status = "INCONCLUSIVE"
            status_title = f"{RESULT_INCONCLUSIVE} — Ambiguous Evidence"
            severity = "caution"
            summary = f"Visual footage analysis was inconclusive: {footage_attempt.get('summary')}"

        return {
            "result_state": result_state,
            "status": status,
            "status_title": status_title,
            "severity": severity,
            "summary": summary,
            "total_analyzed": total_analyzed,
            "evidence_count": evidence_count,
            "repeat_reports_count": repeat_count,
            "incomplete_reports_count": incomplete_count,
            "recap_reports_count": recap_count,
            "filler_reports_count": filler_count,
            "evidence_comments": evidence_list,
            "identified_timestamps": combined_timestamps,
            "community_timestamps": all_timestamps,
            "creator_chapter_timestamps": creator_chapter_timestamps,
            "total_evidence_likes": total_likes_on_evidence,
            "direct_footage_analysis_performed": True,
            "visual_analysis": footage_attempt.get("local_result"),
            "notice": "Direct visual footage analysis completed. Model findings are estimates based on visual evidence.",
            "evidence_pipeline": {
                "metadata": meta_evidence,
                "community_comments": {
                    "source_name": "Public Community Comments",
                    "evaluated": bool(comments),
                    "total_analyzed": total_analyzed,
                    "evidence_count": evidence_count,
                    "repeat_reports": repeat_count,
                    "recap_mentions": recap_count,
                    "incomplete_reports": incomplete_count,
                    "summary": (
                        f"Analyzed {total_analyzed} comments. {evidence_count} flagged community comments found."
                        if evidence_count > 0
                        else f"Analyzed {total_analyzed} comments with zero repetition reports."
                    ),
                },
                "direct_footage": footage_evidence,
            },
        }

    # Priority B: Direct footage analysis is UNAVAILABLE
    # Reflect community reports if present, but NEVER claim absence of comments means absence of loops
    result_state = RESULT_ANALYSIS_UNAVAILABLE

    if comments_disabled:
        status = "COMMENTS_DISABLED"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} ({err_category})"
        severity = "neutral"
        summary = (
            f"Comments are disabled by the creator. Direct AI video footage analysis is unavailable: "
            f"{footage_attempt.get('technical_reason') or err_category}. "
            "Crucially: The absence of viewer comments cannot be used as evidence that a video has no loops."
        )
    elif not comments:
        status = "NO_COMMENTS_AVAILABLE"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} ({err_category})"
        severity = "neutral"
        summary = (
            warning_note or (
                f"No public comments were retrieved for this video. Direct AI footage analysis is unavailable: "
                f"{footage_attempt.get('technical_reason') or err_category}. "
                "Crucially: The absence of viewer comments cannot be used as evidence that a video has no loops."
            )
        )
    elif meta_evidence.get("is_explicit_recap") or (recap_count > 0 and recap_count >= repeat_count):
        status = "INTENTIONAL_RECAP_NOTED"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} — Intentional Recap Noted"
        severity = "caution"
        summary = (
            f"Identified {recap_count} viewer comment(s) and metadata markers referencing a recap or montage. "
            f"Direct AI footage analysis is unavailable: {footage_attempt.get('technical_reason') or err_category}."
        )
    elif repeat_count > 0 and (has_multiple_reports or has_high_likes):
        status = "REPEATED_FOOTAGE_REPORTED"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} — Community Repetition Reported"
        severity = "warning"
        summary = (
            f"Found {repeat_count} viewer comment(s) (with {total_likes_on_evidence} total likes) "
            f"reporting repeated scenes, looping, or duplicated footage. "
            f"Direct AI footage analysis is unavailable: {footage_attempt.get('technical_reason') or err_category}; "
            f"findings reflect authentic viewer evidence."
        )
    elif incomplete_count > 0 and (has_multiple_reports or has_high_likes):
        status = "INCOMPLETE_VIDEO_REPORTED"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} — Community Cut-off Reported"
        severity = "warning"
        summary = (
            f"Found {incomplete_count} viewer comment(s) reporting that the video cuts off or is incomplete. "
            f"Direct AI footage analysis is unavailable: {footage_attempt.get('technical_reason') or err_category}."
        )
    elif repeat_count > 0 or incomplete_count > 0 or filler_count > 0:
        status = "POTENTIAL_ISSUES_NOTED"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} — Isolated Reports Noted"
        severity = "caution"
        summary = (
            f"Identified {evidence_count} viewer comment(s) mentioning potential repetition or padding issues, "
            f"but reports are isolated. Direct AI footage analysis is unavailable: "
            f"{footage_attempt.get('technical_reason') or err_category}. "
            "Crucially: The absence of viewer comments cannot be used as evidence that a video has no loops."
        )
    else:
        status = "CLEAR"
        status_title = f"{RESULT_ANALYSIS_UNAVAILABLE} ({err_category})"
        severity = "neutral"
        summary = (
            f"Analyzed {total_analyzed} public comment(s) with zero repetition reports (inconclusive community evidence). "
            f"Direct AI video footage analysis is unavailable: {footage_attempt.get('technical_reason') or err_category}. "
            "Crucially: The absence of viewer comments cannot be used as evidence that a video has no loops."
        )

    return {
        "result_state": result_state,
        "status": status,
        "status_title": status_title,
        "severity": severity,
        "summary": summary,
        "total_analyzed": total_analyzed,
        "evidence_count": evidence_count,
        "repeat_reports_count": repeat_count,
        "incomplete_reports_count": incomplete_count,
        "recap_reports_count": recap_count,
        "filler_reports_count": filler_count,
        "evidence_comments": evidence_list,
        "identified_timestamps": combined_timestamps,
        "community_timestamps": all_timestamps,
        "creator_chapter_timestamps": creator_chapter_timestamps,
        "total_evidence_likes": total_likes_on_evidence,
        "direct_footage_analysis_performed": False,
        "visual_analysis": None,
        "notice": loop_absence_notice,
        "evidence_pipeline": {
            "metadata": meta_evidence,
            "community_comments": {
                "source_name": "Public Community Comments",
                "evaluated": bool(comments),
                "total_analyzed": total_analyzed,
                "evidence_count": evidence_count,
                "repeat_reports": repeat_count,
                "recap_mentions": recap_count,
                "incomplete_reports": incomplete_count,
                "summary": (
                    f"Analyzed {total_analyzed} comments. {evidence_count} flagged community comments found."
                    if evidence_count > 0
                    else f"Analyzed {total_analyzed} comments with zero repetition reports (inconclusive community evidence)."
                ),
            },
            "direct_footage": footage_evidence,
        },
    }

