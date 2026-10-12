"""Video Frame Analysis Module for LoopLens.

Performs computer-vision based repeated sequence detection on user-provided video files.
Uses perceptual hashing (dHash), luminance profiling, and inter-frame motion analysis
to identify candidate looped or repeated footage segments while filtering out static scenes.

NOTE: This is an experimental prototype using perceptual similarity heuristics.
It is completely separate from YouTube Data API comment analysis and does NOT
interact with or download YouTube streams.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# Configuration Constants
DEFAULT_SAMPLE_INTERVAL_SEC = 1.0
MIN_SAMPLE_INTERVAL_SEC = 0.2
MAX_SAMPLE_INTERVAL_SEC = 10.0

DEFAULT_MIN_SEQUENCE_SEC = 3.0
MIN_GAP_BETWEEN_SEQUENCES_SEC = 3.0  # Sequences must be at least this far apart to be considered duplicates

SIMILARITY_THRESHOLD = 0.88  # Perceptual similarity threshold for single frame candidate match
STRONG_SIMILARITY_THRESHOLD = 0.94
STATIC_MOTION_THRESHOLD = 3.0  # Frame-to-frame pixel change below which consecutive frames are considered static
STATIC_VARIANCE_THRESHOLD = 5.0  # Internal sequence variance below which sequence is flagged as a static scene

MAX_VIDEO_DURATION_SEC = 1800.0  # 30 minutes cap for prototype processing
MAX_SAMPLED_FRAMES = 1200


@dataclass
class FrameSignature:
    dhash: int
    hash_bits: int
    lum_matrix: np.ndarray  # 16x16 normalized luminance
    color_hist: np.ndarray  # 24-bin HSV histogram
    is_blank: bool
    mean_luminance: float


@dataclass
class SampledFrame:
    index: int
    frame_number: int
    timestamp_sec: float
    timestamp_formatted: str
    signature: FrameSignature
    motion_from_prev: float
    thumbnail_b64: str  # Base64 JPEG for visual evidence display


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    sec = max(0.0, seconds)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def compute_dhash(gray_img: np.ndarray, hash_size: int = 16) -> Tuple[int, int]:
    """Compute difference hash (dHash) from grayscale image.
    
    Resizes to (hash_size + 1, hash_size) and compares adjacent columns.
    Returns (hash_integer, total_bits).
    """
    resized = cv2.resize(gray_img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Flatten to bit integer
    total_bits = hash_size * hash_size
    diff_flat = diff.flatten()
    hash_int = 0
    for bit in diff_flat:
        hash_int = (hash_int << 1) | int(bit)
    return hash_int, total_bits


def compute_frame_signature(frame_bgr: np.ndarray) -> FrameSignature:
    """Extract perceptual signature from a video frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))
    
    is_blank = std_lum < 1.5 or mean_lum < 2.0 or mean_lum > 253.0
    
    # 1. dHash (16x16 = 256 bits for high discrimination)
    hash_int, total_bits = compute_dhash(gray, hash_size=16)
    
    # 2. Downscaled normalized luminance profile (16x16)
    lum_small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    # 3. HSV color distribution
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    color_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    norm = np.linalg.norm(color_hist)
    if norm > 0:
        color_hist = color_hist / norm

    return FrameSignature(
        dhash=hash_int,
        hash_bits=total_bits,
        lum_matrix=lum_small,
        color_hist=color_hist,
        is_blank=is_blank,
        mean_luminance=mean_lum,
    )


def compare_signatures(sig1: FrameSignature, sig2: FrameSignature) -> float:
    """Compute perceptual similarity between two frame signatures in range [0.0, 1.0]."""
    # If both are blank (e.g. black screen transition), do not treat as content repetition
    if sig1.is_blank and sig2.is_blank:
        return 0.50
    if sig1.is_blank != sig2.is_blank:
        return 0.0

    # 1. Normalized Hamming correlation on dHash (map 50% random baseline to 0.0)
    xor_val = sig1.dhash ^ sig2.dhash
    hamming_dist = bin(xor_val).count("1")
    raw_hash_sim = 1.0 - (hamming_dist / float(sig1.hash_bits))
    hash_corr = max(0.0, (raw_hash_sim - 0.50) / 0.50)

    # 2. Normalized Cross-Correlation (NCC) on luminance matrix
    std1 = float(np.std(sig1.lum_matrix))
    std2 = float(np.std(sig2.lum_matrix))
    if std1 > 1e-4 and std2 > 1e-4:
        norm1 = (sig1.lum_matrix - float(np.mean(sig1.lum_matrix))) / std1
        norm2 = (sig2.lum_matrix - float(np.mean(sig2.lum_matrix))) / std2
        ncc = float(np.mean(norm1 * norm2))
        lum_sim = max(0.0, ncc)
    else:
        mse = float(np.mean((sig1.lum_matrix - sig2.lum_matrix) ** 2))
        lum_sim = max(0.0, 1.0 - math.sqrt(mse) * 2.0)

    # 3. Color histogram cosine similarity
    color_sim = float(np.dot(sig1.color_hist, sig2.color_hist))

    # Combined weighted score
    combined = 0.50 * hash_corr + 0.35 * lum_sim + 0.15 * color_sim
    return max(0.0, min(1.0, combined))


def frame_to_thumbnail_b64(frame_bgr: np.ndarray, max_width: int = 160) -> str:
    """Resize frame and encode as JPEG Base64 for evidence rendering."""
    h, w = frame_bgr.shape[:2]
    new_w = max_width
    new_h = int(h * (max_width / float(w)))
    thumb = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")


def extract_sampled_frames(
    video_path: Path,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
) -> Tuple[List[SampledFrame], Dict[str, Any]]:
    """Sample frames from video at fixed intervals and extract perceptual features."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path.name}. Unsupported format or corrupted file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    if duration_sec <= 0:
        cap.release()
        raise ValueError("Video duration is zero or could not be determined.")

    if duration_sec > MAX_VIDEO_DURATION_SEC:
        cap.release()
        raise ValueError(f"Video duration ({format_seconds(duration_sec)}) exceeds maximum prototype limit of {int(MAX_VIDEO_DURATION_SEC / 60)} minutes.")

    # Clamp sampling interval
    interval = max(MIN_SAMPLE_INTERVAL_SEC, min(MAX_SAMPLE_INTERVAL_SEC, sample_interval_sec))
    frames_step = max(1, int(round(fps * interval)))

    sampled: List[SampledFrame] = []
    prev_gray: Optional[np.ndarray] = None
    curr_frame_idx = 0
    sample_counter = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_sec = curr_frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion from previous sampled frame
        if prev_gray is not None and prev_gray.shape == gray.shape:
            motion = float(np.mean(cv2.absdiff(gray, prev_gray)))
        else:
            motion = 100.0  # Initial frame baseline
        prev_gray = gray

        signature = compute_frame_signature(frame)
        thumbnail_b64 = frame_to_thumbnail_b64(frame)

        sampled.append(
            SampledFrame(
                index=sample_counter,
                frame_number=curr_frame_idx,
                timestamp_sec=timestamp_sec,
                timestamp_formatted=format_seconds(timestamp_sec),
                signature=signature,
                motion_from_prev=motion,
                thumbnail_b64=thumbnail_b64,
            )
        )

        sample_counter += 1
        if sample_counter >= MAX_SAMPLED_FRAMES:
            break

        curr_frame_idx += frames_step
        if curr_frame_idx >= total_frames:
            break

    cap.release()

    meta = {
        "duration_sec": round(duration_sec, 2),
        "duration_formatted": format_seconds(duration_sec),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "sampled_frames_count": len(sampled),
        "sample_interval_sec": interval,
    }
    return sampled, meta


RESULT_REPEATED_SEQUENCE_DETECTED = "Repeated sequence detected"
RESULT_NO_REPEATED_SEQUENCE = "No qualifying repeated sequence detected in the analysed footage"
RESULT_INCONCLUSIVE = "Inconclusive"
RESULT_FOOTAGE_UNAVAILABLE = "Footage analysis unavailable"


def detect_repeated_sequences(
    sampled_frames: List[SampledFrame],
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    min_sequence_sec: float = DEFAULT_MIN_SEQUENCE_SEC,
    similarity_thresh: float = SIMILARITY_THRESHOLD,
    min_gap_sec: float = MIN_GAP_BETWEEN_SEQUENCES_SEC,
    total_duration_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Detect candidate repeated sequences across sampled frames.
    
    Identifies diagonal runs of consecutively similar frames between two distinct
    time periods in the video, distinguishing dynamic repeated footage from static scenes
    and classifying potential loop-padding vs recaps/montages without inferring intent.
    """
    n = len(sampled_frames)
    if n < 4:
        return []

    video_duration = total_duration_sec or (sampled_frames[-1].timestamp_sec if sampled_frames else 0.0)

    min_sequence_frames = max(2, int(math.ceil(min_sequence_sec / sample_interval_sec)))
    min_gap_frames = max(2, int(math.ceil(min_gap_sec / sample_interval_sec)))

    # Step 1: Compute pairwise similarity matrix for candidate pairs (i < j - min_gap)
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + min_gap_frames, n):
            sim = compare_signatures(sampled_frames[i].signature, sampled_frames[j].signature)
            sim_matrix[i, j] = sim

    # Step 2: Search for diagonal sequence runs (i+k, j+k)
    visited = np.zeros((n, n), dtype=bool)
    raw_matches: List[Tuple[int, int, int, float]] = []  # (start_i, start_j, length, avg_sim)

    for i in range(n - min_sequence_frames):
        for j in range(i + min_gap_frames, n - min_sequence_frames):
            if visited[i, j]:
                continue
            
            if sim_matrix[i, j] >= similarity_thresh:
                # Trace diagonal match
                k = 0
                sim_sum = 0.0
                while (
                    i + k < n
                    and j + k < n
                    and (i + k) + min_gap_frames <= (j + k)
                    and sim_matrix[i + k, j + k] >= similarity_thresh
                ):
                    visited[i + k, j + k] = True
                    sim_sum += sim_matrix[i + k, j + k]
                    k += 1

                if k >= min_sequence_frames:
                    avg_sim = sim_sum / k
                    raw_matches.append((i, j, k, avg_sim))

    # Step 3: Consolidate overlapping / redundant candidate sequences
    consolidated_matches = []
    raw_matches.sort(key=lambda x: (x[2], x[3]), reverse=True)

    used_intervals: List[Tuple[int, int, int, int]] = []

    for start_i, start_j, length, avg_sim in raw_matches:
        end_i = start_i + length - 1
        end_j = start_j + length - 1

        # Check overlap with already accepted intervals
        is_subsumed = False
        for ui_s, ui_e, uj_s, uj_e in used_intervals:
            if ui_s <= start_i and end_i <= ui_e and uj_s <= start_j and end_j <= uj_e:
                is_subsumed = True
                break
        if is_subsumed:
            continue

        used_intervals.append((start_i, end_i, start_j, end_j))

        # Check internal motion and visual variance to detect static scenes
        seq_frames_a = sampled_frames[start_i : end_i + 1]
        seq_frames_b = sampled_frames[start_j : end_j + 1]

        # Calculate average inter-frame motion within sequence A
        motions_a = [f.motion_from_prev for f in seq_frames_a[1:]] if len(seq_frames_a) > 1 else [0.0]
        avg_motion_a = float(np.mean(motions_a)) if motions_a else 0.0

        # Calculate internal luminance standard deviation
        lums_a = [f.signature.mean_luminance for f in seq_frames_a]
        std_lum_a = float(np.std(lums_a)) if len(lums_a) > 1 else 0.0

        is_static_scene = (avg_motion_a < STATIC_MOTION_THRESHOLD) and (std_lum_a < STATIC_VARIANCE_THRESHOLD)
        
        # Calculate duration of the repeated block
        duration_sec = round(length * sample_interval_sec, 2)
        start_a_sec = seq_frames_a[0].timestamp_sec
        end_a_sec = seq_frames_a[-1].timestamp_sec
        start_b_sec = seq_frames_b[0].timestamp_sec
        end_b_sec = seq_frames_b[-1].timestamp_sec

        # Distinguish 'Repeated sequence detected' from 'Possible loop-padding'
        # Loop-padding occurs when repeated footage is placed in the final portion of runtime
        # (e.g. past 65% mark or extending to the end) or accounts for substantial fraction of duration.
        is_end_padding = (
            video_duration > 0
            and (end_b_sec >= video_duration * 0.70 or start_b_sec >= video_duration * 0.60)
        )
        is_substantial_fraction = (
            video_duration > 0
            and (duration_sec / max(1.0, video_duration) >= 0.20)
        )

        if is_static_scene:
            classification = "Static scene / Unchanging frames"
            classification_note = "Minimal inter-frame motion detected. Likely a static slide or title card."
        elif is_end_padding or is_substantial_fraction:
            classification = "Possible loop-padding"
            classification_note = (
                "Sequence repeats towards the end of the video or accounts for a large fraction of runtime, "
                "consistent with duration-extension padding. (Creator intent cannot be inferred from visual similarity alone)."
            )
        else:
            classification = "Repeated sequence detected"
            classification_note = (
                "Candidate repeated sequence within normal content progression. May represent a recap, "
                "flashback, montage, or recurring scene. (Creator intent cannot be inferred from visual similarity alone)."
            )

        # Extract representative keyframes for evidence display (start and mid)
        mid_idx = length // 2
        keyframe_a = seq_frames_a[mid_idx].thumbnail_b64
        keyframe_b = seq_frames_b[mid_idx].thumbnail_b64

        confidence_level = "High" if avg_sim >= STRONG_SIMILARITY_THRESHOLD and not is_static_scene else "Moderate"
        if is_static_scene:
            confidence_level = "Low (Static Scene)"

        match_record = {
            "match_id": f"seq_{start_i}_{start_j}",
            "frames_count": length,
            "duration_sec": duration_sec,
            "duration_formatted": format_seconds(duration_sec),
            "original_sequence": {
                "start_sec": round(start_a_sec, 2),
                "end_sec": round(end_a_sec, 2),
                "start_formatted": format_seconds(start_a_sec),
                "end_formatted": format_seconds(end_a_sec),
                "time_range": f"{format_seconds(start_a_sec)} - {format_seconds(end_a_sec)}",
                "sample_indices": [f.index for f in seq_frames_a],
            },
            "repeated_sequence": {
                "start_sec": round(start_b_sec, 2),
                "end_sec": round(end_b_sec, 2),
                "start_formatted": format_seconds(start_b_sec),
                "end_formatted": format_seconds(end_b_sec),
                "time_range": f"{format_seconds(start_b_sec)} - {format_seconds(end_b_sec)}",
                "sample_indices": [f.index for f in seq_frames_b],
            },
            "similarity_score": round(float(avg_sim) * 100, 1),
            "is_static_scene": is_static_scene,
            "classification": classification,
            "classification_note": classification_note,
            "intent_disclaimer": "Creator intent cannot be inferred from visual similarity alone.",
            "static_scene_warning": (
                "Notice: Frames exhibit minimal inter-frame motion or variance. "
                "This match may be a static presentation, title card, or frozen screen rather than looping motion."
                if is_static_scene
                else None
            ),
            "confidence": confidence_level,
            "evidence_thumbnails": {
                "original_midpoint_b64": keyframe_a,
                "repeated_midpoint_b64": keyframe_b,
                "original_mid_time": format_seconds(seq_frames_a[mid_idx].timestamp_sec),
                "repeated_mid_time": format_seconds(seq_frames_b[mid_idx].timestamp_sec),
            },
        }
        consolidated_matches.append(match_record)

    # Sort final matches by chronological order of original sequence
    consolidated_matches.sort(key=lambda m: m["original_sequence"]["start_sec"])
    return consolidated_matches


def analyze_video_file(
    video_path: Path,
    original_filename: str,
    sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
    min_sequence_sec: float = DEFAULT_MIN_SEQUENCE_SEC,
) -> Dict[str, Any]:
    """Execute end-to-end frame repetition analysis on an uploaded video file.
    
    Adheres strictly to the four evidence-based result states:
    - 'Repeated sequence detected'
    - 'No qualifying repeated sequence detected in the analysed footage'
    - 'Inconclusive'
    - 'Footage analysis unavailable'
    """
    sampled_frames, meta = extract_sampled_frames(
        video_path=video_path,
        sample_interval_sec=sample_interval_sec,
    )

    matches = detect_repeated_sequences(
        sampled_frames=sampled_frames,
        sample_interval_sec=meta["sample_interval_sec"],
        min_sequence_sec=min_sequence_sec,
        total_duration_sec=meta["duration_sec"],
    )

    # Filter into dynamic repetitions and static scene warnings
    dynamic_matches = [m for m in matches if not m["is_static_scene"]]
    static_matches = [m for m in matches if m["is_static_scene"]]

    if dynamic_matches:
        has_loop_padding = any(m["classification"] == "Possible loop-padding" for m in dynamic_matches)
        result_state = RESULT_REPEATED_SEQUENCE_DETECTED
        status = "REPEATED_SEQUENCES_DETECTED"
        status_title = RESULT_REPEATED_SEQUENCE_DETECTED
        sub_classification = "Possible loop-padding" if has_loop_padding else "Repeated sequence detected"
        severity = "warning"
        summary = (
            f"Identified {len(dynamic_matches)} candidate repeated sequence(s) totaling "
            f"{sum(m['duration_sec'] for m in dynamic_matches):.1f}s of duplicated visual footage. "
            f"Distinguished as '{sub_classification}'. Note: Creator intent cannot be inferred from visual similarity alone."
        )
    elif static_matches:
        result_state = RESULT_INCONCLUSIVE
        status = "INCONCLUSIVE"
        status_title = RESULT_INCONCLUSIVE
        sub_classification = "Static scene / Unchanging frames"
        severity = "caution"
        summary = (
            f"Detected {len(static_matches)} sequence match(es), but the underlying frames exhibit "
            f"near-zero motion. These are classified as static content (e.g., presentation slides, pauses, "
            f"or title cards) rather than looped motion footage. Evidence is ambiguous and the scan is inconclusive."
        )
    else:
        result_state = RESULT_NO_REPEATED_SEQUENCE
        status = "NO_REPEATED_SEQUENCES"
        status_title = RESULT_NO_REPEATED_SEQUENCE
        sub_classification = "No repetition detected"
        severity = "safe"
        summary = (
            f"Analyzed {meta['sampled_frames_count']} sampled frames across {meta['duration_formatted']}. "
            f"No qualifying repeated sequence detected in the analysed footage."
        )

    return {
        "success": True,
        "is_prototype": True,
        "evidence_source": "LOCAL_FRAME_PERCEPTUAL_ANALYSIS",
        "file_info": {
            "filename": original_filename,
            "duration_sec": meta["duration_sec"],
            "duration_formatted": meta["duration_formatted"],
            "resolution": f"{meta['width']}x{meta['height']}",
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "sampled_frames_count": meta["sampled_frames_count"],
            "sample_interval_sec": meta["sample_interval_sec"],
            "min_sequence_sec": min_sequence_sec,
        },
        "verdict": {
            "result_state": result_state,
            "status": status,
            "status_title": status_title,
            "sub_classification": sub_classification,
            "severity": severity,
            "summary": summary,
            "dynamic_matches_count": len(dynamic_matches),
            "static_matches_count": len(static_matches),
            "total_candidate_matches": len(matches),
            "intent_disclaimer": "Creator intent cannot be inferred from visual similarity alone.",
        },
        "matches": matches,
        "prototype_disclaimer": (
            "Experimental Computer-Vision Prototype: This analysis samples frames at fixed time intervals "
            "and calculates perceptual visual hash similarity (dHash + luminance profiling). It identifies candidate "
            "repeated sequences and flags potential false positives like static scenes. It does not replace full manual "
            "inspection or editorial review, and creator intent cannot be inferred from visual similarity alone."
        ),
    }


def generate_permitted_sample_video(path: Path, fps: int = 10, width: int = 320, height: int = 240) -> Path:
    """Generate a synthetic 12-second test video containing a deliberate repeated sequence.
    
    Structure:
    - 0s to 3s (30 frames): Sequence A (moving circle and text)
    - 3s to 7s (40 frames): Sequence B (moving rectangle and text)
    - 7s to 11s (40 frames): Sequence A repeated (exact replay of 0s-3s)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # Pre-render Sequence A frames (30 frames)
    seq_a_frames = []
    for f in range(30):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (30, 20, 20)
        x = int(50 + f * 5)
        y = int(60 + f * 3)
        cv2.circle(img, (x, y), 25, (255, 120, 0), -1)
        cv2.putText(img, f"SEQ A: {f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        seq_a_frames.append(img)

    # Pre-render Sequence B frames (40 frames)
    seq_b_frames = []
    for f in range(40):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (20, 40, 20)
        x = int(220 - f * 4)
        y = int(40 + f * 3)
        cv2.rectangle(img, (x, y), (x + 50, y + 40), (0, 220, 50), -1)
        cv2.putText(img, f"SEQ B: {f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        seq_b_frames.append(img)

    for frame in seq_a_frames:
        writer.write(frame)

    for frame in seq_b_frames:
        writer.write(frame)

    for i in range(40):
        writer.write(seq_a_frames[i % 30])

    writer.release()
    return path


def generate_static_sample_video(path: Path, fps: int = 10, width: int = 320, height: int = 240) -> Path:
    """Generate an 8-second synthetic video with a static title slide (zero motion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    static_frame = np.zeros((height, width, 3), dtype=np.uint8)
    static_frame[:, :] = (50, 30, 60)
    cv2.putText(static_frame, "STATIC SLIDE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for _ in range(80):
        writer.write(static_frame)

    writer.release()
    return path

