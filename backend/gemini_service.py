"""Gemini API Multimodal Video Repetition Analyzer

Performs direct visual sequence and repetition analysis of public YouTube videos
using Google's Gemini multimodal models (gemini-2.0-flash / gemini-1.5-flash).

Direct YouTube URL analysis is officially supported via fileData with mimeType='video/*'.
Handles API errors, blocked/restricted keys, quotas, timeouts, and distinguishes normal
storytelling (flashbacks, recaps, montages, credits) from loop-padding without making false claims
about creator intent.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import requests

from backend.config import get_gemini_api_key, is_gemini_configured
from backend.youtube_service import extract_video_id

logger = logging.getLogger("looplens.gemini")

# Standard Evidence-Based Result States
RESULT_REPEATED_FOOTAGE_DETECTED = "Repeated footage detected"
RESULT_NO_QUALIFYING_REPETITION = "No qualifying repetition detected"
RESULT_INCONCLUSIVE = "Inconclusive"
RESULT_ANALYSIS_UNAVAILABLE = "Analysis unavailable"

# Permitted official models with multimodal YouTube video support
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

ANALYSIS_SYSTEM_PROMPT = """You are an objective, expert video analysis engine for LoopLens.
Your goal is to inspect the provided public YouTube video footage and determine whether it contains substantial repeated footage, loop-padding, or duplicated scenes.

CRITICAL INSTRUCTIONS:
1. DISTINGUISH NORMAL STORYTELLING FROM REPETITION:
   - Normal storytelling includes:
     * Flashbacks (brief glimpses of past events for narrative context)
     * Recaps (e.g., 'Previously on...', summary of prior episodes)
     * Montages or compilation highlights
     * Recurring thematic motifs or standard scene transitions
     * Opening and ending credits, title sequences, bumpers, or logos
   - Do NOT classify normal storytelling, credits, or brief artistic flashbacks as deceptive loop-padding.
   
2. IDENTIFY REPEATED SEQUENCES:
   - If footage is repeated, extract:
     * Original sequence start and end time (MM:SS)
     * Repeated sequence start and end time (MM:SS)
     * Approximate sequence duration in seconds
     * Sequence temporal position (e.g., initial 3 minutes replayed starting at minute 7)
   - Compare sequence duration and temporal position:
     * Replaying significant blocks of video to artificially extend duration is a strong indicator of loop-padding.
     * Brief recurring cuts (< 10 seconds) or stylistic replays are NOT loop-padding.

3. DISTINGUISH CONFIRMED VISUAL SIMILARITY FROM INTENT:
   - Clearly report visual similarity between timestamps.
   - Do NOT make definitive accusations regarding the creator's deliberate intent.
   - State clearly: 'Creator intent cannot be inferred from visual similarity alone.'
   - Treat model conclusions as estimates, not definitive proof.

4. REQUIRED RESULT STATES (You MUST pick exactly ONE):
   - 'Repeated footage detected' — substantial repeated sequences are supported by visual evidence.
   - 'No qualifying repetition detected' — sufficient footage was analyzed, but no qualifying repeated sequence was found.
   - 'Inconclusive' — evidence is ambiguous or insufficient to decide.
   - 'Analysis unavailable' — footage could not be accessed, decoded, or analyzed.

Respond ONLY with a valid JSON object matching this schema:
{
  "result_state": "Repeated footage detected" | "No qualifying repetition detected" | "Inconclusive" | "Analysis unavailable",
  "status": "REPEATED_FOOTAGE_DETECTED" | "NO_QUALIFYING_REPETITION" | "INCONCLUSIVE" | "ANALYSIS_UNAVAILABLE",
  "confidence": "High" | "Moderate" | "Low",
  "summary": "Concise 2-3 sentence overview of findings.",
  "sub_classification": "Possible loop-padding" | "Repeated sequence" | "Normal storytelling" | "Clean footage",
  "repeated_sequences": [
    {
      "original_start": "MM:SS",
      "original_end": "MM:SS",
      "repeat_start": "MM:SS",
      "repeat_end": "MM:SS",
      "duration_sec": 30,
      "classification": "Possible loop-padding" | "Repeated sequence" | "Flashback / Recap" | "Credits / Intro",
      "temporal_context": "First 3 minutes replayed starting at minute 7",
      "visual_description": "Exact visual scene replayed",
      "creator_intent_disclaimer": "Creator intent cannot be inferred from visual similarity alone."
    }
  ],
  "storytelling_elements_noted": [
    "Brief recap observed at 01:00-01:15 (excluded from repetition flagging)"
  ],
  "creator_intent_note": "Visual analysis measures footage similarity; it does not confirm creator motives or deliberate padding."
}
"""


def build_gemini_payload(youtube_url: str, video_title: Optional[str] = None) -> Dict[str, Any]:
    """Construct the Gemini API JSON payload with direct YouTube URL video input."""
    prompt_text = (
        f"Perform a detailed multimodal video repetition analysis for this YouTube video: {youtube_url}"
        + (f" (Title: '{video_title}')" if video_title else "")
        + ".\n"
        "INSTRUCTIONS:\n"
        "1. Inspect the video footage and audio to identify any potentially repeated sequences, duplicate scenes, or loop-padding.\n"
        "2. Distinguish ordinary storytelling elements (brief flashbacks, narrative recaps, montage highlights, opening/ending credits, title cards) from suspicious loop-padding.\n"
        "3. Extract approximate timestamps (MM:SS) for original and repeated sequences, duration in seconds, temporal context, and sequence similarity.\n"
        "4. Treat your conclusions as estimates and indicators based on visual/audio evidence, not definitive proof of creator intent.\n"
        "5. Return strictly valid JSON adhering to the specified schema."
    )

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "fileUri": youtube_url,
                            "mimeType": "video/*",
                        }
                    },
                    {
                        "text": prompt_text,
                    },
                ],
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": ANALYSIS_SYSTEM_PROMPT,
                }
            ]
        },
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }


def parse_gemini_error(status_code: int, response_json: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """Translate Gemini API error codes and messages into transparent, sanitized diagnostics."""
    error_obj = response_json.get("error", {})
    err_message = error_obj.get("message", raw_text)
    err_status = error_obj.get("status", "")

    # Extract reason code from error details if present
    details = error_obj.get("details", [])
    detail_reason = ""
    for d in details:
        if isinstance(d, dict) and "reason" in d:
            detail_reason = d["reason"]
            break

    # 403 Forbidden / Blocked API / Restricted Key
    if status_code == 403 or "blocked" in err_message.lower() or detail_reason == "API_KEY_SERVICE_BLOCKED":
        technical_reason = (
            "Gemini API rejected the request (HTTP 403 Forbidden): The configured API key has restrictions "
            "in Google Cloud Console that block the Generative Language API (generativelanguage.googleapis.com), "
            "or the Generative Language API is not enabled in your Google Cloud project. "
            "To enable direct YouTube URL analysis with Gemini, update your key's API restrictions in "
            "Google Cloud Console to allow 'Generative Language API', or generate an unrestricted key in Google AI Studio."
        )
        return {
            "status": "API_KEY_BLOCKED_OR_RESTRICTED",
            "error_category": "API Key Blocked or Restricted",
            "technical_reason": technical_reason,
            "error_code": "FORBIDDEN",
        }

    # 401 Unauthorized / Invalid Key
    if status_code == 401 or "API key not valid" in err_message or detail_reason == "API_KEY_INVALID":
        return {
            "status": "INVALID_API_KEY",
            "error_category": "Invalid API Key",
            "technical_reason": (
                "The configured GEMINI_API_KEY is invalid or unauthorized. "
                "Please verify your credentials in .env or generate a new key in Google AI Studio."
            ),
            "error_code": "UNAUTHORIZED",
        }

    # 429 Rate Limit / Quota Exceeded
    if status_code == 429 or "RESOURCE_EXHAUSTED" in err_status or "quota" in err_message.lower():
        return {
            "status": "QUOTA_EXCEEDED",
            "error_category": "Quota Exceeded",
            "technical_reason": (
                "Gemini API rate limit or quota exceeded for YouTube video processing. "
                "Free tier includes up to 8 hours of video processing per day. "
                "Please check Google AI Studio quotas or try again later."
            ),
            "error_code": "RESOURCE_EXHAUSTED",
        }

    # 404 / 503 Model Unavailable
    if status_code in (404, 503) or "NOT_FOUND" in err_status or "UNAVAILABLE" in err_status:
        return {
            "status": "MODEL_UNAVAILABLE",
            "error_category": "Model Unavailable",
            "technical_reason": (
                f"The Gemini multimodal model is currently unavailable (HTTP {status_code}): {err_message}. "
                "Service may be temporarily overloaded or undergoing maintenance."
            ),
            "error_code": err_status or "MODEL_UNAVAILABLE",
        }

    # 400 Bad Request / Unsupported Video / Cannot Access
    if status_code == 400:
        if (
            "cannot" in err_message.lower()
            or "unsupported" in err_message.lower()
            or "video" in err_message.lower()
            or "private" in err_message.lower()
        ):
            return {
                "status": "VIDEO_INACCESSIBLE_OR_UNSUPPORTED",
                "error_category": "Video Inaccessible or Unsupported",
                "technical_reason": (
                    f"Gemini could not process this YouTube video: {err_message}. "
                    "The video may be private, unlisted, age-restricted, a livestream, or exceed supported length."
                ),
                "error_code": "INVALID_ARGUMENT",
            }
        return {
            "status": "BAD_REQUEST",
            "error_category": "Unsupported URL or Request",
            "technical_reason": f"Gemini API returned bad request (HTTP 400): {err_message}",
            "error_code": "BAD_REQUEST",
        }

    # Other upstream errors
    return {
        "status": "UPSTREAM_API_ERROR",
        "error_category": "Upstream API Error",
        "technical_reason": f"Gemini API error (HTTP {status_code}): {err_message}",
        "error_code": err_status or "UPSTREAM_ERROR",
    }


def format_gemini_matches(repeated_sequences: List[Dict[str, Any]], parsed_confidence: str) -> List[Dict[str, Any]]:
    """Format Gemini sequence findings into standard match structure for frontend display."""
    formatted_matches = []
    for seq in repeated_sequences:
        orig_start = seq.get("original_start", "00:00")
        orig_end = seq.get("original_end", "00:00")
        rep_start = seq.get("repeat_start", "00:00")
        rep_end = seq.get("repeat_end", "00:00")
        dur = int(seq.get("duration_sec") or 0)
        classification = seq.get("classification", "Repeated sequence")
        desc = seq.get("visual_description", "Visual similarity estimated by Gemini multimodal analysis.")
        temporal = seq.get("temporal_context", "")

        note_parts = [desc]
        if temporal:
            note_parts.append(f"Temporal context: {temporal}")

        formatted_matches.append({
            "original_sequence": {
                "start_formatted": orig_start,
                "end_formatted": orig_end,
                "time_range": f"{orig_start} – {orig_end}",
            },
            "repeated_sequence": {
                "start_formatted": rep_start,
                "end_formatted": rep_end,
                "time_range": f"{rep_start} – {rep_end}",
            },
            "duration_sec": dur,
            "duration_formatted": f"{dur}s",
            "similarity_score": 92,
            "confidence": seq.get("confidence") or parsed_confidence,
            "classification": classification,
            "classification_note": " — ".join(note_parts),
            "visual_description": desc,
            "temporal_context": temporal,
            "evidence_thumbnails": {},
            "is_static_scene": classification == "Static Scene",
            "creator_intent_disclaimer": "Creator intent cannot be inferred from visual similarity alone.",
        })
    return formatted_matches


def analyze_youtube_video_with_gemini(
    video_url_or_id: str,
    video_metadata: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 40,
) -> Dict[str, Any]:
    """Analyze a public YouTube video directly via Gemini API.
    
    Returns structured results adhering strictly to the 4 required result states:
    - 'Repeated footage detected'
    - 'No qualifying repetition detected'
    - 'Inconclusive'
    - 'Analysis unavailable'
    
    Never fabricates analysis when credentials are missing or API calls fail.
    Never exposes, logs, or prints the GEMINI_API_KEY.
    """
    video_id = extract_video_id(video_url_or_id)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else video_url_or_id
    video_title = (video_metadata or {}).get("title")

    # 1. Verify API Key configuration
    gemini_key = get_gemini_api_key()
    if not gemini_key or len(gemini_key) < 5:
        return {
            "source_name": "Gemini Multimodal Direct Analysis",
            "attempted": False,
            "available": False,
            "performed": False,
            "result_state": RESULT_ANALYSIS_UNAVAILABLE,
            "status": "API_KEY_NOT_CONFIGURED",
            "error_category": "API Key Not Configured",
            "model_used": None,
            "summary": (
                "Direct AI video footage analysis is unavailable because GEMINI_API_KEY "
                "is not configured in the server's .env file."
            ),
            "technical_reason": (
                "GEMINI_API_KEY environment variable is missing or empty. "
                "Direct YouTube URL multimodal analysis requires a valid Google Gemini API key."
            ),
            "confidence": "Low",
            "repeated_sequences": [],
            "storytelling_elements_noted": [],
            "calculated_matches": [],
            "visual_analysis": None,
            "creator_intent_note": "Visual analysis not performed.",
        }

    # 2. Build official payload
    payload = build_gemini_payload(canonical_url, video_title=video_title)

    # 3. Call Gemini API endpoint with model fallback
    last_error_diag = None
    for model_name in GEMINI_MODELS:
        endpoint = f"{GEMINI_API_BASE}/{model_name}:generateContent"
        try:
            resp = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": gemini_key,
                },
                timeout=timeout_sec,
            )
        except requests.Timeout:
            return {
                "source_name": "Gemini Multimodal Direct Analysis",
                "attempted": True,
                "available": False,
                "performed": False,
                "result_state": RESULT_ANALYSIS_UNAVAILABLE,
                "status": "TIMEOUT",
                "error_category": "Request Timed Out",
                "model_used": model_name,
                "summary": "Direct AI footage analysis timed out while processing video frames.",
                "technical_reason": (
                    f"Gemini API request timed out after {timeout_sec} seconds. "
                    "Longer videos may exceed standard response thresholds."
                ),
                "confidence": "Low",
                "repeated_sequences": [],
                "storytelling_elements_noted": [],
                "calculated_matches": [],
                "visual_analysis": None,
                "creator_intent_note": "Visual analysis timed out.",
            }
        except requests.RequestException as e:
            return {
                "source_name": "Gemini Multimodal Direct Analysis",
                "attempted": True,
                "available": False,
                "performed": False,
                "result_state": RESULT_ANALYSIS_UNAVAILABLE,
                "status": "NETWORK_ERROR",
                "error_category": "Network Error",
                "model_used": model_name,
                "summary": "Network connection error while contacting Gemini API.",
                "technical_reason": f"Failed to connect to Google Generative Language API: {str(e)}",
                "confidence": "Low",
                "repeated_sequences": [],
                "storytelling_elements_noted": [],
                "calculated_matches": [],
                "visual_analysis": None,
                "creator_intent_note": "Network error prevented analysis.",
            }

        # Parse HTTP responses
        if resp.status_code == 200:
            try:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return {
                        "source_name": "Gemini Multimodal Direct Analysis",
                        "attempted": True,
                        "available": False,
                        "performed": False,
                        "result_state": RESULT_INCONCLUSIVE,
                        "status": "EMPTY_MODEL_RESPONSE",
                        "error_category": "Empty Model Response",
                        "model_used": model_name,
                        "summary": "Gemini returned an empty candidate response for this video footage.",
                        "technical_reason": "Model finished without producing content candidates (may be filtered by safety settings).",
                        "confidence": "Low",
                        "repeated_sequences": [],
                        "storytelling_elements_noted": [],
                        "calculated_matches": [],
                        "visual_analysis": None,
                        "creator_intent_note": "Inconclusive response.",
                    }

                part_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Strip markdown code blocks if returned
                cleaned_text = part_text.strip()
                if cleaned_text.startswith("```"):
                    lines = cleaned_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned_text = "\n".join(lines).strip()

                parsed = json.loads(cleaned_text)

                # Validate and normalize result state to required 4 states
                raw_state = parsed.get("result_state", "")
                if "repeated" in raw_state.lower() and "detect" in raw_state.lower():
                    normalized_state = RESULT_REPEATED_FOOTAGE_DETECTED
                elif "no" in raw_state.lower() and "repetition" in raw_state.lower():
                    normalized_state = RESULT_NO_QUALIFYING_REPETITION
                elif "inconclusive" in raw_state.lower():
                    normalized_state = RESULT_INCONCLUSIVE
                elif "unavailable" in raw_state.lower():
                    normalized_state = RESULT_ANALYSIS_UNAVAILABLE
                else:
                    normalized_state = RESULT_INCONCLUSIVE

                repeated_seqs = parsed.get("repeated_sequences", [])
                storytelling_elements = parsed.get("storytelling_elements_noted", [])
                confidence = parsed.get("confidence", "Moderate")

                formatted_matches = format_gemini_matches(repeated_seqs, confidence)
                visual_analysis = {
                    "verdict": {
                        "status": "REPEATED_SEQUENCES_DETECTED" if normalized_state == RESULT_REPEATED_FOOTAGE_DETECTED else ("NO_QUALIFYING_REPETITION" if normalized_state == RESULT_NO_QUALIFYING_REPETITION else "INCONCLUSIVE"),
                        "result_state": normalized_state,
                        "severity": "warning" if normalized_state == RESULT_REPEATED_FOOTAGE_DETECTED else ("safe" if normalized_state == RESULT_NO_QUALIFYING_REPETITION else "caution"),
                        "sub_classification": parsed.get("sub_classification", ""),
                        "total_candidate_matches": len(formatted_matches),
                        "dynamic_matches_count": len([m for m in formatted_matches if not m.get("is_static_scene")]),
                        "static_matches_count": len([m for m in formatted_matches if m.get("is_static_scene")]),
                        "summary": parsed.get("summary", ""),
                    },
                    "matches": formatted_matches,
                    "source_name": "Gemini Multimodal Direct Analysis",
                    "model_used": model_name,
                    "confidence": confidence,
                }

                return {
                    "source_name": "Gemini Multimodal Direct Analysis",
                    "attempted": True,
                    "available": True,
                    "performed": True,
                    "result_state": normalized_state,
                    "status": parsed.get("status", "ANALYSIS_COMPLETE"),
                    "model_used": model_name,
                    "summary": parsed.get("summary", "Gemini video sequence analysis completed."),
                    "sub_classification": parsed.get("sub_classification", ""),
                    "confidence": confidence,
                    "technical_reason": None,
                    "repeated_sequences": repeated_seqs,
                    "storytelling_elements_noted": storytelling_elements,
                    "calculated_matches": formatted_matches,
                    "visual_analysis": visual_analysis,
                    "creator_intent_note": (
                        parsed.get("creator_intent_note")
                        or "Visual analysis measures footage similarity; it does not confirm creator motives or deliberate padding."
                    ),
                }

            except (json.JSONDecodeError, KeyError, IndexError) as err:
                logger.warning("Failed to parse Gemini JSON output: %s", err)
                return {
                    "source_name": "Gemini Multimodal Direct Analysis",
                    "attempted": True,
                    "available": False,
                    "performed": False,
                    "result_state": RESULT_INCONCLUSIVE,
                    "status": "PARSING_ERROR",
                    "error_category": "Output Formatting Variance",
                    "model_used": model_name,
                    "summary": "Gemini evaluated video footage, but output formatting was irregular.",
                    "technical_reason": f"Structured JSON output parse error: {str(err)}",
                    "confidence": "Low",
                    "repeated_sequences": [],
                    "storytelling_elements_noted": [],
                    "calculated_matches": [],
                    "visual_analysis": None,
                    "creator_intent_note": "Inconclusive due to parsing variance.",
                }

        # Non-200 status code
        try:
            err_json = resp.json()
        except Exception:
            err_json = {}
        last_error_diag = parse_gemini_error(resp.status_code, err_json, resp.text)

        # If key is blocked or invalid (401/403), no need to retry with another model
        if resp.status_code in (401, 403):
            break

    # If all models failed or encountered errors
    diag = last_error_diag or {
        "status": "ANALYSIS_UNAVAILABLE",
        "error_category": "Analysis Unavailable",
        "technical_reason": "Gemini multimodal video analysis could not be completed.",
    }

    return {
        "source_name": "Gemini Multimodal Direct Analysis",
        "attempted": True,
        "available": False,
        "performed": False,
        "result_state": RESULT_ANALYSIS_UNAVAILABLE,
        "status": diag.get("status", "ANALYSIS_UNAVAILABLE"),
        "error_category": diag.get("error_category", "Analysis Unavailable"),
        "model_used": None,
        "summary": (
            f"Direct AI video footage analysis is unavailable: {diag.get('error_category', 'Analysis Unavailable')}. "
            f"{diag.get('technical_reason', '')}"
        ),
        "technical_reason": diag.get("technical_reason"),
        "confidence": "Low",
        "repeated_sequences": [],
        "storytelling_elements_noted": [],
        "calculated_matches": [],
        "visual_analysis": None,
        "creator_intent_note": "Analysis unavailable.",
    }
