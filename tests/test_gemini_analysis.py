"""Comprehensive unit and integration tests for LoopLens Gemini video visual analysis.

Tests cover:
1. URL validation and extraction across YouTube formats.
2. Official Gemini API payload structure (mimeType="video/*", fileData, prompt).
3. The four required outcomes:
   - Repeated footage detected
   - No qualifying repetition detected
   - Inconclusive
   - Analysis unavailable
4. Error sanitization and specific diagnostic categories:
   - API key blocked or restricted
   - Model unavailable
   - Unsupported URL / video inaccessible
   - Quota exceeded
   - Invalid or missing API key
   - Request timed out and network errors
5. Fallback behavior and independent community comment scanning.
6. Strict key secrecy (GEMINI_API_KEY is never leaked or logged).
"""

import json
import unittest
from unittest.mock import MagicMock, patch
import requests

from backend.analyzer import (
    RESULT_ANALYSIS_UNAVAILABLE,
    RESULT_INCONCLUSIVE,
    RESULT_NO_QUALIFYING_REPETITION,
    RESULT_REPEATED_FOOTAGE_DETECTED,
    analyze_comments,
    extract_video_id,
)
from backend.gemini_service import (
    ANALYSIS_SYSTEM_PROMPT,
    GEMINI_MODELS,
    build_gemini_payload,
    format_gemini_matches,
    parse_gemini_error,
    analyze_youtube_video_with_gemini,
)


class TestYouTubeUrlValidation(unittest.TestCase):
    """Test validation and canonicalization of YouTube URLs."""

    def test_standard_watch_urls(self):
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
        ]
        for url in urls:
            self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ", f"Failed for {url}")

    def test_shortened_and_mobile_urls(self):
        urls = [
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ?t=10",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in urls:
            self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ", f"Failed for {url}")

    def test_shorts_and_embed_urls(self):
        urls = [
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ]
        for url in urls:
            self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ", f"Failed for {url}")

    def test_invalid_urls(self):
        invalid = [
            "https://vimeo.com/12345678",
            "https://example.com/watch?v=dQw4w9WgXcQ",
            "not-a-url",
            "",
            "   ",
            "https://youtube.com/channel/UC123456",
        ]
        for url in invalid:
            self.assertIsNone(extract_video_id(url), f"Should be None for {url}")


class TestGeminiPayloadAndConfig(unittest.TestCase):
    """Test official Gemini video-understanding request format."""

    def test_official_video_payload_structure(self):
        target_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        payload = build_gemini_payload(target_url)

        self.assertIn("contents", payload)
        self.assertEqual(len(payload["contents"]), 1)
        content_item = payload["contents"][0]
        self.assertEqual(content_item.get("role"), "user")

        parts = content_item.get("parts", [])
        self.assertEqual(len(parts), 2)

        # First part: fileData with video/* mimeType and official canonical YouTube URL
        file_part = parts[0]
        self.assertIn("fileData", file_part)
        self.assertEqual(file_part["fileData"]["mimeType"], "video/*")
        self.assertEqual(file_part["fileData"]["fileUri"], target_url)

        # Second part: Text prompt
        text_part = parts[1]
        self.assertIn("text", text_part)
        self.assertIn("repeated", text_part["text"].lower())
        self.assertIn("timestamp", text_part["text"].lower())

        # System instruction and generationConfig
        self.assertIn("systemInstruction", payload)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_prompt_demands_distinguishing_ordinary_storytelling(self):
        target_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        payload = build_gemini_payload(target_url)
        full_text = payload["contents"][0]["parts"][1]["text"] + ANALYSIS_SYSTEM_PROMPT

        # Must explicitly mention distinguishing recaps, montages, flashbacks, credits
        self.assertIn("flashback", full_text.lower())
        self.assertIn("recap", full_text.lower())
        self.assertIn("montage", full_text.lower())
        self.assertIn("estimate", full_text.lower())

    def test_models_configured(self):
        self.assertIn("gemini-2.0-flash", GEMINI_MODELS)
        self.assertIn("gemini-1.5-flash", GEMINI_MODELS)


class TestGeminiOutcomes(unittest.TestCase):
    """Test the 4 required outcomes when calling Gemini."""

    def _make_gemini_response(self, inner_json_dict, status_code=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": json.dumps(inner_json_dict)}
                        ]
                    }
                }
            ]
        }
        return mock_resp

    @patch("backend.gemini_service.requests.post")
    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    def test_outcome_repeated_footage_detected(self, mock_key, mock_post):
        gemini_mock_inner = {
            "result_state": "Repeated footage detected",
            "status": "REPEATED_FOOTAGE_DETECTED",
            "confidence": "High",
            "summary": "Detected a 3-minute sequence copied and appended at the end of the video.",
            "sub_classification": "Possible loop-padding",
            "repeated_sequences": [
                {
                    "original_start": "01:00",
                    "original_end": "04:00",
                    "repeat_start": "10:00",
                    "repeat_end": "13:00",
                    "duration_sec": 180,
                    "classification": "Possible loop-padding",
                    "temporal_context": "Tutorial steps 1-4 repeated verbatim.",
                    "visual_description": "Repeated sequence with high visual similarity.",
                    "creator_intent_disclaimer": "Creator intent cannot be inferred from visual similarity alone.",
                }
            ],
            "storytelling_elements_noted": [],
            "creator_intent_note": "Visual similarity measures frame correlation, not motives.",
        }
        mock_post.return_value = self._make_gemini_response(gemini_mock_inner)

        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(res["performed"])
        self.assertEqual(res["result_state"], RESULT_REPEATED_FOOTAGE_DETECTED)
        self.assertEqual(res["status"], "REPEATED_FOOTAGE_DETECTED")
        self.assertEqual(len(res["calculated_matches"]), 1)
        self.assertEqual(res["confidence"], "High")

    @patch("backend.gemini_service.requests.post")
    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    def test_outcome_no_qualifying_repetition(self, mock_key, mock_post):
        gemini_mock_inner = {
            "result_state": "No qualifying repetition detected",
            "status": "NO_QUALIFYING_REPETITION",
            "confidence": "High",
            "summary": "Linear narrative progression with no duplicated sequences.",
            "sub_classification": "Clean footage",
            "repeated_sequences": [],
            "storytelling_elements_noted": [],
            "creator_intent_note": "No repeated sequences detected.",
        }
        mock_post.return_value = self._make_gemini_response(gemini_mock_inner)

        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(res["performed"])
        self.assertEqual(res["result_state"], RESULT_NO_QUALIFYING_REPETITION)
        self.assertEqual(res["status"], "NO_QUALIFYING_REPETITION")
        self.assertEqual(len(res["calculated_matches"]), 0)

    @patch("backend.gemini_service.requests.post")
    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    def test_outcome_inconclusive_creative_montage(self, mock_key, mock_post):
        gemini_mock_inner = {
            "result_state": "Inconclusive",
            "status": "INCONCLUSIVE",
            "confidence": "Moderate",
            "summary": "Brief 5-second clip reappears during a deliberate recap montage.",
            "sub_classification": "Normal storytelling",
            "repeated_sequences": [
                {
                    "original_start": "00:30",
                    "original_end": "00:35",
                    "repeat_start": "08:10",
                    "repeat_end": "08:15",
                    "duration_sec": 5,
                    "classification": "Flashback / Recap",
                    "temporal_context": "Recap montage sequence",
                    "visual_description": "Brief artistic flashback.",
                    "creator_intent_disclaimer": "Creator intent cannot be inferred.",
                }
            ],
            "storytelling_elements_noted": ["Brief 5-second recap observed before concluding chapter."],
            "creator_intent_note": "Normal storytelling elements noted.",
        }
        mock_post.return_value = self._make_gemini_response(gemini_mock_inner)

        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(res["performed"])
        self.assertEqual(res["result_state"], RESULT_INCONCLUSIVE)
        self.assertEqual(res["status"], "INCONCLUSIVE")

    @patch("backend.gemini_service.requests.post")
    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    def test_outcome_analysis_unavailable(self, mock_key, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"code": 500, "message": "Internal error"}}
        mock_resp.text = json.dumps(mock_resp.json.return_value)
        mock_post.return_value = mock_resp

        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertFalse(res["performed"])
        self.assertEqual(res["result_state"], RESULT_ANALYSIS_UNAVAILABLE)
        self.assertIn(res["status"], ("ANALYSIS_UNAVAILABLE", "UPSTREAM_API_ERROR"))


class TestGeminiErrorSanitization(unittest.TestCase):
    """Test that errors return transparent sanitized diagnostic categories instead of generic failures."""

    def test_error_403_api_key_service_blocked(self):
        body = {
            "error": {
                "code": 403,
                "message": "Requests to this API generativelanguage.googleapis.com method google.ai.generativelanguage.v1beta.GenerativeService.GenerateContent are blocked.",
                "status": "PERMISSION_DENIED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_SERVICE_BLOCKED",
                        "domain": "googleapis.com",
                    }
                ],
            }
        }
        err = parse_gemini_error(403, body, json.dumps(body))
        self.assertEqual(err["error_category"], "API Key Blocked or Restricted")
        self.assertEqual(err["error_code"], "FORBIDDEN")
        self.assertIn("Google Cloud Console", err["technical_reason"])
        self.assertIn("generativelanguage.googleapis.com", err["technical_reason"])

    def test_error_400_invalid_api_key(self):
        body = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        }
        err = parse_gemini_error(400, body, json.dumps(body))
        self.assertEqual(err["error_category"], "Invalid API Key")
        self.assertIn("valid", err["technical_reason"].lower())

    def test_error_404_model_unavailable(self):
        body = {
            "error": {
                "code": 404,
                "message": "models/gemini-2.0-flash is not found for API version v1beta",
                "status": "NOT_FOUND",
            }
        }
        err = parse_gemini_error(404, body, json.dumps(body))
        self.assertEqual(err["error_category"], "Model Unavailable")

    def test_error_429_quota_exceeded(self):
        body = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
        err = parse_gemini_error(429, body, json.dumps(body))
        self.assertEqual(err["error_category"], "Quota Exceeded")

    def test_error_video_inaccessible_or_unsupported(self):
        body = {
            "error": {
                "code": 400,
                "message": "Failed to load video from YouTube URL or format unsupported.",
                "status": "INVALID_ARGUMENT",
            }
        }
        err = parse_gemini_error(400, body, json.dumps(body))
        self.assertEqual(err["error_category"], "Video Inaccessible or Unsupported")

    def test_missing_api_key(self):
        with patch("backend.gemini_service.get_gemini_api_key", return_value=""):
            res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            self.assertFalse(res["performed"])
            self.assertEqual(res["error_category"], "API Key Not Configured")
            self.assertEqual(res["result_state"], RESULT_ANALYSIS_UNAVAILABLE)

    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    @patch("backend.gemini_service.requests.post")
    def test_request_timeout_categorization(self, mock_post, mock_key):
        mock_post.side_effect = requests.Timeout("Request timed out after 40 seconds")
        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertFalse(res["performed"])
        self.assertEqual(res["error_category"], "Request Timed Out")
        self.assertEqual(res["result_state"], RESULT_ANALYSIS_UNAVAILABLE)

    @patch("backend.gemini_service.get_gemini_api_key", return_value="TEST_KEY")
    @patch("backend.gemini_service.requests.post")
    def test_network_connection_error_categorization(self, mock_post, mock_key):
        mock_post.side_effect = requests.ConnectionError("Could not connect to host")
        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertFalse(res["performed"])
        self.assertEqual(res["error_category"], "Network Error")
        self.assertEqual(res["result_state"], RESULT_ANALYSIS_UNAVAILABLE)


class TestPipelineFallbackAndCommunitySeparation(unittest.TestCase):
    """Test that community scanning is separate and never falsely claims no loops."""

    @patch("backend.analyzer.analyze_youtube_video_with_gemini")
    def test_gemini_repetition_overrides_inconclusive_comments(self, mock_gemini):
        # Gemini finds repetition
        mock_gemini.return_value = {
            "performed": True,
            "result_state": RESULT_REPEATED_FOOTAGE_DETECTED,
            "status": "REPEATED_FOOTAGE_DETECTED",
            "sub_classification": "Repeated video section",
            "confidence": "High",
            "model_used": "gemini-2.0-flash",
            "summary": "Visual repetition detected via Gemini analysis.",
            "calculated_matches": [
                {
                    "original_sequence": {"start_time": "00:10", "end_time": "01:00"},
                    "repeated_sequence": {"start_time": "05:00", "end_time": "05:50"},
                    "similarity_score": 0.95,
                    "description": "Repeated segment",
                }
            ],
            "visual_analysis": {"matches": [{"id": 1}]},
        }

        # Zero repetition in comments
        clean_comments = [
            {"comment_id": "c1", "author": "Alice", "text": "Loved this video!", "like_count": 5},
        ]

        report = analyze_comments(clean_comments, comments_disabled=False, video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(report["direct_footage_analysis_performed"])
        self.assertEqual(report["result_state"], RESULT_REPEATED_FOOTAGE_DETECTED)
        self.assertEqual(len(report["visual_analysis"]["matches"]), 1)

    @patch("backend.analyzer.analyze_youtube_video_with_gemini")
    def test_gemini_failure_preserves_error_and_does_not_claim_no_loop(self, mock_gemini):
        # Gemini fails with 403 blocked
        mock_gemini.return_value = {
            "performed": False,
            "result_state": RESULT_ANALYSIS_UNAVAILABLE,
            "status": "API_KEY_BLOCKED_OR_RESTRICTED",
            "error_category": "API Key Blocked or Restricted",
            "technical_reason": "Google Cloud Console key restrictions block generativelanguage.googleapis.com",
            "summary": "Gemini API rejected request (HTTP 403).",
        }

        # Empty comments (no loop reports from community)
        report = analyze_comments([], comments_disabled=False, video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertFalse(report["direct_footage_analysis_performed"])
        self.assertEqual(report["result_state"], RESULT_ANALYSIS_UNAVAILABLE)

        footage = report["evidence_pipeline"]["direct_footage"]
        self.assertFalse(footage["performed"])
        self.assertEqual(footage["error_category"], "API Key Blocked or Restricted")
        self.assertIn("absence of viewer comments cannot be used as evidence", report["summary"])


class TestApiKeySecurity(unittest.TestCase):
    """Test that API keys are strictly protected from leakage."""

    SECRET_KEY = "AIzaSyFakeSecretKeyForTestingLeakage_12345"

    @patch("backend.gemini_service.get_gemini_api_key")
    @patch("backend.gemini_service.requests.post")
    def test_key_never_leaked_in_output_or_error_data(self, mock_post, mock_key):
        mock_key.return_value = self.SECRET_KEY

        # Mock an error response
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.json.return_value = {"error": {"code": 403, "message": "Blocked key"}}
        mock_resp.text = json.dumps(mock_resp.json.return_value)
        mock_post.return_value = mock_resp

        res = analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        res_str = json.dumps(res)

        self.assertNotIn(self.SECRET_KEY, res_str, "SECRET KEY WAS LEAKED IN GEMINI RESULT!")

    @patch("backend.gemini_service.get_gemini_api_key")
    @patch("backend.gemini_service.requests.post")
    def test_key_passed_in_header_not_query_params(self, mock_post, mock_key):
        mock_key.return_value = self.SECRET_KEY

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "result_state": "No qualifying repetition detected",
                                    "confidence": "High",
                                    "summary": "Clean video",
                                    "repeated_sequences": [],
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        analyze_youtube_video_with_gemini("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Inspect the requests.post call args
        called_args, called_kwargs = mock_post.call_args
        called_url = called_args[0] if called_args else called_kwargs.get("url", "")
        # The URL must NOT have the API key in its query parameters
        self.assertNotIn("key=", called_url)
        self.assertNotIn(self.SECRET_KEY, called_url)

        # The key must be in the x-goog-api-key header
        headers = called_kwargs.get("headers", {})
        self.assertEqual(headers.get("x-goog-api-key"), self.SECRET_KEY)


if __name__ == "__main__":
    unittest.main()
