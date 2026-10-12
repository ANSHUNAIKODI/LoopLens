import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app


class TestLoopLensAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_root_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("LoopLens", response.text)
        self.assertIn("Your time is valuable. Check before you watch.", response.text)
        self.assertIn("video-url-input", response.text)
        self.assertIn("analyze-btn", response.text)

    def test_get_static_assets(self):
        css_res = self.client.get("/styles.css")
        self.assertEqual(css_res.status_code, 200)
        self.assertIn("--mint-400", css_res.text)

        js_res = self.client.get("/app.js")
        self.assertEqual(js_res.status_code, 200)
        self.assertIn("performAnalysis", js_res.text)

    def test_get_api_status(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "online")
        self.assertIn("api_key_configured", data)
        # Verify API key is NOT leaked
        self.assertNotIn("YOUTUBE_API_KEY", data)
        self.assertNotIn("key", data)

    def test_analyze_empty_url(self):
        response = self.client.post("/api/analyze", json={"url": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)

    def test_analyze_invalid_url(self):
        response = self.client.post("/api/analyze", json={"url": "https://example.com/not-youtube"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"]["error_code"], "INVALID_URL")

    def test_analyze_missing_api_key_explanation(self):
        # When no API key is configured in test environment
        # LoopLens must return a clear explanation rather than fake results
        response = self.client.post("/api/analyze", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        # Should return 503 with explanation if no API key is set
        data = response.json()
        if not data.get("success"):
            self.assertEqual(data.get("error_code"), "API_KEY_REQUIRED")
            self.assertIn("YOUTUBE_API_KEY", data.get("message"))
            self.assertEqual(data.get("video_id"), "dQw4w9WgXcQ")

    def test_analyze_video_missing_permission(self):
        fake_bytes = b"fake video bytes"
        response = self.client.post(
            "/api/analyze-video",
            files={"file": ("test.mp4", fake_bytes, "video/mp4")},
            data={"permission_confirmed": "false"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"]["error_code"], "PERMISSION_REQUIRED")

    def test_analyze_video_invalid_file_extension(self):
        fake_bytes = b"fake text file"
        response = self.client.post(
            "/api/analyze-video",
            files={"file": ("document.txt", fake_bytes, "text/plain")},
            data={"permission_confirmed": "true"},
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"]["error_code"], "INVALID_FILE_TYPE")

    def test_analyze_sample_video_endpoint_repeated(self):
        response = self.client.post(
            "/api/analyze-sample-video",
            data={"sample_type": "repeated", "sample_interval_sec": 1.0, "min_sequence_sec": 2.5},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("is_prototype"))
        self.assertEqual(data.get("evidence_source"), "LOCAL_FRAME_PERCEPTUAL_ANALYSIS")
        self.assertEqual(data["verdict"]["status"], "REPEATED_SEQUENCES_DETECTED")
        self.assertGreaterEqual(len(data["matches"]), 1)
        # Verify matched evidence thumbnails and timestamps
        first_match = data["matches"][0]
        self.assertIn("original_sequence", first_match)
        self.assertIn("repeated_sequence", first_match)
        self.assertIn("evidence_thumbnails", first_match)
        self.assertIn("prototype_disclaimer", data)

    def test_analyze_sample_video_endpoint_static(self):
        response = self.client.post(
            "/api/analyze-sample-video",
            data={"sample_type": "static", "sample_interval_sec": 1.0, "min_sequence_sec": 2.0},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        # Static scene should NOT be reported as dynamic repeated footage
        self.assertNotEqual(data["verdict"]["status"], "REPEATED_SEQUENCES_DETECTED")

    def test_analyze_video_upload_endpoint_success(self):
        import tempfile
        from pathlib import Path
        from backend.video_frame_analyzer import generate_permitted_sample_video

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = Path(tmp_dir) / "sample_test.mp4"
            generate_permitted_sample_video(sample_file)
            
            with open(sample_file, "rb") as f:
                response = self.client.post(
                    "/api/analyze-video",
                    files={"file": ("sample_test.mp4", f, "video/mp4")},
                    data={
                        "permission_confirmed": "true",
                        "sample_interval_sec": "1.0",
                        "min_sequence_sec": "2.5",
                    },
                )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("evidence_source"), "LOCAL_FRAME_PERCEPTUAL_ANALYSIS")
            self.assertEqual(data["verdict"]["status"], "REPEATED_SEQUENCES_DETECTED")
            self.assertGreaterEqual(len(data["matches"]), 1)

    def test_analyze_pipeline_evidence_structure(self):
        # Test that analyze endpoint returns full 3-source pipeline and never claims direct frame analysis occurred
        response = self.client.post("/api/analyze", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        data = response.json()
        if data.get("success"):
            analysis = data["analysis"]
            self.assertIn("evidence_pipeline", analysis)
            self.assertFalse(analysis["direct_footage_analysis_performed"])
            pipeline = analysis["evidence_pipeline"]
            self.assertIn("metadata", pipeline)
            self.assertIn("community_comments", pipeline)
            self.assertIn("direct_footage", pipeline)
            self.assertFalse(pipeline["direct_footage"]["performed"])
            self.assertIn(pipeline["direct_footage"]["status"], ("UNAVAILABLE_FOR_REMOTE_YOUTUBE_URL", "API_KEY_BLOCKED_OR_RESTRICTED", "ANALYSIS_UNAVAILABLE", "API_KEY_NOT_CONFIGURED"))
            self.assertEqual(len(pipeline["direct_footage"]["calculated_matches"]), 0)


    @patch("backend.main.is_api_key_configured", return_value=True)
    @patch("backend.main.fetch_video_metadata")
    @patch("backend.main.fetch_video_comments")
    def test_analyze_endpoint_inconclusive_comments_result_state(
        self, mock_comments, mock_metadata, mock_key
    ):
        mock_metadata.return_value = {
            "video_id": "testinconclu",
            "title": "A Standard Video Tutorial",
            "channel_title": "Test Channel",
            "duration_formatted": "10m 0s",
            "duration_seconds": 600,
            "description": "Standard description",
            "view_count": 1000,
            "comment_count": 2,
        }
        mock_comments.return_value = (
            [
                {"comment_id": "c1", "author": "User A", "text": "Very helpful thanks", "like_count": 1},
                {"comment_id": "c2", "author": "User B", "text": "Nice explanation", "like_count": 2},
            ],
            False,
            None,
        )

        response = self.client.post("/api/analyze", json={"url": "https://www.youtube.com/watch?v=testinconclu"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        analysis = data["analysis"]
        self.assertIn(analysis["result_state"], ("Analysis unavailable", "Footage analysis unavailable"))
        self.assertFalse(analysis["direct_footage_analysis_performed"])
        self.assertIn("absence of viewer comments cannot be used as evidence", analysis["notice"])
        self.assertTrue(analysis["evidence_pipeline"]["direct_footage"]["attempted"])
        self.assertFalse(analysis["evidence_pipeline"]["direct_footage"]["performed"])

    @patch("backend.main.is_api_key_configured", return_value=True)
    @patch("backend.main.fetch_video_metadata")
    @patch("backend.main.fetch_video_comments")
    def test_analyze_endpoint_with_authorized_footage_source(
        self, mock_comments, mock_metadata, mock_key
    ):
        from pathlib import Path
        from backend.video_frame_analyzer import generate_permitted_sample_video

        test_id = "testauthr01"
        authorized_file = Path(__file__).resolve().parent.parent / "backend" / "authorized_footage" / f"{test_id}.mp4"
        generate_permitted_sample_video(authorized_file)

        try:
            mock_metadata.return_value = {
                "video_id": test_id,
                "title": "Authorized Video With Repeated Scene",
                "channel_title": "Creator Channel",
                "duration_formatted": "12s",
                "duration_seconds": 12,
                "description": "Creator upload",
                "view_count": 50,
                "comment_count": 0,
            }
            mock_comments.return_value = ([], False, None)

            response = self.client.post("/api/analyze", json={"url": f"https://www.youtube.com/watch?v={test_id}"})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            analysis = data["analysis"]
            self.assertEqual(analysis["result_state"], "Repeated sequence detected")
            self.assertTrue(analysis["direct_footage_analysis_performed"])
            self.assertIsNotNone(analysis["visual_analysis"])
            self.assertGreaterEqual(len(analysis["visual_analysis"]["matches"]), 1)
        finally:
            if authorized_file.exists():
                authorized_file.unlink()


if __name__ == "__main__":
    unittest.main()



