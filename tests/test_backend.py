import unittest
from backend.youtube_service import extract_video_id, parse_iso8601_duration
from backend.analyzer import analyze_comments, analyze_single_comment


class TestLoopLensBackend(unittest.TestCase):
    def test_extract_video_id(self):
        cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&ab_channel=RickAstley", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("not-a-valid-url", None),
            ("", None),
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), expected)

    def test_parse_iso8601_duration(self):
        cases = [
            ("PT1H23M45S", ("1h 23m 45s", 5025)),
            ("PT15M30S", ("15m 30s", 930)),
            ("PT42S", ("42s", 42)),
            ("PT1H", ("1h", 3600)),
            ("P1DT2H", ("1d 2h", 93600)),
        ]
        for duration_iso, (expected_str, expected_sec) in cases:
            with self.subTest(duration_iso=duration_iso):
                res_str, res_sec = parse_iso8601_duration(duration_iso)
                self.assertEqual(res_str, expected_str)
                self.assertEqual(res_sec, expected_sec)

    def test_analyze_comment_repetition_detection(self):
        sample_comment = {
            "comment_id": "c1",
            "author": "Viewer123",
            "author_avatar": "",
            "text": "Why does the same scene repeat at 03:45? The video just loops the second half!",
            "like_count": 25,
            "published_at": "2026-01-01T00:00:00Z",
        }
        evidence = analyze_single_comment(sample_comment)
        self.assertIsNotNone(evidence)
        self.assertIn("repeated_footage", evidence["categories"])
        self.assertIn("03:45", evidence["matched_timestamps"])
        self.assertEqual(evidence["author"], "Viewer123")
        self.assertEqual(evidence["text"], sample_comment["text"])

    def test_analyze_comments_disabled(self):
        report = analyze_comments([], comments_disabled=True)
        self.assertEqual(report["status"], "COMMENTS_DISABLED")
        self.assertEqual(report["evidence_count"], 0)

    def test_analyze_comments_clean(self):
        comments = [
            {"comment_id": "c1", "author": "Alice", "text": "Great tutorial, thanks for sharing!", "like_count": 5},
            {"comment_id": "c2", "author": "Bob", "text": "Very informative and well edited.", "like_count": 2},
        ]
        report = analyze_comments(comments, comments_disabled=False)
        self.assertEqual(report["status"], "CLEAR")
        self.assertEqual(report["evidence_count"], 0)
        self.assertFalse(report["direct_footage_analysis_performed"])
        self.assertIn("evidence_pipeline", report)
        self.assertIn("direct_footage", report["evidence_pipeline"])
        self.assertFalse(report["evidence_pipeline"]["direct_footage"]["performed"])

    def test_recap_false_positive_reduction(self):
        comments = [
            {"comment_id": "c1", "author": "Alice", "text": "This is just a recap of season 1 for new viewers.", "like_count": 30},
        ]
        meta = {
            "title": "Anime Episode 1 - Season 1 Recap & Season 2 Premiere",
            "description": "00:00 Intro\n01:30 Recap of Season 1\n05:00 Season 2 Ep 1",
            "duration_formatted": "24m",
            "duration_seconds": 1440,
        }
        report = analyze_comments(comments, comments_disabled=False, video_metadata=meta)
        self.assertEqual(report["status"], "INTENTIONAL_RECAP_NOTED")
        self.assertNotEqual(report["severity"], "warning")
        self.assertGreaterEqual(report["recap_reports_count"], 1)

    def test_ambient_loop_metadata_detection(self):
        meta = {
            "title": "Study With Me - 1 Hour Loop Relaxing Rain and Lofi Music",
            "description": "Continuous ambient loop for focus.",
            "duration_formatted": "1h 0m",
            "duration_seconds": 3600,
        }
        report = analyze_comments([], comments_disabled=False, video_metadata=meta)
        self.assertEqual(report["status"], "INTENTIONAL_LOOP_CONTENT")
        self.assertFalse(report["direct_footage_analysis_performed"])

    def test_evidence_pipeline_no_false_footage_claims(self):
        sample_comment = {
            "comment_id": "c1",
            "author": "Viewer123",
            "text": "Why does it repeat at 05:00?",
            "like_count": 10,
        }
        report = analyze_comments([sample_comment], comments_disabled=False)
        # Verify direct footage analysis is NOT claimed
        self.assertFalse(report["direct_footage_analysis_performed"])
        footage = report["evidence_pipeline"]["direct_footage"]
        self.assertFalse(footage["performed"])
        self.assertEqual(footage["status"], "UNAVAILABLE_FOR_REMOTE_YOUTUBE_URL")
        self.assertEqual(len(footage["calculated_matches"]), 0)


    def test_result_state_never_claims_no_loop_on_absence_of_comments(self):
        # 1. Comments disabled
        report_disabled = analyze_comments([], comments_disabled=True)
        self.assertIn(report_disabled["result_state"], ("Analysis unavailable", "Footage analysis unavailable"))
        self.assertNotEqual(report_disabled["result_state"], "No qualifying repeated sequence detected in the analysed footage")
        self.assertIn("absence of viewer comments cannot be used as evidence", report_disabled["summary"])

        # 2. No comments available
        report_no_comments = analyze_comments([], comments_disabled=False)
        self.assertIn(report_no_comments["result_state"], ("Analysis unavailable", "Footage analysis unavailable"))
        self.assertNotEqual(report_no_comments["result_state"], "No qualifying repeated sequence detected in the analysed footage")
        self.assertIn("absence of viewer comments cannot be used as evidence", report_no_comments["summary"])

        # 3. Clean comments with zero repetition reports (inconclusive community evidence)
        clean_comments = [
            {"comment_id": "c1", "author": "Alice", "text": "Loved this video!", "like_count": 5},
            {"comment_id": "c2", "author": "Bob", "text": "Super helpful tips.", "like_count": 2},
        ]
        report_clean = analyze_comments(clean_comments, comments_disabled=False)
        self.assertIn(report_clean["result_state"], ("Analysis unavailable", "Footage analysis unavailable"))
        self.assertNotEqual(report_clean["result_state"], "No qualifying repeated sequence detected in the analysed footage")
        self.assertIn("absence of viewer comments cannot be used as evidence", report_clean["summary"])

    def test_inconclusive_comments_automatically_attempts_footage_analysis(self):
        clean_comments = [
            {"comment_id": "c1", "author": "Alice", "text": "Great editing and pacing!", "like_count": 3},
        ]
        report = analyze_comments(clean_comments, comments_disabled=False, video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        # Visual analysis was attempted automatically
        footage = report["evidence_pipeline"]["direct_footage"]
        self.assertTrue(footage["attempted"])
        self.assertFalse(footage["performed"])
        self.assertIn(report["result_state"], ("Analysis unavailable", "Footage analysis unavailable"))
        self.assertIn("technical_reason", footage)
        self.assertTrue(bool(footage["technical_reason"]))

    def test_authorized_footage_analysis_detects_repetition_automatically(self):
        import tempfile
        from pathlib import Path
        from backend.video_frame_analyzer import generate_permitted_sample_video

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = Path(tmp_dir) / "authorized_loop.mp4"
            generate_permitted_sample_video(sample_file)

            # Inconclusive/empty comments with authorized footage source
            report = analyze_comments(
                comments=[],
                comments_disabled=False,
                video_id="test_vid_123",
                local_video_path=str(sample_file),
            )

            # Actual footage was analysed successfully!
            self.assertTrue(report["direct_footage_analysis_performed"])
            self.assertEqual(report["result_state"], "Repeated sequence detected")
            self.assertIsNotNone(report["visual_analysis"])
            self.assertGreaterEqual(len(report["visual_analysis"]["matches"]), 1)
            first_match = report["visual_analysis"]["matches"][0]
            self.assertIn("original_sequence", first_match)
            self.assertIn("repeated_sequence", first_match)
            self.assertIn("Creator intent cannot be inferred", first_match["intent_disclaimer"])

    def test_authorized_footage_static_scene_inconclusive(self):
        import tempfile
        from pathlib import Path
        from backend.video_frame_analyzer import generate_static_sample_video

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = Path(tmp_dir) / "authorized_static.mp4"
            generate_static_sample_video(sample_file)

            report = analyze_comments(
                comments=[],
                comments_disabled=False,
                video_id="test_static_123",
                local_video_path=str(sample_file),
            )

            self.assertTrue(report["direct_footage_analysis_performed"])
            self.assertEqual(report["result_state"], "Inconclusive")


if __name__ == "__main__":
    unittest.main()

