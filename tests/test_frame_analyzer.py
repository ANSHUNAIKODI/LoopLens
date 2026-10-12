import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.video_frame_analyzer import (
    analyze_video_file,
    compute_dhash,
    compute_frame_signature,
    compare_signatures,
    detect_repeated_sequences,
    extract_sampled_frames,
)


def create_synthetic_test_video(path: Path, fps: int = 10, width: int = 320, height: int = 240) -> None:
    """Generate a 12-second synthetic video with deliberate repeated sequence.
    
    Structure:
    - 0s to 3s (30 frames): Sequence A (moving blue circle with counter)
    - 4s to 7s (40 frames): Sequence B (moving green rectangle)
    - 8s to 11s (40 frames): Sequence A repeated (exact replay of 0s-3s)
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # Pre-render Sequence A frames (30 frames)
    seq_a_frames = []
    for f in range(30):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        img[:, :] = (30, 20, 20)
        # Moving circle
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
        # Moving rectangle
        x = int(220 - f * 4)
        y = int(40 + f * 3)
        cv2.rectangle(img, (x, y), (x + 50, y + 40), (0, 220, 50), -1)
        cv2.putText(img, f"SEQ B: {f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        seq_b_frames.append(img)

    # Write Sequence A (frames 0 to 29 -> 0s to 3s)
    for frame in seq_a_frames:
        writer.write(frame)

    # Write Sequence B (frames 30 to 69 -> 3s to 7s)
    for frame in seq_b_frames:
        writer.write(frame)

    # Write Sequence A repeat (frames 70 to 109 -> 7s to 11s)
    # Loop the 30 frames of seq_a
    for i in range(40):
        writer.write(seq_a_frames[i % 30])

    writer.release()


def create_static_test_video(path: Path, fps: int = 10, width: int = 320, height: int = 240) -> None:
    """Generate an 8-second video with a static title slide (zero motion)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    static_frame = np.zeros((height, width, 3), dtype=np.uint8)
    static_frame[:, :] = (50, 30, 60)
    cv2.putText(static_frame, "STATIC SLIDE", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for _ in range(80):  # 80 frames = 8 seconds
        writer.write(static_frame)

    writer.release()


class TestVideoFrameAnalyzer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = Path(self.temp_dir.name) / "test_loop.mp4"
        self.static_video_path = Path(self.temp_dir.name) / "test_static.mp4"
        create_synthetic_test_video(self.test_video_path)
        create_static_test_video(self.static_video_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_perceptual_hash_identical_frames(self):
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.circle(img1, (50, 50), 20, (255, 0, 0), -1)

        sig1 = compute_frame_signature(img1)
        sig2 = compute_frame_signature(img1.copy())
        sim = compare_signatures(sig1, sig2)
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_perceptual_hash_distinct_frames(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img1, (20, 20), 15, (255, 255, 255), -1)

        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img2, (60, 60), (95, 95), (0, 0, 255), -1)

        sig1 = compute_frame_signature(img1)
        sig2 = compute_frame_signature(img2)
        sim = compare_signatures(sig1, sig2)
        self.assertLess(sim, 0.70)

    def test_repeated_sequence_detection_in_synthetic_video(self):
        result = analyze_video_file(
            video_path=self.test_video_path,
            original_filename="test_loop.mp4",
            sample_interval_sec=1.0,
            min_sequence_sec=2.5,
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["is_prototype"])
        self.assertEqual(result["verdict"]["status"], "REPEATED_SEQUENCES_DETECTED")
        self.assertGreaterEqual(result["verdict"]["dynamic_matches_count"], 1)

        # Check matched ranges
        matches = result["matches"]
        match = matches[0]
        self.assertIn("original_sequence", match)
        self.assertIn("repeated_sequence", match)
        self.assertGreaterEqual(match["similarity_score"], 85.0)

        # Sequence A was at 0s-3s, repeated at 7s-11s
        self.assertLessEqual(match["original_sequence"]["start_sec"], 3.0)
        self.assertGreaterEqual(match["repeated_sequence"]["start_sec"], 6.0)

        # Check evidence thumbnails exist and are non-empty
        thumbs = match["evidence_thumbnails"]
        self.assertTrue(len(thumbs["original_midpoint_b64"]) > 100)
        self.assertTrue(len(thumbs["repeated_midpoint_b64"]) > 100)

    def test_static_scene_filtering(self):
        result = analyze_video_file(
            video_path=self.static_video_path,
            original_filename="test_static.mp4",
            sample_interval_sec=1.0,
            min_sequence_sec=2.0,
        )

        self.assertTrue(result["success"])
        # Should flag as static scene or clear, NOT as dynamic repeated footage
        self.assertNotEqual(result["verdict"]["status"], "REPEATED_SEQUENCES_DETECTED")
        if result["matches"]:
            for match in result["matches"]:
                self.assertTrue(match["is_static_scene"])
                self.assertIsNotNone(match["static_scene_warning"])


if __name__ == "__main__":
    unittest.main()
