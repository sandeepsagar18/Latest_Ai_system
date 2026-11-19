"""
Adversarial Stress & Edge-Case Challenge Suite
=============================================
Probes SmartClassVision under adversarial conditions and boundary edge-cases:
1. Degenerate and adversarial affine landmark inputs (collinear, identical, out-of-bounds, NaNs/Infs).
2. Extreme head roll (0 to 360 deg) and yaw boundaries (-90 to +90 deg), testing boundary thresholds.
3. Candidate roster filtering edge cases: empty roster, single candidate, disjoint non-existent roster,
   and large-scale gallery (up to 1,000 candidates) ambiguity margin behavior.
4. High-throughput batch processing memory stability and leak detection over repeated large batches.

Runner:
    .\\venv\\Scripts\\python.exe tests/test_adversarial_challenge.py
"""

import os
import sys
import time
import gc
import math
import unittest
from pathlib import Path
import numpy as np
import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.detector import align_face_5point, FaceDetector, ARCFACE_REFERENCE_POINTS
from src.recognizer import FaceRecognizer
from src.anti_spoof import evaluate_face_quality, evaluate_liveness, LivenessDetector
from utils.config import (
    ARCFACE_SIMILARITY_THRESHOLD, AMBIGUITY_MARGIN,
    MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_CONTRAST, MAX_YAW_RATIO
)


class TestDegenerateAffineLandmarks(unittest.TestCase):
    """
    Challenge Dimension: Degenerate, corrupt, and adversarial 5-point landmark inputs
    to align_face_5point and pipeline detection alignment.
    """

    def setUp(self):
        self.dummy_image = np.full((200, 200, 3), 128, dtype=np.uint8)
        self.raw_crop = np.full((60, 60, 3), 150, dtype=np.uint8)

    def test_all_identical_landmark_points(self):
        """All 5 points mapped to the exact same coordinate (0 area, singular matrix)."""
        degenerate_pts = np.array([[50.0, 50.0]] * 5, dtype=np.float32)
        # Should not crash or throw unhandled cv2 exception; must fall back to raw_crop
        result = align_face_5point(self.dummy_image, degenerate_pts, raw_crop=self.raw_crop)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (112, 112, 3))
        # Should match raw_crop resized, not crash
        expected = cv2.resize(self.raw_crop, (112, 112), interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(result, expected)

    def test_perfectly_collinear_landmarks(self):
        """All 5 landmark points along a single straight line (collinear degeneracy)."""
        collinear_pts = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]], dtype=np.float32)
        result = align_face_5point(self.dummy_image, collinear_pts, raw_crop=self.raw_crop)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (112, 112, 3))

    def test_extreme_negative_and_out_of_bounds_landmarks(self):
        """Landmarks positioned far outside image boundaries (-10000 to +10000)."""
        extreme_pts = np.array([
            [-5000.0, -5000.0],
            [15000.0, -5000.0],
            [5000.0, 5000.0],
            [-2000.0, 12000.0],
            [12000.0, 12000.0]
        ], dtype=np.float32)
        result = align_face_5point(self.dummy_image, extreme_pts, raw_crop=self.raw_crop)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (112, 112, 3))

    def test_nan_and_inf_landmarks(self):
        """Landmarks containing NaN or Infinity values."""
        nan_pts = np.array([
            [np.nan, 50.0],
            [70.0, np.nan],
            [np.inf, -np.inf],
            [40.0, 90.0],
            [70.0, 90.0]
        ], dtype=np.float32)
        # Should handle without crashing Python interpreter or leaking unhandled error
        try:
            result = align_face_5point(self.dummy_image, nan_pts, raw_crop=self.raw_crop)
            self.assertEqual(result.shape, (112, 112, 3))
        except cv2.error:
            # If cv2 raises cv2.error on NaN in estimateAffinePartial2D, note that as an edge case!
            pass

    def test_zero_sized_raw_crop_fallback(self):
        """When raw_crop is empty (0x0x3) and landmarks are degenerate."""
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        degenerate_pts = np.array([[50.0, 50.0]] * 5, dtype=np.float32)
        result = align_face_5point(self.dummy_image, degenerate_pts, raw_crop=empty_crop)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (112, 112, 3))


class TestHeadRollAndYawBoundaries(unittest.TestCase):
    """
    Challenge Dimension: Extreme head roll (0 to 360 deg) and yaw boundaries (-90 to +90 deg).
    Verifies roll-invariance stability and exact cutoff behavior of MAX_YAW_RATIO.
    """

    def _generate_synthetic_face(self, roll_deg: float, yaw_offset_px: float = 0.0, eye_dist: float = 60.0):
        theta = np.radians(roll_deg)
        p_mid = np.array([100.0, 100.0], dtype=np.float32)
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)       # Eye direction
        v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)      # Perpendicular face axis

        pt_r = p_mid - (eye_dist / 2.0) * u
        pt_l = p_mid + (eye_dist / 2.0) * u
        pt_n = p_mid + 30.0 * v + yaw_offset_px * u  # yaw offset shifts nose along u
        pt_rm = p_mid + 55.0 * v - 20.0 * u
        pt_lm = p_mid + 55.0 * v + 20.0 * u

        crop = np.full((200, 200, 3), 130, dtype=np.uint8)
        # Add gradient to pass blur and contrast checks
        for i in range(200):
            crop[i, :] = (i % 256, (i * 2) % 256, (i * 3) % 256)
        return crop, np.array([pt_r, pt_l, pt_n, pt_rm, pt_lm], dtype=np.float32)

    def test_roll_invariance_across_360_degrees(self):
        """Tests pure in-plane head roll from 0 to 360 degrees in 15-degree steps."""
        for roll in range(0, 360, 15):
            crop, lms = self._generate_synthetic_face(roll_deg=roll, yaw_offset_px=0.0)
            passed, reason, metrics = evaluate_face_quality(
                crop, landmarks=lms, min_size=30, blur_thresh=5.0, min_contrast=2.0
            )
            self.assertAlmostEqual(
                metrics["yaw_ratio"], 0.0, delta=0.001,
                msg=f"At roll={roll}°, yaw_ratio should be 0.0, got {metrics['yaw_ratio']}"
            )

    def test_yaw_cutoff_exact_boundary(self):
        """
        Tests yaw ratio response at the critical cutoff boundary MAX_YAW_RATIO (0.50).
        Eye distance = 60px.
        yaw_offset = 60 * 0.50 = 30.0px.
        At 28.5px (yaw_ratio ~0.475 < 0.50) -> PASS.
        At 31.5px (yaw_ratio ~0.525 > 0.50) -> REJECT with FACE_EXTREME_ANGLE.
        """
        crop_pass, lms_pass = self._generate_synthetic_face(roll_deg=0.0, yaw_offset_px=28.5, eye_dist=60.0)
        passed_sub, reason_sub, m_sub = evaluate_face_quality(
            crop_pass, landmarks=lms_pass, min_size=30, blur_thresh=5.0, min_contrast=2.0
        )
        self.assertTrue(passed_sub, f"Expected PASS at yaw_ratio {m_sub['yaw_ratio']} <= 0.50")

        crop_fail, lms_fail = self._generate_synthetic_face(roll_deg=0.0, yaw_offset_px=31.5, eye_dist=60.0)
        passed_sup, reason_sup, m_sup = evaluate_face_quality(
            crop_fail, landmarks=lms_fail, min_size=30, blur_thresh=5.0, min_contrast=2.0
        )
        self.assertFalse(passed_sup, f"Expected REJECT at yaw_ratio {m_sup['yaw_ratio']} > 0.50")
        self.assertIn("FACE_EXTREME_ANGLE", reason_sup)

    def test_coupled_roll_and_yaw(self):
        """Tests simultaneous roll (45 deg) and yaw boundary (20.0px)."""
        crop, lms = self._generate_synthetic_face(roll_deg=45.0, yaw_offset_px=20.0, eye_dist=60.0)
        # yaw_ratio should be 20.0 / 60.0 = 0.333 regardless of 45-deg roll
        passed, reason, metrics = evaluate_face_quality(
            crop, landmarks=lms, min_size=30, blur_thresh=5.0, min_contrast=2.0
        )
        self.assertAlmostEqual(metrics["yaw_ratio"], 0.333, delta=0.01)
        self.assertTrue(passed)

    def test_coincident_eyes_zero_distance(self):
        """When both eye landmarks are at the exact same location (zero eye distance)."""
        degenerate_lms = np.array([
            [100.0, 100.0],
            [100.0, 100.0],  # Coincident eye
            [100.0, 130.0],
            [80.0, 155.0],
            [120.0, 155.0]
        ], dtype=np.float32)
        crop = np.full((200, 200, 3), 130, dtype=np.uint8)
        passed, reason, metrics = evaluate_face_quality(
            crop, landmarks=degenerate_lms, min_size=30, blur_thresh=5.0, min_contrast=2.0
        )
        # Should not raise ZeroDivisionError, yaw_ratio defaults to 0.0
        self.assertEqual(metrics["yaw_ratio"], 0.0)


class TestCandidateRosterCornerCases(unittest.TestCase):
    """
    Challenge Dimension: Candidate roster filtering edge cases:
    - empty candidate roster []
    - single candidate roster [roll]
    - non-existent candidate roster [fake_roll]
    - large-scale gallery (100 to 1,000 mock centroids) to evaluate ambiguity margin behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.recognizer = FaceRecognizer()
        # Seed dummy faces if recognizer has none
        if not cls.recognizer.centroids:
            for i in range(1, 11):
                roll = f"2024TEST{i:03d}"
                vec = np.random.randn(512).astype(np.float32)
                cls.recognizer.centroids[roll] = vec / np.linalg.norm(vec)

    def test_empty_candidate_roster_fallback(self):
        """candidate_rolls=[] passed to recognize_batch should fall back to all centroids."""
        sample_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
        results = self.recognizer.recognize_batch([sample_crop], candidate_rolls=[])
        self.assertEqual(len(results), 1)
        roll, conf, diag = results[0]
        # Should not crash, must return valid diag structure
        self.assertIn("top1_roll", diag)

    def test_single_candidate_roster(self):
        """
        When candidate_rolls has only 1 student:
        Top-2 similarity is absent (0.0 / 'None').
        Ambiguity margin check must NOT trigger false AMBIGUOUS_MATCH rejection.
        """
        target_roll = list(self.recognizer.centroids.keys())[0]
        sample_crop = np.full((112, 112, 3), 128, dtype=np.uint8)

        results = self.recognizer.recognize_batch([sample_crop], candidate_rolls=[target_roll])
        self.assertEqual(len(results), 1)
        roll, conf, diag = results[0]
        self.assertEqual(diag["top1_roll"], target_roll)
        self.assertEqual(diag["top2_roll"], "None")
        # Ensure no AMBIGUOUS_MATCH reason
        self.assertNotIn("AMBIGUOUS_MATCH", diag.get("reason", ""))

    def test_disjoint_nonexistent_candidate_roster(self):
        """When candidate_rolls contains rolls not present in centroids."""
        sample_crop = np.full((112, 112, 3), 128, dtype=np.uint8)
        results = self.recognizer.recognize_batch([sample_crop], candidate_rolls=["GHOST_9999"])
        self.assertEqual(len(results), 1)
        # Should gracefully fall back to full centroid database without throwing KeyErrors
        roll, conf, diag = results[0]
        self.assertIn("top1_roll", diag)

    def test_large_scale_gallery_ambiguity_stress(self):
        """
        Stress test: Scale centroid database to 1,000 students.
        Evaluate cosine similarity distribution, latency scaling, and ambiguity margin impact.
        """
        np.random.seed(42)
        backup_centroids = self.recognizer.centroids.copy()
        try:
            # Populate 1,000 mock centroids
            large_centroids = {}
            for i in range(1000):
                r = f"STUDENT_{i:04d}"
                v = np.random.randn(512).astype(np.float32)
                large_centroids[r] = v / np.linalg.norm(v)
            self.recognizer.centroids = large_centroids

            # Generate query matching target 0 with known similarity 0.75
            target_0 = large_centroids["STUDENT_0000"]
            noise = np.random.randn(512).astype(np.float32)
            noise -= np.dot(noise, target_0) * target_0
            noise /= np.linalg.norm(noise)
            # Query vector with cosine similarity ~ 0.75 to target_0
            query_vec = math.cos(math.acos(0.75)) * target_0 + math.sin(math.acos(0.75)) * noise
            query_vec /= np.linalg.norm(query_vec)

            matched_roll, conf, diag = self.recognizer._match_single_embedding(query_vec)
            self.assertEqual(matched_roll, "STUDENT_0000")
            self.assertAlmostEqual(diag["top1_sim"], 0.75, delta=0.01)

            # In a 1,000-student random gallery, max random dot product is ~ 0.20-0.25
            # Margin = 0.75 - 0.22 ≈ 0.53 > AMBIGUITY_MARGIN (0.06)
            self.assertGreater(diag["margin"], AMBIGUITY_MARGIN)
            self.assertEqual(diag["reason"], "VERIFIED_MATCH")

            # Measure latency for 60 queries against 1,000 centroids
            query_matrix = np.random.randn(60, 512).astype(np.float32)
            query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True)
            c_mat = np.stack(list(large_centroids.values()))

            t0 = time.perf_counter()
            sims = np.dot(query_matrix, c_mat.T)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # 60 queries x 1,000 candidates BLAS dot product should complete smoothly in < 150ms
            self.assertLess(elapsed_ms, 150.0, f"Matrix dot product took {elapsed_ms:.2f}ms, expected < 150ms")
        finally:
            self.recognizer.centroids = backup_centroids


class TestBatchProcessingMemoryStability(unittest.TestCase):
    """
    Challenge Dimension: High-throughput batch processing memory stability.
    Executes repeated batches of size 60 through ArcFace feature extraction
    and vector matching, monitoring RSS memory growth.
    """

    @classmethod
    def setUpClass(cls):
        cls.recognizer = FaceRecognizer()

    def test_arcface_batch_memory_stability_100_iterations(self):
        """
        Executes 50 consecutive batches of size 60 (total 3,000 facial inferences).
        Monitors memory via process info to verify leak-free execution.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
        except ImportError:
            self.skipTest("psutil is not installed in the current environment")

        sample_crop = np.full((112, 112, 3), 140, dtype=np.uint8)
        batch = [sample_crop.copy() for _ in range(60)]

        # Warmup
        for _ in range(5):
            self.recognizer.recognize_batch(batch)

        gc.collect()
        initial_mem_mb = process.memory_info().rss / (1024 * 1024)

        # Run 50 iterations of 60-face batches (3,000 faces)
        for i in range(50):
            res = self.recognizer.recognize_batch(batch)
            self.assertEqual(len(res), 60)

        gc.collect()
        final_mem_mb = process.memory_info().rss / (1024 * 1024)
        mem_diff_mb = final_mem_mb - initial_mem_mb

        print(f"\n[MEMORY STABILITY] Initial: {initial_mem_mb:.2f} MB | Final: {final_mem_mb:.2f} MB | Delta: {mem_diff_mb:+.2f} MB over 3,000 inferences")

        # Memory delta should not exceed 100 MB for 3,000 inferences (no uncontrolled leak)
        self.assertLess(mem_diff_mb, 100.0, f"Memory grew by {mem_diff_mb:.2f} MB, possible memory leak!")


def run_all_adversarial_probes():
    """Runs all adversarial edge case test suites and prints summary."""
    print("=" * 80)
    print("      ADVERSARIAL STRESS & EDGE-CASE CHALLENGE PROBE HARNESS")
    print("=" * 80)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestDegenerateAffineLandmarks))
    suite.addTests(loader.loadTestsFromTestCase(TestHeadRollAndYawBoundaries))
    suite.addTests(loader.loadTestsFromTestCase(TestCandidateRosterCornerCases))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessingMemoryStability))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_all_adversarial_probes()
    sys.exit(0 if result.wasSuccessful() else 1)
