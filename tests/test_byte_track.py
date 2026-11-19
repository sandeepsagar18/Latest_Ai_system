"""
Comprehensive Unit Tests for ByteTrack Implementation Review & Fixes
Covers all 10 required testing suites:
1. Mahalanobis distance accuracy vs direct calculation.
2. Expired track removal and immediate non-repeated processing.
3. Invalid dt values rejection and validation.
4. Invalid Kalman predictions and valid bounding box guarantees.
5. Threshold-aware assignment with rectangular, all-invalid, empty, and competing matrices.
6. Malformed detections handling without crashes.
7. Cholesky and numerical failure recovery.
8. Lost-track reactivation.
9. Multiple tracker instances independent counter allocation.
10. Long-running tracking sessions and bounded memory.
"""

import pytest
import numpy as np
import scipy.linalg
from src.byte_track import (
    ByteTracker,
    STrack,
    KalmanFilterBox,
    linear_assignment,
    validate_detection,
    bbox_to_z,
    z_to_bbox,
    iou_batch,
    TrackState
)


def test_1_mahalanobis_distance_accuracy():
    """
    1. Test Mahalanobis solver interface & numerical correctness compared to direct matrix algebra:
       d^2 = (y - \mu)^T \Sigma^{-1} (y - \mu)
    """
    kf = KalmanFilterBox()
    mean, cov = kf.initiate(bbox_to_z([100, 100, 200, 200]))
    projected_mean, projected_cov = kf.project(mean, cov)

    measurements = [
        bbox_to_z([105, 105, 205, 205]),
        bbox_to_z([120, 110, 220, 210]),
        bbox_to_z([300, 300, 400, 400])
    ]

    # Run gating_distance via Cholesky solver
    dists = kf.gating_distance(mean, cov, measurements)
    assert len(dists) == 3
    assert np.all(np.isfinite(dists))

    # Verify against direct pinv calculation
    inv_cov = np.linalg.pinv(projected_cov)
    for idx, m in enumerate(measurements):
        diff = m - projected_mean
        direct_dist = float(np.dot(np.dot(diff, inv_cov), diff.T))
        assert np.isclose(dists[idx], direct_dist, rtol=1e-3, atol=1e-4)


def test_2_expired_track_removal():
    """
    2. Expired track removal:
    Ensure newly expired tracks are immediately subtracted from lost_stracks
    and never processed repeatedly.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10, track_buffer=3)
    # Frame 1: add track
    t1 = tracker.update([{"coords": (100, 100, 200, 200), "score": 0.95}])
    assert len(t1) == 1
    track_id = t1[0].track_id

    # Frame 2, 3, 4: Lost
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.lost_stracks) == 1
    assert tracker.lost_stracks[0].track_id == track_id

    # Frame 5: Exceeds track_buffer (frame_id 5 - start 1 = 4 > 3)
    tracker.update([])
    # Must be deleted from lost_stracks immediately and placed in removed_stracks
    assert len(tracker.lost_stracks) == 0
    assert len(tracker.tracked_stracks) == 0
    assert any(t.track_id == track_id for t in tracker.removed_stracks)

    # Frame 6: Ensure it is not processed or re-added again
    tracker.update([])
    assert len(tracker.lost_stracks) == 0


def test_3_invalid_dt_values():
    """
    3. Validate dynamic dt:
    Kalman prediction must safely reject NaN, infinite, negative, or excessively large dt values
    and default to a stable interval without throwing exceptions or corrupting state.
    """
    kf = KalmanFilterBox()
    mean, cov = kf.initiate(bbox_to_z([100, 100, 200, 200]))

    invalid_dts = [float('nan'), float('inf'), float('-inf'), -1.0, 0.0, 100.0, None]
    for bad_dt in invalid_dts:
        new_mean, new_cov = kf.predict(mean, cov, dt=bad_dt)
        assert np.all(np.isfinite(new_mean))
        assert np.all(np.isfinite(new_cov))
        assert np.all(np.diag(new_cov) > 0)


def test_4_invalid_kalman_predictions_and_bbox():
    """
    4. Validate predicted bounding boxes:
    z_to_bbox and STrack.tlbr must ensure finite coordinates, positive width and height,
    and prevent degenerate IoU calculations.
    """
    bad_states = [
        np.array([100, 100, -0.5, 50]),       # negative aspect ratio
        np.array([100, 100, 1.0, -20]),       # negative height
        np.array([100, 100, float('nan'), 50]), # NaN aspect ratio
        np.array([float('inf'), 100, 1.0, 50]), # Inf center
        np.array([100, 100, 0.0, 0.0])         # zero dimensions
    ]
    for s in bad_states:
        bbox = z_to_bbox(s)
        assert np.all(np.isfinite(bbox))
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]

    # Verify STrack.tlbr fallback if state ever becomes corrupted
    strack = STrack([10, 10, 50, 50], 0.90)
    strack.mean = np.array([float('nan')] * 8)
    safe_bbox = strack.tlbr
    assert np.all(np.isfinite(safe_bbox))
    assert safe_bbox[2] > safe_bbox[0]


def test_5_threshold_aware_linear_assignment():
    """
    5. Test threshold-aware assignment across edge conditions:
    - Competing valid associations (threshold-aware preference)
    - Rectangular matrices (more tracks than detections, or vice versa)
    - All-invalid matches (costs all > threshold)
    - Empty matrices (0 rows or 0 cols)
    """
    thresh = 0.50

    # Competing valid associations
    competing_cost = np.array([
        [0.05, 0.49],
        [0.10, 0.51]
    ])
    matches, un_a, un_b = linear_assignment(competing_cost, thresh=thresh)
    assert len(matches) == 2
    matched_set = set(tuple(m) for m in matches)
    assert (0, 1) in matched_set and (1, 0) in matched_set

    # Rectangular: 3 tracks, 2 detections
    rect_cost_1 = np.array([
        [0.2, 0.9],
        [0.8, 0.3],
        [0.7, 0.6]
    ])
    matches, un_a, un_b = linear_assignment(rect_cost_1, thresh=thresh)
    assert len(matches) == 2
    assert len(un_a) == 1 and 2 in un_a
    assert len(un_b) == 0

    # Rectangular: 2 tracks, 3 detections
    rect_cost_2 = np.array([
        [0.2, 0.8, 0.9],
        [0.9, 0.3, 0.7]
    ])
    matches, un_a, un_b = linear_assignment(rect_cost_2, thresh=thresh)
    assert len(matches) == 2
    assert len(un_a) == 0
    assert len(un_b) == 1 and 2 in un_b

    # All-invalid matches
    all_invalid = np.array([
        [0.8, 0.9],
        [0.7, 0.85]
    ])
    matches, un_a, un_b = linear_assignment(all_invalid, thresh=thresh)
    assert len(matches) == 0
    assert len(un_a) == 2 and len(un_b) == 2

    # Empty matrices
    m_empty, un_a, un_b = linear_assignment(np.empty((0, 3)), thresh=thresh)
    assert len(m_empty) == 0 and len(un_b) == 3
    m_empty_2, un_a, un_b = linear_assignment(np.empty((3, 0)), thresh=thresh)
    assert len(m_empty_2) == 0 and len(un_a) == 3


def test_6_malformed_detections_handling():
    """
    6. Strengthen detection validation:
    Handle malformed coordinates, wrong shapes, unexpected types, and None inputs safely.
    """
    tracker = ByteTracker()
    malformed = [
        {"coords": [10, 20]},                    # Only 2 coordinates
        {"coords": [[10, 20], [30, 40]]},        # Nested 2D array
        {"coords": "10,20,30,40"},               # String
        {"coords": None},                         # None coords
        {"coords": (100, 100, 200, 200), "score": "bad"}, # Bad score
        {"coords": (100, 100, 200, 200), "score": float('nan')},
        None,                                     # None element
        42                                        # Int element
    ]
    tracks = tracker.update(malformed)
    assert tracks == []
    # Test completely invalid container
    assert tracker.update(None) == []
    assert tracker.update("not_a_list") == []


def test_7_cholesky_numerical_failure_recovery():
    """
    7. Improve Kalman exception handling:
    Safe handling for singular/degenerate covariance matrices without crashing.
    """
    kf = KalmanFilterBox()
    mean = np.zeros(8)
    # Degenerate covariance with zeros
    singular_cov = np.zeros((8, 8))
    # Update should catch exception and return unchanged or safe state
    new_mean, new_cov = kf.update(mean, singular_cov, np.array([10, 10, 1, 20]))
    assert np.all(np.isfinite(new_mean))
    assert np.all(np.isfinite(new_cov))


def test_8_lost_track_reactivation():
    """
    8. Lost-track reactivation:
    Ensure tracks marked as Lost transition back to Tracked upon matching detection.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    tracker.update([{"coords": (200, 200, 300, 300), "score": 0.92}])
    orig_id = tracker.tracked_stracks[0].track_id

    # Lose track
    tracker.update([])
    assert tracker.lost_stracks[0].state == TrackState.Lost

    # Reactivate track with high score detection
    t3 = tracker.update([{"coords": (205, 205, 305, 305), "score": 0.91}])
    assert len(t3) == 1
    assert t3[0].track_id == orig_id
    assert t3[0].state == TrackState.Tracked
    assert len(tracker.lost_stracks) == 0


def test_9_multiple_tracker_instances():
    """
    9. Multiple tracker instances:
    Verifies that creating a second ByteTracker instance does NOT reset or alter
    the track ID counter of another instance.
    """
    tracker_a = ByteTracker(track_thresh=0.50)
    tracker_b = ByteTracker(track_thresh=0.50)

    # Tracker A initializes 2 tracks
    t_a = tracker_a.update([
        {"coords": (10, 10, 50, 50), "score": 0.9},
        {"coords": (60, 60, 100, 100), "score": 0.9}
    ])
    ids_a1 = [t.track_id for t in t_a]
    assert ids_a1 == [1, 2]

    # Tracker B initializes 1 track
    t_b = tracker_b.update([{"coords": (100, 100, 150, 150), "score": 0.9}])
    ids_b = [t.track_id for t in t_b]
    assert ids_b == [1]

    # Tracker A adds another track - its ID counter should continue smoothly to 3
    t_a2 = tracker_a.update([
        {"coords": (10, 10, 50, 50), "score": 0.9},
        {"coords": (60, 60, 100, 100), "score": 0.9},
        {"coords": (200, 200, 250, 250), "score": 0.9}
    ])
    new_tracks_a = [t for t in tracker_a.tracked_stracks if t.track_id not in ids_a1]
    assert len(new_tracks_a) == 1
    assert new_tracks_a[0].track_id == 3


def test_10_long_running_session_memory():
    """
    10. Long-running session memory stability:
    Simulates a 300-frame session with continuous movement, entering and exiting faces.
    Verifies bounded memory and finite covariance values throughout.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10, track_buffer=5, max_removed_buffer=50)

    for f in range(250):
        faces = []
        if f % 12 < 4:
            faces.append({"coords": (100 + (f % 50), 100, 150 + (f % 50), 150), "score": 0.90})
        faces.append({"coords": (300, 300, 400, 400), "score": 0.92})

        tracks = tracker.update(faces, dt=0.033)
        for trk in tracks:
            assert np.all(np.isfinite(trk.mean))
            assert np.all(np.isfinite(trk.covariance))

    assert len(tracker.removed_stracks) <= 50
    assert len(tracker.lost_stracks) <= 10
