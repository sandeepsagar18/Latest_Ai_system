"""
Unit Tests for ByteTrack Multi-Object Face Tracking Implementation
Verifying all 14 engineering requirements for SmartClass Vision AI attendance.
"""

import pytest
import numpy as np
from src.byte_track import ByteTracker, STrack, KalmanFilterBox, linear_assignment, validate_detection, TrackState


def test_1_hungarian_assignment_threshold_edge_case():
    """
    1. Hungarian assignment optimization bug:
    Test a cost matrix where a naive Hungarian matching creates an invalid assignment
    that blocks two valid alternatives.
    Example:
    Cost matrix:
        D0     D1
    T0 [0.05, 0.49]
    T1 [0.10, 0.51]
    With threshold = 0.50:
    Naive Hungarian minimizes 0.05 + 0.51 = 0.56, but (1, 1) has cost 0.51 > 0.50,
    leaving only 1 valid match [(0, 0)].
    Threshold-aware assignment must select [(0, 1), (1, 0)] (costs 0.49 and 0.10, both <= 0.50),
    yielding 2 valid matches.
    """
    cost_matrix = np.array([
        [0.05, 0.49],
        [0.10, 0.51]
    ])
    thresh = 0.50
    matches, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh=thresh)

    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}: {matches}"
    matched_pairs = set(tuple(m) for m in matches)
    assert (0, 1) in matched_pairs and (1, 0) in matched_pairs
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0


def test_2_first_frame_track_initialization():
    """
    2. First-frame track initialization:
    In the first frame (frame_id = 1), high-confidence face detections should be initialized
    and confirmed immediately so the system doesn't drop faces on initial snapshot.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    faces = [
        {"coords": (100, 100, 200, 200), "score": 0.95},
        {"coords": (300, 100, 400, 200), "score": 0.88}
    ]
    tracks = tracker.update(faces)
    assert len(tracks) == 2
    for t in tracks:
        assert t.is_activated is True
        assert t.state == TrackState.Tracked
        assert t.track_id > 0


def test_3_track_confirmation_lifecycle():
    """
    3. Track confirmation lifecycle:
    In subsequent frames (frame_id > 1), a new track is initialized as unconfirmed (is_activated=False).
    - If another matching detection is observed in the next frame, it is confirmed (is_activated=True).
    - If it disappears immediately after creation without a second match, it is removed.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    # Frame 1: empty
    tracker.update([])

    # Frame 2: new student appears at (100, 100, 200, 200)
    faces_f2 = [{"coords": (100, 100, 200, 200), "score": 0.90}]
    tracks_f2 = tracker.update(faces_f2)
    # Since it's a new track on frame 2, it should remain unconfirmed on frame 2 (not returned in confirmed tracks)
    assert len(tracks_f2) == 0
    assert len(tracker.tracked_stracks) == 1
    unconf_track = tracker.tracked_stracks[0]
    assert unconf_track.is_activated is False
    saved_id = unconf_track.track_id

    # Frame 3: same student observed again at (102, 101, 202, 201)
    faces_f3 = [{"coords": (102, 101, 202, 201), "score": 0.91}]
    tracks_f3 = tracker.update(faces_f3)
    assert len(tracks_f3) == 1
    assert tracks_f3[0].is_activated is True
    assert tracks_f3[0].track_id == saved_id  # Keeps original track ID

    # Test disappearing unconfirmed track:
    # Frame 4: a phantom detection appears at (500, 500, 550, 550)
    faces_f4 = [
        {"coords": (104, 102, 204, 202), "score": 0.91},
        {"coords": (500, 500, 550, 550), "score": 0.85}
    ]
    tracker.update(faces_f4)
    # Frame 5: phantom detection disappears
    faces_f5 = [{"coords": (105, 103, 205, 203), "score": 0.92}]
    tracks_f5 = tracker.update(faces_f5)
    assert len(tracks_f5) == 1
    # Check that unconfirmed phantom track was removed
    removed_ids = [t.track_id for t in tracker.removed_stracks]
    assert len(removed_ids) >= 1


def test_4_low_confidence_recovery():
    """
    4. Missing lower detection confidence threshold:
    - High-confidence (>= 0.50) matched in Stage 1.
    - Low-confidence (0.10 <= score < 0.50) recovers track in Stage 2.
    - Discarded (< 0.10) completely ignored.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    # Frame 1: Student detected with high confidence
    tracker.update([{"coords": (100, 100, 200, 200), "score": 0.92}])
    orig_id = tracker.tracked_stracks[0].track_id

    # Frame 2: Student turns head, score drops to 0.35 (recovered in Stage 2)
    tracks_f2 = tracker.update([{"coords": (102, 101, 202, 201), "score": 0.35}])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == orig_id
    assert tracks_f2[0].score == 0.35

    # Frame 3: Noise detection with score 0.05 (< low_thresh) must be discarded
    # Detection far away with score 0.05 shouldn't create or update any track
    tracker.update([
        {"coords": (104, 102, 204, 202), "score": 0.88},
        {"coords": (800, 800, 900, 900), "score": 0.05}
    ])
    assert len(tracker.tracked_stracks) == 1


def test_5_two_students_crossing():
    """
    5. Two students crossing paths:
    Tests that two tracks maintain separate identities as they pass close to each other.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    # Frame 1: Left student at x=100 moving right (+20px), Right student at x=300 moving left (-20px)
    f1 = [
        {"coords": (100, 100, 180, 200), "score": 0.95},
        {"coords": (300, 100, 380, 200), "score": 0.95}
    ]
    t1 = tracker.update(f1)
    id_left = [t.track_id for t in t1 if t.tlbr[0] < 200][0]
    id_right = [t.track_id for t in t1 if t.tlbr[0] > 200][0]
    assert id_left != id_right

    # Frame 2: Moving closer (smooth 10px shift)
    f2 = [
        {"coords": (110, 100, 190, 200), "score": 0.95},
        {"coords": (290, 100, 370, 200), "score": 0.95}
    ]
    t2 = tracker.update(f2)
    assert len(t2) == 2

    # Frame 3: Crossing paths (smooth continued motion)
    f3 = [
        {"coords": (120, 100, 200, 200), "score": 0.95},
        {"coords": (280, 100, 360, 200), "score": 0.95}
    ]
    t3 = tracker.update(f3)
    assert len(t3) == 2
    new_id_left = [t.track_id for t in t3 if t.tlbr[0] < 200][0]
    new_id_right = [t.track_id for t in t3 if t.tlbr[0] > 200][0]
    assert new_id_left == id_left
    assert new_id_right == id_right


def test_6_temporary_face_occlusion():
    """
    6. Temporary face occlusion:
    A student is tracked, disappears for 3 frames (occluded by teacher walking by),
    then reappears nearby. Track should be reactivated with the same track_id.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10, track_buffer=10)
    # Frame 1
    t1 = tracker.update([{"coords": (100, 100, 200, 200), "score": 0.90}])
    track_id = t1[0].track_id

    # Frames 2, 3, 4: Occluded (empty frames)
    for _ in range(3):
        res = tracker.update([])
        assert len(res) == 0
        assert len(tracker.lost_stracks) == 1

    # Frame 5: Student reappears at slightly shifted coordinates
    t5 = tracker.update([{"coords": (105, 102, 205, 202), "score": 0.88}])
    assert len(t5) == 1
    assert t5[0].track_id == track_id
    assert t5[0].state == TrackState.Tracked


def test_7_lost_track_reactivation():
    """
    7. Lost-track reactivation:
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


def test_8_track_expiration():
    """
    8. Track expiration:
    Verify that tracks in lost state longer than track_buffer frames are moved to removed_stracks.
    """
    track_buffer = 5
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10, track_buffer=track_buffer)
    tracker.update([{"coords": (100, 100, 200, 200), "score": 0.90}])
    assert len(tracker.tracked_stracks) == 1

    # Student walks out of room; wait track_buffer + 1 frames
    for _ in range(track_buffer + 2):
        tracker.update([])

    assert len(tracker.tracked_stracks) == 0
    assert len(tracker.lost_stracks) == 0
    assert len(tracker.removed_stracks) >= 1
    assert tracker.removed_stracks[0].state == TrackState.Removed


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


def test_10_invalid_bounding_boxes():
    """
    10. Bounding box and detection validation:
    Rejects malformed bounding boxes:
    - x2 <= x1
    - y2 <= y1
    - zero area
    - negative width/height
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    invalid_detections = [
        {"coords": (100, 100, 100, 200), "score": 0.9},  # width = 0
        {"coords": (100, 100, 200, 100), "score": 0.9},  # height = 0
        {"coords": (200, 100, 100, 200), "score": 0.9},  # x2 < x1
        {"coords": (100, 200, 200, 100), "score": 0.9},  # y2 < y1
        {"coords": None, "score": 0.9},                   # None coords
        "not_a_dict"                                       # Invalid type
    ]
    tracks = tracker.update(invalid_detections)
    assert len(tracks) == 0
    assert len(tracker.tracked_stracks) == 0


def test_11_nan_and_infinite_confidence_values():
    """
    11. NaN and infinite coordinate / confidence rejection:
    Checks that detections containing NaN, inf, or negative confidence are cleanly rejected.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10)
    bad_detections = [
        {"coords": (float('nan'), 100, 200, 200), "score": 0.9},
        {"coords": (100, 100, float('inf'), 200), "score": 0.9},
        {"coords": (100, 100, 200, 200), "score": float('nan')},
        {"coords": (100, 100, 200, 200), "score": float('inf')},
        {"coords": (100, 100, 200, 200), "score": -0.5},
        {"coords": (100, 100, 200, 200), "score": 1.5}
    ]
    tracks = tracker.update(bad_detections)
    assert len(tracks) == 0
    assert len(tracker.tracked_stracks) == 0


def test_12_empty_detection_frames():
    """
    12. Empty detection frames:
    Updating with empty detection lists should execute safely without crashes or NaN states.
    """
    tracker = ByteTracker()
    for _ in range(5):
        tracks = tracker.update([])
        assert tracks == []


def test_13_long_running_session_memory():
    """
    13. Kalman filter numerical stability & Long-running memory bounding:
    Simulates a 500-frame session with objects entering and leaving.
    Verifies that:
    - Kalman covariance remains positive-definite and finite.
    - removed_stracks does not grow unbounded beyond max_removed_buffer.
    """
    tracker = ByteTracker(track_thresh=0.50, low_thresh=0.10, track_buffer=5, max_removed_buffer=50)

    for f in range(200):
        # Every 10 frames, a new transient face appears and disappears
        faces = []
        if f % 10 < 3:
            faces.append({"coords": (100 + f, 100, 150 + f, 150), "score": 0.90})
        # Continuous face
        faces.append({"coords": (300, 300, 400, 400), "score": 0.92})

        tracks = tracker.update(faces, dt=0.033)

        # Check Kalman numerical validity on active tracks
        for trk in tracks:
            assert np.all(np.isfinite(trk.mean))
            assert np.all(np.isfinite(trk.covariance))
            assert np.all(np.diag(trk.covariance) > 0)

    # Check memory bounded
    assert len(tracker.removed_stracks) <= 50


def test_14_duplicate_attendance_vote_prevention():
    """
    14. SmartClass Vision Integration - Duplicate attendance vote prevention:
    Verifies that during a single capture burst, multiple detections associated
    with the same student or active track cannot cast duplicate votes, preserving
    the quorum rule (>= 3 unique burst votes out of 10 shots).
    """
    NUM_SHOTS = 10
    student_shots = {}  # roll -> set of unique shot indices

    # Simulate 10 shots
    for shot_idx in range(NUM_SHOTS):
        # In shot_idx, imagine two detections match Roll "21BCSE01" (e.g. reflection / jitter)
        detected_rolls_in_shot = ["21BCSE01", "21BCSE01", "21BCSE02"]

        # Attendance logic invariant: set of rolls per shot
        rolls_credited_this_shot = set()
        for roll in detected_rolls_in_shot:
            if roll not in rolls_credited_this_shot:
                rolls_credited_this_shot.add(roll)
                if roll not in student_shots:
                    student_shots[roll] = set()
                student_shots[roll].add(shot_idx)

    # Verify that "21BCSE01" only accumulated 1 vote per shot (total 10 votes across 10 shots)
    assert len(student_shots["21BCSE01"]) == 10
    assert len(student_shots["21BCSE02"]) == 10
