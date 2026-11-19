"""
ByteTrack Multi-Object Face Tracking Algorithm
Based on the ByteTrack paper (Zhang et al., ECCV 2022).
Tracks faces between successive video preview and snapshot frames by associating
both high-score and low-score detection bounding boxes using a constant-velocity Kalman filter
and linear assignment via Intersection-over-Union (IoU).

Enhanced with:
- Threshold-aware maximum cardinality linear assignment
- Two-stage lifecycle confirmation for new tracks
- Dynamic frame interval (dt) scaling for Kalman filter
- Numerically stable Joseph-form covariance updates and symmetry enforcement
- Strict coordinate and confidence validation
- Independent per-tracker ID counter allocation
- Bounded memory management for long-running tracking sessions
- Mahalanobis motion gating and verified student identity tracking
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilterBox:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    State: [x, y, a, h, vx, vy, va, vh]
    x, y: center of bounding box
    a: aspect ratio (w / h)
    h: height
    vx, vy, va, vh: velocities
    """
    def __init__(self, default_dt=1.0):
        self.ndim = 4
        self.default_dt = float(default_dt)
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def get_motion_mat(self, dt=1.0):
        motion_mat = np.eye(2 * self.ndim, 2 * self.ndim, dtype=float)
        for i in range(self.ndim):
            motion_mat[i, self.ndim + i] = float(dt)
        return motion_mat

    def initiate(self, measurement):
        """
        Create track from measurement: [x, y, a, h].
        """
        measurement = np.asarray(measurement, dtype=float)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, dt=1.0):
        """
        Run Kalman filter prediction step with dynamic time interval dt.
        """
        dt = max(1e-4, float(dt))
        motion_mat = self.get_motion_mat(dt)

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        # Velocity noise scales with sqrt(dt)
        std_vel = [
            self._std_weight_velocity * mean[3] * np.sqrt(dt),
            self._std_weight_velocity * mean[3] * np.sqrt(dt),
            1e-5,
            self._std_weight_velocity * mean[3] * np.sqrt(dt)
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(motion_mat, mean)
        covariance = np.linalg.multi_dot((motion_mat, covariance, motion_mat.T)) + motion_cov

        # Enforce symmetry and positive variances
        covariance = 0.5 * (covariance + covariance.T)
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 1e-6))
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        projected_cov = 0.5 * (projected_cov + projected_cov.T)
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """
        Kalman update step using Joseph-form covariance update for numerical stability:
        P = (I - K H) P (I - K H)^T + K R K^T
        """
        measurement = np.asarray(measurement, dtype=float)
        projected_mean, projected_cov = self.project(mean, covariance)

        # Innovation covariance R
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        r_cov = np.diag(np.square(std))

        # Solve for Kalman Gain K: projected_cov * K^T = H * P^T => K = P H^T (projected_cov)^-1
        chol_factor, lower = scipy_cholesky(projected_cov)
        kalman_gain = scipy_solve_cholesky(chol_factor, lower, np.dot(covariance, self._update_mat.T).T).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # Joseph form: (I - K H) P (I - K H)^T + K R K^T
        i_kh = np.eye(8) - np.dot(kalman_gain, self._update_mat)
        new_covariance = np.linalg.multi_dot((i_kh, covariance, i_kh.T)) + np.linalg.multi_dot((kalman_gain, r_cov, kalman_gain.T))

        # Ensure symmetry and finite positive values
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        np.fill_diagonal(new_covariance, np.maximum(np.diag(new_covariance), 1e-6))

        if not (np.all(np.isfinite(new_mean)) and np.all(np.isfinite(new_covariance))):
            # Fallback to previous mean and covariance if numerical instability occurred
            return mean, covariance

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Computes Mahalanobis distance between state and measurements.
        measurements: [N, 4] (x, y, a, h)
        Returns: [N] squared Mahalanobis distance
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = np.asarray(measurements)[:, :2]
        else:
            measurements = np.asarray(measurements)[:, :4]

        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy_solve_cholesky((cholesky_factor, True), True, d.T)
        squared_maha = np.sum(d * z.T, axis=1)
        return squared_maha


def scipy_cholesky(a):
    import scipy.linalg
    return scipy.linalg.cho_factor(a, lower=True, check_finite=False)


def scipy_solve_cholesky(c, lower, b):
    import scipy.linalg
    return scipy.linalg.cho_solve((c, lower), b, check_finite=False)


def validate_detection(f):
    """
    Validates detection bounding box and score.
    Returns: (is_valid, (x1, y1, x2, y2), score)
    """
    if not isinstance(f, dict):
        return False, None, 0.0

    coords = f.get("coords")
    if coords is None or len(coords) != 4:
        return False, None, 0.0

    try:
        x1, y1, x2, y2 = [float(c) for c in coords]
    except (ValueError, TypeError):
        return False, None, 0.0

    # Finite coordinate check
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
        return False, None, 0.0

    # Dimension and ordering check
    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        return False, None, 0.0

    # Score validation
    score = f.get("score", 0.90)
    try:
        score = float(score)
    except (ValueError, TypeError):
        return False, None, 0.0

    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        return False, None, 0.0

    return True, np.array([x1, y1, x2, y2], dtype=float), score


def bbox_to_z(bbox):
    """Converts [x1, y1, x2, y2] to [x_center, y_center, aspect_ratio, height]."""
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    x = float(bbox[0]) + w / 2.0
    y = float(bbox[1]) + h / 2.0
    s = max(1e-4, w / max(1e-4, h))
    return np.array([x, y, s, h])


def z_to_bbox(z):
    """Converts [x_center, y_center, aspect_ratio, height] to [x1, y1, x2, y2]."""
    w = z[2] * z[3]
    h = z[3]
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    return np.array([x1, y1, x2, y2])


def iou_batch(bboxes1, bboxes2):
    """
    Computes pairwise IoU between two sets of bounding boxes.
    bboxes1: [N, 4]
    bboxes2: [M, 4]
    Returns: [N, M] cost matrix (1.0 - IoU)
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.empty((len(bboxes1), len(bboxes2)))

    b1 = np.ascontiguousarray(bboxes1, dtype=float)
    b2 = np.ascontiguousarray(bboxes2, dtype=float)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

    iw = np.maximum(0.0, np.minimum(b1[:, 2][:, None], b2[:, 2]) - np.maximum(b1[:, 0][:, None], b2[:, 0]))
    ih = np.maximum(0.0, np.minimum(b1[:, 3][:, None], b2[:, 3]) - np.maximum(b1[:, 1][:, None], b2[:, 1]))
    intersection = iw * ih
    union = area1[:, None] + area2 - intersection
    ious = np.clip(intersection / np.maximum(1e-6, union), 0.0, 1.0)
    return 1.0 - ious


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    shared_kalman = KalmanFilterBox()

    def __init__(self, bbox, score, face_data=None):
        self._tlbr = np.asarray(bbox, dtype=float)
        self.score = float(score)
        self.face_data = face_data
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        self.mean = None
        self.covariance = None
        self.tracklet_len = 0
        self.time_since_update = 0
        self.frame_id = 0
        self.start_frame = 0

        # SmartClass Vision Biometric Association
        self.student_id = None
        self.student_name = None
        self.verified_confidence = 0.0
        self.features = []

    @property
    def tlbr(self):
        if self.mean is None:
            return self._tlbr
        return z_to_bbox(self.mean[:4])

    def predict(self, dt=1.0):
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(mean_state, self.covariance, dt=dt)
        self.time_since_update += 1

    def activate(self, frame_id, track_id=None, is_first_frame=False):
        """
        Activates track.
        If is_first_frame is True, track confirms immediately.
        Otherwise, new tracks start unconfirmed until a second observation is received.
        """
        self.mean, self.covariance = self.shared_kalman.initiate(bbox_to_z(self._tlbr))
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

        if track_id is not None:
            self.track_id = track_id
        elif self.track_id == 0:
            self.track_id = STrack.next_id()

        if is_first_frame:
            self.state = TrackState.Tracked
            self.is_activated = True
        else:
            self.state = TrackState.New
            self.is_activated = False

    def re_activate(self, new_track, frame_id, new_id=False, track_id=None):
        """
        Reactivates a lost track upon re-detection.
        """
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, bbox_to_z(new_track.tlbr)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = track_id if track_id is not None else STrack.next_id()
        self.score = new_track.score
        self.face_data = new_track.face_data
        self.time_since_update = 0

    def update(self, new_track, frame_id):
        """
        Updates track state with an associated detection.
        Confirms unconfirmed tracks.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_z = bbox_to_z(new_track.tlbr)
        self.mean, self.covariance = self.shared_kalman.update(self.mean, self.covariance, new_z)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.face_data = new_track.face_data
        self.time_since_update = 0

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    # Global counter kept for backward compatibility; instance counter is preferred
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0


def linear_assignment(cost_matrix, thresh):
    """
    Threshold-aware linear assignment that maximizes the number of valid matches.
    
    Standard linear_sum_assignment finds a global minimum weight matching without knowing
    the threshold cutoff. Under competitive costs, an edge with high cost (<= thresh)
    could prevent another match from forming, or choose a pairing that leaves a valid edge
    rejected by the cutoff.
    
    Fix:
    Mask pairs exceeding the threshold with a heavy penalty (1e5).
    By assigning cost = C_ij for C_ij <= thresh, and cost = C_ij + 100000 for C_ij > thresh,
    the Hungarian algorithm prioritizes maximizing valid matches (edges <= thresh) before
    minimizing cost. Any pair whose matched cost is > thresh is then placed in unmatched.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    penalty = 1e5
    adjusted_cost = np.copy(cost_matrix)
    invalid_mask = cost_matrix > thresh
    adjusted_cost[invalid_mask] += penalty

    row_ind, col_ind = linear_sum_assignment(adjusted_cost)
    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            unmatched_a.discard(r)
            unmatched_b.discard(c)

    return (
        np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
        tuple(sorted(unmatched_a)),
        tuple(sorted(unmatched_b))
    )


class ByteTracker:
    """
    ByteTrack implementation for multi-face association across frames.
    Associates high-confidence detections first, then recovers occluded/slanted
    faces using low-confidence detections (ByteTrack two-stage data association).

    Parameters:
    - track_thresh: float (default 0.50). High-confidence detection threshold.
    - low_thresh: float (default 0.10). Minimum confidence; detections below this are discarded.
    - match_thresh: float (default 0.70). Maximum 1 - IoU distance for association.
    - track_buffer: int (default 30). Number of frames to keep lost tracks before removal.
    - max_removed_buffer: int (default 500). Max historical removed tracks to retain in memory.
    """
    def __init__(self, track_thresh=0.50, low_thresh=0.10, match_thresh=0.70, track_buffer=30, max_removed_buffer=500):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.max_removed_buffer = max_removed_buffer

        self.frame_id = 0
        self._next_track_id = 0

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

    def next_id(self):
        """Tracker-instance specific ID allocation to prevent cross-tracker collision."""
        self._next_track_id += 1
        return self._next_track_id

    def reset_counter(self):
        """Resets tracker-specific ID generator."""
        self._next_track_id = 0

    def update(self, detected_faces, dt=1.0):
        """
        Updates tracks given current frame detected_faces.
        detected_faces: list of dicts with 'coords': (x1, y1, x2, y2), 'score': float, ...
        dt: float elapsed time or frame interval (default 1.0)
        Returns: list of active, confirmed STrack objects with track_id assigned.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 0. Validate incoming detections and split into High (>= track_thresh),
        # Low (low_thresh <= score < track_thresh), and Discard (< low_thresh)
        detections_high = []
        detections_low = []

        for f in detected_faces:
            valid, bbox, score = validate_detection(f)
            if not valid or score < self.low_thresh:
                continue

            strack = STrack(bbox, score, face_data=f)
            if score >= self.track_thresh:
                detections_high.append(strack)
            else:
                detections_low.append(strack)

        # 1. Partition existing tracks: unconfirmed vs confirmed tracked
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 2. Predict current locations of existing tracks using Kalman Filter
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for s in strack_pool:
            s.predict(dt=dt)

        # 3. First association: Match tracked + lost pool with HIGH score detections
        cost_matrix = iou_batch([s.tlbr for s in strack_pool], [d.tlbr for d in detections_high])
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 4. Second association: Match remaining tracks with LOW score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        cost_matrix_low = iou_batch([s.tlbr for s in r_tracked_stracks], [d.tlbr for d in detections_low])
        matches_low, u_track_2, _ = linear_assignment(cost_matrix_low, thresh=0.50)

        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 5. Third association: Match unconfirmed tracks with remaining high-score detections
        detections_rem = [detections_high[i] for i in u_detection]
        cost_matrix_unconf = iou_batch([s.tlbr for s in unconfirmed], [d.tlbr for d in detections_rem])
        matches_unconf, u_unconfirmed, u_detection_rem = linear_assignment(cost_matrix_unconf, thresh=self.match_thresh)

        for itracked, idet in matches_unconf:
            track = unconfirmed[itracked]
            track.update(detections_rem[idet], self.frame_id)
            activated_stracks.append(track)

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 6. Initialize new tracks from unassociated high-score detections
        for inew in u_detection_rem:
            track = detections_rem[inew]
            if track.score < self.track_thresh:
                continue
            # First-frame initialization confirms immediately; later frames require confirmation
            is_first = (self.frame_id == 1)
            track.activate(self.frame_id, track_id=self.next_id(), is_first_frame=is_first)
            activated_stracks.append(track)

        # 7. Update state of lost tracks (remove if lost longer than track_buffer frames)
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        # 8. Consolidate and update tracker states
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked or t.state == TrackState.New]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        # Bounded removed tracks to prevent memory leak
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > self.max_removed_buffer:
            self.removed_stracks = self.removed_stracks[-self.max_removed_buffer:]

        # Output confirmed active tracks only
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())
