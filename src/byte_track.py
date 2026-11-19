"""
ByteTrack Multi-Object Face Tracking Algorithm
Based on the ByteTrack paper (Zhang et al., ECCV 2022).
Tracks faces between successive video preview and snapshot frames by associating
both high-score and low-score detection bounding boxes using a constant-velocity Kalman filter
and linear assignment via Intersection-over-Union (IoU).
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
    """
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
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

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy_cholesky(projected_cov)
        kalman_gain = scipy_solve_cholesky(chol_factor, lower, np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def scipy_cholesky(a):
    import scipy.linalg
    return scipy.linalg.cho_factor(a, lower=True, check_finite=False)


def scipy_solve_cholesky(c, lower, b):
    import scipy.linalg
    return scipy.linalg.cho_solve((c, lower), b, check_finite=False)


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

    iw = np.maximum(0, np.minimum(b1[:, 2][:, None], b2[:, 2]) - np.maximum(b1[:, 0][:, None], b2[:, 0]))
    ih = np.maximum(0, np.minimum(b1[:, 3][:, None], b2[:, 3]) - np.maximum(b1[:, 1][:, None], b2[:, 1]))
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

    @property
    def tlbr(self):
        if self.mean is None:
            return self._tlbr
        return z_to_bbox(self.mean[:4])

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.shared_kalman.predict(mean_state, self.covariance)

    def activate(self, frame_id, new_id=True):
        self.mean, self.covariance = self.shared_kalman.initiate(bbox_to_z(self._tlbr))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if new_id:
            self.track_id = STrack.next_id()
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, bbox_to_z(new_track.tlbr)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = STrack.next_id()
        self.score = new_track.score
        self.face_data = new_track.face_data
        self.time_since_update = 0

    def update(self, new_track, frame_id):
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

    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0


class ByteTracker:
    """
    ByteTrack implementation for multi-face association across frames.
    Associates high-confidence detections first, then recovers occluded/slanted
    faces using low-confidence detections (ByteTrack two-stage data association).
    """
    def __init__(self, track_thresh=0.50, match_thresh=0.70, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0

        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        STrack.reset_counter()

    def update(self, detected_faces):
        """
        Updates tracks given current frame detected_faces.
        detected_faces: list of dicts with 'coords': (x1, y1, x2, y2), 'score': float, ...
        Returns: list of active STrack objects with track_id assigned.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split detections into high score (D_high) and low score (D_low)
        detections_high = []
        detections_low = []

        for f in detected_faces:
            bbox = f["coords"]
            score = float(f.get("score", 0.90))
            strack = STrack(bbox, score, face_data=f)
            if score >= self.track_thresh:
                detections_high.append(strack)
            else:
                detections_low.append(strack)

        # 1. Predict current locations of existing tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for s in strack_pool:
            s.predict()

        # 2. First association with high score detections
        cost_matrix = iou_batch([s.tlbr for s in strack_pool], [d.tlbr for d in detections_high])
        matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 3. Second association with low score detections (recover occluded / motion blurred faces)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        cost_matrix = iou_batch([s.tlbr for s in r_tracked_stracks], [d.tlbr for d in detections_low])
        matches, u_track_2, u_detection_2 = linear_assignment(cost_matrix, thresh=0.50)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 4. Handle unconfirmed tracks
        detections = [detections_high[i] for i in u_detection]
        cost_matrix = iou_batch([s.tlbr for s in unconfirmed], [d.tlbr for d in detections])
        matches, u_unconfirmed, u_detection = linear_assignment(cost_matrix, thresh=self.match_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 5. Initialize new tracks from unassociated high-score detections
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        # 6. Update state of lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

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


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            if r in unmatched_a: unmatched_a.remove(r)
            if c in unmatched_b: unmatched_b.remove(c)
    return np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int), tuple(unmatched_a), tuple(unmatched_b)
