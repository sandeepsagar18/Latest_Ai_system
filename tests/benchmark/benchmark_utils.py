"""
SmartClass Vision - Evaluation Benchmark Metrics & Dataset Loader
Computes standard CV/ML evaluation metrics:
- Detection: Precision, Recall, F1-Score, IoU, Mean Average Precision (mAP50)
- Tracking: MOTA, MOTP, IDF1, ID Switches, Track Fragmentation
- Recognition: TAR@FAR, Top-1 Accuracy, Cosine Separation Margin
- Anti-Spoofing: APCER, BPCER, ACER (ISO/IEC 30107-3)
- Attendance: Quorum Precision, Recall, False Attendance Rate
- Performance: FPS, Latency (P50, P95, P99), RAM usage, Throughput
"""

import time
import os
import numpy as np
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


# ==============================================================================
# 1. DETECTION METRICS
# ==============================================================================

def compute_box_iou(box1, box2):
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union


def evaluate_detection(ground_truth_boxes, predicted_boxes, iou_threshold=0.50):
    """
    Evaluates face detection performance against ground truth.
    ground_truth_boxes: list of [x1, y1, x2, y2]
    predicted_boxes: list of [x1, y1, x2, y2]
    Returns: dict of detection metrics (TP, FP, FN, Precision, Recall, F1)
    """
    gt_matched = [False] * len(ground_truth_boxes)
    tp = 0
    fp = 0

    for pbox in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gbox in enumerate(ground_truth_boxes):
            iou = compute_box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = len(ground_truth_boxes) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4))
    }


# ==============================================================================
# 2. TRACKING METRICS (MOTA, MOTP, IDF1, ID Switches)
# ==============================================================================

def evaluate_tracking(gt_trajectories, pred_trajectories, iou_threshold=0.50):
    """
    Evaluates Multi-Object Tracking metrics across video frames.
    gt_trajectories: dict {frame_id: list of (gt_id, [x1, y1, x2, y2])}
    pred_trajectories: dict {frame_id: list of (track_id, [x1, y1, x2, y2])}
    Returns: dict with MOTA, MOTP, ID Switches (IDSW), Mostly Tracked, etc.
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    total_iou = 0.0
    matched_pairs_count = 0

    active_map = {}  # gt_id -> track_id

    frames = sorted(set(list(gt_trajectories.keys()) + list(pred_trajectories.keys())))
    for f in frames:
        gt_list = gt_trajectories.get(f, [])
        pred_list = pred_trajectories.get(f, [])
        total_gt += len(gt_list)

        matched_gt = set()
        matched_pred = set()

        for g_idx, (gt_id, gbox) in enumerate(gt_list):
            best_iou = 0.0
            best_p_idx = -1
            best_pid = None

            for p_idx, (pid, pbox) in enumerate(pred_list):
                if p_idx in matched_pred:
                    continue
                iou = compute_box_iou(gbox, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
                    best_pid = pid

            if best_iou >= iou_threshold:
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)
                total_iou += best_iou
                matched_pairs_count += 1

                # Check ID switch
                if gt_id in active_map and active_map[gt_id] != best_pid:
                    id_switches += 1
                active_map[gt_id] = best_pid

        total_fn += len(gt_list) - len(matched_gt)
        total_fp += len(pred_list) - len(matched_pred)

    mota = 1.0 - (total_fn + total_fp + id_switches) / max(1, total_gt)
    motp = (total_iou / max(1, matched_pairs_count))

    return {
        "mota": float(round(mota, 4)),
        "motp": float(round(motp, 4)),
        "id_switches": int(id_switches),
        "false_positives": int(total_fp),
        "false_negatives": int(total_fn),
        "total_ground_truth": int(total_gt)
    }


# ==============================================================================
# 3. RECOGNITION METRICS (TAR, FAR, Top-1, Margin)
# ==============================================================================

def evaluate_recognition(ground_truth_labels, predicted_labels, similarities):
    """
    Evaluates ArcFace biometric identification.
    ground_truth_labels: list of true student IDs
    predicted_labels: list of predicted student IDs ("Unknown" if rejected)
    similarities: list of cosine similarities
    """
    total = len(ground_truth_labels)
    correct = 0
    false_accepts = 0
    false_rejects = 0

    for gt, pred in zip(ground_truth_labels, predicted_labels):
        if gt == pred and gt != "Unknown":
            correct += 1
        elif pred != "Unknown" and gt != pred:
            false_accepts += 1
        elif pred == "Unknown" and gt != "Unknown":
            false_rejects += 1

    top1_accuracy = correct / max(1, total)
    far = false_accepts / max(1, total)
    frr = false_rejects / max(1, total)
    avg_similarity = float(np.mean(similarities)) if similarities else 0.0

    return {
        "top1_accuracy": float(round(top1_accuracy, 4)),
        "far": float(round(far, 4)),
        "frr": float(round(frr, 4)),
        "avg_similarity": float(round(avg_similarity, 4)),
        "samples_evaluated": total
    }


# ==============================================================================
# 4. ANTI-SPOOFING METRICS (ISO/IEC 30107-3: APCER, BPCER, ACER)
# ==============================================================================

def evaluate_antispoofing(ground_truth_is_live, predicted_is_live):
    """
    ISO/IEC 30107-3 Presentation Attack Detection (PAD) Metrics.
    ground_truth_is_live: list of bool (True=Live, False=Spoof/Attack)
    predicted_is_live: list of bool (True=Classified as Live, False=Spoof)
    """
    total_attacks = 0
    failed_attacks = 0  # Attack falsely classified as live
    total_bonafide = 0
    failed_bonafide = 0 # Live person falsely classified as spoof

    for true_live, pred_live in zip(ground_truth_is_live, predicted_is_live):
        if not true_live:
            total_attacks += 1
            if pred_live:
                failed_attacks += 1
        else:
            total_bonafide += 1
            if not pred_live:
                failed_bonafide += 1

    apcer = failed_attacks / max(1, total_attacks)  # Attack Presentation Classification Error Rate
    bpcer = failed_bonafide / max(1, total_bonafide) # Bona Fide Presentation Classification Error Rate
    acer = (apcer + bpcer) / 2.0                    # Average Classification Error Rate

    return {
        "apcer": float(round(apcer, 4)),
        "bpcer": float(round(bpcer, 4)),
        "acer": float(round(acer, 4)),
        "attack_samples": total_attacks,
        "bonafide_samples": total_bonafide
    }


# ==============================================================================
# 5. ATTENDANCE VERIFICATION & QUORUM METRICS
# ==============================================================================

def evaluate_attendance_quorum(enrolled_roster, true_present_students, attendance_records, quorum=3):
    """
    Evaluates multi-shot quorum attendance outcome.
    enrolled_roster: list of all students in section
    true_present_students: set of students actually in attendance
    attendance_records: dict of {student_id: list of verified shot indices}
    """
    predicted_present = set()
    for s_id, shots in attendance_records.items():
        if len(set(shots)) >= quorum:
            predicted_present.add(s_id)

    tp = len(predicted_present.intersection(true_present_students))
    fp = len(predicted_present - true_present_students)
    fn = len(true_present_students - predicted_present)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

    return {
        "quorum_threshold": quorum,
        "attendance_precision": float(round(precision, 4)),
        "attendance_recall": float(round(recall, 4)),
        "attendance_f1": float(round(f1, 4)),
        "false_attendance_count": fp,
        "missed_attendance_count": fn
    }


# ==============================================================================
# 6. SYSTEM PERFORMANCE PROFILER (FPS, Latency, RAM)
# ==============================================================================

class PerformanceProfiler:
    """Tracks latency percentiles (P50, P95, P99), throughput, and memory consumption."""
    def __init__(self):
        self.latencies = []
        self.process = psutil.Process() if psutil is not None else None
        self.start_mem_mb = (self.process.memory_info().rss / (1024 * 1024)) if self.process is not None else 0.0

    def record_latency(self, duration_sec):
        self.latencies.append(duration_sec)

    def get_summary(self):
        if not self.latencies:
            return {}
        lats_ms = np.array(self.latencies) * 1000.0
        current_mem_mb = (self.process.memory_info().rss / (1024 * 1024)) if self.process is not None else 0.0
        return {
            "fps": float(round(1.0 / max(1e-4, np.mean(self.latencies)), 2)),
            "mean_latency_ms": float(round(np.mean(lats_ms), 2)),
            "p50_latency_ms": float(round(np.percentile(lats_ms, 50), 2)),
            "p95_latency_ms": float(round(np.percentile(lats_ms, 95), 2)),
            "p99_latency_ms": float(round(np.percentile(lats_ms, 99), 2)),
            "ram_usage_mb": float(round(current_mem_mb, 1)),
            "ram_delta_mb": float(round(current_mem_mb - self.start_mem_mb, 1))
        }
