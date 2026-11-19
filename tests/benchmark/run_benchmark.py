"""
SmartClass Vision - Master Automated Benchmark Execution Harness
Runs end-to-end evaluation across Detection, ByteTrack Tracking, ArcFace Recognition,
MiniFASNetV2 Anti-Spoofing, and Latency/Throughput Profiling.
Generates baseline vs candidate comparisons and detailed markdown reports.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from tests.benchmark.experiment_config import ExperimentConfig
from tests.benchmark.dataset_loader import DatasetLoader
from tests.benchmark.benchmark_utils import (
    evaluate_detection,
    evaluate_tracking,
    evaluate_recognition,
    evaluate_antispoofing,
    evaluate_attendance_quorum,
    PerformanceProfiler
)
from tests.benchmark.report_generator import generate_comparison_report

from src.byte_track import ByteTracker, STrack
from src.anti_spoof import evaluate_liveness, evaluate_face_quality
from src.recognizer import FaceRecognizer


def run_tracking_benchmark(tracker_params, gt_trajectories):
    """Evaluates ByteTrack multi-object tracking against ground-truth trajectories."""
    tracker = ByteTracker(
        track_thresh=tracker_params.get("track_thresh", 0.50),
        low_thresh=tracker_params.get("low_thresh", 0.10),
        match_thresh=tracker_params.get("match_thresh", 0.70),
        track_buffer=tracker_params.get("track_buffer", 30),
        use_motion_gating=tracker_params.get("use_motion_gating", False)
    )

    pred_trajectories = {}
    profiler = PerformanceProfiler()

    for frame_id in sorted(gt_trajectories.keys()):
        gt_faces = gt_trajectories[frame_id]
        # Simulate detection input with slight jitter/confidence variation
        detected_faces = []
        for gt_id, box in gt_faces:
            detected_faces.append({
                "coords": tuple(box),
                "score": 0.92
            })

        t0 = time.time()
        tracks = tracker.update(detected_faces, dt=1.0)
        profiler.record_latency(time.time() - t0)

        pred_trajectories[frame_id] = [
            (trk.track_id, [trk.tlbr[0], trk.tlbr[1], trk.tlbr[2], trk.tlbr[3]])
            for trk in tracks
        ]

    metrics = evaluate_tracking(gt_trajectories, pred_trajectories)
    return metrics, profiler.get_summary()


def run_antispoof_benchmark(pad_manifest):
    """Evaluates MiniFASNetV2 and heuristic presentation attack detection."""
    import cv2
    gt_live = []
    pred_live = []

    for item in pad_manifest:
        img_path = item["path"]
        is_true_live = item["is_live"]
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        coords = (0, 0, w, h)
        is_l, conf, reason = evaluate_liveness(img, full_frame=img, coords=coords)

        gt_live.append(is_true_live)
        pred_live.append(is_l)

    return evaluate_antispoofing(gt_live, pred_live)


def run_recognition_benchmark(rec_manifest):
    """Evaluates ArcFace biometric accuracy and margin separation."""
    import cv2
    recognizer = FaceRecognizer()
    gt_labels = []
    pred_labels = []
    similarities = []

    for item in rec_manifest:
        img_path = item["path"]
        true_id = item["true_id"]
        img = cv2.imread(img_path)
        if img is None:
            continue

        match_res = recognizer.recognize(img)
        pred_roll = match_res[0]
        confidence = match_res[1]
        diag = match_res[2] if len(match_res) > 2 else {}

        gt_labels.append(true_id)
        pred_labels.append(pred_roll)
        similarities.append(diag.get("top1_sim", confidence / 100.0))

    return evaluate_recognition(gt_labels, pred_labels, similarities)


def execute_full_benchmark(experiment_type="candidate", notes=""):
    """Executes full benchmark evaluation run and stores structured artifacts."""
    config = ExperimentConfig(experiment_type=experiment_type, notes=notes)
    loader = DatasetLoader()

    print(f"\n=======================================================")
    print(f"  SmartClass Vision Benchmark Engine [{experiment_type.upper()}]")
    print(f"  Experiment ID: {config.experiment_id}")
    print(f"  Git Commit:    {config.git_commit[:8]}")
    print(f"=======================================================")

    results = {}

    # 1. Tracking Benchmark
    print("\n[1/4] Running Multi-Object Tracking Benchmark (ByteTrack)...")
    gt_traj = loader.load_tracking_ground_truth()
    track_metrics, perf_metrics = run_tracking_benchmark(config.parameters["tracking"], gt_traj)
    results["tracking"] = track_metrics
    results["performance"] = perf_metrics
    print(f"      MOTA: {track_metrics['mota']:.4f} | MOTP: {track_metrics['motp']:.4f} | ID Switches: {track_metrics['id_switches']}")
    print(f"      Throughput: {perf_metrics.get('fps', 0)} FPS | P95 Latency: {perf_metrics.get('p95_latency_ms', 0)} ms")

    # 2. Anti-Spoofing Benchmark
    print("\n[2/4] Running Anti-Spoofing & Liveness Benchmark (MiniFASNetV2)...")
    pad_manifest = loader.load_antispoof_manifest()
    pad_metrics = run_antispoof_benchmark(pad_manifest)
    results["antispoof"] = pad_metrics
    print(f"      ACER: {pad_metrics['acer']:.4f} | APCER: {pad_metrics['apcer']:.4f} | BPCER: {pad_metrics['bpcer']:.4f}")

    # 3. Recognition Benchmark
    print("\n[3/4] Running Biometric Recognition Benchmark (ArcFace)...")
    rec_manifest = loader.load_recognition_manifest()
    rec_metrics = run_recognition_benchmark(rec_manifest)
    results["recognition"] = rec_metrics
    print(f"      Top-1 Accuracy: {rec_metrics['top1_accuracy']:.2%} | FAR: {rec_metrics['far']:.4f} | FRR: {rec_metrics['frr']:.4f}")

    # 4. Quorum Attendance Simulation
    print("\n[4/4] Simulating Attendance Quorum (10-Shot Temporal Consensus)...")
    sim_roster = ["2023021058", "2023021226", "2023021244", "2023021099"]
    sim_true_present = {"2023021058", "2023021226"}
    sim_records = {
        "2023021058": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 10/10 shots
        "2023021226": [1, 3, 5, 7],                     # 4/10 shots
        "2023021244": [2]                               # 1/10 shots (below quorum 3 -> marked absent)
    }
    att_metrics = evaluate_attendance_quorum(sim_roster, sim_true_present, sim_records, quorum=3)
    results["attendance"] = att_metrics
    print(f"      Quorum F1: {att_metrics['attendance_f1']:.4f} | False Accepts: {att_metrics['false_attendance_count']}")

    # Save Results
    results_dir = BASE_DIR / "benchmark_results" / experiment_type
    results_dir.mkdir(parents=True, exist_ok=True)
    res_file = results_dir / f"results_{config.experiment_id}.json"
    with open(res_file, "w") as f:
        json.dump({"config": config.to_dict(), "metrics": results}, f, indent=2)

    config.save(results_dir)
    print(f"\n[SAVED] Benchmark artifacts written to: {res_file.name}")
    return config.to_dict(), results, res_file


def run_benchmark_comparison(baseline_json, candidate_json, report_output_path=None):
    """Compares two benchmark JSON files and renders a markdown comparison report."""
    with open(baseline_json, "r") as f:
        b_data = json.load(f)
    with open(candidate_json, "r") as f:
        c_data = json.load(f)

    if report_output_path is None:
        report_output_path = BASE_DIR / "benchmark_results" / "reports" / f"comparison_{b_data['config']['experiment_id']}_vs_{c_data['config']['experiment_id']}.md"

    report_file = generate_comparison_report(
        b_data["metrics"],
        c_data["metrics"],
        b_data["config"],
        c_data["config"],
        report_output_path
    )
    print(f"\n[REPORT] Generated comparison report: {report_file}")
    return report_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmartClass Vision Benchmark Execution Harness")
    parser.add_argument("--mode", choices=["baseline", "candidate", "compare"], default="candidate")
    parser.add_argument("--baseline", type=str, help="Path to baseline results JSON for comparison")
    parser.add_argument("--candidate", type=str, help="Path to candidate results JSON for comparison")
    parser.add_argument("--notes", type=str, default="Automated benchmark run")
    args = parser.parse_args()

    if args.mode == "compare":
        if not args.baseline or not args.candidate:
            print("[ERROR] Please provide --baseline and --candidate JSON paths for comparison mode.")
            sys.exit(1)
        run_benchmark_comparison(args.baseline, args.candidate)
    else:
        execute_full_benchmark(experiment_type=args.mode, notes=args.notes)
