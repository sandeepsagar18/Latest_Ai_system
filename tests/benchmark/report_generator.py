"""
SmartClass Vision - Automated Markdown & HTML Benchmark Comparison Report Generator
Generates comprehensive comparative tables showing delta improvements, regressions,
and pass/fail status between Baseline and Candidate experiments.
"""

from pathlib import Path
from datetime import datetime


def generate_comparison_report(baseline_metrics, candidate_metrics, baseline_config, candidate_config, output_path):
    """
    Renders an evaluation report comparing Candidate against Baseline.
    Highlights statistically significant improvements and flags any performance regressions.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Helper to format delta
    def fmt_delta(cand_val, base_val, higher_is_better=True):
        if base_val is None or cand_val is None:
            return "N/A"
        diff = cand_val - base_val
        pct = (diff / max(1e-6, abs(base_val))) * 100.0 if base_val != 0 else 0.0
        sign = "+" if diff >= 0 else ""
        improved = (diff >= 0) if higher_is_better else (diff <= 0)
        indicator = "🟢" if improved else ("🔴" if abs(diff) > 1e-4 else "⚪")
        return f"{cand_val} ({sign}{diff:.3f}, {sign}{pct:.1f}%) {indicator}"

    lines = [
        "# 📊 SmartClass Vision - Evaluation & Benchmark Comparison Report",
        f"**Generated:** {timestamp}  ",
        f"**Baseline ID:** `{baseline_config.get('experiment_id')}` (Commit: `{baseline_config.get('git_commit', 'N/A')[:8]}`)  ",
        f"**Candidate ID:** `{candidate_config.get('experiment_id')}` (Commit: `{candidate_config.get('git_commit', 'N/A')[:8]}`)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Quality Verdict",
        "",
        "| Subsystem / Metric | Baseline | Candidate | Delta Improvement | Status |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    # Tracking MOTA
    b_mota = baseline_metrics.get("tracking", {}).get("mota", 0.0)
    c_mota = candidate_metrics.get("tracking", {}).get("mota", 0.0)
    lines.append(f"| **Tracking MOTA** | {b_mota:.3f} | {c_mota:.3f} | {fmt_delta(c_mota, b_mota, True)} | {'PASS' if c_mota >= b_mota else 'REGRESSION'} |")

    # Tracking ID Switches
    b_idsw = baseline_metrics.get("tracking", {}).get("id_switches", 0)
    c_idsw = candidate_metrics.get("tracking", {}).get("id_switches", 0)
    lines.append(f"| **Tracking ID Switches** | {b_idsw} | {c_idsw} | {fmt_delta(c_idsw, b_idsw, False)} | {'PASS' if c_idsw <= b_idsw else 'REGRESSION'} |")

    # Anti-Spoofing ACER
    b_acer = baseline_metrics.get("antispoof", {}).get("acer", 0.0)
    c_acer = candidate_metrics.get("antispoof", {}).get("acer", 0.0)
    lines.append(f"| **Anti-Spoofing ACER** | {b_acer:.3f} | {c_acer:.3f} | {fmt_delta(c_acer, b_acer, False)} | {'PASS' if c_acer <= b_acer else 'REGRESSION'} |")

    # Recognition Top-1 Accuracy
    b_rec = baseline_metrics.get("recognition", {}).get("top1_accuracy", 0.0)
    c_rec = candidate_metrics.get("recognition", {}).get("top1_accuracy", 0.0)
    lines.append(f"| **Recognition Top-1** | {b_rec:.3f} | {c_rec:.3f} | {fmt_delta(c_rec, b_rec, True)} | {'PASS' if c_rec >= b_rec else 'REGRESSION'} |")

    # Throughput (FPS)
    b_fps = baseline_metrics.get("performance", {}).get("fps", 0.0)
    c_fps = candidate_metrics.get("performance", {}).get("fps", 0.0)
    lines.append(f"| **System FPS (Throughput)** | {b_fps:.1f} | {c_fps:.1f} | {fmt_delta(c_fps, b_fps, True)} | {'PASS' if c_fps >= b_fps * 0.9 else 'WARN (Speed)'} |")

    # Latency P95
    b_lat = baseline_metrics.get("performance", {}).get("p95_latency_ms", 0.0)
    c_lat = candidate_metrics.get("performance", {}).get("p95_latency_ms", 0.0)
    lines.append(f"| **P95 Latency (ms)** | {b_lat:.1f}ms | {c_lat:.1f}ms | {fmt_delta(c_lat, b_lat, False)} | {'PASS' if c_lat <= b_lat * 1.15 else 'WARN'} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Detailed Subsystem Evaluations",
        "",
        "### A. Multi-Object Tracking & Kalman Dynamics (ByteTrack)",
        f"- **Candidate MOTA**: `{c_mota:.4f}` (MOTP: `{candidate_metrics.get('tracking', {}).get('motp', 0.0):.4f}`)",
        f"- **ID Switches**: `{c_idsw}` vs Baseline `{b_idsw}`",
        f"- **False Positives**: `{candidate_metrics.get('tracking', {}).get('false_positives', 0)}`",
        f"- **False Negatives**: `{candidate_metrics.get('tracking', {}).get('false_negatives', 0)}`",
        "",
        "### B. Deep Anti-Spoofing & Liveness (MiniFASNetV2)",
        f"- **APCER (Attack Error Rate)**: `{candidate_metrics.get('antispoof', {}).get('apcer', 0.0):.4f}`",
        f"- **BPCER (Bona Fide Error Rate)**: `{candidate_metrics.get('antispoof', {}).get('bpcer', 0.0):.4f}`",
        f"- **ACER (Combined Error Rate)**: `{c_acer:.4f}`",
        "",
        "### C. Biometric Recognition (ArcFace ResNet-50)",
        f"- **Top-1 Identification Accuracy**: `{c_rec:.2%}`",
        f"- **False Acceptance Rate (FAR)**: `{candidate_metrics.get('recognition', {}).get('far', 0.0):.4f}`",
        f"- **False Rejection Rate (FRR)**: `{candidate_metrics.get('recognition', {}).get('frr', 0.0):.4f}`",
        f"- **Mean Cosine Similarity**: `{candidate_metrics.get('recognition', {}).get('avg_similarity', 0.0):.4f}`",
        "",
        "### D. System Performance & Compute Footprint",
        f"- **Mean Latency**: `{candidate_metrics.get('performance', {}).get('mean_latency_ms', 0.0):.1f} ms`",
        f"- **95th Percentile Latency**: `{c_lat:.1f} ms`",
        f"- **99th Percentile Latency**: `{candidate_metrics.get('performance', {}).get('p99_latency_ms', 0.0):.1f} ms`",
        f"- **Total RAM Resident Set**: `{candidate_metrics.get('performance', {}).get('ram_usage_mb', 0.0):.1f} MB`",
        "",
        "---",
        "*Report automatically compiled by SmartClass Vision Continuous Evaluation Framework.*"
    ])

    report_content = "\n".join(lines)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(report_content)

    return p
