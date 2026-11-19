"""
SmartClass Vision - Benchmark Experiment Configuration & Metadata Tracker
Tracks hardware specs, model SHA-256 hashes, algorithm parameters, and reproducible seeds.
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime

from utils.config import (
    BASE_DIR, MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_CONTRAST, MAX_YAW_RATIO, ARCFACE_SIMILARITY_THRESHOLD, AMBIGUITY_MARGIN
)


def compute_sha256(filepath):
    """Computes SHA-256 hash of a file if it exists."""
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash():
    """Retrieves current git commit hash if available."""
    try:
        git_cmd = BASE_DIR / "tools" / "git" / "cmd" / "git.exe"
        cmd = [str(git_cmd), "rev-parse", "HEAD"] if git_cmd.exists() else ["git", "rev-parse", "HEAD"]
        res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_OR_UNTRACKED"


class ExperimentConfig:
    """
    Structured metadata representing a single benchmark evaluation experiment.
    Guarantees full reproducibility and tracking of all hyperparameters and model hashes.
    """
    def __init__(
        self,
        experiment_type="candidate",  # 'baseline' or 'candidate'
        experiment_name=None,
        dataset_version="v1.0",
        random_seed=42,
        notes=""
    ):
        self.experiment_type = experiment_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_type}_{self.timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        self.experiment_name = experiment_name or f"Benchmark_{self.experiment_id}"
        self.dataset_version = dataset_version
        self.random_seed = random_seed
        self.notes = notes

        # Git commit
        self.git_commit = get_git_commit_hash()

        # Environment & Hardware
        self.environment = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "processor": platform.processor(),
            "machine": platform.machine(),
            "os": platform.system()
        }

        # Model Paths & Hashes
        models_dir = BASE_DIR / "models"
        buffalo_dir = models_dir / "models" / "buffalo_m"
        det_path = buffalo_dir / "det_2.5g.onnx"
        rec_path = buffalo_dir / "w600k_r50.onnx"
        pad_path = models_dir / "minifasnet_v2.onnx"

        self.models = {
            "detector": {
                "name": "SCRFD-2.5GF",
                "path": str(det_path),
                "sha256": compute_sha256(det_path)
            },
            "recognizer": {
                "name": "ArcFace ResNet-50",
                "path": str(rec_path),
                "sha256": compute_sha256(rec_path)
            },
            "antispoof": {
                "name": "MiniFASNetV2",
                "path": str(pad_path),
                "sha256": compute_sha256(pad_path)
            }
        }

        # Hyperparameters and Thresholds
        self.parameters = {
            "detection": {
                "det_size": [640, 640],
                "min_face_size": MIN_FACE_SIZE,
                "blur_threshold": BLUR_THRESHOLD,
                "min_brightness": MIN_BRIGHTNESS,
                "max_brightness": MAX_BRIGHTNESS,
                "min_contrast": MIN_CONTRAST,
                "max_yaw_ratio": MAX_YAW_RATIO
            },
            "tracking": {
                "track_thresh": 0.50,
                "low_thresh": 0.10,
                "match_thresh": 0.70,
                "track_buffer": 30,
                "use_motion_gating": False,
                "gating_threshold": 14.45
            },
            "recognition": {
                "cosine_threshold": ARCFACE_SIMILARITY_THRESHOLD,
                "ambiguity_margin": AMBIGUITY_MARGIN,
                "embedding_dim": 512,
                "vectorized_batch": True
            },
            "antispoof": {
                "liveness_threshold": 0.55,
                "texture_threshold": 35.0,
                "minifasnet_real_thresh": 0.75
            },
            "attendance": {
                "num_shots": 10,
                "quorum_threshold": 3,
                "shot_interval_sec": 1.2
            }
        }

    def to_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "notes": self.notes,
            "environment": self.environment,
            "models": self.models,
            "parameters": self.parameters
        }

    def save(self, output_dir):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"config_{self.experiment_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_file
