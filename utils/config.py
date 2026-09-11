import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
KNOWN_FACES_DIR = BASE_DIR / "data" / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "data" / "unknown_faces"
ATTENDANCE_DIR = BASE_DIR / "data" / "attendance_records"
CLASS_PHOTOS_DIR = BASE_DIR / "data" / "class_photos"

# Model Directories
YUNET_MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DEEPFACE_DIR = BASE_DIR / "models"

# Camera Settings
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.50
ATTENDANCE_WINDOW_MINUTES = 15

# Biometric & Computer Vision Quality Gates
MIN_FACE_SIZE = 45               # Minimum face width/height in pixels (allows background/seated students)
BLUR_THRESHOLD = 80.0            # Minimum Laplacian variance for motion blur rejection
MIN_BRIGHTNESS = 35.0            # Minimum average luminance (reject underexposed)
MAX_BRIGHTNESS = 230.0           # Maximum average luminance (reject overexposed)
MIN_CONTRAST = 18.0              # Minimum standard deviation of pixels (allows varied room lighting)
MAX_YAW_RATIO = 0.42             # Landmark asymmetry cutoff: abs(NoseX - MidEyesX) / EyeDistance (tolerates natural head turns)

# SCRFD Detection & ArcFace Recognition Thresholds (Buffalo_M)
SCRFD_SCORE_THRESHOLD = 0.50     # SCRFD face detection confidence cutoff (detects distant/slanted faces)
YUNET_SCORE_THRESHOLD = 0.60     # Legacy fallback cutoff
ARCFACE_SIMILARITY_THRESHOLD = 0.52 # Optimized for distance (4-8m) while ambiguity margin guards against false swaps
AMBIGUITY_MARGIN = 0.06          # Minimum separation between top-1 and top-2 candidate similarities

# Temporal Consensus & Attendance Logic
TOTAL_CAPTURE_SHOTS = 10         # Total multi-angle shots per session (~35 seconds total capture window)
REQUIRED_CONSISTENT_VOTES = 3    # Minimum consistent matching images required out of 10 shots to mark attendance
SHOT_COUNTDOWN_SECONDS = 3.5     # Countdown interval between shots (gives 50-60 students ample reaction time)

# Features
ENABLE_5POINT_ALIGNMENT = True   # Standard ArcFace 112x112 affine transformation
DEBUG_MODE = True                # Render rich real-time diagnostic overlay