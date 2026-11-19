# 🎓 SmartClass Vision (v2.5)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![InsightFace SCRFD](https://img.shields.io/badge/Face%20Detection-InsightFace%20SCRFD--2.5G%20ONNX-brightgreen.svg)](https://github.com/deepinsight/insightface)
[![ByteTrack](https://img.shields.io/badge/Face%20Tracking-ByteTrack%20ECCV2022-yellowgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![MiniFASNetV2](https://img.shields.io/badge/Anti--Spoofing-MiniFASNetV2%20ONNX-red.svg)](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
[![ArcFace ResNet-50](https://img.shields.io/badge/Face%20Recognition-ArcFace%20w600k__r50%20ONNX-orange.svg)](https://github.com/deepinsight/insightface)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20C++-blue.svg)](https://onnxruntime.ai/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)

An automated, ultra-fast **AI-Powered Face Recognition Attendance & Classroom Management System** engineered for schools, universities, and lecture halls. Combines **InsightFace SCRFD-2.5GF**, **ByteTrack Multi-Face Tracking**, **MiniFASNetV2 Deep Anti-Spoofing (PAD)**, **Two-Tier ArcFace ResNet-50 Matching**, and **CustomTkinter** to eliminate proxy attendance, detect digital screen/photo spoofs, enforce strict section isolation, and provide tamper-proof cryptographic audit logs.

---

## 🚀 Key System Capabilities

### 1. Multi-Scale Face Detection (InsightFace SCRFD-2.5GF) & 5-Point Alignment
- **Anchor-Free Multi-Scale Detection (`det_2.5g.onnx`)**: Detects human faces down to 20x20 px, reliably recognizing students in the back rows of large lecture halls (rows 4–8 up to 8 meters away).
- **5-Point Landmark Normalization**: Warps facial landmarks (eyes, nose, mouth corners) onto standard 112x112 canonical crops using Bicubic interpolation (`INTER_CUBIC`).

### 2. Temporal Face Tracking (ByteTrack Multi-Object Associator)
- **8-State Kalman Motion Prediction**: Tracks students' face bounding boxes between video preview and snapshot frames $(x, y, a, h, v_x, v_y, v_a, v_h)$.
- **Two-Stage IoU Association**: 
  - Associates high-confidence detections ($Score \ge 0.50$).
  - Recovers occluded, blurred, or tilted faces using low-confidence detections ($Score < 0.50$).
- **Persistent Track Overlay**: Renders persistent `ID:1, ID:2, ...` tracking boxes on the live video interface.

### 3. Deep Neural Anti-Spoofing & Liveness Detection (MiniFASNetV2 ONNX)
- **Single-Image Presentation Attack Detection (`minifasnet_v2.onnx`)**: Powered by an 80x80 BGR neural network trained on multi-scale context ($2.7\times$).
- **Replay & Screen Defense**: Detects smartphone screens (iPhone/Android/OLED), tablets, laptops, and printed photographs with $>99\%$ confidence by analyzing subpixel moiré patterns and device bezels.
- **Multi-Spectral Screen-Glare Veto**: Immediately rejects harsh glass reflections and blown-out screen glares.
- **Audit Captures**: Automatically saves snapshots of blocked spoof attacks to `data/unknown_faces/` and logs critical security alerts.

### 4. Two-Tier ArcFace ResNet-50 Recognition Engine (`w600k_r50.onnx`)
- **512-Dimensional Deep Metric Hyperspace**: Extracted via ArcFace (Additive Angular Margin Loss) deep learning feature vectors trained on WebFace600K.
- **Vectorized C++ BLAS Inference**: Batched 4D tensor inference `[B, 3, 112, 112]` operating in sub-20ms per face.
- **Tier 1 (Roster-Filtered Matching)**: Filters initial matching strictly to enrolled candidates of the teacher's selected section (Similarity $\ge 0.52$, Ambiguity Margin $\ge 0.06$).
- **Tier 2 (Global Section Validation)**: Evaluates unmatched candidates against the entire institutional gallery to detect students attending from unauthorized classes/sections.

### 5. Multi-Shot Temporal Consensus & Quorum Engine
- **Classroom 10-Shot Scanning**:
  - Initial **3.0-second preparation countdown** to ensure all students face the camera.
  - Captures 10 multi-angle audit bursts with audio-visual countdown cues.
- **Strict Quorum Rule**: Requires at least 3 verified matching shots ($\ge 3/10$) to mark **PRESENT**.
- **Dual-State Persistence**: Automatically logs explicit **`Present`** and **`Absent`** records for all section-enrolled candidates.
- **Itemized Completion Dialog**: Displays verified attendees alongside discarded wrong-section students with their actual enrolled classes.

### 6. Production Model Architecture & Footprint

The system relies strictly on a minimal, highly optimized trio of production ONNX models (all auxiliary/unused models like 3D mesh `1k3d68`, 106-point `2d106det`, and gender/age classifiers have been eliminated to keep the repository lightweight):

| Production Model | ONNX Weight File | Architecture | Purpose in SmartClass Vision |
| :--- | :--- | :--- | :--- |
| **SCRFD-2.5GF** | `det_2.5g.onnx` | Anchor-free FPN detector | Ultra-fast back-row student face detection down to 20x20 px and 5-point landmark extraction |
| **ArcFace ResNet-50** | `w600k_r50.onnx` | ResNet-50 (WebFace600K) | 512-D hyperspherical biometric embeddings with additive angular margin cosine matching |
| **MiniFASNetV2** | `minifasnet_v2.onnx` | Deep PAD ConvNet | Neural anti-spoofing against digital displays (phones, tablets, laptops) and 2D printed photographs |

### 7. Enterprise Security & Cryptographic Integrity
- **HMAC-SHA256 Tamper-Proof Checksums**: Generates a cryptographic signature for every attendance session CSV file using teacher identity metadata and system salt.
- **Role-Based Access Control (RBAC)**: Secure password hashing via PBKDF2-SHA256 with individual cryptographic salts.

---

## 🛠️ Technology Stack

| Subsystem | Technology | Purpose |
| :--- | :--- | :--- |
| **GUI Framework** | CustomTkinter | Responsive dark-mode desktop interface |
| **Face Detection** | InsightFace SCRFD-2.5GF (ONNX) | Multi-scale face detection down to 20px and 5-point landmark extraction |
| **Face Tracking** | ByteTrack (ECCV 2022) | 8-state Kalman Filter + 2-stage IoU multi-face tracking |
| **Deep Anti-Spoofing** | MiniFASNetV2 (ONNX) | Presentation Attack Detection (screens, phones, print attacks) |
| **Face Recognition** | InsightFace ArcFace ResNet-50 | 512-D angular margin biometric embeddings (>99.8% LFW accuracy) |
| **Inference Engine** | ONNX Runtime (`onnxruntime`) | High-speed C++ CPU SIMD inference |
| **Computer Vision** | OpenCV (cv2), NumPy, SciPy | Image transformations, DirectShow capture, linear assignment |
| **Database** | SQLite3, Pandas, Pickle | Student database, subject enrollment, centroid storage |
| **Security** | PBKDF2-SHA256, HMAC-SHA256 | Credential hashing and tamper-proof attendance sheets |

---

## 📂 Project Architecture

```text
SmartClassVision/
├── data/
│   ├── attendance_records/     # Cryptographically signed CSV sheets
│   ├── backups/                # SQLite database snapshots
│   ├── class_photos/           # Timestamped 10-shot session audit photos
│   ├── database/               # Master SQLite database (smartclass.db)
│   ├── known_faces/            # Student enrollment photos (by Roll Number)
│   ├── unknown_faces/          # Spoof incident snapshots and audit captures
│   └── embeddings.pkl          # Quality-filtered ArcFace centroids for all students
├── models/
│   ├── minifasnet_v2.onnx      # Deep Anti-Spoofing presentation attack detection model
│   └── models/buffalo_m/       # InsightFace SCRFD-2.5G & ArcFace ResNet-50 weights
├── src/
│   ├── anti_spoof.py           # MiniFASNetV2 deep PAD & multi-spectral quality filter
│   ├── attendance_logic.py     # 10-shot multi-angle attendance engine with ByteTrack integration
│   ├── byte_track.py           # ByteTrack multi-face tracking engine (Kalman filter + IoU)
│   ├── database.py             # SQLite schema, HMAC signatures, RBAC queries
│   ├── detector.py             # SCRFD-2.5GF ONNX detector with bicubic affine alignment
│   ├── recognizer.py           # ArcFace ResNet-50 ONNX inference & centroid cosine matching
│   └── registration.py         # 5-pose guided student dataset registration
├── utils/
│   └── config.py               # Central threshold and path configuration
├── tests/                      # Automated fault diagnostic and adversarial challenge suites
├── gui.py                      # Main desktop application interface
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows quick launch script
└── README.md                   # Project documentation
```

---

## ⚙️ Quick Start Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/sandeepsagar18/Latest_Ai_system.git
cd Latest_Ai_system

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python gui.py
```
*(Or double-click `run.bat` on Windows)*

---

## 🔑 Default Credentials

On initial startup, the database creates a default administrative account:

| Role | Employee ID | Default Password |
| :--- | :--- | :--- |
| **System Administrator** | `ADMIN01` | `Admin123` |

> ⚠️ **Note**: Change the default admin password after initial login. Master Key for registering new admin accounts: `SmartClass@Admin#2026`.

---

## 👨‍💻 Author & Maintainer

Developed & Maintained by **Sandeep Sagar**  
- **GitHub**: [@sandeepsagar18](https://github.com/sandeepsagar18)  
- **Repository**: [Latest_Ai_system](https://github.com/sandeepsagar18/Latest_Ai_system.git)  
- **Email**: sandeepsagarkumar85@gmail.com  

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
