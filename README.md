# 🎓 SmartClass Vision (v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![InsightFace SCRFD](https://img.shields.io/badge/Face%20Detection-InsightFace%20SCRFD--2.5G%20ONNX-brightgreen.svg)](https://github.com/deepinsight/insightface)
[![ArcFace ResNet-50](https://img.shields.io/badge/Face%20Recognition-ArcFace%20w600k__r50%20ONNX-orange.svg)](https://github.com/deepinsight/insightface)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20C++-blue.svg)](https://onnxruntime.ai/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey.svg)](https://www.sqlite.org/)

An automated, ultra-fast **AI-Powered Face Recognition Attendance & Classroom Management System** built with **InsightFace SCRFD-2.5GF**, **ArcFace ResNet-50 (w600k_r50.onnx)**, **ONNX Runtime (C++ Execution)**, and **CustomTkinter**. Designed specifically for schools, colleges, and lecture halls to eliminate proxy attendance, enforce strict section isolation, provide hardware-independent anti-spoofing defense, and deliver comprehensive institutional control.

---

## 🚀 Key System Capabilities

### 1. InsightFace SCRFD-2.5GF Face Detection & 5-Point Bicubic Alignment
- **Anchor-Free Multi-Scale Detection (det_2.5g.onnx)**: Detects human faces across a wide range of distances (down to 16x16 px), reliably capturing students sitting in the back rows of large lecture halls (up to 8 meters).
- **5-Point Landmark Normalization**: Automatically extracts 5 key facial coordinates (both eyes, nose tip, both mouth corners) and applies a partial affine transformation with **Bicubic interpolation (INTER_CUBIC)** to warp faces onto standard 112x112 canonical crops before recognition.

### 2. InsightFace ArcFace ResNet-50 Biometric Engine (w600k_r50.onnx)
- **512-Dimensional Deep Metric Hyperspace**: Live faces are encoded with ArcFace (Additive Angular Margin Loss) deep learning feature vectors trained on the massive WebFace600K dataset (12M images, 600K identities).
- **Sub-20ms Native ONNX Inference**: Operates directly through Microsoft onnxruntime with C++ SIMD vectorization (~18ms per face), completely bypassing TensorFlow/Keras overhead.
- **Centroid Profile Matching**: Computes Cosine Similarity against quality-filtered student centroid models.
- **Ambiguity Margin Guard**: Requires top match to hold a >= 0.06 margin over the second-place candidate to avoid false identity swaps.

### 3. Multi-Shot Temporal Consensus Engine
- **Classroom-Optimized Scanning (~35s total)**: 
  - Features an initial **3.0-second preparation buffer** so all students can look towards the camera.
  - Captures 10 multi-angle audit frames with a 3.5-second interval between snaps.
- **Consensus Rule**: A student must achieve at least 3 high-similarity matches across the 10 shots to pass consensus and be marked **PRESENT**.
- **Real-Time Section Roster Update**: Immediately updates attendance status in the dashboard and flags absentees.

### 4. Enterprise Security & Anti-Spoofing
- **Face Quality Gates**: Rejects motion blur via Laplacian variance (>= 80.0), enforces balanced illumination, and discards extreme side-profile poses (MAX_YAW_RATIO = 0.42).
- **Anti-Spoofing Defense**: Texture and chromatic balance analysis to detect printed photo attacks and screen replays.
- **Strict Section & Class Isolation**: Only marks attendance for students registered in the teacher's selected Degree, Year, Branch, and Section. Out-of-section students are tagged **REJECTED - WRONG CLASS** and denied attendance.
- **Cryptographic HMAC-SHA256 Signing**: Every generated CSV attendance sheet is digitally signed with an HMAC-SHA256 signature to guarantee tamper-proof audit records.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **GUI Framework** | CustomTkinter | Modern dark-mode responsive desktop interface |
| **Face Detection** | InsightFace SCRFD-2.5GF (ONNX) | Multi-scale face detection down to 16px and 5-point landmark extraction |
| **Face Alignment** | Affine Partial 2D (INTER_CUBIC) | Standardized 112x112 canonical face crops |
| **Face Recognition** | InsightFace ArcFace ResNet-50 (w600k_r50.onnx) | 512-D angular margin biometric embeddings (>99.8% LFW accuracy) |
| **Inference Engine** | ONNX Runtime (onnxruntime) | High-speed C++ CPU SIMD inference (~18ms latency) |
| **Computer Vision** | OpenCV (cv2), NumPy, PIL | Image transformations, Laplacian filtering, video streaming |
| **Database** | SQLite3, Pandas, Pickle | Student database, subject enrollment, centroid storage |
| **Security** | Argon2 / PBKDF2, HMAC-SHA256 | Credential hashing and tamper-proof attendance sheets |

---

## 📂 Project Architecture

```text
SmartClassVision/
├── data/
│   ├── attendance_records/     # Auto-generated and cryptographically signed CSV sheets
│   ├── backups/                # SQLite database snapshots
│   ├── class_photos/           # Timestamped 10-shot session audit photos
│   ├── database/               # Master SQLite database (smartclass.db)
│   ├── known_faces/            # 5-shot student registration photos (by Roll Number)
│   ├── unknown_faces/          # Spoof incidents and audit captures
│   └── embeddings.pkl          # Quality-filtered ArcFace centroids for all students
├── src/
│   ├── anti_spoof.py           # Texture analysis, blur, and pose quality gates
│   ├── attendance_logic.py     # 10-shot multi-angle attendance engine with 3s prep buffer
│   ├── database.py             # SQLite schema, HMAC signatures, RBAC queries
│   ├── detector.py             # SCRFD-2.5GF ONNX detector with bicubic affine alignment
│   ├── recognizer.py           # ArcFace ResNet-50 ONNX inference & centroid cosine matching
│   └── registration.py         # 5-pose guided student dataset registration
├── utils/
│   └── config.py               # Central threshold and path configuration
├── tests/                      # Automated system and security test suites
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
| **System Administrator** | ADMIN01 | dmin123 |

> ⚠️ **Note**: Change the default admin password after initial login. Master Key for registering new admin accounts: SmartClass@Admin#2026.

---

## 👨‍💻 Author & Maintainer

Developed & Maintained by **Sandeep Sagar**  
- **GitHub**: [@sandeepsagar18](https://github.com/sandeepsagar18)  
- **Repository**: [Latest_Ai_system](https://github.com/sandeepsagar18/Latest_Ai_system.git)  
- **Email**: sandeepsagarkumar85@gmail.com  

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
