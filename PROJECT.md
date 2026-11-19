# Project: SmartClass Vision Biometric Attendance System Audit & Remediation

## Architecture
SmartClass Vision consists of an automated multi-stage biometric processing pipeline:
1. **Camera Ingest**: Video frame capture (`src/attendance_logic.py`) via OpenCV.
2. **Face Detection & Landmark Localization**: Primary SCRFD-2.5G (`det_2.5g.onnx`) with 5-point facial landmarks (`src/detector.py`), fallback to YuNet / Haar.
3. **Quality & Pose Gating**: Multidimensional validation of crop scale, Laplacian blur variance, luminance, contrast, and landmark yaw asymmetry (`src/anti_spoof.py`, `evaluate_face_quality`).
4. **Anti-Spoofing & Liveness**: Multi-criteria passive liveness assessment inspecting texture variance (Laplacian), chromatic skin loci (YCrCb/HSV), and geometric depth gradients (Sobel) (`src/anti_spoof.py`, `LivenessDetector`).
5. **Feature Representation & Vector Matching**: ArcFace ResNet-50 (`w600k_r50.onnx`) generating 512-D L2-normalized embeddings, matched via cosine similarity against enrolled student centroids with an Ambiguity Margin Guard (`src/recognizer.py`).
6. **Consensus & Session Attendance**: 10-shot batch capture session with a 3-second preparation buffer, requiring a consistent voting quorum (>=3 detections out of 10 shots with mean similarity >=0.52) and section validation before committing cryptographically signed HMAC-SHA256 attendance logs and SQLite records (`src/attendance_logic.py`, `src/database.py`).

## Code Layout
- `src/detector.py`: Face detection (SCRFD-2.5G, YuNet), landmark regression, canonical 112x112 affine transformation.
- `src/recognizer.py`: ArcFace ResNet-50 feature extraction, centroid gallery management, cosine matching, ambiguity margin guard, batch inference.
- `src/attendance_logic.py`: Multi-shot capture session, countdown timers, voting quorum consensus, HMAC-SHA256 logging, SQLite attendance persistence.
- `src/anti_spoof.py`: Quality gating, invariant head pose evaluation, multi-spectral passive liveness (texture, chroma, depth gradient).
- `utils/config.py`: Global configurations, thresholds, model paths, camera settings, session parameters.
- `tests/`: Automated stress tests, diagnostic test suites, and hardware benchmarks.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F01 | High-Density Crowd Stress Test | Evaluate SCRFD & ArcFace with 15, 30, and 60 faces simultaneously | M1 | ORIGINAL_REQUEST §R1 |
| F02 | Distance Degradation & Multi-Scale Stress | Evaluate detection & recognition across Rows 1-8 (1m to 8-10m, 16-25px bboxes) | M1 | ORIGINAL_REQUEST §R1 |
| F03 | Mutual Occlusion Stress Test | Evaluate facial overlap and partial occlusions in crowded rows | M1 | ORIGINAL_REQUEST §R1 |
| F04 | Invariant Yaw Calculation Fix (V01) | Fix yaw ratio by projecting nose-mid vector onto interocular vector (roll invariant) | M2 | Codebase Audit V01 |
| F05 | Multiscale SCRFD & Resolution Fix (V02) | Configure dynamic/multiscale det_size and scale-adaptive min face thresholds | M2 | Codebase Audit V02 |
| F06 | Affine Transform Fallback Fix (V03) | Prevent squashing full frame to 112x112; fallback gracefully to raw face crop | M2 | Codebase Audit V03 |
| F07 | Enrolled Section Roster Filtering (V04) | Restrict centroid matching to `candidate_rolls` to avoid DB-wide ambiguity spikes | M2 | Codebase Audit V04 |
| F08 | Distinct Shot Quorum Consensus (V05) | Track unique shot indices per student to prevent intra-shot duplicate vote inflation | M2 | Codebase Audit V05 |
| F09 | True Vectorized 4D Batch Inference (V06) | Pass [B, 3, 112, 112] tensor to ArcFace ONNX and vectorize cosine matching | M2 | Codebase Audit V06 |
| F10 | Config Parameter Consistency Fix (V07) | Synchronize `evaluate_face_quality` defaults with `config.py` | M2 | Codebase Audit V07 |
| F11 | Clean Frame Warping Fix (V08) | Draw visualization bounding boxes on a copy, avoiding green line crop corruption | M2 | Codebase Audit V08 |
| F12 | Absent Student Logging (V09) | Record explicit "Absent" entries in SQLite database for enrolled students failing quorum | M2 | Codebase Audit V09 |
| F13 | Eyeglass Glare & Chroma Robustness (V10, V11) | Prevent false spoof alarms on prescription glasses and broaden skin chromatic loci | M2 | Codebase Audit V10, V11 |
| F14 | Scale-Adaptive Quality & Anti-Spoof (V02, V12) | Normalize blur variance by crop scale, prevent false spoof alerts on distant faces | M2 | Codebase Audit V12 |
| F15 | Camera DirectShow Backend (V13) | Support Windows DirectShow backend (`cv2.CAP_DSHOW`) | M2 | Codebase Audit V13 |
| F16 | Automated Diagnostic Scripts Suite | Standalone reproducible diagnostic scripts verifying each fault V01-V15 | M3 | ORIGINAL_REQUEST §R3 |
| F17 | Production Code Remediation | Implement and integrate tested production-ready fixes for all diagnosed faults | M3 | ORIGINAL_REQUEST §R3 |
| F18 | 1080p vs 4K Empirical Sensor Benchmark | Benchmark detection/recognition distance limits, latency, FPS, memory | M4 | ORIGINAL_REQUEST §R4 |
| F19 | 50+ Student Batch Latency & FPS Profile | Measure end-to-end latency for 50+ student batch inference | M4 | ORIGINAL_REQUEST §R4 |
| F20 | Classroom Deployment Guide | Practical guide for mounting height, tilt angle, lens FoV, lux lighting, seating limits | M4 | ORIGINAL_REQUEST §R4 |
| F21 | Final Verification Gate & Audit | 100% test pass, independent reviewer approvals, adversarial challenge, clean forensic audit | M5 | ORIGINAL_REQUEST Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Crowd & Multi-Scale Stress Testing Suite | Comprehensive synthetic/empirical stress test harness (15, 30, 60 faces, rows 1-8 / 1-10m, 16-25px, occlusions) | none | IN_PROGRESS |
| M2 | Fault Remediation & Performance Engineering | Implement production fixes for V01-V15 across detector, recognizer, attendance_logic, anti_spoof, config | M1 | PLANNED |
| M3 | Automated Fault Diagnostics Suite | Standalone automated reproduction scripts for each vulnerability V01-V15 | M1 | PLANNED |
| M4 | Hardware Benchmark & Classroom Deployment Guide | Empirical 1080p vs 4K benchmark, 50+ student latency/FPS, comprehensive deployment specification | M2 | PLANNED |
| M5 | Final Verification & Forensic Audit Gate | End-to-end regression validation, Reviewer & Challenger verification, Forensic Integrity Audit | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### `src/detector.py`
- `InsightFaceDetector.detect_faces(frame: np.ndarray, det_size: tuple = None) -> List[dict]`
  - Output dict keys: `box` (x1, y1, x2, y2), `confidence` (float), `landmarks` (5x2 np.ndarray), `aligned_face` (112x112 np.ndarray)
  - Contract: Bounding boxes must NOT be drawn on input `frame`; alignment fallback must crop `raw_crop`, not full frame; dynamic `det_size` supported.

### `src/recognizer.py`
- `ArcFaceRecognizer.recognize_batch(aligned_faces: List[np.ndarray], candidate_rolls: List[str] = None) -> List[Tuple[str, float, dict]]`
  - Contract: Batched 4D tensor inference `[B, 3, 112, 112]`; centroid matching restricted to `candidate_rolls` if provided; returns list of `(roll_number, similarity, diag)`.

### `src/attendance_logic.py`
- Quorum calculation: Unique shot indices `student_shots[roll].add(shot_idx)`; quorum met if `len(student_shots[roll]) >= REQUIRED_MATCHES`.
- Persistence: Insert explicit "Absent" records for enrolled students failing quorum.

### `src/anti_spoof.py`
- `evaluate_face_quality(face_crop, landmarks, ...)`
  - Invariant yaw ratio: $(\mathbf{p}_{\text{nose}} - \mathbf{p}_{\text{mid}}) \cdot \mathbf{u} / d_{\text{eyes}}$
  - Scale-adaptive blur: normalize Laplacian variance by face crop resolution.
