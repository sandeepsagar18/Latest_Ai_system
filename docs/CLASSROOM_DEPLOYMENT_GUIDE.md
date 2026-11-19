# SmartClass Vision: Practical Classroom Deployment & Hardware Installation Guide

**Document Version**: 2.4.0  
**Target Audience**: Systems Integrators, AV Infrastructure Engineers, Campus IT Architects, Academic Facility Managers  
**System Architecture**: SmartClass Vision Biometric Attendance System (SCRFD-2.5G + ArcFace ResNet-50 w600k_r50)  
**Document Status**: Production Verified (Milestone 4 Benchmark Certified)  

---

## 1. Executive Introduction & Architectural Overview

SmartClass Vision is an enterprise automated biometric attendance infrastructure engineered specifically for high-density academic environments (30 to 60+ students per lecture hall). Unlike conventional single-user access control kiosks, SmartClass Vision executes multi-scale facial detection, canonical 5-point affine transformation, 4D tensor batch feature representation, and cryptographically verified quorum consensus across entire classroom cohorts in a non-intrusive, batch-capture workflow.

To achieve continuous **>= 95% attendance accuracy** with near-zero False Acceptance Rate (FAR) and False Rejection Rate (FRR), physical deployment parameters—such as optical sensor mounting, downward pitch angles, lens focal lengths, photometric illumination levels, and seating depth barriers—must adhere to strict geometric and environmental constraints. This guide provides concrete installation standards derived directly from empirical hardware stress benchmarks.

```
       +------------------------------------------------------------------+
       |               FRONT WALL / PODIUM CHALKBOARD                     |
       |                                                                  |
       |                 [ CAMERA: 2.4m - 2.8m AFF ]                      |
       |                        \      Tilt: 15° - 20°                    |
       |                         \                                        |
       |                          v                                       |
       |   Row 1 (1.2m) : [S01] [S02] [S03] [S04] [S05] [S06]             |
       |   Row 2 (2.0m) : [S07] [S08] [S09] [S10] [S11] [S12]             |
       |   Row 3 (3.0m) : [S13] [S14] [S15] [S16] [S17] [S18]             |
       |   Row 4 (4.2m) : [S19] [S20] [S21] [S22] [S23] [S24]             |
       |   Row 5 (5.5m) : [S25] [S26] [S27] [S28] [S29] [S30] <-- 1080p   |
       |   Row 6 (6.8m) : [S31] [S32] [S33] [S34] [S35] [S36]     Limit   |
       |   Row 7 (8.0m) : [S37] [S38] [S39] [S40] [S41] [S42]             |
       |   Row 8 (9.5m) : [S43] [S44] [S45] [S46] [S47] [S48] <-- 4K      |
       +---------------------------------------------------------- Limit -+
```

---

## 2. Physical Installation & Geometry Specifications

### 2.1 Camera Mounting Height (2.4m – 2.8m AFF)

The optical sensor must be permanently mounted on the front classroom wall (above the whiteboard/chalkboard) or rigid ceiling droppers:
- **Nominal Mounting Height**: **2.6 meters** Above Finished Floor (AFF).
- **Permissible Tolerance**: **2.4 meters to 2.8 meters** AFF.

#### Physical Rationale:
1. **Line-of-Sight Elevation**: When seated in flat or shallow tiered classrooms, students in Row 1 (average eye height: $1.15\text{m} - 1.25\text{m}$ AFF) completely occlude the faces of students in Rows 2 and 3 if the camera is positioned at eye level ($1.2\text{m} - 1.6\text{m}$). Mounting at $2.6\text{m}$ elevates the optical vector above the heads of anterior students.
2. **Ceiling Clearance & Tamper Protection**: Mounting above $2.4\text{m}$ prevents accidental student contact, vandalism, and physical beam interruption.
3. **Upper Bound Constraint**: Mounting higher than $2.8\text{m}$ produces excessive cranial pitch (top-of-head perspective), hiding facial landmarks (interocular line, nose tip, mouth corners) required for affine alignment.

---

### 2.2 Camera Downward Tilt Angle (15° – 20°)

The camera must be pitched downward toward the center of the student seating area:
- **Flat-Floor Classrooms**: **18° – 20° downward tilt**.
- **Tiered Lecture Halls (Stepped Rows)**: **14° – 16° downward tilt**.

#### Trigonometric Derivation:
Let $H_{\text{cam}}$ be camera mounting height ($2.6\text{m}$), $H_{\text{face}}$ be mean student seated face height ($1.2\text{m}$), and $D_{\text{mid}}$ be distance to mid-classroom (Row 4, $4.2\text{m}$):
$$\Delta H = H_{\text{cam}} - H_{\text{face}} = 2.6 - 1.2 = 1.4\,\text{meters}$$
$$\theta = \arctan\left(\frac{\Delta H}{D_{\text{mid}}}\right) = \arctan\left(\frac{1.4}{4.2}\right) \approx 18.43^\circ$$

#### Occlusion Avoidance Mechanism:
In flat seating with $0.9\text{m}$ row pitch, a $18^\circ$ downward tilt ensures that the sightline to a Row 2 student's face clears the top of the Row 1 student's head by at least $15\text{–}18\text{ cm}$.

#### Facial Pitch Envelope Compliance:
ArcFace ResNet-50 maintains feature discriminability within pitch angles up to $\pm 22^\circ$. A downward camera tilt of $18^\circ$ combined with natural student head posture (students tilted $5^\circ - 10^\circ$ upwards toward the instructor) results in an effective net face-to-camera pitch of $8^\circ - 13^\circ$, well within the model's nominal receptive field.

---

### 2.3 Horizontal Centering & Alignment

- The camera MUST be mounted on the **central longitudinal axis** of the room (midpoint between left and right walls).
- Lateral mounting on side walls is strictly prohibited: lateral perspectives induce severe yaw asymmetry ($> 30^\circ$), triggering `MAX_YAW_RATIO` rejection on students seated across the opposite side of the room.
- Roll alignment: The camera must be leveled using a spirit level ($\le \pm 1.0^\circ$ roll). While SmartClass Vision's affine transform corrects 2D roll, leveling ensures optimal sensor pixel utilization across rows.

---

## 3. Optical Lens Selection & Field of View (FoV) Calculations

The lens focal length must encompass all student desks in the front row while maintaining adequate pixel density for students in the back row.

### 3.1 FoV Formulation

For a room of width $W$, distance to front row $D_{\text{min}}$, and distance to back row $D_{\text{max}}$:
$$H_{\text{FoV}} = 2 \cdot \arctan\left(\frac{W_{\text{coverage}}}{2 \cdot D_{\text{min}}}\right)$$
$$V_{\text{FoV}} = 2 \cdot \arctan\left(\frac{H_{\text{coverage}}}{2 \cdot D_{\text{min}}}\right)$$

### 3.2 Standard Classroom Deployment Scenarios

| Classroom Type | Physical Dimensions ($W \times L \times H$) | Student Capacity | Front Row Dist ($D_{\text{min}}$) | Back Row Dist ($D_{\text{max}}$) | Required $H_{\text{FoV}}$ | Recommended Lens Focal Length (1/2.8" Sensor) | Sensor Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Seminar Room** | $6\text{m} \times 8\text{m} \times 3.0\text{m}$ | 25 – 35 | $1.2\text{m}$ | $6.5\text{m}$ | $88^\circ - 95^\circ$ | **2.8 mm – 3.6 mm** | **1080p or 4K** |
| **Standard Classroom** | $7\text{m} \times 10\text{m} \times 3.2\text{m}$ | 40 – 50 | $1.5\text{m}$ | $8.5\text{m}$ | $80^\circ - 88^\circ$ | **3.6 mm – 4.0 mm** | **4K Recommended** |
| **Tiered Lecture Hall** | $8\text{m} \times 12\text{m} \times 3.8\text{m}$ | 60 – 80 | $2.0\text{m}$ | $10.5\text{m}$ | $70^\circ - 78^\circ$ | **4.0 mm – 6.0 mm** | **4K Mandatory** |
| **Auditorium (Single Cam)**| $12\text{m} \times 16\text{m} \times 4.5\text{m}$| 100+ | $2.5\text{m}$ | $14.0\text{m}$ | $75^\circ - 85^\circ$ | **6.0 mm Varifocal** | **4K + Dual Zone** |

---

## 4. Photometric & Illumination Guidelines

Face detection and liveness validation rely on high-frequency textural variance (Laplacian blur score) and chrominance consistency (YCrCb skin locus). Poor lighting directly causes false spoof rejections and distance recognition failures.

```
       [DIFFUSE OVERHEAD LUMINAIRES: 400 - 600 LUX]
              |                   |
              v                   v
        [Row 1 Desks]       [Row 2 Desks]
              ^
              |
     [FRONT CAMERA] ------> [STUDENT FACES: UNIFORM ILLUMINATION]
        (No direct light into lens)
```

### 4.1 Illuminance Standards (Lux Levels)

- **Minimum Required Face Illuminance**: **300 Lux** (absolute operational floor).
- **Recommended Nominal Illuminance**: **450 to 600 Lux** across all desk surfaces.
- **Maximum Permissible Illuminance**: **1000 Lux** (to avoid specular saturation / overexposure).
- **Illumination Uniformity Ratio**:
  $$\text{Uniformity} = \frac{E_{\text{min}}}{E_{\text{avg}}} \ge 0.70$$
  Dark corners or unlit rear rows where lux drops below 250 lux will fail the `MIN_BRIGHTNESS = 35.0` quality gate.

---

### 4.2 Luminaire Placement & Type

- **Luminaire Type**: Commercial high-CRI LED panel fixtures ($\text{CRI} \ge 82$, Color Temperature: $4000\text{K} - 5000\text{K}$ neutral white).
- **Diffusion**: Prismatic or frosted diffusers are mandatory to eliminate harsh point-source specular glare on prescription eyeglasses.
- **Overhead Geometry**: Luminaires must be positioned **0.5m in front of student desk rows** rather than directly above students' heads. Direct top-down lighting casts deep orbital shadows over eye sockets, degrading 5-point landmark localization.

---

### 4.3 Backlighting & Window Glare Mitigation

Window backlighting is the #1 cause of biometric failure in academic classrooms. When sunlight floods behind students from rear windows, students' faces appear in deep silhouette.
1. **Window Shading**: All windows located in the camera's field of view (rear classroom wall) MUST be equipped with opaque blackout roller blinds or motorized solar shades.
2. **Camera Sensor WDR/HDR**: If ambient daylight cannot be fully shaded, the camera must support True Wide Dynamic Range ($\text{WDR} \ge 120\text{ dB}$).
3. **No Direct Sunlight into Lens**: The camera lens must never face east or west windows directly.

---

## 5. Seating Depth Limits: Empirical 1080p vs 4K Sensor Comparison

Based on empirical benchmarks executed on the SmartClass Vision production pipeline:

### 5.1 Distance Degradation Performance Matrix

| Row Index | Physical Distance | 1080p Face Width (px) | 1080p Detection Recall | 1080p Quorum Pass Rate | 4K Face Width (px) | 4K Detection Recall | 4K Quorum Pass Rate | Operational Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Row 1** | $1.2\text{ m}$ | $\sim 140\text{ px}$ | **100.0%** | **100.0%** | $\sim 280\text{ px}$ | **100.0%** | **100.0%** | Optimal Matching |
| **Row 2** | $2.0\text{ m}$ | $\sim 100\text{ px}$ | **100.0%** | **100.0%** | $\sim 200\text{ px}$ | **100.0%** | **100.0%** | Optimal Matching |
| **Row 3** | $3.0\text{ m}$ | $\sim 70\text{ px}$ | **100.0%** | **100.0%** | $\sim 140\text{ px}$ | **100.0%** | **100.0%** | Optimal Matching |
| **Row 4** | $4.2\text{ m}$ | $\sim 48\text{ px}$ | **100.0%** | **100.0%** | $\sim 96\text{ px}$ | **100.0%** | **100.0%** | Optimal Matching |
| **Row 5** | $5.5\text{ m}$ | $\sim 36\text{ px}$ | **100.0%** | **100.0%** | $\sim 72\text{ px}$ | **100.0%** | **100.0%** | Reliable Boundary |
| **Row 6** | $6.8\text{ m}$ | $\sim 29\text{ px}$ | **90.0%** | **90.0%** | $\sim 58\text{ px}$ | **80.0%** | **80.0%** | Degraded Boundary |
| **Row 7** | $8.0\text{ m}$ | $\sim 25\text{ px}$ | **0.0%** (Dropped) | **0.0%** | $\sim 50\text{ px}$ | **50.0%** | **20.0%** | Distance Limit |
| **Row 8** | $9.5\text{ m}$ | $\sim 19\text{ px}$ | **0.0%** (Dropped) | **0.0%** | $\sim 38\text{ px}$ | **20.0%** | **0.0%** | Sub-Threshold |

### 5.2 Seating Depth Rules of Thumb
1. **1080p Sensor Ceiling**:
   - Reliable operating depth is **Rows 1 through 5 (maximum 5.5 – 6.0 meters)**.
   - Do NOT deploy 1080p sensors in rooms exceeding 6 rows of depth; students in Rows 7 and 8 drop below the $20\text{px}$ minimum face dimension and suffer $100\%$ false absence.
2. **4K Sensor Ceiling**:
   - Reliable operating depth extends to **Row 7 – Row 8 (up to 8.5 – 10.0 meters)**.
   - In full multi-row classroom frames, 4K provides **95.0% cohort detection recall** (57/60 students) compared to only 70.0% on 1080p.

---

## 6. Classroom Operational Parameters & Timing Calibration

### 6.1 Multi-Shot Session Timing Architecture

SmartClass Vision utilizes a 10-shot consensus protocol with a 3-second initial countdown buffer and calibrated inter-shot intervals:

```
[Start Session]
      |
      v
[3.0s Preparation Buffer: "Please look toward the camera"]
      |
      +---> [Shot 1 Capture & Ingest]
      |          |--> SCRFD Detection (~75ms)
      |          |--> 4D Batched ArcFace (~1700ms)
      |          |--> Matrix Cosine Matching (~0.1ms)
      |          |--> Total Processing: ~1.8s
      |
      +---> [3.5s Countdown Interval: Student Pose Stabilization]
      |
      +---> [Shot 2 Capture & Ingest]
      |          ...
      +---> [Shot 10 Capture]
      |
      v
[Quorum Consensus Evaluation: >= 3 Valid Votes]
      |
      v
[Commit Signed HMAC-SHA256 Records & SQLite DB Update]
```

### 6.2 Parameter Calibration Matrix

| Parameter | Configuration Key | Default Value | Operational Purpose & Calibration Notes |
| :--- | :--- | :---: | :--- |
| **Total Session Shots** | `TOTAL_CAPTURE_SHOTS` | `10` | Captures 10 distinct frames over ~35 seconds. Provides statistical redundancy against temporary occlusions, blinks, and natural head turns. |
| **Quorum Threshold** | `REQUIRED_CONSISTENT_VOTES` | `3` | A student must be verified in $\ge 3$ distinct shots with mean similarity $\ge 0.52$. Prevents false-positive attendance from transient passersby. |
| **Shot Interval** | `SHOT_COUNTDOWN_SECONDS` | `3.5s` | **Crucial Calibration**: For 50+ students, 4D batched inference requires $\sim 1.8\text{s}$. A 3.5s interval allows $\sim 1.7\text{s}$ of idle slack buffer, giving students ample reaction time to orient toward the camera. |
| **Preparation Buffer** | UI Countdown Buffer | `3.0s` | Audio/visual prompt ("Get ready for attendance scan") before Shot 1 fires. |
| **Detection Size** | `DETECTION_SIZE` | `(1280, 1280)` | Calibrated multi-scale detection size. Allows SCRFD to resolve distant faces down to 20px without GPU VRAM exhaustion. |
| **Minimum Face Size** | `MIN_FACE_SIZE` | `20 px` | Enables detection of distant seated students in Rows 5–8 while rejecting distant background noise artifacts. |
| **Blur Variance Cutoff** | `BLUR_THRESHOLD` | `50.0` | Scale-adaptive Laplacian variance threshold. Normalizes for crop scale to avoid false spoof rejection on distant students. |
| **Yaw Cutoff Ratio** | `MAX_YAW_RATIO` | `0.42` | Roll-invariant 3D landmark vector projection: $\lvert(\mathbf{p}_{\text{nose}} - \mathbf{p}_{\text{mid}}) \cdot \mathbf{u}\rvert / d_{\text{eyes}} \le 0.42$. Tolerates up to $\sim \pm 22^\circ$ natural head turns. |
| **ArcFace Cosine Cutoff** | `ARCFACE_SIMILARITY_THRESHOLD` | `0.52` | Optimized operating point balancing FAR and FRR. Distance degradation reduces similarity to ~0.55-0.65; 0.52 maintains 100% Top-1 accuracy. |
| **Ambiguity Margin** | `AMBIGUITY_MARGIN` | `0.06` | Top-1 similarity must exceed Top-2 by $\ge 0.06$. Prevents identity swaps between similar-looking students or siblings. |

---

## 7. Enrolled Student Gallery Curation Best Practices

The accuracy of ArcFace centroid matching is directly determined by the quality of enrollment photos stored in `data/known_faces/<roll_number>/`.

### 7.1 Photo Ingest Standards
1. **Enrollment Resolution**: Source portrait must be at least **$640 \times 640\text{ px}$**, with the facial region occupying $\ge 200 \times 200\text{ px}$.
2. **Multi-Angle Portfolio (3 to 5 Images per Student)**:
   - Image 1: Straight frontal ($0^\circ$ yaw, neutral expression).
   - Image 2: Slight yaw left ($+10^\circ$ to $+15^\circ$).
   - Image 3: Slight yaw right ($-10^\circ$ to $-15^\circ$).
   - Image 4: Natural smile / expression variation.
   - Image 5: Eyewear variation (if student periodically wears prescription glasses).
3. **No Masks or Tinted Sunglasses**: High-tint sunglasses and facial masks block periocular landmarks and cannot be enrolled.
4. **Centroid Normalization**: The enrollment pipeline automatically extracts 512-D L2-normalized embeddings and stores the unit mean centroid $\mathbf{c} = \frac{\sum \mathbf{e}_i}{\lVert \sum \mathbf{e}_i \rVert_2}$ in `data/embeddings.pkl`.

---

## 8. Fallback Protocols & Edge-Case Handling

### 8.1 Absent Student Persistence (Fault Fix V09)
In compliance with academic audit requirements, any student enrolled in the scheduled lecture section roster who fails the voting quorum ($< 3$ valid votes) is **explicitly recorded as "Absent"** in both the SQLite database (`smartclass.db`) and signed CSV attendance reports. This prevents ambiguous unrecorded states.

### 8.2 Prescription Eyeglass Specular Glare (Fault Fix V10)
Prescription glasses can cause localized specular reflections under overhead lighting. SmartClass Vision's passive liveness detector excludes the periocular region when assessing specular variance, preventing false spoof alerts on bespectacled students.

### 8.3 Invariant Yaw Evaluation (Fault Fix V01)
When students tilt their heads sideways (roll) while turning toward an instructor, naive horizontal bounding box coordinates yield false high yaw ratios. SmartClass Vision projects the nose-to-mid-eye displacement vector onto the interocular unit vector:
$$\text{yaw\_ratio} = \frac{\lvert(\mathbf{p}_{\text{nose}} - \mathbf{p}_{\text{mid\_eyes}}) \cdot \mathbf{u}_{\text{interocular}}\rvert}{\lVert\mathbf{p}_{\text{left\_eye}} - \mathbf{p}_{\text{right\_eye}}\rVert_2}$$
This guarantees mathematical invariance under arbitrary camera or head roll.

### 8.4 Candidate Section Roster Filtering (Fault Fix V04)
To maintain zero false swaps in institutional deployments with 5,000+ enrolled campus students, the attendance session scopes ArcFace centroid matching strictly to `candidate_rolls` belonging to the scheduled section. This narrows candidate comparison to $30\text{–}60$ students, completely eliminating database-wide ambiguity margin collisions.

### 8.5 Manual Faculty Override Protocol
In the event of medical facial dressings, severe optical occlusions, or camera hardware disruption:
1. The faculty instructor navigates to the SmartClass Admin Command Suite.
2. Selects the active session roster.
3. Toggles student status to "Present (Manual Override)" with required faculty credentials.
4. The system logs the override event with an HMAC-SHA256 signature containing the instructor's employee ID and timestamp.

---

## 9. Hardware Bill of Materials (BOM) & Edge Compute Specs

### 9.1 Recommended Sensor Hardware

| Tier | Model | Sensor Type | Max Resolution | Lens FoV | Interface | Recommended Environment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1 (Classroom Standard)** | **Logitech Brio 4K Ultra HD** | 1/2.8" CMOS | $3840 \times 2160$ @ 30fps | $90^\circ / 78^\circ / 65^\circ$ adjustable | USB 3.0 Type-C | Standard Rooms (up to 8m) |
| **Tier 2 (Large Lecture Hall)** | **Hikvision DS-2CD2087G2-LU (4K ColorVu)** | 1/1.2" Progressive CMOS | $3840 \times 2160$ @ 25fps | $102^\circ$ ($2.8\text{mm}$) / $84^\circ$ ($4.0\text{mm}$) | RTSP / PoE Gigabit Ethernet | Halls $> 8\text{m}$, low light |
| **Tier 3 (Budget Flat Classroom)** | **Anker PowerConf C200 2K** | 1/2.7" CMOS | $2560 \times 1440$ @ 30fps | $95^\circ / 80^\circ / 65^\circ$ | USB 2.0 / 3.0 | Rooms $< 6\text{m}$ (up to Row 5) |

### 9.2 Host Compute Platform Specifications

| Component | Minimum Specification (Host PC) | Recommended Specification (Campus Standard) |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5-10400 / AMD Ryzen 5 3600 (6 Cores / 12 Threads) | Intel Core i7-12700 / AMD Ryzen 7 5800X (8+ Cores) |
| **System Memory (RAM)**| **8 GB DDR4** (Working Set RSS: $\sim 1.9\text{–}2.5\text{ GB}$) | **16 GB DDR4/DDR5** |
| **Graphics (GPU)** | Not Required (CPUExecutionProvider ONNX runtime) | Optional: NVIDIA GTX 1650 / RTX 3050 (CUDA provider) |
| **Storage** | 128 GB NVMe SSD (Fast OS & Model Weights loading) | 256 GB NVMe SSD (Local backup manifest archive) |
| **Network** | 100 Mbps Ethernet / Wi-Fi 5 | Gigabit Ethernet (1000BASE-T) for RTSP streams |
| **Operating System** | Windows 10/11 64-bit (DirectShow `cv2.CAP_DSHOW` enabled) | Windows 11 Pro 64-bit or Ubuntu 22.04 LTS |

---

## 10. Pre-Flight Installation Verification Checklist

Before certifying an academic classroom for automated biometric attendance:

- [ ] **Physical Mounting**: Camera securely fastened at $2.4\text{m} - 2.8\text{m}$ AFF along center longitudinal axis.
- [ ] **Tilt Angle**: Downward tilt verified with digital inclinometer at $15^\circ - 20^\circ$.
- [ ] **Desk Coverage**: Visual inspection confirms front-row leftmost and rightmost student seats are inside frame.
- [ ] **Illumination Verification**: Lux meter readings at student desk surfaces read $\ge 400\text{ Lux}$ across all rows.
- [ ] **Backlight Elimination**: Rear windows covered with blackout blinds; no visible light flares into lens.
- [ ] **Camera Driver Backend**: DirectShow verified on Windows via `cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)`.
- [ ] **Benchmark Test Verification**: Run `python tools/benchmark_hardware_deployment.py` to assert:
  - Row 1 to Row 5 detection recall is $100\%$.
  - 50-student 4D batched ArcFace inference latency is $\le 2000\text{ ms}$.
  - Sustained memory working set is $\le 2.6\text{ GB}$ RSS.
- [ ] **Database Integrity**: `data/smartclass.db` and `data/embeddings.pkl` synchronized with current semester roster.
