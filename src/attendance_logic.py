import cv2
import time
import re
import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.anti_spoof import evaluate_liveness, evaluate_face_quality
from src.database import get_student_info, save_attendance_entry, get_students_by_class, generate_file_checksum, log_security_event
from utils.config import (
    ATTENDANCE_DIR, CLASS_PHOTOS_DIR, UNKNOWN_FACES_DIR, CAMERA_ID, BASE_DIR,
    MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_CONTRAST,
    MAX_YAW_RATIO, TOTAL_CAPTURE_SHOTS, REQUIRED_CONSISTENT_VOTES, SHOT_COUNTDOWN_SECONDS,
    ARCFACE_SIMILARITY_THRESHOLD, DEBUG_MODE
)

DB_PATH = BASE_DIR / "data" / "smartclass.db"


def clean_filename(text):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(text))


def start_attendance(subject_info=None):
    print("\n" + "=" * 55)
    print("   SUBJECT & SECTION-SPECIFIC BATCH ATTENDANCE")
    print("=" * 55)

    if subject_info is None:
        subject_info = {
            "id": 0,
            "subject_code": "ALL",
            "subject_name": "General Session",
            "teacher_name": "Faculty",
            "teacher_emp_id": "",
            "degree": None,
            "year": None,
            "branch": None,
            "section": None
        }

    sub_id = subject_info.get("id", 0)
    sub_code = clean_filename(subject_info.get("subject_code", "GEN"))
    sub_name = subject_info.get("subject_name", "General Session")
    teacher_name = subject_info.get("teacher_name", "Faculty")
    teacher_emp_id = subject_info.get("teacher_emp_id", "")
    target_degree = subject_info.get("degree")
    target_year = subject_info.get("year")
    target_branch = subject_info.get("branch")
    target_section = subject_info.get("section")

    class_title = f"{target_degree or ''} {target_year or ''} {target_branch or 'ALL'} Sec {target_section or 'ALL'}".strip()

    # Automatically fetch enrolled students belonging to this section
    enrolled_students = get_students_by_class(target_degree, target_year, target_branch, target_section)
    enrolled_count = len(enrolled_students)

    print(f"[TEACHER]  {teacher_name}" + (f" (Emp ID: {teacher_emp_id})" if teacher_emp_id else ""))
    print(f"[SUBJECT]  {sub_code} - {sub_name}")
    print(f"[CLASS]    {class_title} -> {enrolled_count} Students Enrolled in Section")
    print("=" * 55)

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if not recognizer.known_embeddings:
        print("[ERROR] AI Brain empty. Train model first!")
        return {"error": "AI Brain is empty. Please train the model first."}

    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID)

    # Force High Definition
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    NUM_SHOTS = TOTAL_CAPTURE_SHOTS
    REQUIRED_MATCHES = REQUIRED_CONSISTENT_VOTES
    SHOT_INTERVAL = SHOT_COUNTDOWN_SECONDS
    captured_frames = []

    # Extract target section student roll numbers for context-aware priority matching
    enrolled_roll_numbers = [str(s[0]) for s in enrolled_students] if enrolled_students else []

    print(f"[INFO] Initiating {NUM_SHOTS}-Shot Multi-Angle Batch Capture (Consensus: {REQUIRED_MATCHES}/{NUM_SHOTS})...")

    # PHASE 1: BATCH CAPTURE WITH REAL-TIME MULTI-FACE DETECTION
    try:
        import winsound
    except Exception:
        winsound = None

    # Initial 3-second preparation buffer: allows all students to settle and face the camera before Shot 1
    prep_start = time.time()
    prep_duration = 3.0
    while time.time() - prep_start < prep_duration:
        ret, frame = cap.read()
        if not ret: break
        prep_frame = frame.copy()
        time_rem = max(0.0, prep_duration - (time.time() - prep_start))
        _, p_faces = detector.detect_faces(prep_frame)
        cv2.rectangle(prep_frame, (10, 10), (760, 160), (20, 20, 20), -1)
        cv2.rectangle(prep_frame, (10, 10), (760, 160), (0, 255, 255), 2)
        cv2.putText(prep_frame, f"CLASS ATTENDANCE STARTING: {sub_code}", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(prep_frame, f"ALL STUDENTS PLEASE FACE THE CAMERA! ({len(p_faces)} faces in view)", (25, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        cv2.putText(prep_frame, f"First Capture in: {time_rem:.1f}s", (25, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.imshow("SmartClass Vision - Section Attendance Scan", prep_frame)
        cv2.waitKey(1)

    for i in range(NUM_SHOTS):
        start_wait = time.time()
        beep_played = False

        while time.time() - start_wait < SHOT_INTERVAL:
            ret, frame = cap.read()
            if not ret: break

            display_frame = frame.copy()
            time_left = max(0.0, SHOT_INTERVAL - (time.time() - start_wait))

            # Detect multiple faces in live preview so teacher sees all students being captured
            _, live_faces = detector.detect_faces(display_frame)
            active_count = len(live_faces)

            # Pre-shot alert: audio beep & visual border pulse 0.5s before snap
            is_imminent = time_left <= 0.6
            if is_imminent and not beep_played and winsound:
                try:
                    winsound.Beep(1200, 120)  # Gentle 1200Hz alert beep
                except Exception:
                    pass
                beep_played = True

            # Draw session banner
            banner_border = (0, 0, 255) if is_imminent else (0, 200, 255)
            cv2.rectangle(display_frame, (10, 10), (760, 160), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (760, 160), banner_border, 2)

            cv2.putText(display_frame, f"Subject: {sub_code} ({class_title})", (25, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            teacher_display = f"Teacher: {teacher_name}" + (f" [{teacher_emp_id}]" if teacher_emp_id else "")
            cv2.putText(display_frame, teacher_display, (25, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Section Enrolled: {enrolled_count} Students | Live Faces: {active_count}", (25, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

            snap_text = f"LOOK AT CAMERA! SNAPPING IN {time_left:.1f}s" if is_imminent else f"Scan Shot {i + 1}/{NUM_SHOTS} - Next Snap: {time_left:.1f}s"
            snap_color = (0, 140, 255) if is_imminent else (0, 255, 0)
            cv2.putText(display_frame, snap_text, (25, 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, snap_color, 2)

            if is_imminent:
                # Flashing border alert for the classroom
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 165, 255), 6)

            cv2.imshow("SmartClass Vision - Section Attendance Scan", display_frame)
            cv2.waitKey(1)

        # Flush camera buffer to guarantee a fresh snapshot
        for _ in range(2):
            cap.grab()
        ret, frame = cap.read()
        if ret:
            captured_frames.append(frame.copy())
            photo_name = f"ClassAudit_{sub_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_Shot{i + 1}.jpg"
            cv2.imwrite(str(CLASS_PHOTOS_DIR / photo_name), frame)
            print(f"[CAPTURE] Secured High-Res Image: {photo_name}")

            # Shutter sound
            if winsound:
                try:
                    winsound.Beep(2000, 80)
                except Exception:
                    pass

            # Elegant capture visual confirmation
            feedback_frame = frame.copy()
            cv2.rectangle(feedback_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            cv2.rectangle(feedback_frame, (frame.shape[1] // 2 - 280, 20), (frame.shape[1] // 2 + 280, 85), (20, 20, 20), -1)
            cv2.rectangle(feedback_frame, (frame.shape[1] // 2 - 280, 20), (frame.shape[1] // 2 + 280, 85), (0, 255, 0), 2)
            cv2.putText(feedback_frame, f"SHOT {i + 1}/{NUM_SHOTS} SECURED", (frame.shape[1] // 2 - 210, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("SmartClass Vision - Section Attendance Scan", feedback_frame)
            cv2.waitKey(80)

    cap.release()

    # PHASE 2: AI PROCESSING (SECTION-AWARE MULTI-STUDENT RECOGNITION)
    print(f"\n[INFO] Cross-referencing captures with Section Roster ({len(enrolled_roll_numbers)} enrolled candidates)...")
    student_detections = {}
    student_similarities = {}
    spoof_incidents = 0

    for f_idx, frame in enumerate(captured_frames):
        # Update live visual progress window
        processing_screen = np.zeros((400, 800, 3), dtype="uint8")
        progress_pct = int(((f_idx + 1) / NUM_SHOTS) * 100)
        cv2.putText(processing_screen, "AI Biometric Verification & Voting...", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(processing_screen, f"Evaluating Shot {f_idx + 1} of {NUM_SHOTS} ({progress_pct}%)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # Progress bar
        cv2.rectangle(processing_screen, (50, 180), (750, 210), (50, 50, 50), -1)
        bar_w = int(700 * ((f_idx + 1) / NUM_SHOTS))
        cv2.rectangle(processing_screen, (50, 180), (50 + bar_w, 210), (0, 255, 0), -1)
        cv2.putText(processing_screen, f"Auditing against {class_title} ({enrolled_count} students)...",
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.imshow("SmartClass Vision - Section Attendance Scan", processing_screen)
        cv2.waitKey(1)

        processed_frame, detected_faces = detector.detect_faces(frame)
        print(f"\n[SHOT {f_idx + 1}] Detected {len(detected_faces)} raw face bounding boxes in frame")

        valid_live_crops = []
        for face_idx, face_data in enumerate(detected_faces):
            raw_crop = face_data.get("raw_crop", face_data["image"])
            aligned_face = face_data["image"]
            coords = face_data["coords"]
            det_score = face_data.get("score", 0.0)
            landmarks = face_data.get("landmarks")

            # STEP 4: Comprehensive Face-Quality Gate (Size, Blur, Brightness, Contrast, Pose)
            q_pass, q_reason, q_metrics = evaluate_face_quality(
                raw_crop, landmarks=landmarks, min_size=MIN_FACE_SIZE, blur_thresh=BLUR_THRESHOLD,
                min_b=MIN_BRIGHTNESS, max_b=MAX_BRIGHTNESS, min_contrast=MIN_CONTRAST,
                max_yaw_ratio=MAX_YAW_RATIO
            )
            if not q_pass:
                if DEBUG_MODE:
                    print(f"\n--- FACE #{face_idx + 1} (SHOT {f_idx + 1}) ---")
                    print(f"Detection confidence: {det_score:.2f}")
                    print(f"BBox: {coords[2]-coords[0]}x{coords[3]-coords[1]}")
                    print(f"Sharpness: {q_metrics.get('sharpness', 0.0)}")
                    print(f"Brightness: {q_metrics.get('brightness', 0.0)}")
                    print(f"Contrast: {q_metrics.get('contrast', 0.0)}")
                    print(f"Yaw ratio: {q_metrics.get('yaw_ratio', 0.0)}")
                    print(f"Quality: REJECTED ({q_reason})")
                    print("FINAL: UNKNOWN (Quality Filter)")
                    print("-" * 30)
                continue

            # Biometric Liveness & Anti-Spoofing Assessment
            is_live, l_score, l_reason = evaluate_liveness(raw_crop)
            if not is_live:
                spoof_incidents += 1
                spoof_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                UNKNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
                spoof_path = UNKNOWN_FACES_DIR / f"spoof_{sub_code}_{spoof_ts}.jpg"
                cv2.imwrite(str(spoof_path), raw_crop)
                print(f"  [SECURITY ALERT - SPOOF DETECTED] Score: {l_score}%, Reason: {l_reason}")
                log_security_event("SPOOF_ATTEMPT", "CRITICAL", teacher_name,
                                   f"Biometric spoof rejected in {sub_code} ({l_reason}, Score: {l_score}%). Snapshot: {spoof_path.name}")
                continue

            valid_live_crops.append({
                "aligned": aligned_face,
                "raw": raw_crop,
                "coords": coords,
                "det_score": det_score,
                "metrics": q_metrics
            })

        # STEP 5 & 6: Vectorized Canonical ArcFace Inference with Global Cosine Matching & UNKNOWN Gating
        if valid_live_crops:
            aligned_batch = [item["aligned"] for item in valid_live_crops]
            batch_matches = recognizer.recognize_batch(aligned_batch)
            for item_idx, match_res in enumerate(batch_matches):
                roll_number = match_res[0]
                confidence = match_res[1]
                diag = match_res[2] if len(match_res) > 2 else {}

                item_info = valid_live_crops[item_idx]
                w, h = item_info["metrics"]["width"], item_info["metrics"]["height"]
                det_score = item_info["det_score"]
                q_met = item_info["metrics"]

                # Step 11: Structured Debug Output for every detected face
                if DEBUG_MODE:
                    print(f"\n--- FACE #{item_idx + 1} (SHOT {f_idx + 1}) ---")
                    print(f"Detection confidence: {det_score:.2f}")
                    print(f"BBox: {w}x{h}")
                    print(f"Sharpness: {q_met.get('sharpness', 0.0)}")
                    print(f"Brightness: {q_met.get('brightness', 0.0)}")
                    print(f"Contrast: {q_met.get('contrast', 0.0)}")
                    print(f"Yaw ratio: {q_met.get('yaw_ratio', 0.0)}")
                    print("Quality: GOOD")
                    print("\nTOP MATCHES:")
                    for top_m in diag.get("top_5", []):
                        print(f"  {top_m.get('name')} ({top_m.get('roll')}): {top_m.get('sim', 0.0):.4f}")
                    print(f"\nTop1: {diag.get('top1_name', 'None')} ({diag.get('top1_roll', 'None')})")
                    print(f"Top1 score: {diag.get('top1_sim', 0.0):.4f}")
                    print(f"Top2 score: {diag.get('top2_sim', 0.0):.4f}")
                    print(f"Margin: {diag.get('margin', 0.0):.4f}")
                    print(f"FINAL: {'UNKNOWN (' + diag.get('reason', '') + ')' if roll_number == 'Unknown' else diag.get('top1_name', roll_number) + ' -> VERIFIED'}")
                    print("-" * 30)

                if roll_number != "Unknown":
                    student_detections[roll_number] = student_detections.get(roll_number, 0) + 1
                    if roll_number not in student_similarities:
                        student_similarities[roll_number] = []
                    student_similarities[roll_number].append(diag.get("top1_sim", confidence / 100.0))

    cv2.destroyAllWindows()

    # PHASE 3: CONSENSUS, SECTION FILTER & EXPORT
    attendance_list = []
    rejected_wrong_section = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    today_date = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 55)
    print("   SECTION VALIDATION & VERIFICATION RESULTS")
    print("=" * 55)

    for roll, count in student_detections.items():
        info = get_student_info(roll)
        if not info:
            continue

        s_roll, s_name, s_gender, s_degree, s_year, s_branch, s_section, _ = info
        sim_scores = student_similarities.get(roll, [])
        avg_similarity = float(np.mean(sim_scores)) if sim_scores else 0.0

        # Strict Section & Branch validation
        match_degree = (target_degree is None or s_degree.strip().upper() == target_degree.strip().upper())
        match_year = (target_year is None or s_year.strip().upper() == target_year.strip().upper())
        match_branch = (target_branch is None or s_branch.strip().upper() == target_branch.strip().upper())
        match_section = (target_section is None or s_section.strip().upper() == target_section.strip().upper())

        is_correct_section = match_degree and match_year and match_branch and match_section

        # Step 5 & 6: Strengthened Temporal Consensus (at least 5/7 votes AND mean similarity >= 0.65)
        passed_consensus = (count >= REQUIRED_MATCHES) and (avg_similarity >= ARCFACE_SIMILARITY_THRESHOLD)

        if passed_consensus:
            if is_correct_section:
                print(f"[VERIFIED - SECTION MATCH] {s_name} ({s_roll}) -> {s_branch} Sec {s_section} -> PRESENT (Votes: {count}/{NUM_SHOTS}, Avg Sim: {avg_similarity:.3f})")
                attendance_list.append({
                    "Subject Code": sub_code,
                    "Subject Name": sub_name,
                    "Teacher Name": teacher_name,
                    "Degree": s_degree,
                    "Year": s_year,
                    "Branch": s_branch,
                    "Section": s_section,
                    "Roll Number": s_roll,
                    "Name": s_name,
                    "Time Marked": timestamp,
                    "Status": "Present"
                })
                # Persist to database
                save_attendance_entry(
                    sub_id, sub_code, sub_name, teacher_name,
                    s_roll, s_name, s_branch, s_section,
                    today_date, timestamp, "Present"
                )
            else:
                mismatch_reasons = []
                if not match_degree: mismatch_reasons.append(f"Degree '{s_degree}' != target '{target_degree}'")
                if not match_year: mismatch_reasons.append(f"Year '{s_year}' != target '{target_year}'")
                if not match_branch: mismatch_reasons.append(f"Branch '{s_branch}' != target '{target_branch}'")
                if not match_section: mismatch_reasons.append(f"Section '{s_section}' != target '{target_section}'")
                reason_str = ", ".join(mismatch_reasons)
                print(f"[REJECTED - WRONG CLASS/SECTION] Student {s_name} ({s_roll}) [{s_degree} {s_year} {s_branch} Sec {s_section}] does not match target class [{target_degree} {target_year} {target_branch} Sec {target_section}]! ({reason_str}, Votes: {count}/{NUM_SHOTS}, Avg Sim: {avg_similarity:.3f}). Attendance denied.")
                rejected_wrong_section.append({
                    "roll": s_roll, "name": s_name, "enrolled": f"{s_degree} {s_year} {s_branch} Sec {s_section}"
                })
        else:
            # Only report insufficient consensus if the student is actually enrolled in this target class/section
            if is_correct_section:
                print(f"[REJECTED - INSUFFICIENT CONSENSUS] {s_name} ({s_roll}): Votes={count}/{NUM_SHOTS} (Required: {REQUIRED_MATCHES}), Avg Sim={avg_similarity:.3f} (Required: {ARCFACE_SIMILARITY_THRESHOLD}) -> Attendance denied.")


    filename = None
    hmac_sig = None
    if attendance_list:
        clean_sec = clean_filename(target_section or "ALL")
        clean_br = clean_filename(target_branch or "ALL")
        filename = f"Attendance_{sub_code}_{clean_br}_{clean_sec}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = ATTENDANCE_DIR / filename
        df = pd.DataFrame(attendance_list)
        df.to_csv(filepath, index=False)
        print(f"\n[SUCCESS] Saved section attendance to: {filename}")

        # Cryptographically sign the attendance file with HMAC-SHA256
        f_hash, hmac_sig = generate_file_checksum(filepath, created_by=f"{teacher_name} ({teacher_emp_id})")
        print(f"[SECURITY] Cryptographic HMAC-SHA256 Registered: {hmac_sig[:16]}...")
    else:
        print("\n[INFO] No students from the target section passed verification threshold.")

    return {
        "verified": len(attendance_list),
        "rejected_section": len(rejected_wrong_section),
        "spoofs_rejected": spoof_incidents,
        "enrolled_total": enrolled_count,
        "filename": filename,
        "hmac_signature": hmac_sig,
        "verified_students": [{"roll": a["Roll Number"], "name": a["Name"], "time": a["Time Marked"]} for a in attendance_list],
        "rejected_students": rejected_wrong_section
    }