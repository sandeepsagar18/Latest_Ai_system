import cv2
import time
import shutil
import numpy as np
from pathlib import Path
from src.database import add_student, delete_student, student_exists, get_student_info
from src.detector import FaceDetector
from utils.config import KNOWN_FACES_DIR, CAMERA_ID


def search_student_record(roll_number):
    record = get_student_info(roll_number)
    if record:
        student_dir = Path(KNOWN_FACES_DIR) / roll_number
        image_count = len(list(student_dir.glob("*.jpg"))) if student_dir.exists() else 0
        return True, record, image_count
    return False, None, 0


def delete_existing_student(roll_number):
    if not student_exists(roll_number):
        return False, f"Roll Number {roll_number} not found in database."

    delete_student(roll_number)
    student_dir = Path(KNOWN_FACES_DIR) / roll_number
    if student_dir.exists():
        shutil.rmtree(student_dir)
    return True, f"All records and images for {roll_number} deleted successfully."


def get_camera(camera_id=CAMERA_ID):
    """Attempt to open camera with DirectShow backend first for fast, reliable Windows access."""
    for backend in [cv2.CAP_DSHOW, None]:
        for cid in [camera_id, 0, 1]:
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(cid, backend)
                else:
                    cap = cv2.VideoCapture(cid)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        return cap
                cap.release()
            except Exception:
                pass
    return None


def register_student(roll_number, name, gender, degree, year, branch, section):
    if student_exists(roll_number):
        return False, f"Roll Number {roll_number} is already registered!"

    if not add_student(roll_number, name, gender, degree, year, branch, section):
        return False, "Database insertion failed."

    student_dir = Path(KNOWN_FACES_DIR) / roll_number
    student_dir.mkdir(parents=True, exist_ok=True)

    instructions = [
        "Look STRAIGHT at the camera (Frontal View)",
        "Turn head GENTLY LEFT (slight angle, max 15 deg)",
        "Turn head GENTLY RIGHT (slight angle, max 15 deg)",
        "Tilt head GENTLY UP (slight angle)",
        "Tilt head GENTLY DOWN (slight angle)"
    ]

    # Open camera with DirectShow fallback
    cap = get_camera(CAMERA_ID)
    if cap is None or not cap.isOpened():
        delete_student(roll_number)
        if student_dir.exists():
            shutil.rmtree(student_dir)
        return False, f"Unable to open camera (Device ID {CAMERA_ID}). Please verify that your webcam is connected and not in use by another app."

    # Warm up camera sensor
    for _ in range(5):
        cap.read()

    try:
        detector = FaceDetector()
    except Exception as e:
        cap.release()
        delete_student(roll_number)
        if student_dir.exists():
            shutil.rmtree(student_dir)
        return False, f"Failed to initialize Face Detection model: {str(e)}"

    from src.anti_spoof import evaluate_face_quality
    from utils.config import (
        MIN_FACE_SIZE, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
        MIN_CONTRAST, MAX_YAW_RATIO
    )

    count = 0
    aborted = False

    win_title = "SmartClass Vision - Student Registration Capture"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 780, 580)
    cv2.setWindowProperty(win_title, cv2.WND_PROP_TOPMOST, 1)

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if window was closed via 'X'
        try:
            if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                aborted = True
                break
        except Exception:
            pass

        clean_frame = frame.copy()
        h, w = frame.shape[:2]

        processed_frame, cropped_faces = detector.detect_faces(frame)
        captured_crop = None
        quality_passed = False
        quality_msg = ""

        if len(cropped_faces) == 1:
            face_item = cropped_faces[0]
            fx1, fy1, fx2, fy2 = face_item["coords"]
            fw, fh = (fx2 - fx1), (fy2 - fy1)

            # Dynamic crop with 30% padding around detected face
            pad_w = int(fw * 0.3)
            pad_h = int(fh * 0.4)
            cx1 = max(0, fx1 - pad_w)
            cy1 = max(0, fy1 - pad_h)
            cx2 = min(w, fx2 + pad_w)
            cy2 = min(h, fy2 + int(pad_h * 0.4))

            captured_crop = clean_frame[cy1:cy2, cx1:cx2]
            raw_face_crop = face_item.get("raw_crop", clean_frame[fy1:fy2, fx1:fx2])
            landmarks = face_item.get("landmarks")

            # -------------------------------------------------------------
            # UNIFIED QUALITY GATE (Identical to training pipeline)
            # -------------------------------------------------------------
            q_pass, q_reason, metrics = evaluate_face_quality(
                raw_face_crop,
                landmarks=landmarks,
                min_size=max(MIN_FACE_SIZE, 70),
                blur_thresh=75.0,
                min_b=MIN_BRIGHTNESS,
                max_b=MAX_BRIGHTNESS,
                min_contrast=MIN_CONTRAST,
                max_yaw_ratio=MAX_YAW_RATIO
            )

            margin = 15
            pos_pass = (fx1 >= margin and fy1 >= margin and fx2 <= (w - margin) and fy2 <= (h - margin))

            sharpness = metrics.get("sharpness", 0.0)
            brightness = metrics.get("brightness", 0.0)
            yaw = metrics.get("yaw_ratio", 0.0)

            if not pos_pass:
                quality_passed = False
                color = (0, 165, 255)
                quality_msg = "Center your face in camera frame"
            elif not q_pass:
                quality_passed = False
                color = (0, 165, 255)
                if "BLURRY" in q_reason:
                    quality_msg = f"Motion blur detected (Sharpness: {int(sharpness)}/75) - Hold Still"
                elif "LOW_LIGHT" in q_reason:
                    quality_msg = "Face too dark! Move into better lighting"
                elif "OVEREXPOSED" in q_reason:
                    quality_msg = "Face too bright! Avoid harsh direct glare"
                elif "LOW_CONTRAST" in q_reason:
                    quality_msg = "Low image contrast. Adjust room light"
                elif "FACE_EXTREME_ANGLE" in q_reason:
                    quality_msg = f"Face turned too far (Yaw {yaw:.2f}) - Turn gently towards center"
                elif "FACE_TOO_SMALL" in q_reason:
                    quality_msg = "Come closer: Face too small in frame"
                else:
                    quality_msg = q_reason
            else:
                quality_passed = True
                color = (0, 255, 0)
                quality_msg = f"READY: Quality Good (Sharpness: {int(sharpness)}) - Press Space or 'C' to Capture"

            # Draw bounding box and quality stats
            cv2.rectangle(processed_frame, (fx1, fy1), (fx2, fy2), color, 3)
            cv2.circle(processed_frame, ((fx1 + fx2) // 2, (fy1 + fy2) // 2), 4, color, -1)

            # Quality meter overlay
            badge_text = f"Sharpness: {int(sharpness)} | Light: {int(brightness)} | Yaw: {yaw:.2f}"
            cv2.rectangle(processed_frame, (fx1, fy2 + 5), (fx1 + 270, fy2 + 30), (20, 20, 20), -1)
            cv2.putText(processed_frame, badge_text, (fx1 + 6, fy2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255) if quality_passed else (0, 165, 255), 1)

            msg = quality_msg

        elif len(cropped_faces) > 1:
            msg = "Multiple faces detected! Only 1 student allowed in frame."
            color = (0, 0, 255)
        else:
            msg = "Looking for student face... Face the camera directly"
            color = (0, 165, 255)

        # Header Info Banner
        cv2.rectangle(processed_frame, (0, 0), (w, 58), (20, 20, 20), -1)
        cv2.putText(processed_frame, f"Student: {name} ({roll_number}) | Pose {count + 1}/5", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Instruction: {instructions[count]}", (15, 49),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

        # Status text in center
        cv2.rectangle(processed_frame, (10, h - 85), (w - 10, h - 50), (20, 20, 20), -1)
        cv2.putText(processed_frame, msg, (20, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        # Footer Hint Banner
        cv2.rectangle(processed_frame, (0, h - 45), (w, h), (15, 15, 15), -1)
        hint = "📸 PRESS SPACE OR 'C' TO CAPTURE  |  'Q' or ESC to Cancel"
        cv2.putText(processed_frame, hint, (20, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow(win_title, processed_frame)
        key = cv2.waitKey(20) & 0xFF

        is_capture_key = key in (ord('c'), ord('C'), 32, 13)  # User presses Space, Enter, or C
        is_cancel_key = key in (ord('q'), ord('Q'), 27)        # 'q', 'Q', ESC

        if is_cancel_key:
            aborted = True
            break

        # -------------------------------------------------------------
        # HUMAN-TRIGGERED CAPTURE WITH AI QUALITY & BIOMETRIC VERIFICATION
        # -------------------------------------------------------------
        if is_capture_key:
            if captured_crop is None or captured_crop.size == 0 or len(cropped_faces) != 1:
                # No valid single face present when user clicked
                flash = processed_frame.copy()
                cv2.rectangle(flash, (w // 8, h // 3), (7 * w // 8, h // 3 + 85), (0, 0, 180), -1)
                cv2.putText(flash, "NO FACE DETECTED - PLEASE RETAKE", (w // 8 + 20, h // 3 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                cv2.putText(flash, "Ensure your face is clearly centered in camera frame.", (w // 8 + 20, h // 3 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 255), 1)
                cv2.imshow(win_title, flash)
                cv2.waitKey(850)
                continue

            # Run full quality analysis on the captured snapshot
            snap_pass, snap_reason, snap_metrics = evaluate_face_quality(
                raw_face_crop,
                landmarks=landmarks,
                min_size=max(MIN_FACE_SIZE, 70),
                blur_thresh=75.0,
                min_b=MIN_BRIGHTNESS,
                max_b=MAX_BRIGHTNESS,
                min_contrast=MIN_CONTRAST,
                max_yaw_ratio=MAX_YAW_RATIO
            )

            # Test biometric detectability with ArcFace ONNX neural network
            biometric_verified = False
            if snap_pass and pos_pass:
                try:
                    from src.recognizer import FaceRecognizer
                    rec_inst = FaceRecognizer()
                    emb = rec_inst._extract_embedding(captured_crop)
                    if emb is not None and len(emb) == 512:
                        biometric_verified = True
                except Exception:
                    biometric_verified = False

            if snap_pass and pos_pass and biometric_verified:
                # Image quality is high and verified for attendance recognition -> ACCEPT!
                img_path = student_dir / f"{roll_number}_{count}.jpg"
                cv2.imwrite(str(img_path), captured_crop)
                count += 1

                # Visual confirmation
                flash = processed_frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (0, 80, 0), 12)
                cv2.rectangle(flash, (w // 8, h // 3), (7 * w // 8, h // 3 + 95), (0, 140, 0), -1)
                cv2.putText(flash, f"POSE {count}/5 ACCEPTED! (HIGH QUALITY)", (w // 8 + 25, h // 3 + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2)
                cv2.putText(flash, f"Sharpness: {int(snap_metrics.get('sharpness', 0))} | Stored for Attendance", (w // 8 + 25, h // 3 + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
                cv2.imshow(win_title, flash)
                cv2.waitKey(650)
            else:
                # Image quality is NOT better/sufficient -> Tell user to RETAKE!
                failure_msg = "Please hold still and ensure good lighting."
                if not pos_pass:
                    failure_msg = "Face was near frame edge. Center face and retake."
                elif "BLURRY" in snap_reason:
                    failure_msg = f"Blurry image (Sharpness {int(snap_metrics.get('sharpness',0))}/75). Hold still and retake."
                elif "LOW_LIGHT" in snap_reason:
                    failure_msg = "Face was too dark. Move into better light and retake."
                elif "OVEREXPOSED" in snap_reason:
                    failure_msg = "Too much bright glare on face. Adjust lighting and retake."
                elif "FACE_EXTREME_ANGLE" in snap_reason:
                    failure_msg = "Head turned too far. Face closer to center and retake."
                elif not biometric_verified:
                    failure_msg = "Biometric neural network could not clearly read face. Retake photo."

                flash = processed_frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 150), 10)
                cv2.rectangle(flash, (w // 10, h // 3), (9 * w // 10, h // 3 + 105), (0, 0, 180), -1)
                cv2.putText(flash, "IMAGE QUALITY NOT SUFFICIENT - PLEASE RETAKE", (w // 10 + 18, h // 3 + 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
                cv2.putText(flash, failure_msg, (w // 10 + 20, h // 3 + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 255), 1)
                cv2.imshow(win_title, flash)
                cv2.waitKey(1100)

    cap.release()
    cv2.destroyAllWindows()

    if count < 5 or aborted:
        delete_student(roll_number)
        if student_dir.exists():
            shutil.rmtree(student_dir)
        if aborted:
            return False, "Registration cancelled by user. Incomplete data cleaned up."
        else:
            return False, f"Registration stopped: Only captured {count}/5 poses. Please try again."

    return True, f"Registration complete for {name} ({roll_number})! All 5 poses captured successfully."