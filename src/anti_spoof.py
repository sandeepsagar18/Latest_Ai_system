import cv2
import numpy as np
from pathlib import Path
from utils.config import (
    MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_CONTRAST, MAX_YAW_RATIO, BASE_DIR
)

MINIFASNET_MODEL_PATH = BASE_DIR / "models" / "minifasnet_v2.onnx"


class LivenessDetector:
    """
    Biometric Anti-Spoofing & Liveness Detection Engine
    Powered by Deep-Learning MiniFASNetV2 (ONNX) with multi-spectral heuristic fallback.
    Defends against:
      1. Digital smartphone / tablet / laptop display screens (moiré & pixel grids)
      2. Printed 2D paper photographs
      3. Flat static cutouts
    """

    def __init__(self, texture_threshold=35.0, liveness_threshold=0.55):
        self.texture_threshold = texture_threshold
        self.liveness_threshold = liveness_threshold
        self.minifasnet_session = None

        # Load MiniFASNetV2 ONNX session for high-accuracy neural screen & replay detection
        if MINIFASNET_MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self.minifasnet_session = ort.InferenceSession(
                    str(MINIFASNET_MODEL_PATH),
                    providers=['CPUExecutionProvider']
                )
            except Exception as e:
                print(f"[INFO] MiniFASNetV2 init fallback: {e}")
                self.minifasnet_session = None

    @staticmethod
    def _get_expanded_box(src_w, src_h, bbox, scale=2.7):
        """
        Calculates expanded context bounding box matching MiniFASNet 2.7x scale specification.
        Includes device edges, phone borders, and natural context around the face.
        """
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / float(box_h), min((src_w - 1) / float(box_w), scale))
        new_w = box_w * scale
        new_h = box_h * scale
        cx, cy = box_w / 2.0 + x, box_h / 2.0 + y

        left_top_x = cx - new_w / 2.0
        left_top_y = cy - new_h / 2.0
        right_bottom_x = cx + new_w / 2.0
        right_bottom_y = cy + new_h / 2.0

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def evaluate_minifasnet(self, full_frame, bbox):
        """
        Runs MiniFASNetV2 deep neural inference on a 2.7x context patch.
        Returns:
            is_live (bool): True if neural model classifies as real human face (Class 1)
            real_score (float): Probability score in [0.0, 1.0]
            label (int): Predicted class index (1 = Real, 0 = Print, 2 = Replay/Screen)
        """
        if self.minifasnet_session is None or full_frame is None:
            return None, 0.0, -1

        src_h, src_w = full_frame.shape[:2]
        nx1, ny1, nx2, ny2 = self._get_expanded_box(src_w, src_h, bbox, scale=2.7)
        context_patch = full_frame[ny1:ny2 + 1, nx1:nx2 + 1]
        if context_patch.size == 0:
            return None, 0.0, -1

        patch_80 = cv2.resize(context_patch, (80, 80))
        # Minivision input format: float32 in [0, 255], BGR, shape [1, 3, 80, 80]
        tensor = patch_80.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        input_name = self.minifasnet_session.get_inputs()[0].name
        logits = self.minifasnet_session.run(None, {input_name: tensor})[0][0]

        # Softmax over 3 classes: [Class 0: Print, Class 1: Real Face, Class 2: Screen Replay]
        e = np.exp(logits - np.max(logits))
        probs = e / np.sum(e)
        pred_label = int(np.argmax(probs))
        real_score = float(probs[1])

        # Real face requires argmax == 1 and probability >= 0.75
        is_live = (pred_label == 1 and real_score >= 0.75)
        return is_live, real_score, pred_label

    def evaluate_texture(self, face_crop):
        """
        Analyzes micro-texture and frequency characteristics using the Laplacian operator.
        Screens and printed papers exhibit either low variance (blur) or spiky high frequencies.
        Scale-adaptive with crop resolution so sharp distant faces (rows 5-8) are not rejected.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())

        # Scale-adaptive texture threshold: small distant crops have lower high-frequency pixel count
        h, w = face_crop.shape[:2]
        scale = min(1.0, max(0.15, (w * h) / (112.0 * 112.0)))
        adaptive_tex_thresh = max(8.0, self.texture_threshold * scale)

        # Healthy live human face typically has variance in [40, 1500] depending on lighting and resolution
        if variance < adaptive_tex_thresh:
            return False, variance, "Insufficient texture depth (Flat / Blurry image)"
        elif variance > 3000.0:
            # Extremely sharp high-frequency noise is typical of digital screen pixel grid / moire
            return False, variance, "Unnatural high-frequency noise (Possible Screen Moiré)"
        return True, variance, "Natural facial skin texture"

    def evaluate_color_chroma(self, face_crop, landmarks=None):
        """
        Analyzes human skin tone characteristics in YCrCb and HSV color spaces.
        Broadened skin chroma checks with adaptive luminance compensation.
        Excludes detected eye/spectacle orbital regions from screen-glare rejection.
        """
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb)

        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)

        mean_y = float(np.mean(y_channel))

        # Fix V11: Adaptive luminance compensation across diverse Fitzpatrick skin types (I-VI)
        cr_min = 115 if mean_y < 100 else 120
        cr_max = 185 if mean_y > 160 else 180
        cb_min = 65 if mean_y < 100 else 70
        cb_max = 145 if mean_y > 160 else 140

        skin_ycrcb = (cr >= cr_min) & (cr <= cr_max) & (cb >= cb_min) & (cb <= cb_max)
        skin_hsv = ((h_c <= 28) | (h_c >= 165)) & (s_c >= 15) & (s_c <= 230) & (v_c >= 25)
        skin_mask = skin_ycrcb | skin_hsv
        skin_ratio = float(np.sum(skin_mask)) / float(face_crop.shape[0] * face_crop.shape[1])

        # Check for harsh specular glass glare (pure white blown-out clusters common on phone screens)
        glare_mask = (v_c > 250) & (s_c < 20)
        h_img, w_img = face_crop.shape[:2]

        # Fix V10: Exclude detected eye/spectacle orbital regions from screen-glare rejection
        if landmarks is not None and len(landmarks) >= 2:
            lm = np.array(landmarks, dtype=np.float32)
            pt_r = lm[0]
            pt_l = lm[1]
            eye_dist = float(np.linalg.norm(pt_l - pt_r))
            radius = max(3, int(0.40 * eye_dist))
            orbital_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(orbital_mask, (int(round(pt_r[0])), int(round(pt_r[1]))), radius, 255, -1)
            cv2.circle(orbital_mask, (int(round(pt_l[0])), int(round(pt_l[1]))), radius, 255, -1)
            glare_mask[orbital_mask > 0] = False
        else:
            # Heuristic spectacle orbital band exclusion when landmarks not provided
            orb_y1, orb_y2 = int(0.20 * h_img), int(0.45 * h_img)
            orb_x1, orb_x2 = int(0.15 * w_img), int(0.85 * w_img)
            total_glare_px = np.sum(glare_mask)
            orbital_glare_px = np.sum(glare_mask[orb_y1:orb_y2, orb_x1:orb_x2])
            if total_glare_px > 0 and (orbital_glare_px / float(total_glare_px)) > 0.70:
                glare_mask[orb_y1:orb_y2, orb_x1:orb_x2] = False

        glare_ratio = float(np.sum(glare_mask)) / float(face_crop.shape[0] * face_crop.shape[1])

        if glare_ratio > 0.12:
            return False, 0.25, "Severe glass reflection / screen glare detected"

        if skin_ratio < 0.18:
            return False, 0.35, "Abnormal chromatic distribution for biological skin"

        return True, float(skin_ratio), "Normal biological skin chromaticity"

    def evaluate_gradient_depth(self, face_crop):
        """
        Evaluates 3D facial depth gradients around facial features (nose bridge, cheekbones).
        Flat 2D surfaces lack natural biological gradient transitions.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mean_grad = float(np.mean(mag))

        if mean_grad < 5.0:
            return False, mean_grad, "Low geometric gradient depth"
        return True, mean_grad, "Healthy geometric depth"

    def check_liveness(self, face_crop, landmarks=None, full_frame=None, coords=None):
        """
        Performs multi-criteria liveness assessment on a face.
        Uses Deep-Learning MiniFASNetV2 as primary neural gate, with multi-spectral heuristic validation.
        Returns:
            is_live (bool): True if verified live human, False if suspected spoof.
            confidence (float): Score between 0.0 and 100.0.
            reason (str): Assessment details.
        """
        # Guard against empty or degenerate crops (<10px)
        if face_crop is None or face_crop.size == 0:
            return False, 0.0, "Empty face crop"
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return False, 0.0, "Degenerate face crop (<10px)"

        # 1. Primary Deep Neural Gate: MiniFASNetV2 (evaluates screen moire, device bezels, 3D context)
        if self.minifasnet_session is not None and full_frame is not None and coords is not None:
            x1, y1, x2, y2 = coords
            fw, fh = x2 - x1, y2 - y1
            bbox = [x1, y1, fw, fh]
            net_live, real_score, pred_label = self.evaluate_minifasnet(full_frame, bbox)
            if net_live is not None:
                if not net_live:
                    attack_type = "Digital Screen Replay (Phone/Tablet)" if pred_label == 2 else "Printed Paper Photo"
                    return False, round(real_score * 100.0, 1), f"MiniFASNetV2 Rejected: {attack_type} (Score: {real_score:.3f})"

        # 2. Heuristic Multi-Spectral Checks (Texture, Skin Chroma, Depth Gradients)
        live_tex, tex_score, tex_msg = self.evaluate_texture(face_crop)
        live_col, col_score, col_msg = self.evaluate_color_chroma(face_crop, landmarks=landmarks)
        live_grad, grad_score, grad_msg = self.evaluate_gradient_depth(face_crop)

        # Severe screen glare or harsh glass reflection veto
        if not live_col and "screen glare" in col_msg.lower():
            return False, 25.0, f"Suspected Spoof: {col_msg}"

        # Aggregate weighted score
        score = 0.0
        if live_tex: score += 0.40
        if live_col: score += 0.35
        if live_grad: score += 0.25

        confidence = round(score * 100.0, 1)

        if score >= self.liveness_threshold:
            return True, confidence, "Live Human Face Verified"
        else:
            reasons = []
            if not live_tex: reasons.append(tex_msg)
            if not live_col: reasons.append(col_msg)
            if not live_grad: reasons.append(grad_msg)
            return False, confidence, f"Suspected Spoof: {'; '.join(reasons)}"


# Module-level singleton instance
_detector = LivenessDetector()


def evaluate_liveness(face_crop, landmarks=None, full_frame=None, coords=None):
    return _detector.check_liveness(face_crop, landmarks=landmarks, full_frame=full_frame, coords=coords)


def evaluate_face_quality(
    face_crop,
    landmarks=None,
    min_size=None,
    blur_thresh=None,
    min_b=None,
    max_b=None,
    min_contrast=None,
    max_yaw_ratio=None
):
    """
    Step 4 Quality Gate: Validates bounding-box dimensions, blur, brightness, contrast, and landmark pose asymmetry.
    Synchronized with utils.config defaults (Fix V07).
    Roll-invariant yaw ratio calculation (Fix V01).
    Scale-adaptive blur variance (Fix V12).
    Returns:
        passed (bool): Whether the face crop qualifies for biometric recognition.
        reason (str): Human-readable failure reason if rejected.
        metrics (dict): Numeric quality indicators.
    """
    if min_size is None:
        min_size = MIN_FACE_SIZE
    if blur_thresh is None:
        blur_thresh = BLUR_THRESHOLD
    if min_b is None:
        min_b = MIN_BRIGHTNESS
    if max_b is None:
        max_b = MAX_BRIGHTNESS
    if min_contrast is None:
        min_contrast = MIN_CONTRAST
    if max_yaw_ratio is None:
        max_yaw_ratio = MAX_YAW_RATIO

    if face_crop is None or face_crop.size == 0:
        return False, "EMPTY_CROP", {}

    h, w = face_crop.shape[:2]
    if w < min_size or h < min_size:
        return False, f"FACE_TOO_SMALL ({w}x{h} < {min_size}px)", {"width": w, "height": h}

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    # Standardize blur evaluation on a canonical 112x112 scale to eliminate resolution-dependent variance artifacts
    norm_gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA) if (w, h) != (112, 112) else gray
    blur_score = float(cv2.Laplacian(norm_gray, cv2.CV_64F).var())

    # Fix V01: Pose Asymmetry Heuristic with Roll-Invariant Projection
    # u = (pt_left - pt_right) / norm(pt_left - pt_right)
    # yaw_ratio = abs(dot(pt_nose - pt_mid, u)) / eye_distance
    yaw_ratio = 0.0
    if landmarks is not None and len(landmarks) >= 5:
        pt_right_eye = np.array(landmarks[0], dtype=np.float32)
        pt_left_eye = np.array(landmarks[1], dtype=np.float32)
        pt_nose = np.array(landmarks[2], dtype=np.float32)
        eye_vec = pt_left_eye - pt_right_eye
        eye_distance = float(np.linalg.norm(eye_vec))
        if eye_distance > 1e-4:
            u = eye_vec / eye_distance
            mid_eyes = (pt_right_eye + pt_left_eye) / 2.0
            yaw_ratio = float(abs(np.dot(pt_nose - mid_eyes, u)) / eye_distance)

    metrics = {
        "width": w,
        "height": h,
        "sharpness": round(blur_score, 1),
        "brightness": round(mean_brightness, 1),
        "contrast": round(std_contrast, 1),
        "yaw_ratio": round(yaw_ratio, 3)
    }

    effective_blur_thresh = blur_thresh

    if blur_score < effective_blur_thresh:
        return False, f"BLURRY_FACE (Sharpness {blur_score:.1f} < {effective_blur_thresh:.1f})", metrics

    if mean_brightness < min_b:
        return False, f"LOW_LIGHT (Brightness {mean_brightness:.1f} < {min_b})", metrics

    if mean_brightness > max_b:
        return False, f"OVEREXPOSED (Brightness {mean_brightness:.1f} > {max_b})", metrics

    if std_contrast < min_contrast:
        return False, f"LOW_CONTRAST (Contrast {std_contrast:.1f} < {min_contrast})", metrics

    if landmarks is not None and yaw_ratio > max_yaw_ratio:
        return False, f"FACE_EXTREME_ANGLE (Yaw ratio {yaw_ratio:.2f} > {max_yaw_ratio:.2f})", metrics

    return True, "QUALITY_PASSED", metrics
