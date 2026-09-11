import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import pickle
import numpy as np
from pathlib import Path
from utils.config import (
    KNOWN_FACES_DIR, BASE_DIR, DEEPFACE_DIR,
    ARCFACE_SIMILARITY_THRESHOLD, AMBIGUITY_MARGIN,
    MIN_FACE_SIZE, BLUR_THRESHOLD, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    MIN_CONTRAST, MAX_YAW_RATIO
)

ARCFACE_ONNX_PATH = Path.home() / ".insightface" / "models" / "buffalo_m" / "w600k_r50.onnx"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.pkl"


class FaceRecognizer:
    def __init__(self):
        self.model_name = "ArcFace-w600k_r50"
        self.known_embeddings = {}
        self.centroids = {}
        self.min_similarity = ARCFACE_SIMILARITY_THRESHOLD
        self.ambiguity_margin = AMBIGUITY_MARGIN
        self.onnx_rec = None

        # Load Native ONNX ArcFace ResNet-50
        if ARCFACE_ONNX_PATH.exists():
            try:
                from insightface.model_zoo import get_model
                self.onnx_rec = get_model(str(ARCFACE_ONNX_PATH), providers=['CPUExecutionProvider'])
                self.onnx_rec.prepare(ctx_id=0)
            except Exception as e:
                print(f"[INFO] ArcFace ONNX fallback: {e}")
                self.onnx_rec = None

        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "rb") as f:
                self.known_embeddings = pickle.load(f)
            self._compute_centroids()

    def _extract_embedding(self, face_crop):
        """Extract 512-D L2-normalized embedding using ONNX ArcFace R50 with DeepFace fallback."""
        if self.onnx_rec is not None:
            aligned = cv2.resize(face_crop, (112, 112)) if face_crop.shape[:2] != (112, 112) else face_crop
            feat = self.onnx_rec.get_feat(aligned)
            if feat is not None and len(feat) > 0:
                vec = feat[0]
                norm = np.linalg.norm(vec)
                return (vec / (norm + 1e-10)).tolist() if norm > 0 else None
        try:
            os.environ["DEEPFACE_HOME"] = str(DEEPFACE_DIR)
            from deepface import DeepFace
            res = DeepFace.represent(face_crop, model_name="ArcFace", enforce_detection=False)
            if res and len(res) > 0:
                return res[0]["embedding"]
        except Exception:
            pass
        return None

    def _compute_centroids(self):
        self.centroids = {}
        for roll, embs in self.known_embeddings.items():
            if not embs: continue
            normed = [np.array(e) / (np.linalg.norm(e) + 1e-10) for e in embs]
            c = np.mean(normed, axis=0)
            c = c / (np.linalg.norm(c) + 1e-10)
            self.centroids[roll] = c

    def train_system(self):
        from src.detector import FaceDetector
        from src.anti_spoof import evaluate_face_quality
        from src.database import get_student_info

        print(f"\n[INFO] Training High-Definition {self.model_name} Engine with Quality-Filtered Centroids...")
        detector = FaceDetector()
        embeddings_dict = {}
        enrollment_stats = {}

        for student_folder in sorted(os.listdir(KNOWN_FACES_DIR)):
            folder_path = Path(KNOWN_FACES_DIR) / student_folder
            if not folder_path.is_dir(): continue

            roll_number = student_folder
            embeddings_dict[roll_number] = []
            accepted = 0
            rejected = 0

            img_files = sorted(list(folder_path.glob("*.jpg")))
            for img_path_obj in img_files:
                img_path = str(img_path_obj)
                try:
                    raw_img = cv2.imread(img_path)
                    if raw_img is None:
                        rejected += 1
                        continue
                    _, faces = detector.detect_faces(raw_img)
                    if not faces:
                        rejected += 1
                        continue

                    f = faces[0]
                    raw_crop = f.get("raw_crop", f["image"])
                    landmarks = f.get("landmarks")

                    # Step 4: Quality & Pose Filter for Enrollment Images
                    q_pass, q_reason, _ = evaluate_face_quality(
                        raw_crop, landmarks=landmarks,
                        min_size=MIN_FACE_SIZE,
                        blur_thresh=50.0, # Lenient blur threshold for webcam enrollment
                        min_b=MIN_BRIGHTNESS, max_b=MAX_BRIGHTNESS,
                        min_contrast=MIN_CONTRAST,
                        max_yaw_ratio=0.35 # Allow gentle multi-angle poses captured during registration
                    )

                    if not q_pass:
                        rejected += 1
                        continue

                    face_crop = f["image"]
                    emb = self._extract_embedding(face_crop)
                    if emb is not None:
                        embeddings_dict[roll_number].append(emb)
                        accepted += 1
                    else:
                        rejected += 1
                except Exception as e:
                    print(f"[WARNING] Could not process {img_path}: {e}")
                    rejected += 1

            # Fallback: if all images were filtered out, accept the first detected crop to prevent empty student profile
            if len(embeddings_dict[roll_number]) == 0 and len(img_files) > 0:
                first_img = cv2.imread(str(img_files[0]))
                if first_img is not None:
                    _, f_faces = detector.detect_faces(first_img)
                    if f_faces:
                        emb = self._extract_embedding(f_faces[0]["image"])
                        if emb is not None:
                            embeddings_dict[roll_number].append(emb)
                            accepted = 1
                            rejected = len(img_files) - 1

            info = get_student_info(roll_number)
            s_name = info[1] if info else roll_number
            enrollment_stats[roll_number] = {"name": s_name, "total": len(img_files), "accepted": accepted, "rejected": rejected}
            print(f"  [{roll_number}] {s_name}: {len(img_files)} images -> {accepted} accepted, {rejected} rejected")

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings_dict, f)

        self.known_embeddings = embeddings_dict
        self._compute_centroids()
        print(f"[SUCCESS] Training complete! Loaded {len(self.known_embeddings)} students with canonical centroid profiles into HD memory.")
        return enrollment_stats

    def _match_single_embedding(self, live_embedding, candidate_rolls=None):
        """
        Calculates cosine similarity against all registered student centroids:
        Cosine Similarity = dot(A, B) / (||A|| * ||B||)
        Enforces:
          1. Strict minimum similarity threshold (e.g. >= 0.65).
          2. Ambiguity margin: ensures top match is clearly separated from 2nd closest student.
        """
        from src.database import get_student_info
        live_norm = live_embedding / (np.linalg.norm(live_embedding) + 1e-10)
        
        all_similarities = []
        for roll_number, centroid in self.centroids.items():
            sim = float(np.dot(live_norm, centroid))
            all_similarities.append((roll_number, sim))

        if not all_similarities:
            return "Unknown", 0.0, {"top1_sim": 0.0, "top2_sim": 0.0, "reason": "NO_EMBEDDINGS", "top_5": []}

        all_similarities.sort(key=lambda x: x[1], reverse=True)
        top1_roll, top1_sim = all_similarities[0]
        top2_roll, top2_sim = all_similarities[1] if len(all_similarities) > 1 else ("None", 0.0)

        # Build top 5 matches with names
        top_5 = []
        for r, s in all_similarities[:5]:
            info = get_student_info(r)
            name = info[1] if info else r
            top_5.append({"roll": r, "name": name, "sim": round(s, 4)})

        top1_info = get_student_info(top1_roll)
        top1_name = top1_info[1] if top1_info else top1_roll
        top2_info = get_student_info(top2_roll) if top2_roll != "None" else None
        top2_name = top2_info[1] if top2_info else top2_roll

        diag = {
            "top1_roll": top1_roll,
            "top1_name": top1_name,
            "top1_sim": round(top1_sim, 4),
            "top2_roll": top2_roll,
            "top2_name": top2_name,
            "top2_sim": round(top2_sim, 4),
            "margin": round(top1_sim - top2_sim, 4),
            "top_5": top_5
        }

        # Step 6 & 8: Strict UNKNOWN condition - NEVER force a match
        if top1_sim < self.min_similarity:
            diag["reason"] = f"SIMILARITY_BELOW_THRESHOLD ({top1_sim:.3f} < {self.min_similarity:.2f})"
            return "Unknown", round(top1_sim * 100, 2), diag

        # Step 7: Ambiguity check: prevent confusing similar-looking students
        if len(all_similarities) > 1 and (top1_sim - top2_sim) < self.ambiguity_margin:
            diag["reason"] = f"AMBIGUOUS_MATCH (Top1 {top1_name}={top1_sim:.2f} vs Top2 {top2_name}={top2_sim:.2f}, Margin={diag['margin']:.3f} < {self.ambiguity_margin:.2f})"
            return "Unknown", round(top1_sim * 100, 2), diag

        confidence = round(min(100.0, (top1_sim / 1.0) * 100.0), 2)
        diag["reason"] = "VERIFIED_MATCH"
        return top1_roll, confidence, diag

    def recognize(self, face_crop, candidate_rolls=None):
        if not self.centroids:
            return "Unknown", 0.0, {}
        try:
            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                return "Unknown", 0.0, {"reason": "CROP_TOO_SMALL"}

            emb = self._extract_embedding(face_crop)
            if emb is None:
                return "Unknown", 0.0, {"reason": "NO_REPRESENTATION"}

            live_embedding = np.array(emb)
            return self._match_single_embedding(live_embedding, candidate_rolls=candidate_rolls)
        except Exception as e:
            return "Unknown", 0.0, {"reason": str(e)}

    def recognize_batch(self, face_crops, candidate_rolls=None):
        """
        Batch vectorized ArcFace inference:
        Processes multiple face crops in a single forward pass for high-speed crowd attendance.
        """
        if not face_crops or not self.centroids:
            return [("Unknown", 0.0, {})] * len(face_crops)
        try:
            valid_indices = []
            valid_crops = []
            for idx, crop in enumerate(face_crops):
                if crop.shape[0] >= 20 and crop.shape[1] >= 20:
                    valid_indices.append(idx)
                    valid_crops.append(crop)

            if not valid_crops:
                return [("Unknown", 0.0, {})] * len(face_crops)

            output = [("Unknown", 0.0, {})] * len(face_crops)

            # Fast Native ONNX ArcFace Inference
            if self.onnx_rec is not None:
                for res_idx, orig_idx in enumerate(valid_indices):
                    crop = valid_crops[res_idx]
                    aligned = cv2.resize(crop, (112, 112)) if crop.shape[:2] != (112, 112) else crop
                    feat = self.onnx_rec.get_feat(aligned)
                    if feat is not None and len(feat) > 0:
                        vec = feat[0]
                        norm = np.linalg.norm(vec)
                        emb = vec / (norm + 1e-10) if norm > 0 else vec
                        output[orig_idx] = self._match_single_embedding(emb, candidate_rolls=candidate_rolls)
                return output

            # DeepFace Fallback
            from deepface import DeepFace
            batch_results = DeepFace.represent(valid_crops, model_name="ArcFace", enforce_detection=False)
            for res_idx, orig_idx in enumerate(valid_indices):
                if res_idx < len(batch_results):
                    emb = np.array(batch_results[res_idx]["embedding"])
                    output[orig_idx] = self._match_single_embedding(emb, candidate_rolls=candidate_rolls)
            return output
        except Exception:
            return [self.recognize(c, candidate_rolls=candidate_rolls) for c in face_crops]