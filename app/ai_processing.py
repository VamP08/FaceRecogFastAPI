# # app/ai_processing.py

# import logging
# import cv2
# import numpy as np
# import onnxruntime as ort
# from mtcnn import MTCNN
# from typing import Optional, Tuple, List
# from werkzeug.utils import secure_filename
# import os
# from .config import settings
# import time

# # --- Model and Detector Initialization ---
# # These are loaded once when the module is imported.
# logging.info("Loading ArcFace ONNX model...")
# ort_session = ort.InferenceSession(settings.MODEL_PATH)
# logging.info("Model loaded successfully.")

# logging.info("Loading MTCNN face detector...")
# detector = MTCNN()
# logging.info("MTCNN loaded successfully.")


# # --- Core AI/CV Functions ---

# def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
#     """Normalize the embedding vector to unit length."""
#     norm = np.linalg.norm(embedding)
#     return embedding / norm if norm != 0 else embedding

# def preprocess_face(image: np.ndarray) -> np.ndarray:
#     """Preprocess a cropped face image for the ArcFace model."""
#     image = cv2.resize(image, (112, 112))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image.astype(np.float32)
#     image = (image / 127.5) - 1.0
#     image = np.transpose(image, (2, 0, 1))
#     return np.expand_dims(image, axis=0)

# def generate_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
#     """Generate a normalized 512-dim embedding for a single face ROI."""
#     if face_roi is None or face_roi.size == 0:
#         return None
#     try:
#         preprocessed = preprocess_face(face_roi)
#         input_name = ort_session.get_inputs()[0].name
#         output_name = ort_session.get_outputs()[0].name
#         embedding = ort_session.run([output_name], {input_name: preprocessed})[0]
#         return normalize_embedding(embedding[0])
#     except Exception as e:
#         logging.error(f"Error in embedding generation: {e}", exc_info=True)
#         return None

# def detect_and_recognize_faces(image_bgr: np.ndarray, cache_data: Tuple) -> List[dict]:
#     """
#     Detect all faces in an image and recognize them against the cached embeddings.
#     """
#     if image_bgr is None or image_bgr.size == 0:
#         return []

#     names, stored_embeddings, _, member_codes = cache_data
#     if not names or stored_embeddings.size == 0:
#         return []
    
#     try:
#         rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         detected_faces = detector.detect_faces(rgb_image)
#     except Exception as e:
#         logging.exception("Face detection failed: %s", e)
#         return []

#     recognized_faces_list = []
#     for face in detected_faces:
#         x, y, w, h = face['box']
#         face_roi = image_bgr[y:y+h, x:x+w]

#         if face_roi.size == 0:
#             continue

#         embedding = generate_embedding(face_roi)
#         if embedding is None:
#             continue

#         normalized_embedding = normalize_embedding(embedding)
#         sims = np.dot(stored_embeddings, normalized_embedding.T)
#         best_idx = int(np.argmax(sims))
#         best_score = float(sims[best_idx])
        
#         recognized_name = "Unknown"
#         member_code = None
#         if best_score >= settings.RECOGNITION_THRESHOLD:
#             recognized_name = names[best_idx]
#             member_code = member_codes[best_idx]

#         recognized_faces_list.append({
#             "name": recognized_name,
#             "member_code": member_code,
#             "box": [int(coord) for coord in face['box']],
#             "score": best_score
#         })
    
#     return recognized_faces_list

# def get_image_sharpness(image_bgr: np.ndarray) -> float:
#     """Calculates the sharpness of an image using Laplacian variance."""
#     if image_bgr is None or image_bgr.size == 0:
#         return 0.0
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()

# def process_employee_images(employee_name: str, employee_id: str, files_data: List[Tuple[str, bytes]]) -> Tuple[Optional[np.ndarray], Optional[str]]:
#     """
#     Finds the best quality image from a list, generates an embedding for only that one,
#     and saves it. This process is done in-memory for performance.
#     """
#     best_image_bgr = None
#     best_filename = None
#     highest_sharpness = -1.0

#     if not files_data:
#         return None, None

#     # Step 1: Find the best image by checking sharpness (in-memory)
#     for filename, contents in files_data:
#         nparr = np.frombuffer(contents, np.uint8)
#         image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if image_bgr is None:
#             continue
#         sharpness = get_image_sharpness(image_bgr)
#         if sharpness > highest_sharpness:
#             highest_sharpness = sharpness
#             best_image_bgr = image_bgr
#             best_filename = filename

#     if best_image_bgr is None:
#         logging.warning("No valid images found for processing during upload.")
#         return None, None

#     # Step 2: Run the AI process on ONLY the best image (in-memory)
#     try:
#         rgb_image = cv2.cvtColor(best_image_bgr, cv2.COLOR_BGR2RGB)
#         results = detector.detect_faces(rgb_image)
#         if not results:
#             logging.error("No face detected in the best quality image for upload.")
#             return None, None

#         main_face = max(results, key=lambda r: r["box"][2] * r["box"][3])
#         x, y, w, h = main_face["box"]
#         face_roi = best_image_bgr[y:y+h, x:x+w]

#         embedding = generate_embedding(face_roi)
#         if embedding is None:
#             logging.error("Failed to generate embedding for the best quality image.")
#             return None, None
            
#         # Step 3: Save the final, processed image to disk
#         secure_name = f"{employee_id}_best_{secure_filename(best_filename)}"
#         saved_path = os.path.join(settings.IMAGE_UPLOAD_FOLDER, secure_name)
#         cv2.imwrite(saved_path, best_image_bgr)

#         return embedding, saved_path

#     except Exception as e:
#         logging.error(f"Failed to process the best image {best_filename}: {e}", exc_info=True)
#         return None, None




import logging
import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort
from mtcnn import MTCNN
from werkzeug.utils import secure_filename

from .config import settings


# ----------------- Initialization -----------------
logging.getLogger().setLevel(logging.INFO)

if not hasattr(settings, "MODEL_PATH"):
    raise RuntimeError("settings.MODEL_PATH is required")

logging.info("Loading ArcFace ONNX model: %s", settings.MODEL_PATH)
ort_session = ort.InferenceSession(settings.MODEL_PATH)
logging.info("ArcFace model loaded")

logging.info("Initializing MTCNN detector")
detector = MTCNN()

# Haar cascade fallback
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure upload folder exists
IMAGE_UPLOAD_FOLDER = getattr(settings, "IMAGE_UPLOAD_FOLDER", "uploads")
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

# Configurable thresholds (tune these)
MIN_FACE_SIZE = getattr(settings, "MIN_FACE_SIZE", 50)  # px
MIN_SHARPNESS = getattr(settings, "MIN_SHARPNESS", 25.0)  # variance
RECOGNITION_THRESHOLD = getattr(settings, "RECOGNITION_THRESHOLD", 0.40)


# ----------------- Helpers -----------------

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """Return L2-normalized embedding (safe against zero norm)."""
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def preprocess_face(image_bgr: np.ndarray) -> np.ndarray:
    """Resize/cvt/scale/transpose to model input (1,3,112,112)."""
    image = cv2.resize(image_bgr, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def get_image_sharpness(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ----------------- Face Alignment -----------------

ARCREF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def align_face_by_keypoints(image: np.ndarray, keypoints: dict) -> Optional[np.ndarray]:
    """Return a 112x112 aligned face using the five MTCNN keypoints.
    If alignment fails, return None.
    """
    try:
        src_pts = np.array([
            keypoints["left_eye"],
            keypoints["right_eye"],
            keypoints["nose"],
            keypoints["mouth_left"],
            keypoints["mouth_right"]
        ], dtype=np.float32)

        # estimate transform from src_pts -> ARCREF
        tfm, _ = cv2.estimateAffinePartial2D(src_pts, ARCREF, method=cv2.LMEDS)
        if tfm is None:
            logging.debug("Alignment transform returned None")
            return None
        aligned = cv2.warpAffine(image, tfm, (112, 112), borderValue=0.0)
        return aligned
    except Exception as e:
        logging.exception("align_face_by_keypoints failed: %s", e)
        return None


def crop_face_from_box(image: np.ndarray, box: List[int], margin: float = 0.2) -> Optional[np.ndarray]:
    """Crop face bounding box with optional margin (relative).
    Handles negative/out-of-range coordinates.
    """
    try:
        x, y, w, h = box
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(image.shape[1], int(x + w + w * margin))
        y2 = min(image.shape[0], int(y + h + h * margin))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]
    except Exception:
        return None


# ----------------- Embedding Generation -----------------

def generate_embedding_from_face(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Given a cropped/aligned face (BGR), return normalized embedding or None."""
    if face_bgr is None or face_bgr.size == 0:
        return None

    h, w = face_bgr.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        logging.warning("Face rejected: too small (%dx%d)", w, h)
        return None

    sharp = get_image_sharpness(face_bgr)
    if sharp < MIN_SHARPNESS:
        logging.warning("Face rejected: too blurry (lap_var=%.2f)", sharp)
        return None

    try:
        inp = preprocess_face(face_bgr)
        in_name = ort_session.get_inputs()[0].name
        out_name = ort_session.get_outputs()[0].name
        emb = ort_session.run([out_name], {in_name: inp})[0]
        return normalize_embedding(emb[0])
    except Exception:
        logging.exception("Embedding generation failed")
        return None


# ----------------- Detection with fallback -----------------

def detect_faces_with_fallback(image_bgr: np.ndarray) -> List[dict]:
    """Return a list of face dicts. Try MTCNN first, then Haar fallback.
    MTCNN returns dicts with keys: box, confidence, keypoints
    Haar fallback returns dicts with box and no keypoints.
    """
    faces_out = []
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mtcnn_results = detector.detect_faces(rgb)
        if mtcnn_results:
            logging.debug("MTCNN detected %d faces", len(mtcnn_results))
            # normalize box to [x,y,w,h]
            faces_out.extend(mtcnn_results)
            return faces_out
    except Exception:
        logging.exception("MTCNN detection failed, falling back to Haar")

    # Haar fallback
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        haar_faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        for (x, y, w, h) in haar_faces:
            faces_out.append({"box": [int(x), int(y), int(w), int(h)], "confidence": None, "keypoints": {}})
        logging.debug("Haar detected %d faces", len(haar_faces))
    except Exception:
        logging.exception("Haar fallback detection failed")

    return faces_out


# ----------------- Recognition -----------------

def detect_and_recognize_faces(image_bgr: np.ndarray, cache_data: Tuple) -> List[dict]:
    """Detect faces and recognize using cached embeddings.
    cache_data: (names, stored_embeddings, ids, member_codes)
    stored_embeddings must be a (N,512) numpy array of normalized vectors.
    Returns list of dicts with: name, member_code, box, score
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    names, stored_embeddings, ids, member_codes = cache_data
    if stored_embeddings is None or stored_embeddings.size == 0:
        logging.warning("No stored embeddings available")
        return []

    faces = detect_faces_with_fallback(image_bgr)
    if not faces:
        return []

    results = []
    for face in faces:
        box = face.get("box")
        keypoints = face.get("keypoints", {})

        # Align if keypoints available else crop box
        aligned = None
        if keypoints and all(k in keypoints for k in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")):
            aligned = align_face_by_keypoints(image_bgr, keypoints)
        if aligned is None and box is not None:
            aligned = crop_face_from_box(image_bgr, box)

        if aligned is None:
            logging.debug("Skipping face: cannot align or crop")
            continue

        emb = generate_embedding_from_face(aligned)
        if emb is None:
            continue

        # Both stored_embeddings and emb are normalized → cosine = dot
        sims = stored_embeddings @ emb  # shape (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        recognized_name = "Unknown"
        member_code = None
        threshold = max(RECOGNITION_THRESHOLD, getattr(settings, "MIN_RECOGNITION_THRESHOLD", 0.35))
        if best_score >= threshold:
            recognized_name = names[best_idx]
            member_code = member_codes[best_idx] if member_codes is not None else None

        results.append({
            "name": recognized_name,
            "member_code": member_code,
            "box": [int(x) for x in box] if box is not None else None,
            "score": best_score
        })

    return results


# ----------------- Employee Image Processing -----------------

def process_employee_images(employee_name: str, employee_id: str, files_data: List[Tuple[str, bytes]]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Process uploaded employee images (filename, bytes) and return average embedding and saved representative path."""
    embeddings = []
    rep_img_path = None

    if not files_data:
        return None, None

    for filename, contents in files_data:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        faces = detect_faces_with_fallback(image)
        if not faces:
            continue

        main_face = max(faces, key=lambda r: r["box"][2] * r["box"][3])
        keypoints = main_face.get("keypoints", {})

        aligned = None
        if keypoints and all(k in keypoints for k in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")):
            aligned = align_face_by_keypoints(image, keypoints)
        if aligned is None:
            aligned = crop_face_from_box(image, main_face["box"], margin=0.25)

        if aligned is None:
            continue

        emb = generate_embedding_from_face(aligned)
        if emb is not None:
            embeddings.append(emb)
            if rep_img_path is None:
                safe = secure_filename(f"{employee_id}_rep_{filename}")
                rep_img_path = os.path.join(IMAGE_UPLOAD_FOLDER, safe)
                # Save aligned face for future comparisons
                cv2.imwrite(rep_img_path, aligned)

    if embeddings:
        avg = normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32))
        return avg, rep_img_path

    return None, None
