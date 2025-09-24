# app/ai_processing.py

import logging
import cv2
import numpy as np
import onnxruntime as ort
from mtcnn import MTCNN
from typing import Optional, Tuple, List
from werkzeug.utils import secure_filename
import os
from .config import settings
import time

# --- Model and Detector Initialization ---
# These are loaded once when the module is imported.
logging.info("Loading ArcFace ONNX model...")
ort_session = ort.InferenceSession(settings.MODEL_PATH)
logging.info("Model loaded successfully.")

logging.info("Loading MTCNN face detector...")
detector = MTCNN()
logging.info("MTCNN loaded successfully.")


# --- Core AI/CV Functions ---

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize the embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

def preprocess_face(image: np.ndarray) -> np.ndarray:
    """Preprocess a cropped face image for the ArcFace model."""
    image = cv2.resize(image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)

def generate_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """Generate a normalized 512-dim embedding for a single face ROI."""
    if face_roi is None or face_roi.size == 0:
        return None
    try:
        preprocessed = preprocess_face(face_roi)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        embedding = ort_session.run([output_name], {input_name: preprocessed})[0]
        return normalize_embedding(embedding[0])
    except Exception as e:
        logging.error(f"Error in embedding generation: {e}", exc_info=True)
        return None

def detect_and_recognize_faces(image_bgr: np.ndarray, cache_data: Tuple) -> List[dict]:
    """Detect all faces in an image and recognize them against the cached embeddings."""
    if image_bgr is None or image_bgr.size == 0:
        return []

    names, stored_embeddings, _ = cache_data
    if not names or stored_embeddings.size == 0:
        return []

    try:
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(rgb_image)
    except Exception as e:
        logging.exception("Face detection failed: %s", e)
        return []

    recognized_faces_list = []
    for face in detected_faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_roi = image_bgr[y:y+h, x:x+w]

        if face_roi.size == 0:
            continue

        embedding = generate_embedding(face_roi)
        if embedding is None:
            continue

        sims = np.dot(stored_embeddings, embedding.T)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        
        recognized_name = "Unknown"
        if best_score >= settings.RECOGNITION_THRESHOLD:
            recognized_name = names[best_idx]

        recognized_faces_list.append({
            "name": recognized_name,
            "box": [x, y, w, h],
            "score": best_score
        })
    
    return recognized_faces_list

def get_image_sharpness(image_bgr: np.ndarray) -> float:
    """Calculates the sharpness of an image using Laplacian variance."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian of the gray image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_employee_images(employee_name: str, employee_id: str, files_data: List[Tuple[str, bytes]]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Finds the best quality image from a list, generates an embedding for only that one,
    and saves it. This process is done in-memory for performance.
    """
    best_image_bgr = None
    best_filename = None
    highest_sharpness = -1.0

    if not files_data:
        return None, None

    # Step 1: Find the best image by checking sharpness (in-memory)
    for filename, contents in files_data:
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        sharpness = get_image_sharpness(image_bgr)
        if sharpness > highest_sharpness:
            highest_sharpness = sharpness
            best_image_bgr = image_bgr
            best_filename = filename

    if best_image_bgr is None:
        logging.warning("No valid images found for processing during upload.")
        return None, None

    # Step 2: Run the AI process on ONLY the best image (in-memory)
    try:
        rgb_image = cv2.cvtColor(best_image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_image)
        if not results:
            logging.error("No face detected in the best quality image for upload.")
            return None, None

        main_face = max(results, key=lambda r: r["box"][2] * r["box"][3])
        x, y, w, h = main_face["box"]
        face_roi = best_image_bgr[y:y+h, x:x+w]

        embedding = generate_embedding(face_roi)
        if embedding is None:
            logging.error("Failed to generate embedding for the best quality image.")
            return None, None
            
        # Step 3: Save the final, processed image to disk
        secure_name = f"{employee_id}_best_{secure_filename(best_filename)}"
        saved_path = os.path.join(settings.IMAGE_UPLOAD_FOLDER, secure_name)
        cv2.imwrite(saved_path, best_image_bgr)

        return embedding, saved_path

    except Exception as e:
        logging.error(f"Failed to process the best image {best_filename}: {e}", exc_info=True)
        return None, None
