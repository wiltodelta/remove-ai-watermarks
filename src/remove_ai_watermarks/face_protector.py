"""YOLO-based face detection and soft-blend restoration for diffusion pipelines."""

import logging
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

logger = logging.getLogger(__name__)


class FaceProtector:
    """
    Detects faces in an image and provides methods to seamlessly paste them back
    onto the an upscaled/processed image to preserve facial details that may have
    been destroyed by latent diffusion or other algorithms.
    """

    def __init__(self, use_yolo: bool = True, model_name: str = "yolov8n.pt"):
        self.use_yolo = use_yolo and HAS_YOLO
        self.detector = None
        self.haar_cascade = None

        if self.use_yolo:
            # Fix SSL certificate issues on macOS (fresh Python installs)
            self._fix_ssl_certs()
            logger.info("Loading YOLO model '%s' for face protection...", model_name)
            self.detector = YOLO(model_name)
        else:
            if use_yolo and not HAS_YOLO:
                logger.warning(
                    "ultralytics YOLO is not installed. Falling back to OpenCV Haar "
                    "Cascades. Install ultralytics with `pip install ultralytics` "
                    "for better face detection."
                )
            logger.info("Loading OpenCV Haar Cascade for face protection...")
            cascade_path = Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
            if not cascade_path.exists():
                cascade_path = "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(str(cascade_path))

    def detect_face_bboxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces and return bounding boxes as (x1, y1, x2, y2).
        """
        if self.use_yolo and self.detector is not None:
            # For standard YOLOv8n, 'person' is class 0. We'll use person bounding boxes
            # as a proxy for faces/people to protect them. If using a specific face model, adjust classes.
            results = self.detector(image, verbose=False, classes=[0])
            bboxes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
            return bboxes

        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            bboxes = []
            for x, y, w, h in faces:
                # Add a 20% margin around the haar cascade face box
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - int(margin_y * 1.5))  # more margin on top for hair
                x2 = min(image.shape[1], x + w + margin_x)
                y2 = min(image.shape[0], y + h + margin_y)
                bboxes.append((x1, y1, x2, y2))
            return bboxes

    @staticmethod
    def _fix_ssl_certs() -> None:
        """Set SSL_CERT_FILE from certifi if not already set (macOS fix)."""
        import os

        if os.environ.get("SSL_CERT_FILE"):
            return
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass

    def extract_faces(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        """
        Extract faces from the image.
        Returns a list of (bbox, face_crop) tuples.
        """
        bboxes = self.detect_face_bboxes(image)
        faces = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            faces.append((bbox, image[y1:y2, x1:x2].copy()))
        return faces

    def restore_faces(
        self, processed_image: np.ndarray, original_faces: list[tuple[tuple[int, int, int, int], np.ndarray]]
    ) -> np.ndarray:
        """
        Paste original faces back onto the processed image using seamless cloning
        or soft blending so the edges don't show.
        """
        if not original_faces:
            return processed_image

        result = processed_image.copy()
        for (x1, y1, x2, y2), face_crop in original_faces:
            h, w = face_crop.shape[:2]

            # If the processed image was resized, we'd need to resize face_crop, but
            # pipeline ensures the output from InvisibleEngine is the same size or we resize it back before this.
            if result.shape[:2] != processed_image.shape[:2]:
                continue  # Safety bypass

            try:
                # Create a soft alpha mask for the face crop to smoothly blend it
                mask = np.zeros((h, w), dtype=np.float32)

                # Inner ellipse is pure white
                cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.4), int(h * 0.4)), 0, 0, 360, 1.0, -1)

                # Blur the mask heavily for soft edges
                blur_size = max(w, h) // 4
                if blur_size % 2 == 0:
                    blur_size += 1
                mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                mask = cv2.merge([mask, mask, mask])

                # Blend
                target_roi = result[y1:y2, x1:x2].astype(np.float32)
                src_roi = face_crop.astype(np.float32)

                blended = src_roi * mask + target_roi * (1.0 - mask)
                result[y1:y2, x1:x2] = blended.astype(np.uint8)
            except Exception as e:
                logger.warning("Failed to restore face at %d,%d to %d,%d: %s", x1, y1, x2, y2, e)

        return result
