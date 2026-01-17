import logging
from pathlib import Path

import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.model_zoo import get_model

from app.config import settings

log = logging.getLogger("faceswap")

class LocalFaceSwapper:
    def __init__(self):
        if not Path(settings.INSWAPPER_PATH).exists():
            raise RuntimeError(f"inswapper model not found at: {settings.INSWAPPER_PATH}")

        # Face detector/recognition pipeline
        self.app = FaceAnalysis(name="buffalo_l")
        # ctx_id=0 uses GPU if available; on CPU-only machines this still works in many cases
        # If it fails, set ctx_id=-1
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load local ONNX swapper (NO download)
        self.swapper = get_model(settings.INSWAPPER_PATH, download=False)

        log.info("InsightFace version=%s", insightface.__version__)
        log.info("Loaded inswapper model from=%s", settings.INSWAPPER_PATH)

    @staticmethod
    def _pick_largest_face(faces):
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

    def swap(self, base_path: str, selfie_path: str, out_path: str):
        log.info("Reading images base=%s selfie=%s", base_path, selfie_path)
        base_img = cv2.imread(base_path)
        selfie_img = cv2.imread(selfie_path)

        if base_img is None:
            raise ValueError("Failed to read BASE image via OpenCV.")
        if selfie_img is None:
            raise ValueError("Failed to read SELFIE image via OpenCV.")

        base_faces = self.app.get(base_img)
        selfie_faces = self.app.get(selfie_img)

        if not base_faces:
            raise ValueError("No face detected in BASE image.")
        if not selfie_faces:
            raise ValueError("No face detected in SELFIE image.")

        dst_face = self._pick_largest_face(base_faces)
        src_face = self._pick_largest_face(selfie_faces)

        log.info("Swapping face (selfie -> base)")
        result = self.swapper.get(base_img, dst_face, src_face, paste_back=True)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(out_path, result)
        if not ok:
            raise RuntimeError("Failed to write output image.")
        log.info("Wrote output=%s", out_path)

_swapper = None

def _get_swapper():
    global _swapper
    if _swapper is None:
        _swapper = LocalFaceSwapper()
    return _swapper

def face_swap_local(base_path: str, selfie_path: str, out_path: str):
    return _get_swapper().swap(base_path, selfie_path, out_path)

