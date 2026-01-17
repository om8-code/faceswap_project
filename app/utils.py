from pathlib import Path
from PIL import Image
import shutil

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

def ensure_allowed_image(path: str):
    try:
        with Image.open(path) as img:
            img.verify()
            fmt = (img.format or "").upper()
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format: {fmt or 'unknown'}. Allowed: {sorted(ALLOWED_FORMATS)}")

def save_upload_to_path(upload_file, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
