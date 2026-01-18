import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
import requests

load_dotenv()

# --------------------------
# Logging setup
# --------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("openrouter_gemini_faceswap")

# Toggle to log more/less of the request/response (avoid huge logs)
LOG_PAYLOAD_PREVIEW_CHARS = int(os.getenv("LOG_PAYLOAD_PREVIEW_CHARS", "1200"))
LOG_RESPONSE_PREVIEW_CHARS = int(os.getenv("LOG_RESPONSE_PREVIEW_CHARS", "2000"))


# --------------------------
# Helpers
# --------------------------
def encode_image_to_data_url(image_path: str) -> str:
    """
    Reads image bytes and returns a data URL like:
    data:image/jpeg;base64,....
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Cheap MIME guess based on extension (works fine for most cases)
    ext = p.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        # Default; still works often
        mime = "image/jpeg"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def decode_data_url_to_bytes(data_url: str) -> Tuple[bytes, str]:
    """
    Takes data:image/png;base64,... and returns (bytes, mime_type).
    """
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data URL: data:image/...;base64,....")

    header, b64data = data_url.split(",", 1)
    # header example: data:image/png;base64
    mime = header.split(";")[0].replace("data:", "").strip()
    return base64.b64decode(b64data), mime


def safe_preview(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "...(truncated)"


def extract_openrouter_error(resp_json: dict) -> Optional[str]:
    """
    OpenRouter errors often appear as:
    { "error": { "message": "...", "code": ... } }
    """
    err = resp_json.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or json.dumps(err)
        return msg
    return None


# --------------------------
# Main function
# --------------------------
def face_swap_gemini(
    base_path: str,
    selfie_path: str,
    out_path: str,
    model: str = "google/gemini-2.5-flash-image",
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    log.info("Starting face swap with OpenRouter model=%s", model)
    log.info("Base image: %s", base_path)
    log.info("Selfie image: %s", selfie_path)

    # 1) Encode local images as data URLs
    try:
        base_data_url = encode_image_to_data_url(base_path)
        selfie_data_url = encode_image_to_data_url(selfie_path)
        log.info("Encoded images to data URLs (lengths: base=%d selfie=%d)",
                 len(base_data_url), len(selfie_data_url))
    except Exception as e:
        log.exception("Failed to encode images: %s", str(e))
        raise

    # 2) Prompt (don’t ask for base64 text; let API return image in message.images)
    prompt_text = (
        "You are an expert photo editor.\n"
        "We have two images:\n"
        "1) BASE: keep everything from this image (background, pose, clothing, hair, lighting).\n"
        "2) SELFIE: use ONLY the face identity from this image.\n\n"
        "Task: Replace ONLY the face of the person in BASE with the face identity from SELFIE.\n"
        "Rules:\n"
        "- Preserve BASE composition: background, body, pose, clothes, hair style, camera angle.\n"
        "- Match BASE lighting and shadows.\n"
        "- Photorealistic.\n"
        "- Output a single edited image.\n"
        "- If no clear face is visible in either image, output text 'NO_FACE'.\n"
    )

    # 3) Construct payload (OpenRouter image output must set modalities)
    payload = {
        "model": model,
        "modalities": ["image", "text"],  # required for image output 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "text", "text": "BASE IMAGE:"},
                    {"type": "image_url", "image_url": {"url": base_data_url}},
                    {"type": "text", "text": "SELFIE IMAGE:"},
                    {"type": "image_url", "image_url": {"url": selfie_data_url}},
                ],
            }
        ],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # These help OpenRouter attribution, not required but recommended
        "HTTP-Referer": os.getenv("https://openrouter.ai/api/v1/chat/completions", "http://localhost:8000"),
        #"X-Title": os.getenv("OPENROUTER_APP_NAME", "Face Swap Debugger"),
    }

    # Log a sanitized preview (no giant base64 dump)
    payload_preview = json.dumps(
        {**payload, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": "BASE IMAGE:"},
            {"type": "image_url", "image_url": {"url": f"<data_url length={len(base_data_url)}>" }},
            {"type": "text", "text": "SELFIE IMAGE:"},
            {"type": "image_url", "image_url": {"url": f"<data_url length={len(selfie_data_url)}>" }},
        ]}]},
        ensure_ascii=False
    )
    log.info("Payload preview: %s", safe_preview(payload_preview, LOG_PAYLOAD_PREVIEW_CHARS))

    # 4) Send request
    url = "https://openrouter.ai/api/v1/chat/completions"
    log.info("POST %s", url)

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        log.info("HTTP status=%s", resp.status_code)
    except Exception as e:
        log.exception("Request failed before getting response: %s", str(e))
        raise

    # 5) Parse response JSON
    try:
        resp_json = resp.json()
    except Exception:
        log.error("Non-JSON response body (first 1000 chars): %s", safe_preview(resp.text, 1000))
        resp.raise_for_status()
        raise

    # Log response keys + preview
    log.info("Response top-level keys: %s", list(resp_json.keys()))
    log.info("Response preview: %s", safe_preview(json.dumps(resp_json), LOG_RESPONSE_PREVIEW_CHARS))

    # Handle OpenRouter errors
    err_msg = extract_openrouter_error(resp_json)
    if err_msg:
        log.error("OpenRouter returned error: %s", err_msg)
        raise RuntimeError(f"OpenRouter error: {err_msg}")

    # Handle non-2xx
    if resp.status_code >= 400:
        log.error("HTTP error status=%s body=%s", resp.status_code, safe_preview(resp.text, 1000))
        resp.raise_for_status()

    # 6) Extract image output (OpenRouter images are in message.images) 
    try:
        choice0 = resp_json["choices"][0]
        message = choice0.get("message", {})
        content_text = message.get("content")  # might contain NO_FACE or extra text
        images = message.get("images")

        log.info("Message content preview: %s", safe_preview(content_text or "", 300))
        log.info("Has images field? %s", bool(images))

        # If model replied NO_FACE and no images
        if (not images) and content_text and "NO_FACE" in str(content_text):
            raise ValueError("Model indicated NO_FACE (no clear face detected).")

        if not images:
            # Dump helpful debug info
            provider = resp_json.get("provider")
            usage = resp_json.get("usage")
            log.warning("No images returned. provider=%s usage=%s", provider, usage)
            raise ValueError("No images returned in response. Check model/provider supports image output.")

        # Get first image data URL
        data_url = images[0]["image_url"]["url"]
        log.info("Got image data URL length=%d", len(data_url))

        img_bytes, mime = decode_data_url_to_bytes(data_url)
        log.info("Decoded image bytes=%d mime=%s", len(img_bytes), mime)

        # Choose file extension based on mime
        ext = ".png"
        if mime == "image/jpeg":
            ext = ".jpg"
        elif mime == "image/webp":
            ext = ".webp"

        out_path_final = str(Path(out_path).with_suffix(ext))
        Path(out_path_final).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path_final).write_bytes(img_bytes)

        log.info("Saved output image: %s", out_path_final)

    except Exception as e:
        log.exception("Failed to extract/save image: %s", str(e))
        # Print deeper debug hints
        try:
            log.info("choices[0] keys: %s", list(resp_json.get("choices", [{}])[0].keys()))
            log.info("message keys: %s", list(resp_json.get("choices", [{}])[0].get("message", {}).keys()))
        except Exception:
            pass
        raise
