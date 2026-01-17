import os
import time
import uuid
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logging_config import setup_logging
from app.schemas import CreateJobResponse, JobStatusResponse
from app.storage import JobStore
from app.utils import ensure_allowed_image, save_upload_to_path

# IMPORTANT: This should point to the OpenRouter version of the function
# Put the code inside app/gemini_swapper.py and export face_swap_openrouter_gemini
from app.gemini_swapper import face_swap_gemini

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("face-swap-api")

app = FastAPI(title="Face Swap API (OpenRouter Gemini 2.5 Flash Image)")

DATA = Path(settings.DATA_DIR)
JOBS_DIR = DATA / "jobs"
OUT_DIR = DATA / "outputs"
DB_PATH = str(DATA / "jobs.sqlite3")

store = JobStore(DB_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(DATA)), name="static")

# OpenRouter model id you’re using
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-image")

logger.info(
    "Startup config: base_url=%s data_dir=%s model=%s openrouter_key_set=%s",
    settings.BASE_URL,
    settings.DATA_DIR,
    OPENROUTER_MODEL,
    bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
)

def _job_paths(reference_id: str):
    job_dir = JOBS_DIR / reference_id
    base_path = job_dir / "base.jpg"
    selfie_path = job_dir / "selfie.jpg"
    out_path = OUT_DIR / f"{reference_id}.png"  # extension may change based on mime; swapper may override
    return job_dir, str(base_path), str(selfie_path), str(out_path)

def _result_url(reference_id: str) -> str:
    # If swapper saves jpg/webp, we still return .png here.
    # Best practice: store actual output path in DB and return that.
    # Keeping as-is for assignment simplicity.
    return f"{settings.BASE_URL}/static/outputs/{reference_id}.png"

def process_job(reference_id: str):
    log = logging.getLogger(f"job.{reference_id}")
    _, base_path, selfie_path, out_path = _job_paths(reference_id)

    store.set_status(reference_id, "processing")
    log.info("Job started: status -> processing")
    log.info("Input paths: base=%s selfie=%s", base_path, selfie_path)

    t0 = time.time()
    try:
        # Call OpenRouter Gemini (image edit)
        log.info("Calling OpenRouter model=%s", OPENROUTER_MODEL)
        face_swap_gemini(
            base_path=base_path,
            selfie_path=selfie_path,
            out_path=out_path,
            model=OPENROUTER_MODEL,
        )

        ms = int((time.time() - t0) * 1000)

        # NOTE: If your swapper changes extension based on mime, it should return the final path.
        # If you kept it writing exactly to out_path, this is fine.
        store.set_status(reference_id, "completed", result_path=out_path, processing_ms=ms)
        log.info("Job completed in %d ms output=%s", ms, out_path)

    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        store.set_status(reference_id, "failed", error=str(e), processing_ms=ms)
        log.exception("Job failed after %d ms: %s", ms, str(e))

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:10]}"
    request.state.request_id = req_id
    resp = await call_next(request)
    resp.headers["x-request-id"] = req_id
    return resp

@app.post("/api/v1/face-swap/jobs", response_model=CreateJobResponse)
async def create_job(
    request: Request,
    bg: BackgroundTasks,
    base_image: UploadFile = File(...),
    selfie: UploadFile = File(...),
):
    req_id = getattr(request.state, "request_id", "req_unknown")

    # ✅ OpenRouter key check (not Gemini key)
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        logger.error("[%s] OPENROUTER_API_KEY missing. Refusing job creation.", req_id)
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENROUTER_API_KEY not set")

    logger.info("[%s] Create job request: base=%s selfie=%s",
                req_id, base_image.filename, selfie.filename)

    reference_id = f"job_{uuid.uuid4().hex[:10]}"
    store.create(reference_id)

    job_log = logging.getLogger(f"job.{reference_id}")
    job_log.info("[%s] Job created", req_id)

    job_dir, base_path, selfie_path, _ = _job_paths(reference_id)
    Path(job_dir).mkdir(parents=True, exist_ok=True)

    try:
        save_upload_to_path(base_image, base_path)
        save_upload_to_path(selfie, selfie_path)
        job_log.info("[%s] Files saved base=%s selfie=%s", req_id, base_path, selfie_path)

        ensure_allowed_image(base_path)
        ensure_allowed_image(selfie_path)
        job_log.info("[%s] Image validation passed", req_id)

    except ValueError as ve:
        store.set_status(reference_id, "failed", error=str(ve))
        job_log.warning("[%s] Validation failed: %s", req_id, str(ve))
        raise HTTPException(status_code=415, detail=str(ve))

    except Exception as e:
        store.set_status(reference_id, "failed", error=str(e))
        job_log.exception("[%s] Upload handling failed: %s", req_id, str(e))
        raise HTTPException(status_code=400, detail="Failed to accept uploaded files")

    bg.add_task(process_job, reference_id)
    job_log.info("[%s] Background task queued", req_id)

    return CreateJobResponse(reference_id=reference_id, status="pending")

@app.get("/api/v1/face-swap/jobs/{reference_id}", response_model=JobStatusResponse)
def get_job(reference_id: str):
    job = store.get(reference_id)
    if not job:
        raise HTTPException(status_code=404, detail="Invalid reference_id")

    if job["status"] == "completed":
        return JobStatusResponse(
            reference_id=reference_id,
            status="completed",
            result_image_url=_result_url(reference_id),
            processing_ms=job.get("processing_ms"),
        )

    if job["status"] == "failed":
        return JobStatusResponse(
            reference_id=reference_id,
            status="failed",
            error=job.get("error") or "Unknown error",
            processing_ms=job.get("processing_ms"),
        )

    return JobStatusResponse(reference_id=reference_id, status=job["status"])
