"""
Civic Issue Verifier - FastAPI service (port 5001)

    python main.py                                   # or:
    uvicorn main:app --host 0.0.0.0 --port 5001

Interactive docs:  http://localhost:5001/docs

Environment variables (all optional)
    PORT=5001  HOST=0.0.0.0
    CORS_ORIGINS=http://localhost:3000,http://localhost:5173   (comma separated, or *)
    API_KEY=...            if set, every /api/v1 request must send the header  X-API-Key: ...
    MAX_UPLOAD_MB=100
    OLLAMA_URL=http://localhost:11434   OLLAMA_MODEL=llama3
    PRELOAD_MODELS=1       load all models at startup (0 = load on first request)
"""
import json
import logging
import os
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import image as core

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("civic.api")

PORT = int(os.getenv("PORT", "5001"))
HOST = os.getenv("HOST", "0.0.0.0")
API_KEY = os.getenv("API_KEY", "")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:4200").split(",") if o.strip()]
PRELOAD = os.getenv("PRELOAD_MODELS", "1") == "1"


# ============================================================
# RESPONSE SCHEMAS
# ============================================================
class Detector(BaseModel):
    model: str
    score_pct: float
    threshold_pct: float
    calibrated: bool
    flagged: bool

    model_config = {"protected_namespaces": ()}


class Authenticity(BaseModel):
    is_ai: bool
    detectors_flagging: str
    top_score_pct: float
    note: str = ""
    detectors: List[Detector] = []


class IssueFinding(BaseModel):
    key: str
    issue: str
    department: str
    confirmed: bool
    verification_status: Optional[str] = None
    confidence_pct: Optional[float] = None
    detected_by: List[str] = []
    instances_boxed: Optional[int] = None
    photo_area_covered_pct: Optional[float] = None
    seen_in_frames: Optional[str] = None
    field_verification_needed: bool = False
    reliability: str = "ok"
    note: Optional[str] = None


class Box(BaseModel):
    issue_key: str
    issue: str
    bbox: List[int]               # [x1, y1, x2, y2] in the annotated image's pixel coordinates
    confidence: float
    source: str
    reported: bool


class AnnotatedImage(BaseModel):
    image: str                    # data:image/jpeg;base64,...
    width: int
    height: int
    boxes: List[Box] = []


class Report(BaseModel):
    generated: bool
    model: str
    markdown: Optional[str] = None
    error: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class AnalyzeResponse(BaseModel):
    status: Literal["ai_generated", "no_issues_detected", "issues_detected"]
    message: str
    media_type: Literal["image", "video"]
    reported_issue_keys: List[str] = []
    authenticity: Authenticity
    reported_issues: List[IssueFinding] = []
    additional_issues: List[IssueFinding] = []
    unverified_claims: List[str] = []
    annotated_images: List[AnnotatedImage] = []
    report: Optional[Report] = None
    payload: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = {}


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


# ============================================================
# APP
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        log.info("Loading models (first run downloads them - this can take a while)...")
        await run_in_threadpool(core.warmup)
        log.info("Models ready: %s", core.loaded_models())
    yield


app = FastAPI(title="Civic Issue Verifier API", version="1.0.0", lifespan=lifespan,
              description="AI-generated-media check, civic issue detection with bounding boxes, "
                          "and an LLM-written technical report for municipal admins.")

app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=CORS_ORIGINS != ["*"])


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


# The models run on CPU and are not meant to be used concurrently: requests queue up here.
_PIPELINE_LOCK = threading.Lock()


def _run_serialized(**kwargs):
    with _PIPELINE_LOCK:
        return core.run_pipeline(**kwargs)


async def _read_upload(file: UploadFile) -> bytes:
    name = (file.filename or "").lower()
    if not name.endswith(core.ALLOWED_EXT):
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type. Allowed: {', '.join(core.ALLOWED_EXT)}")
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return bytes(data)


def _parse_reported(raw: str) -> List[str]:
    """Accepts a JSON array or a comma-separated string of issue keys or labels."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except ValueError:
            raise HTTPException(status_code=422, detail="reported_issues is not valid JSON.")
    else:
        items = [x.strip() for x in raw.split(",")]
    resolved, unknown = core.resolve_issue_names(items)
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f"Unknown issue type(s): {unknown}. Valid keys: {list(core.ISSUES)}")
    return resolved


# ============================================================
# ROUTES
# ============================================================
@app.get("/health", tags=["system"])
def health():
    """Service status, loaded models and Ollama reachability (no auth needed)."""
    models = core.list_ollama_models()
    return dict(status="ok", port=PORT, models=core.loaded_models(),
                ollama=dict(url=core.OLLAMA_URL, reachable=models is not None,
                            installed_models=models or [], default_model=core.OLLAMA_MODEL))


@app.get("/api/v1/issue-types", tags=["meta"], dependencies=[Depends(require_key)])
def issue_types():
    """Issue types the citizen can report (use `key` in reported_issues). For your dropdown."""
    return core.issue_catalog()


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"],
          dependencies=[Depends(require_key)])
async def analyze(
    file: UploadFile = File(..., description="Citizen's photo or video"),
    reported_issues: str = Form("", description='Issue keys, JSON array or comma-separated, e.g. ["pothole","garbage"]'),
    description: str = Form("", description="Citizen's description"),
    location: str = Form("", description="Location text"),
    ai_rule: Literal["any", "majority", "all"] = Form("any", description="When to flag media as AI-generated"),
    ai_threshold: float = Form(0.5, ge=0.1, le=0.95, description="Threshold for uncalibrated AI detectors"),
    detection_confidence: float = Form(0.15, ge=0.05, le=0.6, description="YOLO confidence"),
    llm_model: Optional[str] = Form(None, description="Ollama model override"),
    generate_report: bool = Form(True, description="false = detections only (much faster)"),
    include_images: bool = Form(True, description="Include annotated images (base64)"),
):
    """
    Synchronous analysis. Can take 30-120 s on CPU (the LLM report is the slowest part), so for a web UI
    prefer POST /api/v1/jobs + polling, or call this with generate_report=false.
    """
    data = await _read_upload(file)
    keys = _parse_reported(reported_issues)
    try:
        return await run_in_threadpool(
            _run_serialized, data=data, filename=file.filename, reported_selected=keys,
            description=description, location=location, ai_rule=ai_rule, ai_threshold=ai_threshold,
            detection_conf=detection_confidence, llm_model=llm_model or None,
            generate_report=generate_report, include_images=include_images)
    except core.MediaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed. See the server log.")


# ---------- async jobs (recommended for browsers: no long-held HTTP request) ----------
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
MAX_JOBS_KEPT = 100


def _run_job(job_id: str, kwargs: dict):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        result = _run_serialized(**kwargs)
        with _jobs_lock:
            _jobs[job_id].update(status="done", result=result)
    except core.MediaError as e:
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(e))
    except Exception:
        log.exception("job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error="Analysis failed. See the server log.")


@app.post("/api/v1/jobs", response_model=JobCreated, status_code=202, tags=["analysis"],
          dependencies=[Depends(require_key)])
async def create_job(
    file: UploadFile = File(...),
    reported_issues: str = Form(""),
    description: str = Form(""),
    location: str = Form(""),
    ai_rule: Literal["any", "majority", "all"] = Form("any"),
    ai_threshold: float = Form(0.5, ge=0.1, le=0.95),
    detection_confidence: float = Form(0.15, ge=0.05, le=0.6),
    llm_model: Optional[str] = Form(None),
    generate_report: bool = Form(True),
    include_images: bool = Form(True),
):
    """Same inputs as /analyze. Returns a job_id immediately; poll GET /api/v1/jobs/{job_id}."""
    data = await _read_upload(file)
    keys = _parse_reported(reported_issues)
    job_id = uuid.uuid4().hex
    kwargs = dict(data=data, filename=file.filename, reported_selected=keys, description=description,
                  location=location, ai_rule=ai_rule, ai_threshold=ai_threshold,
                  detection_conf=detection_confidence, llm_model=llm_model or None,
                  generate_report=generate_report, include_images=include_images)
    with _jobs_lock:
        _jobs[job_id] = dict(status="queued", result=None, error=None, created=time.time())
        for old in sorted(_jobs, key=lambda j: _jobs[j]["created"])[:-MAX_JOBS_KEPT]:
            _jobs.pop(old, None)
    _executor.submit(_run_job, job_id, kwargs)
    return JobCreated(job_id=job_id, status="queued")


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus, tags=["analysis"],
         dependencies=[Depends(require_key)])
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id (jobs are kept in memory only).")
        return JobStatus(job_id=job_id, status=job["status"], result=job["result"], error=job["error"])


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)