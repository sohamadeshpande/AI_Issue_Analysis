"""
civic_core - the analysis pipeline (no web framework, no UI).

    1. AI-generated check (per-model calibrated thresholds)  -> stop if AI
    2. Civic issue detection with bounding boxes (trained YOLO models)
    3. Structured payload -> local Ollama LLM -> technical report

main.py exposes this through FastAPI.
"""
import base64
import io
import json
import logging
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import pipeline
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

from ai_detect import DEFAULT_MODELS, CALIB_PATH, score_image

log = logging.getLogger("civic")

# ============================================================
# CONFIG (override with environment variables)
# ============================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DEVICE = os.getenv("DEVICE", "cpu")

MAX_VIDEO_FRAMES = 8
DETECT_FRAMES = 6
MAX_SIDE = 1280
MAX_BOXES_PER_ISSUE = 5
YOLO_IMGSZ = 1024
YOLO_IOU = 0.45
EXTRA_SINGLE_CONF = 0.50

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
ALLOWED_EXT = IMAGE_EXT + VIDEO_EXT

YOLO_SOURCES = [
    dict(name="PotholeNet", repo="Vansh180/PotholeNet-V1", min_conf=0.0,
         classes={"pothole": "pothole", "road_damage": "road_crack", "garbage": "garbage"}),
    dict(name="GarbagePothole", repo="utkarsh-23/yolov8m-garbage-pothole-detector", min_conf=0.25,
         classes={"garbage": "garbage", "container": None, "crocodile crack": "road_crack",
                  "longitudinal crack": "road_crack", "pothole": "pothole", "streetlight": "streetlight",
                  "traffic-light": "streetlight", "hv-switch": None, "crossarm": None, "transformer": None}),
    dict(name="DuoQueue", repo="tejas-varshney/duoqueue-civic-hazard-detector", min_conf=0.30,
         classes={"pothole": "pothole", "waterlogged_road": "waterlogging", "drain_overflow": "sewage"}),
    dict(name="COCO", repo=None, weights="yolov8n.pt", min_conf=0.40, strict=True,
         classes={"cow": "stray_animals", "dog": "stray_animals", "horse": "stray_animals",
                  "sheep": "stray_animals"}),
]

CLUSTER = {"garbage", "construction_debris", "waterlogging", "sewage"}
REPORTED_ONLY = {"streetlight"}                     # models detect lights, not whether they are broken
STRICT_EXTRA = {"pothole", "road_crack", "waterlogging", "sewage"}

ISSUES = {
    "pothole": dict(label="Pothole", dept="PWD / Roads Dept", keywords=["pothole", "gaddha", "gadda"]),
    "road_crack": dict(label="Road cracks / damaged surface", dept="PWD / Roads Dept",
                       keywords=["crack", "damaged road", "broken road"]),
    "garbage": dict(label="Garbage / waste dumping", dept="Solid Waste Management",
                    keywords=["garbage", "trash", "waste", "kachra", "litter", "dump"]),
    "construction_debris": dict(label="Construction debris / rubble", dept="Building / Solid Waste Dept",
                                keywords=["debris", "rubble", "malba", "bricks"]),
    "waterlogging": dict(label="Waterlogging / flooding", dept="Drainage Dept",
                         keywords=["waterlog", "flood", "puddle", "water"]),
    "open_drain": dict(label="Open manhole / uncovered drain", dept="Drainage / Sewerage Dept",
                       keywords=["manhole", "drain", "gutter", "nala"]),
    "sewage": dict(label="Sewage overflow", dept="Drainage / Sewerage Dept",
                   keywords=["sewage", "sewer", "overflow"]),
    "streetlight": dict(label="Broken streetlight / pole / hanging wires", dept="Electrical Dept",
                        keywords=["streetlight", "street light", "lamp", "pole", "wire"]),
    "fallen_tree": dict(label="Fallen tree / branches", dept="Garden / Tree Authority",
                        keywords=["tree", "branch"]),
    "stray_animals": dict(label="Stray animals on road", dept="Animal Control",
                          keywords=["stray", "cattle", "cow", "dog", "animal"]),
    "broken_footpath": dict(label="Broken footpath / kerb", dept="PWD / Roads Dept",
                            keywords=["footpath", "pavement", "sidewalk", "kerb"]),
    "damaged_sign": dict(label="Damaged road sign / divider", dept="Traffic / PWD Dept",
                         keywords=["sign", "divider", "signal", "barricade"]),
    "illegal_hoarding": dict(label="Illegal hoarding / banner", dept="Licensing / Advertisement Dept",
                             keywords=["hoarding", "banner", "poster"]),
    "encroachment": dict(label="Road / footpath encroachment", dept="Anti-Encroachment Dept",
                         keywords=["encroach", "vendor", "stall", "cart"]),
}
KEYS = list(ISSUES)


class MediaError(ValueError):
    """The uploaded file could not be decoded."""


def info(k):
    if k in ISSUES:
        return ISSUES[k]
    return dict(label=k.replace("yolo:", "").replace("_", " ").title(), dept="Municipal Corporation (general)")


def issue_catalog():
    return [dict(key=k, label=v["label"], department=v["dept"]) for k, v in ISSUES.items()]


def resolve_issue_names(items):
    """Accept issue keys OR labels (case-insensitive). Returns (resolved_keys, unknown_items)."""
    by_label = {v["label"].lower(): k for k, v in ISSUES.items()}
    resolved, unknown = [], []
    for item in items:
        s = str(item).strip()
        key = s.lower() if s.lower() in ISSUES else by_label.get(s.lower())
        if key:
            if key not in resolved:
                resolved.append(key)
        elif s:
            unknown.append(s)
    return resolved, unknown


def reported_from_inputs(selected_keys, text):
    keys = set(selected_keys)
    text = (text or "").lower()
    keys |= {k for k, v in ISSUES.items() if any(w in text for w in v["keywords"])}
    return [k for k in KEYS if k in keys]


def issue_color(key):
    i = KEYS.index(key) if key in KEYS else len(KEYS) + sum(map(ord, key)) % 5
    h = int(170 * i / (len(KEYS) + 5))
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


# ============================================================
# MODEL LOADING (loaded once, reused by every request)
# ============================================================
_load_lock = threading.Lock()
_ai_cache = {}
_yolo_cache = None


def load_calibration():
    try:
        with open(CALIB_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def ai_model_names(calib):
    return tuple(sorted(calib, key=lambda n: -calib[n]["auc"])[:3]) if calib else tuple(DEFAULT_MODELS)


def get_ai_detectors(names):
    with _load_lock:
        for n in names:
            if n not in _ai_cache:
                try:
                    log.info("Loading AI detector %s", n)
                    _ai_cache[n] = pipeline("image-classification", model=n)
                except Exception as e:
                    log.warning("Could not load AI detector %s: %s", n, e)
        return {n: _ai_cache[n] for n in names if n in _ai_cache}


def get_yolo_models():
    global _yolo_cache
    with _load_lock:
        if _yolo_cache is not None:
            return _yolo_cache
        loaded = []
        for cfg in YOLO_SOURCES:
            try:
                if cfg.get("repo"):
                    path = None
                    for repo_type in ("model", "space"):       # some authors publish weights in a Space
                        try:
                            pts = [f for f in list_repo_files(cfg["repo"], repo_type=repo_type)
                                   if f.endswith(".pt")]
                            if pts:
                                fname = "best.pt" if "best.pt" in pts else pts[0]
                                path = hf_hub_download(repo_id=cfg["repo"], filename=fname, repo_type=repo_type)
                                break
                        except Exception:
                            continue
                    if not path:
                        raise FileNotFoundError("No .pt weights found")
                else:
                    path = cfg["weights"]
                model = YOLO(path)
                log.info("YOLO %s classes: %s", cfg["name"], list(model.names.values()))
                loaded.append((model, cfg))
            except Exception as e:
                log.warning("Detector %s unavailable: %s", cfg["name"], e)
        _yolo_cache = loaded
        return loaded


def warmup():
    """Load every model once (called at server start)."""
    get_ai_detectors(ai_model_names(load_calibration()))
    get_yolo_models()


def loaded_models():
    return dict(ai_detectors=list(_ai_cache),
                yolo_detectors=[cfg["name"] for _, cfg in (_yolo_cache or [])],
                ai_calibrated=bool(load_calibration()))


# ============================================================
# STEP 1: AI-GENERATED CHECK
# ============================================================
def check_ai(frames_bgr, detectors, calib, fallback_thr, rule):
    """rule: 'any' | 'majority' | 'all'"""
    if not detectors:
        return dict(is_ai=False, top_score=0.0, votes="0/0", models=[], note="No AI detector could be loaded")

    per = {n: [] for n in detectors}
    for frame in frames_bgr:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for name, detector in detectors.items():
            score = score_image(pil, detector)
            if score is not None:
                per[name].append(score)

    rows = []
    for name, values in per.items():
        if not values:
            continue
        score = float(np.mean(values))
        thr = calib[name]["threshold"] if name in calib else fallback_thr
        rows.append(dict(model=name, score_pct=round(score * 100, 1), threshold_pct=round(thr * 100, 1),
                         calibrated=name in calib, flagged=bool(score >= thr)))

    if not rows:
        return dict(is_ai=False, top_score=0.0, votes="0/0", models=[], note="No detector produced a score")

    votes = sum(r["flagged"] for r in rows)
    if rule == "any":
        is_ai = votes >= 1
    elif rule == "majority":
        is_ai = votes > len(rows) / 2
    else:
        is_ai = votes == len(rows)

    note = "" if calib else "AI detectors are NOT calibrated yet. Run calibrate_ai_detector.py on your test images."
    return dict(is_ai=bool(is_ai), top_score=max(r["score_pct"] for r in rows),
                votes=f"{votes}/{len(rows)}", models=rows, note=note)


# ============================================================
# STEP 2: CIVIC ISSUE DETECTION
# ============================================================
YOLO_KEYWORDS = {
    "pothole": ["pothole", "d40"], "road_crack": ["crack", "d00", "d10", "d20"],
    "garbage": ["garbage", "waste", "trash", "litter", "dump"], "open_drain": ["manhole", "drain"],
    "waterlogging": ["water", "flood"], "sewage": ["sewage", "sewer"],
    "stray_animals": ["cow", "cattle", "dog", "animal", "stray"], "streetlight": ["streetlight", "pole", "lamp"],
    "fallen_tree": ["tree"], "broken_footpath": ["footpath", "pavement"],
    "construction_debris": ["debris", "rubble"],
}


def yolo_key(name):
    n = name.lower().replace(" ", "_")
    for key, words in YOLO_KEYWORDS.items():
        if any(w in n for w in words):
            return key
    return "yolo:" + n


def map_class(cfg, name):
    n = name.lower().strip()
    for cand in (n, n.replace("_", " "), n.replace(" ", "_"), n.replace("-", " ")):
        if cand in cfg["classes"]:
            return cfg["classes"][cand]
    return None if cfg.get("strict") else yolo_key(name)


def run_yolo(frame, loaded, conf):
    out = []
    for model, cfg in loaded:
        for result in model(frame, device=DEVICE, conf=conf, imgsz=YOLO_IMGSZ, iou=YOLO_IOU, verbose=False):
            for box in result.boxes:
                key = map_class(cfg, model.names[int(box.cls[0])])
                if key:
                    out.append(dict(key=key, src=cfg["name"], min_conf=cfg["min_conf"],
                                    confidence=float(box.conf[0]), bbox=list(map(int, box.xyxy[0]))))
    return out


def nms_boxes(boxes, iou=0.5):
    if len(boxes) < 2:
        return boxes
    keep = nms(torch.tensor([b["bbox"] for b in boxes], dtype=torch.float32),
               torch.tensor([b["confidence"] for b in boxes]), iou).tolist()
    return [boxes[i] for i in keep]


def merge_boxes(boxes, grow=0.5):
    boxes = [dict(b) for b in boxes]

    def grown(bb):
        m = grow * max(bb[2] - bb[0], bb[3] - bb[1])
        return bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m

    changed = True
    while changed:
        changed, out = False, []
        for box in boxes:
            for other in out:
                ax1, ay1, ax2, ay2 = grown(box["bbox"])
                bx1, by1, bx2, by2 = grown(other["bbox"])
                if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
                    other["bbox"] = [min(other["bbox"][0], box["bbox"][0]), min(other["bbox"][1], box["bbox"][1]),
                                     max(other["bbox"][2], box["bbox"][2]), max(other["bbox"][3], box["bbox"][3])]
                    other["confidence"] = max(other["confidence"], box["confidence"])
                    if box["src"] not in other["src"].split("+"):
                        other["src"] += "+" + box["src"]
                    changed = True
                    break
            else:
                out.append(box)
        boxes = out
    return boxes


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def _corroborated(box, same_key_boxes):
    mine = set(box["src"].split("+"))
    return any(mine.isdisjoint(o["src"].split("+")) and o["confidence"] >= 0.25
               and _iou(box["bbox"], o["bbox"]) >= 0.3 for o in same_key_boxes)


def analyze_frame(frame, reported, yolo_models, yolo_thr):
    raw = run_yolo(frame, yolo_models, yolo_thr * 0.7)
    fh, fw = frame.shape[:2]
    all_keys = KEYS + sorted({b["key"] for b in raw if b["key"] not in ISSUES})

    cands = {}
    for key in all_keys:
        is_reported = key in reported
        if key in REPORTED_ONLY and not is_reported:
            continue
        same = [b for b in raw if b["key"] == key]
        cand = [b for b in same if b["confidence"] >= max(yolo_thr, b["min_conf"])]
        if key in STRICT_EXTRA:
            cand = [b for b in cand if b["confidence"] >= EXTRA_SINGLE_CONF or _corroborated(b, same)]
        cand = merge_boxes(cand) if key in CLUSTER else nms_boxes(cand)
        cands[key] = sorted(cand, key=lambda b: -b["confidence"])[:MAX_BOXES_PER_ISSUE]

    # an unreported pothole/crack inside a garbage pile is almost always the pile itself
    garbage = [b["bbox"] for b in cands.get("garbage", [])]
    for key in ("pothole", "road_crack"):
        if key in cands and key not in reported:
            cands[key] = [b for b in cands[key] if not any(_inside(b["bbox"], g) >= 0.5 for g in garbage)]

    found, kept = {}, []
    for key, cand in cands.items():
        if not cand:
            continue                                    # a citizen's claim alone never creates a finding
        area = sum(max(0, b["bbox"][2] - b["bbox"][0]) * max(0, b["bbox"][3] - b["bbox"][1]) for b in cand)
        found[key] = dict(confidence=max(b["confidence"] for b in cand), count=len(cand),
                          sources=sorted({s for b in cand for s in b["src"].split("+")}),
                          area_pct=round(min(100.0, 100.0 * area / (fh * fw)), 1))
        kept += [dict(b, reported=key in reported) for b in cand]
    return found, kept


def draw_boxes(frame, boxes):
    out = frame.copy()
    t = max(2, out.shape[1] // 300)
    fs = max(0.55, out.shape[1] / 1500)
    for b in boxes:
        color = issue_color(b["key"])
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t + (2 if b["reported"] else 0))
        label = f"{b['key'].replace('yolo:', '').upper()} {b['confidence']:.2f} [{b['src']}]"
        if not b["reported"]:
            label += " (extra)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(y1 - 8, th + 8)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2)
    return out


def analyze_media(frames, reported, yolo_models, yolo_thr):
    """Returns (aggregated findings, gallery [(annotated_bgr, boxes)], frames_used)."""
    if not frames:
        return {}, [], 0
    idx = np.unique(np.linspace(0, len(frames) - 1, min(DETECT_FRAMES, len(frames)), dtype=int))
    agg, scored = {}, []
    for i in idx:
        frame = frames[int(i)]
        found, boxes = analyze_frame(frame, reported, yolo_models, yolo_thr)
        for key, v in found.items():
            a = agg.setdefault(key, dict(confidence=0.0, count=0, frames_hit=0, sources=set(), area_pct=0.0))
            a["confidence"] = max(a["confidence"], v["confidence"])
            a["count"] = max(a["count"], v["count"])
            a["area_pct"] = max(a["area_pct"], v["area_pct"])
            a["frames_hit"] += 1
            a["sources"].update(v["sources"])
        scored.append((sum(v["confidence"] for v in found.values()), frame, boxes))
    scored.sort(key=lambda x: x[0], reverse=True)
    gallery = [(draw_boxes(f, b), b) for s, f, b in scored[:3] if s > 0]
    for a in agg.values():
        a["sources"] = sorted(a["sources"])
    return agg, gallery, len(idx)


# ============================================================
# STEP 3: STRUCTURED PAYLOAD + OLLAMA (the LLM writes the advice)
# ============================================================
def issue_row(key, f, n_frames):
    conf = round(f["confidence"] * 100, 1)
    return dict(
        key=key, issue=info(key)["label"], department=info(key)["dept"], confirmed=True,
        confidence_pct=conf, detected_by=f["sources"], instances_boxed=f["count"],
        photo_area_covered_pct=f["area_pct"], seen_in_frames=f"{f['frames_hit']}/{n_frames}",
        field_verification_needed=bool(conf < 40 or f["count"] == 0),
        reliability=("LOW - report only as a POSSIBLE issue that needs on-site verification" if conf < 50 else "ok"),
    )


def unverified_row(key):
    return dict(
        key=key, issue=info(key)["label"], department=info(key)["dept"], confirmed=False,
        verification_status="not_verified", field_verification_needed=True, reliability="UNVERIFIED",
        note="The citizen reported this issue, but no qualifying bounding box was found.",
    )


def build_payload(meta, ai, findings, reported, n_frames):
    reported_rows = [issue_row(k, findings[k], n_frames) | {"verification_status": "detected_with_box"}
                     if k in findings else unverified_row(k) for k in reported]
    additional_rows = [issue_row(k, f, n_frames) | {"verification_status": "background_detection"}
                       for k, f in findings.items() if k not in reported]
    return dict(
        complaint=dict(media_type=meta["media_type"], location=meta["location"] or "not provided",
                       citizen_description=meta["description"] or "not provided"),
        authenticity=dict(ai_generated=False, detectors_flagging=ai["votes"]),
        citizen_reported_issues=reported_rows,
        additional_issues_in_background=additional_rows,
    )


def build_prompt(payload):
    return (
        "You are a senior civil / municipal engineer advising the admin of an Indian urban local body "
        "(municipal corporation).\n\n"
        "The submitted media passed an AI-generated-media screening. That does not prove every citizen "
        "claim is true.\n\n"
        f"DETECTIONS (authoritative for what was found and what is unverified):\n{json.dumps(payload, indent=2)}\n\n"
        "YOUR TASK\n"
        "Write a technical action report for the municipal admin, using your own engineering knowledge of "
        "standard Indian municipal and road-maintenance practice.\n\n"
        "QUALITY RULES\n"
        "- Be concrete and technical: name the repair method, materials, equipment, order of work and the "
        "responsible department. Never write vague advice such as 'fix the pothole' or 'clean the area'.\n"
        "- Tailor the advice to the evidence: instances_boxed, photo_area_covered_pct, confidence_pct, the "
        "citizen's description and the location.\n"
        "- Include root-cause checks (why did this happen?) and prevention, not only the repair.\n"
        "- Mention coordination with other departments only where it is genuinely needed.\n\n"
        "ACCURACY RULES\n"
        "- Never invent dimensions, depths, costs, quantities, crew sizes or deadlines. "
        "photo_area_covered_pct is the share of the PHOTO, not a real-world size. "
        "Say 'measure on site' when a measurement is needed.\n"
        "- Mention a code, standard or Act (IRC, BIS, SWM Rules, etc.) only if you are certain it exists and "
        "applies. Otherwise describe the practice without a code number. Add '(verify)' after any code you cite.\n"
        "- Never mention an issue that is not in DETECTIONS.\n"
        "- Severity must reflect public-safety risk and extent. Low = nuisance, no direct safety risk. "
        "Medium = inconvenience or moderate risk (vehicle damage, slipping, hygiene) that can be scheduled. "
        "High = significant safety or health risk that needs urgent attention. "
        "Critical = immediate danger to life (for example an open manhole on a live road or exposed live wires). "
        "Only use High or Critical when the detection is reliable.\n"
        "- If reliability starts with LOW, describe the finding as possible/unverified, give severity at most "
        "Medium, and make on-site verification the first action.\n"
        "- If field_verification_needed is true, make verification the first action.\n\n"
        "UNVERIFIED CLAIMS\n"
        "- A citizen-reported issue with confirmed=false is an unverified claim, NOT a detected hazard.\n"
        "- For every such issue write exactly: 'No <Issue Name> detected in the image.' and recommend "
        "on-site verification only.\n"
        "- Do not give repair suggestions, severity, immediate actions or preventive measures for an issue "
        "with confirmed=false. Put them only in the '### Unverified citizen claims' section.\n"
        "- Issues in additional_issues_in_background were not reported by the citizen. Identify them "
        "explicitly as background findings.\n\n"
        "WRITE THE REPORT IN MARKDOWN WITH THIS STRUCTURE\n"
        "### Summary\nSummarize the verified findings and any unverified claims in 2-3 sentences.\n\n"
        "### Priority order\nRank the confirmed detections by urgency, each with its severity and a one-line "
        "reason. Keep unverified claims out of this ranking.\n\n"
        "Then, for each confirmed issue (citizen-reported first, then background):\n"
        "#### <Issue> - Severity: <Low|Medium|High|Critical>\n"
        "- **Risk if ignored:** ...\n"
        "- **Immediate action (0-24 h):** ...\n"
        "- **Technical repair method:** numbered steps\n"
        "- **Root-cause checks and prevention:** ...\n"
        "- **Coordinate with:** ...\n\n"
        "### Unverified citizen claims\nList each claim with confirmed=false (write 'None' if there are none).\n\n"
        "### Admin notes\nState the limits of automated detection, which findings need on-site verification, and "
        "that these recommendations are AI-generated and must be checked by a qualified engineer before work "
        "orders are issued.\n"
    )


def list_ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None           # None = Ollama not reachable


def call_ollama(model, prompt):
    try:
        # num_ctx: Ollama's default context (2048 tokens) would silently cut a long prompt
        r = requests.post(f"{OLLAMA_URL}/api/generate", timeout=400,
                          json=dict(model=model, prompt=prompt, stream=False,
                                    options=dict(temperature=0.2, num_ctx=8192, num_predict=2500)))
        if r.status_code == 200:
            return r.json().get("response", "").strip(), None
        return None, f"Ollama returned {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach Ollama at {OLLAMA_URL}. Start Ollama and run `ollama pull {model}`."
    except Exception as exc:
        return None, str(exc)


# ============================================================
# MEDIA LOADING
# ============================================================
def _resize(frame):
    h, w = frame.shape[:2]
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame


def load_frames(data, filename):
    """Returns (frames for detection, media_type, frames for the AI check)."""
    name = filename.lower()
    if name.endswith(VIDEO_EXT):
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1]) as tmp:
                tmp.write(data)
                path = tmp.name
            cap = cv2.VideoCapture(path)
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                if total > 0:
                    for i in np.unique(np.linspace(0, total - 1, MAX_VIDEO_FRAMES, dtype=int)):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                        ok, fr = cap.read()
                        if ok:
                            frames.append(_resize(fr))
                if not frames:
                    raise MediaError("The video could not be decoded.")
                return frames, "video", frames
            finally:
                cap.release()
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise MediaError("The image could not be decoded.")
    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return [_resize(orig)], "image", [orig]          # the AI check uses the ORIGINAL pixels


def to_data_uri(bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ============================================================
# FULL PIPELINE (what the API calls)
# ============================================================
def run_pipeline(data, filename, reported_selected=(), description="", location="", ai_rule="any",
                 ai_threshold=0.5, detection_conf=0.15, llm_model=None, generate_report=True,
                 include_images=True):
    t_start = time.time()
    timings = {}
    frames, media_type, ai_frames = load_frames(data, filename)
    reported = reported_from_inputs(reported_selected, description)

    result = dict(status="", message="", media_type=media_type, reported_issue_keys=reported,
                  authenticity=None, reported_issues=[], additional_issues=[], unverified_claims=[],
                  annotated_images=[], report=None, payload=None, timings=timings)

    # ---- step 1: authenticity ----
    t = time.time()
    calib = load_calibration()
    ai = check_ai(ai_frames, get_ai_detectors(ai_model_names(calib)), calib, ai_threshold, ai_rule)
    timings["authenticity_s"] = round(time.time() - t, 2)
    result["authenticity"] = dict(is_ai=ai["is_ai"], detectors_flagging=ai["votes"],
                                  top_score_pct=ai["top_score"], note=ai["note"], detectors=ai["models"])
    if ai["is_ai"]:
        result["status"] = "ai_generated"
        result["message"] = (f"This {media_type} appears to be AI-generated "
                             f"({ai['votes']} detectors flagged it). Analysis stopped. "
                             "Please review or reject this complaint.")
        timings["total_s"] = round(time.time() - t_start, 2)
        return result

    # ---- step 2: detection ----
    t = time.time()
    findings, gallery, n_used = analyze_media(frames, reported, get_yolo_models(), detection_conf)
    timings["detection_s"] = round(time.time() - t, 2)

    result["reported_issues"] = [issue_row(k, findings[k], n_used) | {"verification_status": "detected_with_box"}
                                 if k in findings else unverified_row(k) for k in reported]
    result["additional_issues"] = [issue_row(k, f, n_used) | {"verification_status": "background_detection"}
                                   for k, f in findings.items() if k not in reported]
    result["unverified_claims"] = [info(k)["label"] for k in reported if k not in findings]

    if include_images:
        for img, boxes in gallery:
            result["annotated_images"].append(dict(
                image=to_data_uri(img), width=img.shape[1], height=img.shape[0],
                boxes=[dict(issue_key=b["key"], issue=info(b["key"])["label"], bbox=b["bbox"],
                            confidence=round(b["confidence"], 3), source=b["src"], reported=b["reported"])
                       for b in boxes]))

    if not findings:
        result["status"] = "no_issues_detected"
        result["message"] = "No civic issues detected."
        if reported:
            result["message"] += (" The citizen-reported issue(s) could not be verified: "
                                  + ", ".join(result["unverified_claims"]) + ". Manual verification recommended.")
        timings["total_s"] = round(time.time() - t_start, 2)
        return result                         # nothing confirmed -> nothing is sent to the LLM

    result["status"] = "issues_detected"
    n_rep = sum(1 for r in result["reported_issues"] if r["confirmed"])
    result["message"] = (f"{len(findings)} issue(s) detected: {n_rep} citizen-reported, "
                         f"{len(result['additional_issues'])} additional in the background.")

    # ---- step 3: LLM report ----
    payload = build_payload(dict(media_type=media_type, location=location, description=description),
                            ai, findings, reported, n_used)
    result["payload"] = payload
    model = llm_model or OLLAMA_MODEL
    if generate_report:
        t = time.time()
        text, err = call_ollama(model, build_prompt(payload))
        timings["report_s"] = round(time.time() - t, 2)
        result["report"] = dict(generated=bool(text), model=model, markdown=text, error=err)
    else:
        result["report"] = dict(generated=False, model=model, markdown=None, error=None)

    timings["total_s"] = round(time.time() - t_start, 2)
    return result