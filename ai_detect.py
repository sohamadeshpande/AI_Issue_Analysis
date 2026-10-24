"""Shared AI-generated-image detection helpers (used by the app AND the calibration script)."""
import os
import re
import numpy as np

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_calibration.json")

# Models to try when no calibration file exists yet
DEFAULT_MODELS = [
    "Ateeqq/ai-vs-human-image-detector",             # labels: ai / hum
    "capcheck/ai-human-generated-image-detection",   # labels: AI-generated / human
    "Modotte/AIRealNet",                             # labels: artificial / real
]
# Models the calibration script will evaluate
CANDIDATES = DEFAULT_MODELS + [
    "dima806/ai_vs_real_image_detection",
    "prithivMLmods/Deepfake-Detect-Siglip2",
    "Organika/sdxl-detector",
]

AI_TOKENS = {"fake", "deepfake", "ai", "artificial", "synthetic", "generated"}
REAL_TOKENS = {"real", "human", "hum", "authentic", "natural", "genuine"}


def fake_prob(results):
    """Probability the image is AI. None if the model's labels are not understood."""
    for r in results:
        toks = set(re.findall(r"[a-z]+", r["label"].lower()))
        if toks & AI_TOKENS:
            return r["score"]
        if toks & REAL_TOKENS:
            return 1.0 - r["score"]
    return None


def patches(pil):
    w, h = pil.size
    out = [pil]
    if w >= 400 and h >= 400:
        cw, ch = int(w * .5), int(h * .5)
        out += [pil.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)),
                pil.crop((0, 0, cw, ch)),
                pil.crop((w - cw, h - ch, w, h))]
    return out


def score_image(pil, detector):
    """AI score in [0,1] for one image: 50% whole image + 50% mean of crops."""
    probs = [fake_prob(detector(p, top_k=None)) for p in patches(pil)]
    probs = [p for p in probs if p is not None]
    if not probs:
        return None
    if len(probs) == 1:
        return probs[0]
    return 0.5 * probs[0] + 0.5 * float(np.mean(probs[1:]))