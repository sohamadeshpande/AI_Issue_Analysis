"""
Find out which Hugging Face AI-image detector works on YOUR images, and save
per-model thresholds to ai_calibration.json (the app reads this file automatically).

Folder layout:
    test_data/real/   <- genuine photos (ideally real phone photos of roads)
    test_data/ai/     <- AI-generated images

Run:
    python calibrate_ai_detector.py --real test_data/real --ai test_data/ai
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline
from ai_detect import CANDIDATES, CALIB_PATH, score_image

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def auc(pos, neg):  # P(score_ai > score_real)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def files(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(EXT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--ai", required=True)
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    a = ap.parse_args()

    real_files, ai_files = files(a.real), files(a.ai)
    print(f"{len(real_files)} real images, {len(ai_files)} AI images\n")
    results = {}

    for name in a.models:
        try:
            det = pipeline("image-classification", model=name)
        except Exception as e:
            print(f"[skip] {name}: {e}\n")
            continue
        print(f"=== {name} ===\nlabels: {list(det.model.config.id2label.values())}")

        def run(path):
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            return score_image(img, det)

        r = [run(f) for f in real_files]
        i = [run(f) for f in ai_files]
        if None in r or None in i:
            print("labels not recognised -> add them to AI_TOKENS / REAL_TOKENS in ai_detect.py\n")
            continue

        r_, i_ = np.array(r), np.array(i)
        best_t, best_j = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            j = np.mean(i_ >= t) - np.mean(r_ >= t)      # TPR - FPR
            if j > best_j:
                best_t, best_j = float(t), float(j)
        acc = float((np.mean(i_ >= best_t) + np.mean(r_ < best_t)) / 2)
        a_ = auc(i, r)
        print(f"mean AI-score  real={r_.mean():.3f}   ai={i_.mean():.3f}")
        print(f"AUC={a_:.3f}  best_threshold={best_t:.2f}  balanced_acc={acc:.3f}\n")
        results[name] = dict(auc=round(a_, 4), threshold=round(best_t, 3), balanced_acc=round(acc, 4))

    print("=" * 60 + "\nRANKING (AUC 0.5 = useless, 1.0 = perfect)")
    for n, v in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        print(f"{v['auc']:.3f}  thr={v['threshold']:.2f}  acc={v['balanced_acc']:.3f}  {n}")

    good = {n: v for n, v in results.items() if v["auc"] >= 0.7}
    if good:
        with open(CALIB_PATH, "w") as f:
            json.dump(good, f, indent=2)
        print(f"\nSaved {len(good)} usable model(s) to {CALIB_PATH}. The app will now use them with their own thresholds.")
    else:
        print("\nNo model reached AUC >= 0.7 on your data; nothing saved. Add more test images or try other models.")


if __name__ == "__main__":
    main()