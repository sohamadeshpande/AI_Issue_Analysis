import os
import cv2
import json
import requests
import numpy as np
from PIL import Image
from transformers import pipeline
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

# ==========================================
# 1. LOAD HIGH-PRECISION AI IMAGE DETECTOR
# ==========================================
AI_MODEL_NAME = "prithivMLmods/Deepfake-Detect-Siglip2"

print(f"Loading AI Classifier ('{AI_MODEL_NAME}')...")
try:
    ai_detector = pipeline("image-classification", model=AI_MODEL_NAME)
    print("SigLIP2 AI Detector loaded successfully.")
except Exception as e:
    print(f"Warning: Primary AI model failed ({e}). Falling back to dima806...")
    AI_MODEL_NAME = "dima806/ai_vs_real_image_detection"
    ai_detector = pipeline("image-classification", model=AI_MODEL_NAME)

def extract_image_patches(pil_img):
    """Extracts high-resolution crops to inspect fine-grained AI artifacts."""
    w, h = pil_img.size
    patches = [pil_img]

    if w >= 400 and h >= 400:
        crop_w, crop_h = int(w * 0.5), int(h * 0.5)
        patches.append(pil_img.crop(((w - crop_w) // 2, (h - crop_h) // 2, (w + crop_w) // 2, (h + crop_h) // 2)))
        patches.append(pil_img.crop((0, 0, crop_w, crop_h)))
        patches.append(pil_img.crop(((w - crop_w) // 2, h - crop_h, (w + crop_w) // 2, h)))

    return patches

def check_ai_generated(image_input, confidence_threshold=0.75):
    """Evaluates image authenticity using multi-crop patch analysis."""
    if ai_detector is None:
        return {"is_ai": False, "confidence": 0.0}

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return {"is_ai": False, "confidence": 0.0}
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        rgb_frame = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

    patches = extract_image_patches(pil_img)
    ai_scores = []

    for patch in patches:
        results = ai_detector(patch)
        for res in results:
            label = res['label'].lower()
            score = res['score']
            if any(kw in label for kw in ['fake', 'artificial', 'ai', 'synthetic']):
                ai_scores.append(score)
                break
            elif 'real' in label:
                ai_scores.append(1.0 - score)
                break

    global_score = ai_scores[0] if ai_scores else 0.0
    local_max_score = max(ai_scores[1:]) if len(ai_scores) > 1 else global_score

    final_score = (global_score * 0.40) + (local_max_score * 0.60)
    final_confidence = round(final_score * 100, 2)

    return {
        "is_ai": final_score >= confidence_threshold,
        "confidence": final_confidence
    }

def draw_ai_warning(frame, confidence_score):
    """Draws a red warning header over AI-generated images."""
    annotated = frame.copy()
    h, w, _ = annotated.shape
    
    header_height = max(60, int(h * 0.08))
    cv2.rectangle(annotated, (0, 0), (w, header_height), (0, 0, 220), -1)
    
    text = f"This is an AI generated image ({confidence_score:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, w / 1100.0)
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = max(10, (w - text_w) // 2)
    text_y = (header_height + text_h) // 2
    
    cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return annotated

# ==========================================
# 2. DOWNLOAD & LOAD POTHOLE DETECTOR
# ==========================================
REPO_ID = "Vansh180/PotholeNet-V1"
print(f"Checking repository: {REPO_ID}...")

try:
    repo_files = list_repo_files(repo_id=REPO_ID)
    pt_files = [f for f in repo_files if f.endswith(".pt")]
    
    if not pt_files:
        raise FileNotFoundError("No .pt weights file found in the repository.")
    
    weights_filename = pt_files[0]
    print(f"Found model file: '{weights_filename}'. Downloading...")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=weights_filename)

except Exception as e:
    print(f"Warning: Could not fetch from {REPO_ID} ({e}). Using yolov8n.pt...")
    weights_path = "yolov8n.pt"

print("Loading PotholeNet model into memory...")
model = YOLO(weights_path)

# ==========================================
# 3. LOCAL LLM (OLLAMA / LLAMA 3) INTEGRATION
# ==========================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Change to llama3.2, mistral, etc., depending on your local setup

def generate_civic_report(detection_summary, source_name="Image"):
    """
    Sends structured detection metadata to local Ollama LLM 
    to generate an automated municipal complaint & assessment report.
    """
    if not detection_summary:
        prompt = (
            f"Source: {source_name}\n"
            "Computer Vision Analysis Results: No civic issues or road hazards detected.\n"
            "Provide a brief 1-2 sentence confirmation that the roadway appears clear."
        )
    else:
        # Group counts & confidence scores
        issue_counts = {}
        confidence_list = []
        for det in detection_summary:
            cls = det['class']
            issue_counts[cls] = issue_counts.get(cls, 0) + 1
            confidence_list.append(f"{cls} ({det['confidence']*100:.1f}%)")

        issues_str = ", ".join([f"{count}x {cls.upper()}" for cls, count in issue_counts.items()])
        det_details = ", ".join(confidence_list)

        prompt = (
            f"You are an automated AI Municipal Inspector for Indian Roadways.\n"
            f"Analyzed Input Source: {source_name}\n"
            f"Computer Vision Detections: {issues_str}\n"
            f"Detection Confidence Breakdown: {det_details}\n\n"
            "Based on these automated detections, generate a concise Municipal Civic Issue Report with:\n"
            "1. Hazard Severity Level (Low / Medium / High / Critical)\n"
            "2. Brief Description of Public Safety Risks\n"
            "3. Actionable Remediation Steps for Local Authorities (e.g., PWD / Municipal Corporation)\n"
            "Keep the report brief, structured, and professional."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    print("\n" + "="*50)
    print(f"Sending vision outputs to Local LLM ({OLLAMA_MODEL})...")
    print("="*50)

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result_json = response.json()
            llm_text = result_json.get("response", "No response received.")
            print("\n--- LLM CIVIC COMPLAINT & SEVERITY REPORT ---")
            print(llm_text)
            print("-" * 50 + "\n")
            return llm_text
        else:
            print(f"[ERROR] Ollama returned status code {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to local Ollama server at http://localhost:11434.")
        print("Make sure Ollama application is running (`ollama run llama3`).")
        return None
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return None

# ==========================================
# 4. CONFIGURATION & INFERENCE PIPELINE
# ==========================================
INPUT_SOURCE = "./images"  # Path to folder, image file, video file, or 0 for webcam

WINDOW_NAME = "Civic Issue Detector (PotholeNet + Ollama LLM)"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

CONF_THRESHOLD = 0.15
INFERENCE_SIZE = 1024
IOU_THRESHOLD = 0.45

def process_frame(frame, model, conf=CONF_THRESHOLD, imgsz=INFERENCE_SIZE, iou=IOU_THRESHOLD):
    """
    Runs YOLO object detection and extracts structured metadata.
    Returns: (annotated_frame, list_of_detections)
    """
    results = model(frame, device="cpu", conf=conf, imgsz=imgsz, iou=iou, verbose=False)
    annotated_frame = frame.copy()
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            # Store detection metadata for LLM ingestion
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2]
            })
            
            box_color = (0, 255, 255) if "garbage" in class_name.lower() else (0, 0, 255)

            # 1. Draw Bounding Box Rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 4)

            # 2. Draw Text Label
            label_text = f"{class_name.upper()} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, font_thickness = 0.7, 2
            
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_y1 = max(y1 - 10, text_h + 10)
            
            cv2.rectangle(annotated_frame, (x1, text_y1 - text_h - 5), (x1 + text_w + 10, text_y1 + 5), box_color, -1)
            cv2.putText(annotated_frame, label_text, (x1 + 5, text_y1), font, font_scale, (0, 0, 0), font_thickness)

    return annotated_frame, detections

def evaluate_and_process(image_input, source_label="Image"):
    """
    1. AI Authenticity Check via SigLIP2
    2. Civic Issue Detection via PotholeNet (YOLOv8)
    3. LLM Report Generation via Ollama (Llama 3)
    """
    frame = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if frame is None:
        return None, False

    # Step 1: AI Authenticity Check
    ai_check = check_ai_generated(image_input, confidence_threshold=0.75)

    if ai_check["is_ai"]:
        print(f" -> [ALERT] AI generated image detected ({ai_check['confidence']}%). Skipping LLM civic processing.")
        output_frame = draw_ai_warning(frame, ai_check['confidence'])
        return output_frame, True
    else:
        print(f" -> Authenticity Verified (AI Score: {ai_check['confidence']}%). Running pothole detection...")
        # Step 2: Object Detection
        output_frame, detections = process_frame(frame, model)

        # Step 3: Pass detection metadata to local LLM
        generate_civic_report(detections, source_name=source_label)

        return output_frame, False

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if isinstance(INPUT_SOURCE, str) and os.path.isdir(INPUT_SOURCE):
    image_files = [f for f in os.listdir(INPUT_SOURCE) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_files:
        print(f"[ERROR] No image files found in '{INPUT_SOURCE}'")
        exit()

    output_dir = os.path.join(INPUT_SOURCE, "annotated_results")
    os.makedirs(output_dir, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    for idx, filename in enumerate(image_files, 1):
        file_path = os.path.join(INPUT_SOURCE, filename)
        print(f"\n[{idx}/{len(image_files)}] Analyzing file: {filename}")

        output_frame, is_ai = evaluate_and_process(file_path, source_label=filename)
        if output_frame is None:
            continue

        cv2.imshow(WINDOW_NAME, output_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(os.path.join(output_dir, f"annotated_{filename}"), output_frame)

    cv2.destroyAllWindows()

elif isinstance(INPUT_SOURCE, str) and INPUT_SOURCE.lower().endswith(IMAGE_EXTENSIONS):
    if not os.path.exists(INPUT_SOURCE):
        print(f"\n[ERROR] Image file '{INPUT_SOURCE}' was not found.")
        exit()

    print(f"Analyzing single image: {INPUT_SOURCE}")
    output_frame, is_ai = evaluate_and_process(INPUT_SOURCE, source_label=os.path.basename(INPUT_SOURCE))

    if output_frame is not None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
        cv2.imshow(WINDOW_NAME, output_frame)
        
        print("\nPress 's' to save or any other key to exit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            save_path = "detected_output.jpg"
            cv2.imwrite(save_path, output_frame)
            print(f"Saved output to: {os.path.abspath(save_path)}")

        cv2.destroyAllWindows()

else:
    if isinstance(INPUT_SOURCE, str) and not os.path.exists(INPUT_SOURCE):
        print(f"\n[ERROR] Video file '{INPUT_SOURCE}' was not found.")
        exit()

    cap = cv2.VideoCapture(INPUT_SOURCE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    print("Starting video analysis. Press 'q' to stop playback.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream reached.")
            break

        output_frame, is_ai = evaluate_and_process(frame, source_label="Video Stream Frame")

        if output_frame is not None:
            cv2.imshow(WINDOW_NAME, output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline closed successfully.")