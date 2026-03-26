import json
import re
import time
import io
import base64
import requests
import cv2
import numpy as np

import gradio as gr
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"
REQUEST_TIMEOUT = 160

# ── PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a quality inspector. Look at this image and find ALL defects.

You MUST respond with ONLY this JSON and nothing else. No markdown. No explanation. Just JSON:

{
  "quality_score": 60,
  "verdict": "FAIL",
  "summary": "describe main defect here",
  "defects": [
    {
      "name": "Scratch",
      "severity": "WARNING",
      "confidence": 85,
      "location": "center",
      "size_estimate": "5cm long"
    }
  ],
  "root_cause": "cause here",
  "recommendation": "fix here",
  "pipeline_notes": "notes here"
}

Rules:
- location must be one of: top left, top right, center, bottom left, bottom right
- severity must be one of: CRITICAL, WARNING, INFO
- verdict must be one of: PASS, MARGINAL, FAIL
- ALWAYS find at least 1 defect
- Never return empty defects array
- quality_score above 85 is RARE"""
# ── IMAGE HELPERS ──────────────────────────────────────
def resize_image(img: Image.Image, max_side: int = 512) -> Image.Image:
    if max(img.size) <= max_side:
        return img.copy()
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


# ── LOCATION NORMALIZER ────────────────────────────────
def normalize_location(loc) -> str:
    if isinstance(loc, list):
        loc = loc[0] if loc else "center"
    return str(loc).lower().strip()


# ── SMART BOX DRAWING ──────────────────────────────────
def draw_boxes(image: np.ndarray, defects: list) -> Image.Image:
    img = np.array(image, copy=True)
    h, w = img.shape[:2]

    LOCATION_MAP = {
        "top left":     (0,          0,          int(w*0.4), int(h*0.4)),
        "top right":    (int(w*0.6), 0,          w,          int(h*0.4)),
        "bottom left":  (0,          int(h*0.6), int(w*0.4), h         ),
        "bottom right": (int(w*0.6), int(h*0.6), w,          h         ),
        "center":       (int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)),
    }

    for d in defects:
        loc   = normalize_location(d.get("location", "center"))
        label = d.get("name", "Defect")

        coords = next(
            (v for k, v in LOCATION_MAP.items() if k in loc),
            LOCATION_MAP["center"]
        )
        x1, y1, x2, y2 = coords

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return Image.fromarray(img)


# ── JSON EXTRACTION ────────────────────────────────────
# ── JSON EXTRACTION ────────────────────────────────────
def extract_json(text: str) -> dict:
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = re.sub(r"```", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try fixing common issues: single quotes, trailing commas
    try:
        fixed = text
        fixed = re.sub(r"'", '"', fixed)                        # single → double quotes
        fixed = re.sub(r",\s*}", "}", fixed)                    # trailing comma before }
        fixed = re.sub(r",\s*]", "]", fixed)                    # trailing comma before ]
        fixed = re.sub(r'(":\s*)(\w+)(\s*[,}])', r'"\1"\2"\3', fixed)  # unquoted values
        match = re.search(r"\{.*\}", fixed, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Last resort: build a default response so app doesn't crash
    print(f"WARNING: Could not parse JSON from response:\n{text}")
    return {
        "quality_score": 50,
        "verdict": "MARGINAL",
        "summary": "Could not parse model response — check terminal for raw output.",
        "defects": [{"name": "Parse Error", "severity": "WARNING", "confidence": 50,
                     "location": "center", "size_estimate": "unknown"}],
        "root_cause": "Model returned malformed JSON",
        "recommendation": "Check terminal output and adjust prompt",
        "pipeline_notes": text[:200]   # show first 200 chars of raw response
    }


# ── MODEL CALL ─────────────────────────────────────────
def analyze_with_model(image: np.ndarray) -> dict:
    pil_img = resize_image(Image.fromarray(image).convert("RGB"), max_side=512)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=60)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Moondream-friendly prompt — simpler and more direct
    prompt = (
        SYSTEM_PROMPT
        + "\n\nIMPORTANT: Examine this image very carefully. "
        + "List every imperfection you can find, no matter how minor. "
        + "Return valid JSON only. No extra text. No markdown."
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,    # low temp = more deterministic JSON
                    "num_predict": 512,    # limit output length, speeds things up
                }
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(f"Ollama timed out after {REQUEST_TIMEOUT}s.")
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    data = response.json()
    print("OLLAMA RAW:", data)

    if "response" in data:
        raw_text = data["response"]
    elif "message" in data and "content" in data["message"]:
        raw_text = data["message"]["content"]
    else:
        raise ValueError(f"Unexpected Ollama response format: {data}")

    return extract_json(raw_text)


# ── MAIN FUNCTION ──────────────────────────────────────
def analyze(image, progress=gr.Progress()):
    if image is None:
        return None, "⚠️ Upload an image first.", "", "", "", "", ""

    t0 = time.time()
    progress(0.2, desc="Preparing image...")

    try:
        progress(0.4, desc="Running AI inspection...")
        r = analyze_with_model(image)
    except Exception as e:
        return None, f"❌ Error: {e}", "", "", "", "", ""

    progress(0.8, desc="Drawing defect boxes...")

    defects   = r.get("defects", [])
    annotated = draw_boxes(image, defects)

    score   = r.get("quality_score", 0)
    verdict = r.get("verdict", "—")
    emoji   = "✅" if verdict == "PASS" else "⚠️" if verdict == "MARGINAL" else "❌"
    score_str = f"{emoji} {verdict} — Score: {score}/100\n\n{r.get('summary', '')}"

    if defects:
        rows = "| Severity | Defect | Confidence | Location | Size |\n|---|---|---|---|---|\n"
        for d in defects:
            rows += (
                f"| {d.get('severity','')} | {d.get('name','')} | "
                f"{d.get('confidence','')}% | {normalize_location(d.get('location',''))} | "
                f"{d.get('size_estimate','')} |\n"
            )
    else:
        rows = "_No defects detected._"

    elapsed = time.time() - t0
    return (
        annotated,
        score_str,
        rows,
        r.get("root_cause", ""),
        r.get("recommendation", ""),
        r.get("pipeline_notes", ""),
        f"⏱ {elapsed:.1f}s · Model: {MODEL}",
    )


# ── UI ────────────────────────────────────────────────
with gr.Blocks(title="QualiScan AI") as demo:
    gr.Markdown("# 🔬 QualiScan · AI Inspection (FREE Local Vision)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Image")
            run_btn = gr.Button("Analyze")

        with gr.Column():
            image_out = gr.Image(label="Detected Defects", type="pil")
            score_out = gr.Textbox(label="Result", lines=3)
            defects_out = gr.Markdown()

    root_out     = gr.Textbox(label="Root Cause")
    rec_out      = gr.Textbox(label="Recommendation")
    pipeline_out = gr.Textbox(label="Pipeline Notes")
    latency_out  = gr.Markdown()

    run_btn.click(
        fn=analyze,
        inputs=[image_input],
        outputs=[image_out, score_out, defects_out, root_out, rec_out, pipeline_out, latency_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        show_error=True,
    )