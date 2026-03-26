# QualiScan · AI Quality Inspection

A local AI-powered manufacturing quality inspection system. No API keys. No cloud. Runs entirely on your machine.

## How It Works

Two model pipeline:
- **Moondream** — fast vision model that analyzes the image and describes defects
- **Llama 3.2** — converts the description into structured JSON output

## Features

- Detects scratches, cracks, stains, dents, corrosion and more
- Draws bounding boxes around defect regions
- Quality score from 0-100 with PASS / MARGINAL / FAIL verdict
- Root cause analysis and actionable recommendations
- Runs 100% locally via Ollama — free, private, no internet needed

## Setup

1. Install [Ollama](https://ollama.com)
2. Pull the models:
```bash
ollama pull llava
ollama pull moondream
ollama pull llama3.2
```
3. Install dependencies:
```bash
pip install gradio opencv-python pillow requests numpy
```
4. Run:
```bash
python app.py
```
5. Open `http://127.0.0.1:7860`

## Tech Stack

- Python
- Gradio
- OpenCV
- Ollama
- Moondream + Llama 3.2
