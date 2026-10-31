# Sketch to 3D

Draw a rough sketch (or upload one), optionally add a text prompt, and get a textured 3D model as `.glb`, with an interactive preview and download.

## Architecture

```
Browser (HTML/CSS/JS: canvas, upload, prompt, <model-viewer>)
        | HTTP (job based: POST /generate -> GET /status/<id> -> GET /result/<id>)
Flask backend (CPU only: OpenCV preprocessing, fidelity check, trimesh validation)
        | gradio_client
Hugging Face Spaces (free shared GPU)
   1) ControlNet scribble : sketch + prompt -> clean product-style image (optional step)
   2) TRELLIS             : image -> background removal -> 3D -> textured .glb
```

## Pipeline

1. **Input preparation (backend, OpenCV/Pillow):** transparent pixels flattened to white, EXIF rotation applied, downscale to 640px, denoise, contrast stretch, adaptive threshold plus a dark-pixel mask (so thick brush strokes stay solid), crop, centre, resize to 512x512. Output is white lines on black (scribble format). Images over 4000px per side are rejected.
2. **Sketch-guided image (HF Space, ControlNet scribble):** keeps the sketch's shape and takes style from the prompt. Arguments are mapped by name from the Space's live API, and a fixed seed (0) is used.
3. **Fidelity check (backend):** a Canny edge match between the sketch lines and the generated image. If less than 50% of the sketch lines are matched (`FAITHFUL_THRESHOLD`), the generated image is rejected so the result stays related to the sketch.
4. **Fallback image (backend):** if ControlNet is down or rejected, the sketch outline is filled with a flat, lightly shaded colour (taken from the prompt, e.g. "wooden", "blue") with thin outlines.
5. **Image to 3D (HF Space, TRELLIS):** removes the background, reconstructs geometry and texture, exports `.glb`. Spaces are tried in the order listed below.
6. **Validation (trimesh):** non-empty mesh, real volume (not a flat plane), texture present. Stats, stage timings, which Space was used and the ControlNet status are returned by `/status/<id>`.

## Models and third-party services (disclosure)

| Component | Used for | Runs on | Licence (verify before commercial use) |
|---|---|---|---|
| Stable Diffusion 1.5 + ControlNet scribble, Space `hysts/ControlNet` (configurable) | sketch-guided image | Hugging Face Space | CreativeML OpenRAIL-M / Apache-2.0 |
| Microsoft TRELLIS via Spaces, tried in order: `trellis-community/TRELLIS`, `JeffreyXiang/TRELLIS`, `vaibhavpandeyvpz/trellis-image-to-3d`, `crevelop/Trellis` | background removal + image-to-3D | Hugging Face Space (ZeroGPU) | MIT |
| OpenCV, Pillow, NumPy, trimesh, Flask, flask-cors, gradio_client, huggingface_hub | preprocessing, validation, API, Space calls | Backend (CPU) | Apache-2.0 / HPND / BSD / MIT |
| `<model-viewer>` (Google) | 3D preview | Browser | Apache-2.0 |

No model weights are downloaded locally. All Spaces are external services and can change or go offline.

## Setup and run

Requirements: Python 3.11 or 3.12, internet access, a free Hugging Face account and a **Read** access token.

**A token is required in practice.** Without it, the free ZeroGPU quota is shared with anonymous users and is usually exhausted ("exceeded your ZeroGPU quota").

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# set the token without echoing it
read -s "HF_TOKEN?Paste token: "       # zsh   (bash: read -s -p "Paste token: " HF_TOKEN)
export HF_TOKEN; echo

python check_spaces.py                 # which Spaces are alive right now
python app.py                          # http://127.0.0.1:5000
```

In a second terminal, serve the frontend **from its own folder**:

```bash
cd frontend
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000, draw or upload a sketch, and click **Generate 3D model**. `BACKEND_URL` in `frontend/app.js` must point at the backend (`http://localhost:5000` locally).

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | none | Hugging Face token (needed for usable GPU quota) |
| `TRELLIS_SPACES` | list above | Comma-separated Spaces, tried in order |
| `CONTROLNET_SPACE` | `hysts/ControlNet` | Set to empty to disable the ControlNet step |
| `CONTROLNET_API` | `/scribble` | Endpoint name on that Space |
| `FAITHFUL_THRESHOLD` | `0.5` | Minimum sketch/edge match to accept the ControlNet image |
| `PORT` | `5000` | Backend port |

## Deploy (optional)

- **Backend:** Render Web Service, build `pip install -r requirements.txt`, start `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT` (the in-memory job store needs a single worker). Set `HF_TOKEN` as a secret.
- **Frontend:** any static host (e.g. GitHub Pages). Set `BACKEND_URL` to the backend's https URL, with no trailing slash.

## Hardware and timing

| Item | Value |
|---|---|
| Backend machine | Ubuntu 24.04 laptop, `<CPU model>`, `<RAM>`, no GPU used |
| GPU work | Hugging Face free shared ZeroGPU (TRELLIS and ControlNet) |
| Python / key libs | 3.12; see `requirements.txt` |

Measured end-to-end times (draw to downloadable .glb). Fill in from your own runs:

| Run | Input | Prompt | ControlNet | Total |
|---|---|---|---|---|
| 1 | chair, thick brush | "chair" | unavailable (fallback) | 57 s (warm) |
| 2 | `<mug>` | `<...>` | `<...>` | `<...>` |
| 3 | `<table>` | `<...>` | `<...>` | `<...>` |
| Cold start (first request after idle) | | | | `<...>` |

Per-stage timings for each run are in the `/status` response (`stats.timings`).

## Sample results

`samples/` contains, for each test: the input sketch, the generated `.glb`, and a screenshot of the viewer.

## Limitations

- GPU work depends on free shared Spaces, which can queue, sleep, change their API, or run out of the daily ZeroGPU quota. A run may fail with a quota message; retry later or use a fresh token/account.
- If ControlNet is unavailable, the pipeline falls back to a filled silhouette. The result then follows the outline but has less style detail.
- A single view cannot define the hidden sides. Geometry is plausible, not exact, and the reconstructed viewpoint can differ from the drawn one (a side-view sketch may come out front-facing).
- Closed, bold outlines work best. Very thin, abstract or overlapping sketches can give weak shapes.
- One job at a time on the backend; job state is in memory and lost on restart.
- Not intended for print-ready geometry or production animation topology.

## Troubleshooting

- **"exceeded your ZeroGPU quota":** the token is missing or not exported in the terminal that runs `python app.py`. Check with `python -c "import os; print(len(os.getenv('HF_TOKEN') or ''))"` (about 37 for a valid token).
- **"All 3D Spaces failed":** run `python check_spaces.py`; Spaces showing `RUNTIME_ERROR` or `CONFIG_ERROR` are skipped. Set `TRELLIS_SPACES` to a live Space.
- **Failed right away in the browser:** open `<BACKEND_URL>/health`; confirm the backend is running and `BACKEND_URL` is correct.
- **CORS error:** confirm `BACKEND_URL` has no trailing slash and the backend is running.
- **`pip install` says "externally-managed-environment":** use a virtual environment as shown above.