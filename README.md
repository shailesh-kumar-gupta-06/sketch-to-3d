# Sketch to 3D

Draw a rough sketch (or upload one), optionally add a text prompt, and get a textured 3D model as `.glb` with an interactive preview and download. Built for the 24-hour *Sketch2Shape AI* hackathon.

- **Live app:** https://shailesh-kumar-gupta-06.github.io/sketch-to-3d/
- **Backend:** https://sketch-to-3d-0dgb.onrender.com (`/health` for a status check)

> The backend runs on a free Render instance that sleeps when idle. Open the app a minute before use. GPU work uses free Hugging Face Spaces with a shared daily quota (see Limitations).

## Features
- In-app canvas (brush, eraser, size, reset) and image upload (PNG/JPG/WebP)
- Optional text prompt for colour, material and style
- Preprocessing: denoise, contrast stretch, thick-stroke-safe thresholding, crop and centre
- Sketch-guided image generation, with a fidelity check and automatic fallback
- Image-to-3D with texture, mesh validation, `.glb` export, in-browser 3D viewer

## Architecture
```
Browser (canvas, upload, prompt, <model-viewer>)
   | POST /generate -> GET /status/<id> -> GET /result/<id>
Flask backend (CPU: OpenCV, trimesh)  [Render]
   | gradio_client
Hugging Face Spaces (free GPU)
   1) ControlNet scribble : sketch + prompt -> product-style image (optional)
   2) TRELLIS             : background removal -> 3D -> textured .glb
```

## Pipeline
1. **Clean:** flatten transparency, fix EXIF rotation, downscale, denoise, adaptive threshold plus dark-pixel mask, crop, centre, 512x512 scribble map.
2. **Refine:** ControlNet scribble generates a clean image from the sketch and prompt (fixed seed, arguments mapped by name from the Space's live API).
3. **Fidelity check:** Canny edge match between sketch and generated image. If under 50% (`FAITHFUL_THRESHOLD`), the image is rejected.
4. **Fallback:** if ControlNet is down or rejected, the outline is filled with a lightly shaded colour (taken from the prompt, e.g. "wooden") with thin outlines.
5. **Reconstruct:** TRELLIS removes the background and builds a textured mesh. Spaces are tried in order.
6. **Validate:** trimesh checks for non-empty geometry, real volume (not a flat plane) and texture. Stats and per-stage timings are returned by `/status/<id>`.

## Models and third-party services (disclosure)
| Component | Used for | Runs on | Licence (verify before commercial use) |
|---|---|---|---|
| Stable Diffusion 1.5 + ControlNet scribble, Space `hysts/ControlNet` | sketch-guided image | HF Space | CreativeML OpenRAIL-M / Apache-2.0 |
| Microsoft TRELLIS via Spaces `trellis-community/TRELLIS`, `JeffreyXiang/TRELLIS`, `vaibhavpandeyvpz/trellis-image-to-3d`, `crevelop/Trellis` (tried in this order) | background removal, image-to-3D | HF Space (ZeroGPU) | MIT |
| OpenCV, Pillow, NumPy, trimesh, Flask, flask-cors, gradio_client, huggingface_hub | preprocessing, validation, API | Backend (CPU) | Apache-2.0 / HPND / BSD / MIT |
| `<model-viewer>` (Google) | 3D preview | Browser | Apache-2.0 |

No weights are downloaded locally. Spaces are external services and may change or go offline.

## Run locally
Requirements: Python 3.11 or 3.12, internet, a free Hugging Face account and a **Read** token. A token is needed in practice: anonymous ZeroGPU quota is usually exhausted.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
read -s "HF_TOKEN?Paste token: "; export HF_TOKEN; echo   # zsh. bash: read -s -p "Paste token: " HF_TOKEN
python check_spaces.py        # which Spaces are alive
python app.py                 # http://127.0.0.1:5000
```
In another terminal, serve the frontend from its own folder:
```bash
python3 -m http.server 8000 --bind 127.0.0.1
```
Open http://localhost:8000. Set `BACKEND_URL` at the top of `app.js` to `http://localhost:5000` for local use.

### Configuration
| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | none | Hugging Face token |
| `TRELLIS_SPACES` | list above | Comma-separated Spaces, tried in order |
| `CONTROLNET_SPACE` | `hysts/ControlNet` | Empty string disables the refine step |
| `CONTROLNET_API` | `/scribble` | Endpoint name on that Space |
| `FAITHFUL_THRESHOLD` | `0.5` | Minimum sketch/edge match |

## Deployment
- **Backend (Render):** Root Directory `backend`, build `pip install -r requirements.txt`, start `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`, env `HF_TOKEN` (secret) and `PYTHON_VERSION=3.11.9`. Use one worker: jobs are held in memory.
- **Frontend (GitHub Pages):** serve `index.html`, `app.js`, `style.css`; set `BACKEND_URL` to the Render URL without a trailing slash.

## Hardware and timing
| Item | Value |
|---|---|
| Backend | Render free instance (CPU only, 512 MB RAM); dev machine Ubuntu 24.04, `<CPU>`, `<RAM>`, no GPU used |
| GPU work | Hugging Face free shared ZeroGPU |

| Run | Input | ControlNet | Total |
|---|---|---|---|
| Chair, thick brush, prompt "chair" | in-app drawing | unavailable (fallback) | 57 s (warm) |
| `<fill in>` | | | |
| Cold start (first request after idle) | | | `<fill in>` |

## Sample results
See `samples/` for input sketches, generated `.glb` files and viewer screenshots.

## Limitations
- Free Spaces can queue, sleep, change API, or run out of the daily ZeroGPU quota; a run may fail with a quota message until it resets.
- Without ControlNet the result follows the outline but has less style detail.
- A single view cannot define hidden sides: geometry is plausible, not exact, and the viewpoint may differ from the drawing.
- Bold, closed outlines of everyday objects work best; thin, abstract or tonal ink drawings give weak shapes.
- One job at a time; job state is in memory. Not intended for print-ready or production-topology meshes.

## Troubleshooting
- **"exceeded your ZeroGPU quota":** token missing or not exported in the shell running `app.py`, or the account's daily quota is used.
- **"All 3D Spaces failed":** run `python check_spaces.py`; set `TRELLIS_SPACES` to a live Space.
- **Fails immediately in browser:** open `<BACKEND_URL>/health`; check `BACKEND_URL` has no trailing slash.
- **`externally-managed-environment` on pip install:** use a virtual environment as above.

## Project structure
```
backend/   app.py  pipeline.py  check_spaces.py  requirements.txt  render.yaml
index.html  app.js  style.css      (frontend)
samples/                            (inputs, .glb outputs, screenshots)
```

## Team
`<OFF-GRID>` · Licence: `<add a LICENSE file>`
