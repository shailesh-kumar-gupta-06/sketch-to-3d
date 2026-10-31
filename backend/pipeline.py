"""Sketch -> clean -> (optional ControlNet refine + fidelity check) -> TRELLIS image-to-3D -> validate .glb
Heavy GPU work runs on free Hugging Face Spaces. This server only does CPU-light steps."""
import os, io, time, logging
import cv2, numpy as np, trimesh
from PIL import Image, ImageOps
from gradio_client import Client, handle_file

log = logging.getLogger("pipeline")
HF_TOKEN = os.getenv("HF_TOKEN") or None                       # strongly recommended: anonymous ZeroGPU quota is usually empty
# Tried in order; Spaces that are broken (CONFIG_ERROR etc.) are skipped automatically.
TRELLIS_SPACES = [x.strip() for x in os.getenv(
    "TRELLIS_SPACES", "trellis-community/TRELLIS,JeffreyXiang/TRELLIS,vaibhavpandeyvpz/trellis-image-to-3d,crevelop/Trellis"
).split(",") if x.strip()]
CONTROLNET_SPACE = os.getenv("CONTROLNET_SPACE", "hysts/ControlNet")  # set to "" to disable
CONTROLNET_API = os.getenv("CONTROLNET_API", "/scribble")
FAITHFUL_THRESHOLD = float(os.getenv("FAITHFUL_THRESHOLD", "0.5"))    # share of sketch lines that must match edges in the refined image
MAX_SIDE = 4000                                                        # reject larger uploads (protects 512 MB RAM)


def space_ok(space: str) -> bool:
    """True if the Space is running or can wake up. False for CONFIG_ERROR, BUILD_ERROR, PAUSED, etc."""
    try:
        from huggingface_hub import HfApi
        stage = HfApi(token=HF_TOKEN).get_space_runtime(space).stage
        log.info("Space %s stage=%s", space, stage)
        return str(stage).upper() in ("RUNNING", "SLEEPING", "RUNNING_BUILDING", "BUILDING")
    except Exception as e:
        log.warning("Could not check %s: %s", space, e)
        return True                                              # unknown: let the real call decide


def _client(space):
    """gradio_client renamed hf_token -> token in newer versions, so try both."""
    if not HF_TOKEN:
        return Client(space)
    for kw in ("token", "hf_token"):
        try:
            return Client(space, **{kw: HF_TOKEN})
        except TypeError:
            continue
    return Client(space)


# ---------- 1. sketch cleanup (OpenCV, CPU) ----------
def clean_sketch(raw: bytes, size=512) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(raw))
        w, h = im.size                                           # header only, no full decode yet
    except Exception:
        raise ValueError("Could not read the image.")
    if max(w, h) > MAX_SIDE:
        raise ValueError(f"Image is too large ({w}x{h}). Use one under {MAX_SIDE}px per side.")
    im = ImageOps.exif_transpose(im)                             # phone photos: respect rotation
    im = im.convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")  # transparent -> white
    scale = 640 / max(im.size)
    if scale < 1:                                                # downscale first: denoising is slow on a 0.1 CPU
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)

    g = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    g = cv2.fastNlMeansDenoising(g, None, 7, 7, 15)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)          # contrast stretch

    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    # adaptive threshold only catches thin strokes; thick brush strokes come out hollow. Add very dark pixels as well.
    dark = ((g < 60) * 255).astype(np.uint8)
    if dark.mean() < 0.3 * 255:                                  # skip for very dark photos so shadows don't become ink
        mask = cv2.bitwise_or(mask, dark)

    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        raise ValueError("The sketch looks empty. Draw bolder lines.")
    pad = 30
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    crop = mask[y0:y1, x0:x1]
    s = max(crop.shape)                                            # square canvas, centred
    sq = np.zeros((s, s), np.uint8)
    sq[(s - crop.shape[0]) // 2:(s - crop.shape[0]) // 2 + crop.shape[0],
       (s - crop.shape[1]) // 2:(s - crop.shape[1]) // 2 + crop.shape[1]] = crop
    sq = cv2.dilate(cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), np.ones((2, 2), np.uint8))
    return Image.fromarray(sq).convert("RGB")                      # white lines on black = scribble format


def as_black_on_white(scribble: Image.Image) -> Image.Image:
    return Image.fromarray(255 - np.array(scribble))


COLORS = {"blue": (80, 125, 220), "red": (210, 70, 70), "green": (80, 170, 110), "yellow": (235, 200, 70),
          "orange": (235, 140, 60), "purple": (150, 95, 200), "pink": (235, 140, 180), "brown": (140, 95, 60),
          "wood": (150, 105, 65), "wooden": (150, 105, 65), "black": (70, 70, 75), "white": (238, 238, 238),
          "gray": (150, 150, 155), "grey": (150, 150, 155), "silver": (190, 195, 200), "gold": (212, 175, 55)}


def fill_sketch(scribble: Image.Image, prompt: str) -> Image.Image:
    """Fallback when ControlNet is unavailable (or drifts from the sketch): turn the line drawing into a filled,
    lightly shaded, coloured object on white, which image-to-3D models handle far better than bare outlines.
    Outlines are kept thin and only slightly darker than the fill so they don't turn into black patches in 3D."""
    m = (np.array(scribble.convert("L")) > 127).astype(np.uint8) * 255
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # bridge small gaps
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return as_black_on_white(scribble)
    color = (205, 208, 214)
    for w in (prompt or "").lower().replace(",", " ").split():
        if w in COLORS:
            color = COLORS[w]; break
    use = [c for c in cnts if cv2.contourArea(c) > 0.01 * m.size] or cnts

    fill = np.zeros(m.shape, np.uint8)
    cv2.drawContours(fill, use, -1, 255, thickness=cv2.FILLED)
    shade = np.linspace(1.08, 0.88, m.shape[0], dtype=np.float32)[:, None, None]  # soft top-to-bottom light gradient
    base = np.clip(np.array(color, np.float32)[None, None, :] * shade, 0, 255).astype(np.uint8)
    base = np.broadcast_to(base, (*m.shape, 3))
    img = np.where(fill[..., None] > 0, base, np.uint8(255)).astype(np.uint8)

    thin = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)                 # thin thick strokes down to a line
    if not thin.any():
        thin = m
    img[thin > 0] = tuple(int(c * 0.72) for c in color)                          # outlines: slightly darker, not black
    return Image.fromarray(img)


def faithful(scribble: Image.Image, refined: Image.Image, thr=FAITHFUL_THRESHOLD) -> bool:
    """True if most sketch line pixels have a Canny edge of the refined image nearby (the output follows the sketch)."""
    s = (np.array(scribble.convert("L")) > 127).astype(np.uint8)
    g = cv2.cvtColor(np.array(refined.convert("RGB").resize(scribble.size)), cv2.COLOR_RGB2GRAY)
    e = cv2.dilate((cv2.Canny(g, 80, 160) > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    score = float((s & e).sum()) / max(int(s.sum()), 1)
    log.info("Sketch fidelity score %.2f (threshold %.2f)", score, thr)
    return score >= thr


# ---------- 2. sketch-guided refinement (ControlNet scribble on a free HF Space) ----------
def _first_path(x):
    """Gradio galleries return paths, dicts or nested lists depending on version. Pull out an image path."""
    if isinstance(x, (list, tuple)):
        for item in reversed(x):                       # the generated image is last; the scribble map comes first
            p = _first_path(item)
            if p: return p
        return None
    if isinstance(x, dict):
        return _first_path(x.get("image") or x.get("path") or x.get("value"))
    return x if isinstance(x, str) and os.path.exists(x) else None


# Different ControlNet Space versions name the same argument differently. Map by name, not by position.
_CN_ALIASES = {
    "prompt": ("prompt",),
    "additional_prompt": ("additional_prompt", "a_prompt"),
    "negative_prompt": ("negative_prompt", "n_prompt"),
    "num_images": ("num_images", "num_samples", "n_samples"),
    "image_resolution": ("image_resolution",),
    "num_steps": ("num_steps", "ddim_steps", "steps"),
    "guidance_scale": ("guidance_scale", "scale", "cfg"),
    "seed": ("seed",),
}
_CN_LOOKUP = {alias: key for key, aliases in _CN_ALIASES.items() for alias in aliases}


def _controlnet_kwargs(params, values, image_path):
    kw = {}
    for prm in params:
        pn = prm["parameter_name"]; low = pn.lower()
        if low in ("image", "input_image"):
            kw[pn] = handle_file(image_path)
        elif low in _CN_LOOKUP:
            kw[pn] = values[_CN_LOOKUP[low]]
        elif "preprocess" in low and "resolution" not in low:
            kw[pn] = "None"                                        # our input is already a scribble map
        elif not prm.get("parameter_has_default"):
            raise ValueError(f"ControlNet: required parameter '{pn}' not handled")
    return kw


def refine(scribble: Image.Image, prompt: str, tmp: str):
    """Returns a PIL image, or None if the Space is down/changed. The job then continues without it."""
    if not CONTROLNET_SPACE or not space_ok(CONTROLNET_SPACE):
        return None
    try:
        p = os.path.join(tmp, "scribble.png"); as_black_on_white(scribble).save(p)   # Space expects dark lines on white
        c = _client(CONTROLNET_SPACE)
        eps = c.view_api(return_format="dict", print_info=False).get("named_endpoints", {})
        name = CONTROLNET_API if CONTROLNET_API in eps else next((n for n in eps if "scribble" in n.lower()), None)
        if not name:
            raise RuntimeError(f"no scribble endpoint, endpoints: {list(eps)}")
        values = {
            "prompt": (prompt or "a single object") + ", plain white background, studio lighting, product photo",
            "additional_prompt": "best quality, extremely detailed",
            "negative_prompt": "lowres, cropped, worst quality, low quality, multiple objects",
            "num_images": 1, "image_resolution": 512, "num_steps": 20, "guidance_scale": 9.0,
            "seed": 0,                                              # fixed seed: reproducible runs
        }
        res = c.predict(api_name=name, **_controlnet_kwargs(eps[name]["parameters"], values, p))
        out = _first_path(res)
        if not out:
            raise ValueError("no image in response")
        return Image.open(out).convert("RGB")
    except Exception as e:
        log.warning("ControlNet step failed, continuing without it: %s", e)
        return None


# ---------- 3. image -> 3D on a free GPU Space ----------
def _pick(eps, *keys):
    """Find an endpoint name: exact match first, then substring (ignores multi-image variants)."""
    for k in keys:
        for n in eps:
            if n.strip("/").lower() == k: return n
    for k in keys:
        for n in eps:
            if k in n.lower() and "images" not in n.lower(): return n
    return None


def _call(c, eps, name, path, extra):
    """Call an endpoint, filling only the required params; the Space's own defaults cover the rest."""
    kw = {}
    for prm in eps[name]["parameters"]:
        pn = prm["parameter_name"]
        if pn in extra: kw[pn] = extra[pn]
        elif not prm.get("parameter_has_default"):
            if "image" in pn: kw[pn] = handle_file(path)
            else: raise ValueError(f"{name}: required parameter '{pn}' not handled")
    return c.predict(api_name=name, **kw)


def _trellis_run(space: str, p: str) -> str:
    c = _client(space)
    eps = c.view_api(return_format="dict", print_info=False).get("named_endpoints", {})
    log.info("%s endpoints: %s", space, list(eps))
    pre, gen, glb, start = (_pick(eps, "preprocess_image", "preprocess"), _pick(eps, "image_to_3d"),
                            _pick(eps, "extract_glb"), _pick(eps, "start_session"))
    combined = _pick(eps, "generate_and_extract_glb")          # newer Spaces do 3D + GLB in a single call
    if not combined and (not gen or not glb):
        raise RuntimeError(f"unexpected API, endpoints: {list(eps)}")
    if start: _call(c, eps, start, p, {})
    proc, trial = p, None
    if pre:                                                     # background removal happens on the Space
        out = _call(c, eps, pre, p, {})
        proc = _first_path(out) or p
        for it in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(it, str) and not os.path.exists(it): trial = it   # some versions return a trial id
    if combined:
        out = _call(c, eps, combined, proc, {"image": handle_file(proc), "multiimages": [],
                                             "mesh_simplify": 0.95, "texture_size": 1024})
        path = _first_path(out)                                  # last item = download_glb
        if not path: raise RuntimeError("no .glb returned")
        return path
    extra = {"image": handle_file(proc), "multiimages": []}
    if trial: extra["trial_id"] = trial
    _call(c, eps, gen, proc, extra)
    out = _call(c, eps, glb, proc, {"mesh_simplify": 0.95, "texture_size": 1024})
    path = _first_path(out)
    if not path: raise RuntimeError("no .glb returned")
    return path


def image_to_glb(img: Image.Image, tmp: str):
    """Returns (glb_path, space_name)."""
    p = os.path.join(tmp, "input.png"); img.save(p)
    errors = []
    for space in TRELLIS_SPACES:
        if not space_ok(space):
            errors.append(f"{space}: not running"); continue
        try:
            log.info("Trying 3D Space %s", space)
            return _trellis_run(space, p), space
        except Exception as e:
            log.warning("3D Space %s failed: %s", space, e)
            errors.append(f"{space}: {str(e)[:120]}")
    raise RuntimeError("All 3D Spaces failed. " + " | ".join(errors))


# ---------- 4. mesh validation (trimesh) ----------
def validate(glb_path: str) -> dict:
    scene = trimesh.load(glb_path, force="scene")
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("Exported file has no mesh.")
    v = sum(len(m.vertices) for m in meshes); f = sum(len(m.faces) for m in meshes)
    ext = np.sort(scene.extents)
    if v == 0 or f == 0:
        raise ValueError("Mesh is empty.")
    if ext[0] < 1e-3 * ext[-1]:
        raise ValueError("Mesh is flat (no real volume).")
    textured = any(getattr(m.visual, "material", None) is not None and
                   getattr(m.visual.material, "baseColorTexture", None) is not None for m in meshes)
    return {"vertices": v, "faces": f, "textured": textured, "extents": [round(float(x), 3) for x in scene.extents]}


def run(raw: bytes, prompt: str, tmp: str, progress=lambda m: None):
    timings, t = {}, time.time()

    def lap(name):
        nonlocal t
        now = time.time(); timings[name] = round(now - t, 1); t = now

    progress("Cleaning sketch")
    scribble = clean_sketch(raw)
    lap("clean_s")

    progress("Refining sketch into an image" if CONTROLNET_SPACE else "Preparing image")
    refined = refine(scribble, prompt, tmp)
    if refined is None:
        note, img = "unavailable", fill_sketch(scribble, prompt)
    elif faithful(scribble, refined):
        note, img = "used", refined
    else:
        note, img = "rejected: output drifted from the sketch", fill_sketch(scribble, prompt)
    lap("refine_s")

    progress("Building 3D model on GPU (1-4 min)")
    glb, space = image_to_glb(img, tmp)
    lap("image_to_3d_s")

    progress("Validating mesh")
    stats = validate(glb)
    lap("validate_s")

    stats.update(used_controlnet=(note == "used"), controlnet_note=note, space=space, timings=timings)
    return glb, img, stats