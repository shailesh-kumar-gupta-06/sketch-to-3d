import os, io, uuid, time, shutil, tempfile, threading, base64, logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pipeline

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)                                                          # lets GitHub Pages / localhost frontends call this API
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
JOBS, LOCK = {}, threading.Lock()                                  # in-memory; run gunicorn with 1 worker
SEM = threading.Semaphore(1)                                       # one job at a time (512 MB RAM)


def cleanup():
    now = time.time()
    with LOCK:
        for k in [k for k, j in JOBS.items() if now - j["t"] > 1800]:
            shutil.rmtree(JOBS[k]["dir"], ignore_errors=True); JOBS.pop(k)


def worker(jid, raw, prompt):
    job = JOBS[jid]
    with SEM:
        try:
            glb, img, stats = pipeline.run(raw, prompt, job["dir"], lambda m: job.update(msg=m))
            dst = os.path.join(job["dir"], "model.glb"); shutil.copy(glb, dst)
            thumb = img.copy(); thumb.thumbnail((128, 128)); b = io.BytesIO(); thumb.save(b, "PNG")
            job.update(state="done", msg="Done", stats=stats, thumb=base64.b64encode(b.getvalue()).decode(),
                       took=round(time.time() - job["t"], 1))
        except Exception as e:
            logging.exception("job failed")
            job.update(state="error", error=str(e)[:300])


@app.get("/health")
def health(): return jsonify(ok=True)


@app.post("/generate")
def generate():
    f = request.files.get("sketch")
    if not f: return jsonify(error="No sketch uploaded."), 400
    cleanup()
    jid = uuid.uuid4().hex
    with LOCK:
        JOBS[jid] = {"state": "running", "msg": "Queued", "t": time.time(), "dir": tempfile.mkdtemp()}
    threading.Thread(target=worker, args=(jid, f.read(), request.form.get("prompt", "")[:200]), daemon=True).start()
    return jsonify(job_id=jid), 202


@app.get("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j: return jsonify(error="Unknown or expired job."), 404
    return jsonify({k: j[k] for k in ("state", "msg", "error", "stats", "thumb", "took") if k in j})


@app.get("/result/<jid>")
def result(jid):
    j = JOBS.get(jid)
    if not j or j["state"] != "done": return jsonify(error="Not ready."), 404
    return send_file(os.path.join(j["dir"], "model.glb"), mimetype="model/gltf-binary", download_name="model.glb")


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)))