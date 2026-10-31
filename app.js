const BACKEND_URL = "https://sketch-to-3d-0dgb.onrender.com";

const $ = id => document.getElementById(id);
const pad = $("pad"), ctx = pad.getContext("2d");
let stage = "", tool = "brush", drawing = false, last = null, dirty = false, blobUrl = null;

function clearPad() {
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, pad.width, pad.height);
  dirty = false;
}
clearPad();

function pos(e) {
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * pad.width / r.width, y: (e.clientY - r.top) * pad.height / r.height };
}
pad.addEventListener("pointerdown", e => { drawing = true; last = pos(e); pad.setPointerCapture(e.pointerId); stroke(last, last); });
pad.addEventListener("pointermove", e => { if (!drawing) return; const p = pos(e); stroke(last, p); last = p; });
["pointerup", "pointercancel"].forEach(t => pad.addEventListener(t, () => drawing = false));

function stroke(a, b) {
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.lineWidth = +$("size").value * (tool === "eraser" ? 2 : 1);
  ctx.strokeStyle = tool === "eraser" ? "#fff" : "#111";
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  if (tool === "brush") dirty = true;
}

function setTool(t) {
  tool = t;
  $("brush").classList.toggle("active", t === "brush");
  $("eraser").classList.toggle("active", t === "eraser");
}
$("brush").onclick = () => setTool("brush");
$("eraser").onclick = () => setTool("eraser");
$("reset").onclick = clearPad;

$("file").addEventListener("change", e => {
  const f = e.target.files[0]; if (!f) return;
  const img = new Image();
  img.onload = () => {
    clearPad();
    const s = Math.min(pad.width / img.width, pad.height / img.height);
    const w = img.width * s, h = img.height * s;
    ctx.drawImage(img, (pad.width - w) / 2, (pad.height - h) / 2, w, h);
    dirty = true; URL.revokeObjectURL(img.src);
  };
  img.onerror = () => status("Could not read that image. Try a PNG or JPG.", true);
  img.src = URL.createObjectURL(f);
  e.target.value = "";
});

function status(msg, err = false) { const s = $("status"); s.textContent = msg; s.className = err ? "err" : ""; }

fetch(BACKEND_URL + "/health").catch(() => {});

$("go").onclick = async () => {
  if (!dirty) return status("Draw something or upload a sketch first.", true);
  const btn = $("go"); btn.disabled = true;
  const t0 = Date.now();
  const timer = setInterval(() => status(`${stage || "Starting"}... ${Math.round((Date.now() - t0) / 1000)}s (first run can take 1-3 min)`), 500);
  try {
    const blob = await new Promise(r => pad.toBlob(r, "image/png"));
    const fd = new FormData();
    fd.append("sketch", blob, "sketch.png");
    fd.append("prompt", $("prompt").value.trim());
    const start = await fetch(BACKEND_URL + "/generate", { method: "POST", body: fd });
    if (!start.ok) throw new Error((await start.json().catch(() => ({}))).error || "Server error " + start.status);
    const { job_id } = await start.json();
    let info;
    while (true) {
      await new Promise(r => setTimeout(r, 3000));
      if (Date.now() - t0 > 8 * 60 * 1000) throw new Error("Timed out. Try again.");
      const sr = await fetch(BACKEND_URL + "/status/" + job_id);
      info = await sr.json();
      if (!sr.ok) throw new Error(info.error || "Lost the job.");
      if (info.state === "error") throw new Error(info.error);
      if (info.state === "done") break;
      stage = info.msg;
    }
    const res = await fetch(BACKEND_URL + "/result/" + job_id);
    if (!res.ok) throw new Error("Could not download the model.");
    const glb = await res.blob();
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(glb);
    $("viewer").src = blobUrl;
    $("empty").hidden = true;
    const dl = $("dl"); dl.href = blobUrl; dl.classList.remove("disabled"); dl.removeAttribute("aria-disabled");
    if (info.thumb) { $("refined").src = "data:image/png;base64," + info.thumb; $("refined").hidden = false; }
    status(`Done in ${Math.round((Date.now() - t0) / 1000)}s.`);
  } catch (e) {
    status(e.name === "AbortError" ? "Timed out. Try again." : "Failed: " + e.message, true);
  } finally {
    clearInterval(timer); btn.disabled = false;
  }
};
