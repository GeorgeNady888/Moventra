// Tremor Vision - front-end logic (tabs, camera, hand tracking, recording, results)

const $ = (id) => document.getElementById(id);

const TITLES = { dashboard: "Tremor Vision", assess: "New Assessment", results: "Results", records: "Records" };
const PROTOCOLS = {
  rest:     ["Rest tremor", "Rest the forearm on a support, palm down, hand relaxed. Do not move it for the whole recording."],
  postural: ["Postural tremor", "Extend the arm straight in front, palm down, and hold it still against gravity."],
  kinetic:  ["Kinetic tremor", "Slowly touch your nose, then the camera, with your index finger. Repeat smoothly."]
};

const VISION_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
                     [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const CSV_COLUMNS = ["time_seconds", "wrist_x", "wrist_y", "hand_detected", "hand_size", "aspect_ratio",
                     "condition", "hand_selected", "handedness", "handedness_score",
                     "frame_index", "frame_interval_seconds", "elapsed_fps"];

const S = {
  state: "idle",            // idle | loading | ready | countdown | recording | analyzing
  protocol: "rest",
  hand: "RIGHT",
  duration: 20,
  stream: null, landmarker: null, delegate: "",
  rows: [], t0: null, prevT: null, detected: 0,
  lastTs: 0, lastFrameT: null, dtEma: null, lastVT: -1, frameCount: 0, recent: [], lightLabel: "--",
  timeSource: "", cd: null, lastCsv: null, lastName: null, recId: null,
  facing: "user", camFps: null, camSize: "", detMs: null,
  // record-then-track: record at the full camera rate, run the tracker afterwards
  useOffline: typeof MediaRecorder !== "undefined" && "requestVideoFrameCallback" in HTMLVideoElement.prototype,
  rec: null, recFail: null, recChunks: [], recMime: "", recStart: 0, recTimer: null,
  pf0: null, pt0: null, camMeasured: null
};

// ---------------------------------------------------------------- tabs
function showTab(name) {
  if (!TITLES[name]) name = "dashboard";
  document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("active", s.id === "screen-" + name));
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("screenTitle").textContent = TITLES[name];
  history.replaceState(null, "", "#" + name);
  window.scrollTo(0, 0);
  if (name !== "assess" && ["ready", "idle"].includes(S.state)) stopCamera();
}
document.querySelectorAll(".nav-btn").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll(".go").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.go)));

// ---------------------------------------------------------------- settings controls
document.querySelectorAll("#protocolSeg .seg").forEach((b) => b.addEventListener("click", () => {
  if (isBusy()) return;
  S.protocol = b.dataset.p;
  document.querySelectorAll("#protocolSeg .seg").forEach((x) => x.classList.toggle("active", x === b));
  $("protoTitle").textContent = PROTOCOLS[S.protocol][0];
  $("protoText").textContent = PROTOCOLS[S.protocol][1];
}));
$("btnHand").addEventListener("click", () => {
  if (isBusy()) return;
  S.hand = S.hand === "RIGHT" ? "LEFT" : "RIGHT";
  $("handLabel").textContent = S.hand;
});
$("durSel").addEventListener("change", (e) => { S.duration = Number(e.target.value); });
$("btnCam").addEventListener("click", async () => {
  if (isBusy()) return;
  S.facing = S.facing === "user" ? "environment" : "user";
  $("camLabel").textContent = S.facing === "user" ? "FRONT" : "BACK";
  if (S.stream) { stopCamera(); await startCamera(); }
});

const isBusy = () => ["countdown", "recording", "processing", "analyzing", "loading"].includes(S.state);

// ---------------------------------------------------------------- UI state
function setStatus(msg, isError = false) {
  const el = $("statusMsg");
  el.textContent = msg || "";
  el.classList.toggle("text-danger", isError);
  el.classList.toggle("text-ink-soft", !isError);
}

function setState(state) {
  S.state = state;
  const b = $("mainBtn");
  const cfg = {
    idle:      ["Enable camera", "btn-navy", false],
    loading:   ["Loading...", "btn-navy", true],
    ready:     ["Start recording", "btn-navy", false],
    countdown: ["Cancel", "btn-navy", false],
    recording: ["End test", "btn-danger", false],
    processing: ["Tracking...", "btn-navy", true],
    analyzing: ["Analyzing...", "btn-navy", true]
  }[state];
  b.textContent = cfg[0];
  b.classList.remove("btn-navy", "btn-danger");
  b.classList.add(cfg[1]);
  b.disabled = cfg[2];
  const locked = isBusy();
  $("durSel").disabled = locked;
  $("btnHand").classList.toggle("opacity-50", locked);
  $("btnCam").classList.toggle("opacity-50", locked);
  $("camPlaceholder").classList.toggle("hidden", Boolean(S.stream) || state === "processing");
  $("camBadge").classList.toggle("hidden", state !== "recording");
  $("camBadge").classList.toggle("flex", state === "recording");
  $("progressWrap").classList.toggle("hidden", state !== "recording" && state !== "processing");
}

$("mainBtn").addEventListener("click", () => {
  if (S.state === "idle") startCamera();
  else if (S.state === "ready") beginCountdown();
  else if (S.state === "countdown") cancelCountdown();
  else if (S.state === "recording") finish();
});

// ---------------------------------------------------------------- camera + model
async function loadLandmarker() {
  if (S.landmarker) return;
  const mod = await import(VISION_BASE + "/vision_bundle.mjs");
  const files = await mod.FilesetResolver.forVisionTasks(VISION_BASE + "/wasm");
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numHands: 1,
    minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6
  });
  try {
    S.landmarker = await mod.HandLandmarker.createFromOptions(files, opts("GPU"));
    S.delegate = "GPU";
  } catch (e) {
    S.landmarker = await mod.HandLandmarker.createFromOptions(files, opts("CPU"));
    S.delegate = "CPU";
  }
}

async function startCamera() {
  setState("loading");
  setStatus("Starting camera...");
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: S.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } }, audio: false
    });
  } catch (e) {
    S.stream = null;
    setState("idle");
    setStatus(e.name === "NotAllowedError"
      ? "Camera permission was denied. Allow the camera from the address bar and try again."
      : e.name === "NotFoundError" ? "No camera was found on this device." : "Could not start the camera: " + e.message, true);
    return;
  }
  const st = S.stream.getVideoTracks()[0].getSettings();
  S.camFps = st.frameRate || null;
  S.camSize = (st.width || "?") + "x" + (st.height || "?");
  S.detMs = null;
  $("mirrorWrap").style.transform = S.facing === "user" ? "scaleX(-1)" : "none";
  const v = $("video");
  v.srcObject = S.stream;
  await v.play();
  if (v.readyState < 1) await new Promise((r) => v.addEventListener("loadedmetadata", r, { once: true }));
  $("overlay").width = v.videoWidth;
  $("overlay").height = v.videoHeight;
  $("camBox").style.aspectRatio = v.videoWidth + " / " + v.videoHeight;

  setStatus("Loading hand-tracking model...");
  try {
    await loadLandmarker();
  } catch (e) {
    console.error(e);
    stopCamera();
    setStatus("Could not load the hand-tracking model. Check your internet connection and try again.", true);
    return;
  }
  setState("ready");
  setStatus("Camera ready (" + S.delegate + ", " + S.camSize + (S.camFps ? " @ " + Math.round(S.camFps) + " fps" : "") + "). Place your hand in view, then press Start recording.");
  S.lastFrameT = null; S.dtEma = null; S.recent = []; S.lastVT = -1; S.frameCount = 0;
  S.pf0 = null; S.pt0 = null; S.camMeasured = null;
  scheduleFrame();
}

function stopCamera() {
  if (S.cd) { clearInterval(S.cd); S.cd = null; }
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;
  $("video").srcObject = null;
  const c = $("overlay"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
  ["pillLight", "pillHand", "pillFps"].forEach((id) => { $(id).textContent = "--"; });
  $("diagLine").textContent = "";
  if (!["analyzing"].includes(S.state)) setState("idle");
  if (S.state === "idle") setStatus("");
}

// ---------------------------------------------------------------- frame loop
function scheduleFrame() {
  if (!S.stream) return;
  const v = $("video");
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame((t) => onFrame(t, null));
}

function onFrame(now, meta) {
  if (!S.stream) return;
  const v = $("video");
  if (!meta) {                                   // fallback path: skip repeated frames
    if (v.currentTime === S.lastVT) { scheduleFrame(); return; }
    S.lastVT = v.currentTime;
  }
  const tFrame = meta ? (meta.captureTime ?? meta.presentationTime ?? now) : now;
  S.timeSource = meta ? (meta.captureTime != null ? "captureTime" : "presentationTime") : "animationFrame";

  // true camera frame rate (counts every frame the camera delivered, even those we did not track)
  if (meta && meta.presentedFrames != null) {
    if (S.pf0 == null) { S.pf0 = meta.presentedFrames; S.pt0 = tFrame; }
    else if (tFrame - S.pt0 >= 1000) {
      S.camMeasured = (meta.presentedFrames - S.pf0) * 1000 / (tFrame - S.pt0);
      S.pf0 = meta.presentedFrames; S.pt0 = tFrame;
    }
  }
  if (S.useOffline && S.state === "recording") {   // recording only: no tracking now, so no frames are lost
    const oc = $("overlay"); oc.getContext("2d").clearRect(0, 0, oc.width, oc.height);
    scheduleFrame();
    return;
  }

  let lm = null, hd = null;
  try {
    const ts = Math.max(now, S.lastTs + 1);
    S.lastTs = ts;
    const t0 = performance.now();
    const res = S.landmarker.detectForVideo(v, ts);
    const dm = performance.now() - t0;
    S.detMs = S.detMs == null ? dm : 0.9 * S.detMs + 0.1 * dm;
    if (res.landmarks && res.landmarks.length) {
      lm = res.landmarks[0];
      hd = (res.handedness && res.handedness[0] && res.handedness[0][0]) || null;
    }
  } catch (e) { console.error(e); }

  drawHand(lm);
  if (S.lastFrameT != null) {
    const dt = tFrame - S.lastFrameT;
    if (dt > 0) S.dtEma = S.dtEma == null ? dt : 0.9 * S.dtEma + 0.1 * dt;
  }
  S.lastFrameT = tFrame;
  S.recent.push(lm ? 1 : 0);
  if (S.recent.length > 30) S.recent.shift();
  if (S.frameCount++ % 10 === 0) updatePills();
  if (S.state === "recording") recordRow(tFrame, lm, hd);
  scheduleFrame();
}

function drawHand(lm) {
  const c = $("overlay"), ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!lm) return;
  ctx.lineWidth = Math.max(2, w / 320);
  ctx.strokeStyle = "#4cd7f6";
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath(); ctx.moveTo(lm[a].x * w, lm[a].y * h); ctx.lineTo(lm[b].x * w, lm[b].y * h); ctx.stroke();
  }
  const r = Math.max(3, w / 220);
  ctx.fillStyle = "#89f5e7";
  lm.forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, i === 0 ? r * 1.8 : r, 0, 2 * Math.PI); ctx.fill(); });
}

const lightCanvas = document.createElement("canvas");
lightCanvas.width = 32; lightCanvas.height = 24;
function updatePills() {
  const v = $("video");
  try {
    const g = lightCanvas.getContext("2d", { willReadFrequently: true });
    g.drawImage(v, 0, 0, 32, 24);
    const d = g.getImageData(0, 0, 32, 24).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    const m = sum / (d.length / 4);
    S.lightLabel = m < 60 ? "Too dark" : m > 210 ? "Too bright" : "OK";
  } catch (e) { S.lightLabel = "--"; }
  $("pillLight").textContent = S.lightLabel;
  const rate = S.recent.length ? S.recent.reduce((a, b) => a + b, 0) / S.recent.length : 0;
  $("pillHand").textContent = Math.round(rate * 100) + "%";
  const fps = S.dtEma ? 1000 / S.dtEma : 0;
  const camShown = S.camMeasured || S.camFps || 0;
  if (S.useOffline) $("pillFps").textContent = camShown ? Math.round(camShown) + " fps" + (camShown < 24 ? " (low)" : "") : "--";
  else $("pillFps").textContent = fps ? fps.toFixed(0) + " fps" + (fps < 24 ? " (low)" : "") : "--";
  let diag = "";
  if (fps && S.detMs != null) {
    const cam = camShown ? Math.round(camShown) : null;
    if (S.useOffline) {
      diag = "Camera " + (cam ? cam + " fps" : "? fps") + " | tracking " + S.detMs.toFixed(0) + " ms/frame | recording uses the full camera rate";
    } else {
      let verdict = "OK";
      if (fps < 24) verdict = (cam && cam < 24) ? "limit: camera" : (cam && fps < 0.8 * cam) ? "limit: tracking model" : "limit: camera or model";
      diag = "Camera " + (cam ? cam + " fps" : "? fps") + " | tracking " + S.detMs.toFixed(0) + " ms/frame | processing " + fps.toFixed(0) + " fps | " + verdict;
    }
  }
  $("diagLine").textContent = diag;
}

// ---------------------------------------------------------------- recording
function beginCountdown() {
  let n = 3;
  setState("countdown");
  setStatus("Hold your hand in position...");
  const box = $("countdown");
  box.textContent = n; box.classList.remove("hidden"); box.classList.add("flex");
  S.cd = setInterval(() => {
    n--;
    if (n > 0) { box.textContent = n; return; }
    clearInterval(S.cd); S.cd = null;
    box.classList.add("hidden"); box.classList.remove("flex");
    startRecording();
  }, 1000);
}

function cancelCountdown() {
  if (S.cd) { clearInterval(S.cd); S.cd = null; }
  $("countdown").classList.add("hidden"); $("countdown").classList.remove("flex");
  setState("ready");
  setStatus("Cancelled.");
}

function startRecording() {
  S.rows = []; S.t0 = null; S.prevT = null; S.detected = 0;
  setState("recording");
  setStatus("Recording... keep the posture steady.");
  updateTimer(0);
  if (S.useOffline) startCapture();
}

function startCapture() {
  S.recChunks = [];
  S.recStart = performance.now();
  try {
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
    S.recMime = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
    S.rec = S.recMime ? new MediaRecorder(S.stream, { mimeType: S.recMime, videoBitsPerSecond: 4000000 }) : new MediaRecorder(S.stream);
    S.rec.ondataavailable = (e) => { if (e.data && e.data.size) S.recChunks.push(e.data); };
    S.rec.start(1000);
  } catch (e) {
    console.error(e);
    S.rec = null; S.useOffline = false;              // fall back to live tracking so the run still works
    setStatus("Recorder unavailable (" + e.message + "). Using live tracking.", true);
  }
  clearInterval(S.recTimer);
  S.recTimer = setInterval(() => {
    try {
      const el = (performance.now() - S.recStart) / 1000;
      updateTimer(el);
      if (el >= S.duration) finish();
    } catch (e) { console.error(e); setStatus("Timer error: " + e.message, true); }
  }, 100);
  // hard failsafe: whatever happens, stop on time
  clearTimeout(S.recFail);
  S.recFail = setTimeout(() => { if (S.state === "recording") finish(); }, S.duration * 1000 + 700);
}

function fmt(sec) { sec = Math.max(0, Math.floor(sec)); return String(Math.floor(sec / 60)).padStart(2, "0") + ":" + String(sec % 60).padStart(2, "0"); }
function updateTimer(el) {
  $("recTime").textContent = fmt(el) + " / " + fmt(S.duration);
  $("progressBar").style.width = Math.min(100, (el / S.duration) * 100) + "%";
  $("progressText").textContent = Math.min(100, Math.round((el / S.duration) * 100)) + "% completed";
  $("remainText").textContent = fmt(S.duration - el) + " remaining";
}

// one CSV row for one tracked frame (el = seconds since the first frame)
function pushRow(el, lm, hd) {
  const aspect = $("video").videoWidth / $("video").videoHeight;
  const row = {
    time_seconds: el.toFixed(6), wrist_x: "", wrist_y: "", hand_detected: 0, hand_size: "",
    aspect_ratio: aspect.toFixed(6), condition: S.protocol, hand_selected: S.hand,
    handedness: "", handedness_score: "", frame_index: S.rows.length + 1,
    frame_interval_seconds: S.prevT == null ? "" : (el - S.prevT).toFixed(6),
    elapsed_fps: el > 0 ? ((S.rows.length + 1) / el).toFixed(2) : ""
  };
  if (lm) {
    const w = lm[0], m = lm[9];                       // 0 = wrist, 9 = middle-finger MCP
    row.wrist_x = w.x.toFixed(6);
    row.wrist_y = w.y.toFixed(6);
    row.hand_size = Math.hypot((m.x - w.x) * aspect, m.y - w.y).toFixed(6);
    row.hand_detected = 1;
    S.detected++;
    if (hd) { row.handedness = hd.categoryName; row.handedness_score = Number(hd.score).toFixed(4); }
  }
  S.prevT = el;
  S.rows.push(row);
}

// live mode (used only if the browser cannot record video)
function recordRow(t, lm, hd) {
  if (S.t0 == null) S.t0 = t;
  const el = (t - S.t0) / 1000;
  pushRow(el, lm, hd);
  updateTimer(el);
  if (el >= S.duration) finish();
}

const toCsv = (rows) => CSV_COLUMNS.join(",") + "\n" + rows.map((r) => CSV_COLUMNS.map((c) => r[c]).join(",")).join("\n") + "\n";

// replay the recorded video through the tracker; every recorded frame is tracked, timing = video media time
function trackRecording(blob, wallSeconds) {
  return new Promise((resolve, reject) => {
    const v = $("video"), c = $("overlay"), ctx = c.getContext("2d");
    const url = URL.createObjectURL(blob);
    S.rows = []; S.prevT = null; S.detected = 0;
    let first = null, lastTm = -1, lastCb = performance.now(), finished = false;
    const base = S.lastTs + 1000;                       // keep tracker timestamps increasing
    const rate = Math.min(1, Math.max(0.25, 18 / (S.detMs || 30)));
    const end = (err) => {
      if (finished) return;
      finished = true;
      clearInterval(watch);
      v.pause(); v.onended = null; v.onerror = null;
      URL.revokeObjectURL(url);
      v.removeAttribute("src"); v.load();
      ctx.clearRect(0, 0, c.width, c.height);
      if (err) reject(err); else resolve();
    };
    const watch = setInterval(() => { if (performance.now() - lastCb > 5000) end(); }, 500);
    v.onended = () => end();
    v.onerror = () => end(new Error("the recording could not be played back"));
    const onF = (now, meta) => {
      lastCb = performance.now();
      const tm = meta.mediaTime;
      if (tm !== lastTm) {
        lastTm = tm;
        if (first == null) first = tm;
        let lm = null, hd = null;
        try {
          const ts = Math.max(base + tm * 1000, S.lastTs + 1);
          S.lastTs = ts;
          const t0 = performance.now();
          const res = S.landmarker.detectForVideo(v, ts);
          const dm = performance.now() - t0;
          S.detMs = S.detMs == null ? dm : 0.9 * S.detMs + 0.1 * dm;
          if (res.landmarks && res.landmarks.length) {
            lm = res.landmarks[0];
            hd = (res.handedness && res.handedness[0] && res.handedness[0][0]) || null;
          }
        } catch (e) { console.error(e); }
        drawHand(lm);
        pushRow(tm - first, lm, hd);
        const pct = Math.min(100, Math.round(((tm - first) / Math.max(wallSeconds, 1)) * 100));
        $("progressBar").style.width = pct + "%";
        $("progressText").textContent = "Tracking recording " + pct + "%";
        $("remainText").textContent = "";
      }
      if (!v.ended && !finished) v.requestVideoFrameCallback(onF);
    };
    v.srcObject = null;
    v.muted = true;
    v.src = url;
    v.playbackRate = rate;
    v.play().then(() => v.requestVideoFrameCallback(onF)).catch((e) => end(e));
  });
}

function afterRun() {
  if (S.stream) setState("ready");
  else { $("video").srcObject = null; setState("idle"); }
}

async function submitRows(meta) {
  const rows = S.rows;
  meta.frames = rows.length; meta.detected = S.detected;
  meta.detectionRate = rows.length ? S.detected / rows.length : 0;
  if (rows.length < 32 || S.detected < 32) {
    afterRun();
    setStatus("Not enough frames with a detected hand (" + S.detected + " of " + rows.length + "). Improve lighting, keep the hand in view, and try again.", true);
    return;
  }
  S.lastCsv = toCsv(rows);
  setState("analyzing");
  setStatus("Analyzing...");
  try {
    const fd = new FormData();
    fd.append("file", new Blob([S.lastCsv], { type: "text/csv" }), "recording.csv");
    fd.append("condition", S.protocol);
    fd.append("meta", JSON.stringify(meta));
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    S.lastName = null;
    renderResults(data, meta);
    afterRun();
    setStatus("");
    showTab("results");
    refreshRecords();
    if (data.save_error) setStatus("Result shown, but it could not be saved: " + data.save_error, true);
  } catch (e) {
    afterRun();
    setStatus("Analysis failed: " + e.message, true);
  }
}

async function finish() {
  if (S.state !== "recording") return;
  const meta = {
    protocol: S.protocol, hand: S.hand, timeSource: S.timeSource, delegate: S.delegate,
    camFps: S.camMeasured || S.camFps, detMs: S.detMs, facing: S.facing, mode: S.useOffline ? "record, then track" : "live"
  };
  if (!S.useOffline) {                               // live mode: rows were collected while recording
    setState("analyzing");
    await submitRows(meta);
    return;
  }
  clearInterval(S.recTimer); S.recTimer = null; clearTimeout(S.recFail);
  const wall = (performance.now() - S.recStart) / 1000;
  setState("processing");
  setStatus("Finishing the recording...");
  let blob;
  try {
    blob = await new Promise((resolve, reject) => {
      let done = false;
      const out = () => { if (!done) { done = true; resolve(new Blob(S.recChunks, { type: S.recMime || "video/webm" })); } };
      S.rec.onstop = out;
      S.rec.onerror = (e) => reject(e.error || new Error("recording failed"));
      setTimeout(out, 3000);                         // onstop never fired: use what we have
      if (S.rec.state !== "inactive") S.rec.stop(); else out();
    });
  } catch (e) {
    stopStream(); afterRun(); setStatus("Recording failed: " + e.message, true); return;
  }
  stopStream();                                      // free the camera; the replay needs only the file
  setStatus("Tracking the recording (about " + Math.max(5, Math.round(wall / Math.min(1, Math.max(0.25, 18 / (S.detMs || 30))))) + " s)...");
  try {
    await trackRecording(blob, wall);
  } catch (e) {
    console.error(e);
    afterRun();
    setStatus("Could not process the recording: " + e.message, true);
    return;
  }
  await submitRows(meta);
}

function stopStream() {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  S.stream = null;                                   // also ends the live frame loop
}

// ---------------------------------------------------------------- results
function tile(label, value) {
  const d = document.createElement("div");
  d.className = "bg-surface-low rounded-2xl p-3";
  const a = document.createElement("div");
  a.className = "text-[11px] font-semibold uppercase tracking-wide text-ink-soft";
  a.textContent = label;
  const b = document.createElement("div");
  b.className = "font-head font-semibold text-xl mt-0.5";
  b.textContent = value;
  d.append(a, b);
  return d;
}

function renderResults(d, meta) {
  $("resultsEmpty").classList.add("hidden");
  $("resultsBody").classList.remove("hidden");
  $("resultsBody").classList.add("flex");

  const steady = d.status === "steady";
  $("resFreq").textContent = d.reliable ? Number(d.frequency_hz).toFixed(2) : (steady ? "None" : "--");
  $("resUnit").classList.toggle("hidden", !d.reliable);
  const badge = $("resBadge");
  badge.textContent = d.reliable ? "Reliable peak" : (steady ? "Steady hand - no significant tremor" : "Inconclusive - repeat the test");
  badge.className = "inline-block mt-2 rounded-full px-3 py-1 text-xs font-semibold " +
    ((d.reliable || steady) ? "bg-teal-soft text-teal-ink" : "bg-surface-high text-ink-soft");
  $("resTitle").textContent = d.result_title;
  $("resMeaning").textContent = d.result_meaning;
  $("resNext").textContent = d.result_next_step;

  const m = $("resMetrics");
  m.replaceChildren(
    tile("Peak concentration", d.concentration),
    tile("Signal-to-noise", d.snr),
    tile("Amplitude (RMS)", d.rms_cm != null ? d.rms_cm + " cm" : d.rms),
    tile("Duration analyzed", d.duration_s + " s"),
    tile("Sample rate", d.sample_rate_hz + " Hz"),
    tile("Hand detected", Math.round(meta.detectionRate * 100) + "%")
  );

  const w = $("resWarnings");
  const warns = d.warnings.slice();
  if (meta.detectionRate < 0.9) warns.push("The hand was detected in only " + Math.round(meta.detectionRate * 100) + "% of frames.");
  w.replaceChildren(...warns.map((t) => { const p = document.createElement("p"); p.textContent = t; return p; }));
  w.classList.toggle("hidden", warns.length === 0);

  $("resFoot").textContent = "Test: " + meta.protocol + " | hand: " + meta.hand + " | " + meta.frames + " frames | timing: " + meta.timeSource +
    " | tracking: " + meta.delegate + (meta.mode ? " (" + meta.mode + ")" : "") + " | camera: " + meta.facing + (meta.camFps ? " " + Math.round(meta.camFps) + " fps" : "") +
    (meta.detMs != null ? " | model: " + meta.detMs.toFixed(0) + " ms/frame" : "") + " | " + d.disclaimer;
  if (typeof window.drawCharts === "function") window.drawCharts(d);
  S.recId = d.record_id || null;
  btnPdf.classList.toggle("hidden", !S.recId);
}

$("btnDownload").addEventListener("click", () => {
  if (!S.lastCsv) return;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([S.lastCsv], { type: "text/csv" }));
  a.download = S.lastName || ("tremor_" + S.protocol + "_" + stamp + ".csv");
  a.click();
  URL.revokeObjectURL(a.href);
});
$("btnAgain").addEventListener("click", () => showTab("assess"));
const btnPdf = document.createElement("button");
btnPdf.type = "button";
btnPdf.className = $("btnDownload").className;
btnPdf.textContent = "PDF report";
btnPdf.classList.add("hidden", "col-span-2");
$("btnDownload").parentNode.insertBefore(btnPdf, $("btnDownload").nextSibling);
btnPdf.addEventListener("click", () => { if (S.recId) window.open("/records/" + S.recId + "/report", "_blank"); });

// ---------------------------------------------------------------- records + dashboard
function recFreq(r) {
  if (r.status === "tremor" && r.frequency_hz != null) return Number(r.frequency_hz).toFixed(2) + " Hz";
  return r.status === "steady" ? "No tremor" : "Inconclusive";
}
function recWhen(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("en-GB", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}
function recRow(r, withActions) {
  const row = document.createElement("div");
  row.className = "bg-surface-low rounded-2xl p-3 lg:p-4 flex items-center gap-3";
  const main = document.createElement("button");
  main.type = "button";
  main.className = "flex-1 text-left min-w-0";
  const top = document.createElement("div");
  top.className = "font-head font-semibold text-lg";
  top.textContent = recFreq(r);
  const sub = document.createElement("div");
  sub.className = "text-xs text-ink-soft truncate";
  sub.textContent = recWhen(r.created_at) + " | " + r.condition + (r.hand ? " | " + String(r.hand).toLowerCase() + " hand" : "") +
    (r.rms_cm != null ? " | " + Number(r.rms_cm).toFixed(2) + " cm" : "");
  main.append(top, sub);
  main.addEventListener("click", () => openRecord(r.id));
  row.append(main);
  if (withActions) {
    const csv = document.createElement("a");
    csv.href = "/api/records/" + r.id + "/csv";
    csv.className = "text-xs font-semibold text-teal-ink rounded-full bg-surface-mid px-3 py-1.5";
    csv.textContent = "CSV";
    const del = document.createElement("button");
    del.type = "button";
    del.className = "text-xs font-semibold text-danger rounded-full bg-surface-mid px-3 py-1.5";
    del.textContent = "Delete";
    del.addEventListener("click", () => deleteRecord(r.id));
    const pdf = document.createElement("a");
    pdf.href = "/records/" + r.id + "/report";
    pdf.target = "_blank";
    pdf.className = "text-xs font-semibold text-teal-ink rounded-full bg-surface-mid px-3 py-1.5";
    pdf.textContent = "PDF";
    row.append(pdf, csv, del);
  }
  return row;
}
async function refreshRecords() {
  let data;
  try {
    const res = await fetch("/api/records");
    data = await res.json();
    if (!res.ok) throw new Error(data.error || "failed");
  } catch (e) { console.error(e); return; }
  const list = data.records || [];
  $("statTotal").textContent = data.total;
  const last = list[0];
  $("statLast").textContent = last ? (last.status === "tremor" && last.frequency_hz != null ? Number(last.frequency_hz).toFixed(2) : (last.status === "steady" ? "None" : "--")) : "--";
  const recent = $("recentList"), all = $("recordsList");
  recent.replaceChildren(); all.replaceChildren();
  $("recordsEmpty").classList.toggle("hidden", list.length > 0);
  if (!list.length) {
    const p = document.createElement("p");
    p.className = "text-ink-soft";
    p.textContent = "No assessments yet. Start one and it will appear here.";
    recent.append(p);
    return;
  }
  list.slice(0, 5).forEach((r) => recent.append(recRow(r, false)));
  list.forEach((r) => all.append(recRow(r, true)));
}
async function openRecord(id) {
  try {
    const res = await fetch("/api/records/" + id);
    const rec = await res.json();
    if (!res.ok) throw new Error(rec.error || "failed");
    const csv = await fetch("/api/records/" + id + "/csv");
    S.lastCsv = csv.ok ? await csv.text() : null;
    S.lastName = "tremor_" + rec.condition + "_" + String(rec.created_at).slice(0, 19).replace(/:/g, "-") + ".csv";
    const meta = Object.assign({ protocol: rec.condition, hand: rec.hand || "", frames: "?", timeSource: "", delegate: "", facing: "" }, rec.meta || {});
    renderResults(rec.result, meta);
    S.recId = rec.id; btnPdf.classList.remove("hidden");
    showTab("results");
  } catch (e) { alert("Could not open the record: " + e.message); }
}
async function deleteRecord(id) {
  if (!confirm("Delete this record? This cannot be undone.")) return;
  try {
    const res = await fetch("/api/records/" + id, { method: "DELETE" });
    if (!res.ok) throw new Error("failed");
  } catch (e) { alert("Could not delete: " + e.message); return; }
  refreshRecords();
}

// ---------------------------------------------------------------- start
setState("idle");
refreshRecords();
showTab(location.hash.replace("#", "") || "dashboard");


// ---------------------------------------------------------------- charts (inline SVG, no libraries)
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = { ink: "#0b1c30", soft: "#45464d", teal: "#006a61", tealSoft: "#86f2e4", grid: "#dce9ff", warn: "#ba1a1a" };

  function el(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function niceStep(range, ticks) {
    const raw = range / ticks, mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }
  function card(title, note) {
    const box = document.createElement("div");
    box.className = "bg-surface-low rounded-2xl p-4 lg:p-5";
    const h = document.createElement("div");
    h.className = "font-head font-semibold text-sm lg:text-base";
    h.textContent = title;
    const p = document.createElement("p");
    p.className = "text-xs text-ink-soft mt-0.5 mb-2";
    p.textContent = note;
    box.append(h, p);
    return box;
  }
  // generic line chart; returns the svg
  function lineChart(o) {
    const W = 640, H = 240, L = 54, R = 14, T = 12, B = 40;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "w-full h-auto", role: "img", "aria-label": o.label });
    const x0 = o.xmin, x1 = o.xmax;
    let y0 = o.ymin, y1 = o.ymax;
    const X = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    if (o.band) svg.append(el("rect", { x: X(o.band[0]), y: T, width: X(o.band[1]) - X(o.band[0]), height: H - T - B, fill: C.tealSoft, opacity: 0.25 }));
    const ys = niceStep(y1 - y0, 4);
    for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) {
      svg.append(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: C.grid, "stroke-width": 1 }));
      svg.append(el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: C.soft }, +v.toFixed(3)));
    }
    const xs = niceStep(x1 - x0, 7);
    for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) {
      svg.append(el("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: C.grid, "stroke-width": 1 }));
      svg.append(el("text", { x: X(v), y: H - B + 15, "text-anchor": "middle", "font-size": 11, fill: C.soft }, +v.toFixed(2)));
    }
    svg.append(el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: C.soft }, o.xlabel));
    svg.append(el("text", { x: 13, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 12, fill: C.soft, transform: `rotate(-90 13 ${(T + H - B) / 2})` }, o.ylabel));
    let dpath = "";
    o.x.forEach((xv, i) => { if (o.y[i] != null && xv != null) dpath += (dpath ? "L" : "M") + X(xv).toFixed(1) + " " + Y(o.y[i]).toFixed(1); });
    svg.append(el("path", { d: dpath, fill: "none", stroke: C.teal, "stroke-width": 1.8, "stroke-linejoin": "round" }));
    if (o.mark) {
      const mx = X(o.mark.x);
      svg.append(el("line", { x1: mx, x2: mx, y1: T, y2: H - B, stroke: o.mark.color, "stroke-width": 1.6, "stroke-dasharray": o.mark.dash ? "5 4" : "0" }));
      const right = mx > W * 0.7;
      svg.append(el("text", { x: right ? mx - 6 : mx + 6, y: T + 13, "text-anchor": right ? "end" : "start", "font-size": 12, "font-weight": 700, fill: o.mark.color }, o.mark.text));
    }
    return svg;
  }

  window.drawCharts = function (d) {
    const host = document.getElementById("resCharts");
    host.replaceChildren();
    const wt = d.waveform && d.waveform.t, wy = d.waveform && d.waveform.y;
    const sf = d.spectrum && d.spectrum.f, sp = d.spectrum && d.spectrum.p;
    if (!wt || !wt.length || !sf || !sf.length) {
      const m = document.createElement("div");
      m.className = "bg-surface-low rounded-2xl p-6 text-center text-ink-soft";
      m.textContent = "No chart data for this recording.";
      host.append(m);
      return;
    }
    // waveform: image-height units -> estimated cm (same scale the RMS uses)
    const k = d.rms && d.rms_cm ? d.rms_cm / d.rms : null;
    const wv = wy.map((v) => (v == null ? null : (k ? v * k : v)));
    const wmax = Math.max(0.05, ...wv.filter((v) => v != null).map(Math.abs)) * 1.15;
    const c1 = card("Hand movement (2-12 Hz band)", "Tremor-band motion of the wrist along its main direction. A steady hand stays flat; a tremor looks like a regular wave.");
    c1.append(lineChart({
      label: "Waveform", x: wt, y: wv, xmin: 0, xmax: wt[wt.length - 1], ymin: -wmax, ymax: wmax,
      xlabel: "Time (s)", ylabel: k ? "Displacement (cm, est.)" : "Displacement (image units)"
    }));
    // spectrum: normalised power so the peak is easy to compare
    const pmax = Math.max(...sp.filter((v) => v != null)) || 1;
    const pn = sp.map((v) => (v == null ? null : v / pmax));
    const c2 = card("Frequency spectrum", "Where the movement energy sits. The shaded band (2-12 Hz) is where we look for tremor; a tall narrow peak means a regular rhythm.");
    const mark = d.reliable
      ? { x: d.frequency_hz, color: C.teal, text: d.frequency_hz.toFixed(2) + " Hz", dash: false }
      : (d.strongest_bin_hz ? { x: d.strongest_bin_hz, color: C.soft, text: "strongest bin " + d.strongest_bin_hz.toFixed(2) + " Hz (not reliable)", dash: true } : null);
    c2.append(lineChart({
      label: "Spectrum", x: sf, y: pn, xmin: 0, xmax: Math.min(15, sf[sf.length - 1]), ymin: 0, ymax: 1.05,
      band: [2, 12], mark, xlabel: "Frequency (Hz)", ylabel: "Relative power"
    }));
    host.append(c1, c2);
  };
})();