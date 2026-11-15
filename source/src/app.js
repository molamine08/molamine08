import {
  Calibrator, GazeModel, ErrorMap, FrameRing, FixationSmoother, HeadFilter, TargetSelector, summarizeWindow, windowWeight,
  pursuitPath, pursuitSamples, estimateLag, measureHeadShares, packInt8, median, percentile, clamp,
} from './engine.js';
import {
  FEATURE_NAMES, MODEL_SPECS, DEFAULT_MODEL, MOVE_SAFE, DEFAULT_SHARES, POSE_IDX, POSE_SIGMA, EYE_REF, N_BASE, N_FULL, N_PIX, OFF_PIX, OFF_LOG, PIX_W, PIX_H,
  extractFeatures, setGeometry, getGeometry, headParts,
} from './features.js';
import { SimUser, mulberry32 } from './sim.js';

const VERSION = '0.4';
// Frames and touches are timed with performance.now(); stored samples carry wall-clock time.
// Converted at the moment of use: the page clock stops while the browser is in the background.
const wall = (tPerf) => Date.now() - (performance.now() - tPerf);
const PURSUIT_MS = 20000;
const MOVE_MS = 30000;
// A touch or a dot is one deliberate look at a known place; a pursuit frame is one of hundreds.
const TOUCH_BOOST = 3;
const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const SIM_URL = /\bsim\b/.test(location.search + location.hash);
// Inside another app's preview (an iframe, or a page not served over http/https) the camera is blocked.
const EMBEDDED = (() => {
  try {
    return window.top !== window.self;
  } catch (e) {
    return true;
  }
})() || !/^https?:$/.test(location.protocol);

const TITLES = { home: 'معايرة النظر', boot: 'معايرة 9 نقاط', pursue: 'معايرة بالتتبّع', pursueMove: 'معايرة متحركة', test: 'قياس الدقة', free: 'استخدام حر', pick: 'نظر + لمس', stats: 'النتائج' };
const IMMERSIVE = { boot: true, pursue: true, pursueMove: true, test: true, pick: true };

const makeCal = () => new Calibrator({
  models: MODEL_SPECS.map((spec) => new GazeModel(spec)),
  poseIdx: POSE_IDX,
  poseSigma: POSE_SIGMA,
  defaultModel: DEFAULT_MODEL,
  prefer: MOVE_SAFE, // the model that survives a moving phone leads unless another is clearly better
  cvEvery: 24,
  deferCv: true, // learning waits for the idle timer instead of delaying a touch
  deferFit: true,
  poseRefitMs: 1000,
});

function guessPxPerCm() {
  const ua = navigator.userAgent;
  const dpr = window.devicePixelRatio || 1;
  if (/iPhone/.test(ua)) return dpr >= 3 ? 60.4 : 64.2;
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) return 52;
  if (/Android/.test(ua)) return 63;
  return 38;
}

const S = {
  W: window.innerWidth,
  H: window.innerHeight,
  pxPerCm: guessPxPerCm(),
  sim: SIM_URL,
  simUser: null,
  running: false,
  face: false,
  fps: 0,
  lastFrameT: 0,
  distCm: NaN,
  delegate: SIM_URL ? 'sim' : 'CPU',
  cal: makeCal(),
  ring: new FrameRing(),
  errMap: new ErrorMap(3, 4),
  smooth: new FixationSmoother(),
  headFilter: new HeadFilter(),
  geo: null, // what the last moving calibration measured about the head's share
  gaze: null,
  video: null,
  pursuitLog: [],
  anchor: null, // when and in which head pose the last labelled sample was taken
  onFrame: null,
  settings: { cursor: true, learn: true, preview: true, gpu: false, motion: false },
  log: [],
  picks: [],
  rejects: {},
  looPx: NaN,
  motion: 0,
  motionT: -1e9,
  rot: [0, 0, 0, 1], // how the phone has turned since the page opened (quaternion), summed from the gyroscope
  grav: null, // gravity as the phone feels it: how the phone is held
  screen: null,
  cleanup: null,
  onGaze: null,
  onSample: null,
  selector: null,
  pickLevel: 0,
};

setGeometry({ pxPerCm: S.pxPerCm });

// ---------- small helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);

function el(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const targetsLabel = (n) => `${n} ${n <= 10 ? 'أهداف' : 'هدفًا'}`;
const cm = (px, digits = 1) => (Number.isFinite(px) ? (px / S.pxPerCm).toFixed(digits) : '—');

let toastTimer = 0;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

function eventTime(e) {
  const now = performance.now();
  const t = e.timeStamp;
  return t > 0 && t <= now + 5 && now - t < 500 ? t : now;
}

function shuffle(a, rand = Math.random) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ---------- persistence (one calibration per screen width and orientation) ----------
const profileKey = () => `gazecal:v3:${S.W}:${S.H >= S.W ? 'p' : 'l'}`;
let saveTimer = 0;

function save() {
  if (S.sim) return;
  try {
    localStorage.setItem('gazecal:settings', JSON.stringify(S.settings));
    localStorage.setItem(profileKey(), JSON.stringify({ cal: S.cal.toJSON({ nBase: N_BASE }), loo: S.looPx, shares: getGeometry().shares, geo: S.geo }));
  } catch (e) { /* storage can be unavailable; the app works without it */ }
}

function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 600);
}

function load() {
  if (S.sim) return;
  try {
    const s = JSON.parse(localStorage.getItem('gazecal:settings') || 'null');
    const motion = S.settings.motion; // granted or refused in this visit; a stored answer means nothing
    if (s) Object.assign(S.settings, s);
    S.settings.motion = motion;
    const p = JSON.parse(localStorage.getItem(profileKey()) || 'null');
    // the head shares belong to the profile and have to be in place before its models are rebuilt
    setGeometry({ shares: DEFAULT_SHARES });
    S.geo = null;
    if (p && p.cal) {
      setGeometry({ shares: p.shares });
      if (S.cal.load(p.cal, { nFeat: N_FULL, nowT: Date.now() })) {
        S.looPx = Number.isFinite(p.loo) ? p.loo : NaN;
        S.geo = p.geo || null;
      } else {
        setGeometry({ shares: DEFAULT_SHARES });
      }
    }
  } catch (e) { /* ignore a corrupt or missing profile */ }
  S.cal.learning = S.settings.learn;
}

// ---------- frames ----------
const sigmaPx = () => clamp(S.errMap.recentMedian(12) || 90, 45, 220);

function onFrame(f, t) {
  S.face = !!f.face;
  S.distCm = f.face ? f.distCm : NaN;
  if (S.lastFrameT) {
    const inst = 1000 / Math.max(1, t - S.lastFrameT);
    S.fps = S.fps ? S.fps + 0.1 * (inst - S.fps) : inst;
  }
  S.lastFrameT = t;
  const gyro = t - S.motionT < 400;
  f.device = gyro ? { rot: S.rot.map((v) => Math.round(v * 1e4) / 1e4), grav: S.grav ? S.grav.map((v) => Math.round(v * 100) / 100) : null } : null;
  if (S.face) {
    // The head numbers are steadied while the gyroscope says the phone rests; the camera's own
    // reading is kept beside them for the log.
    const turn = S.simUser ? (f.turn >= 0 ? f.turn : null) : gyro ? S.motion : null;
    f.rawHead = Array.from(f.phi.subarray(22, 27));
    f.phi.set(S.headFilter.push(f.rawHead, t / 1000, turn), 22);
  } else {
    S.headFilter.reset();
  }
  S.ring.push({ t, face: S.face, phi: f.phi || null, blink: f.blink || 0, motion: gyro ? S.motion : 0, hi: f.hi || null, raw: f.rawHead || null });
  if (S.face) S.cal.setPose(f.phi);

  if (!S.face) {
    S.gaze = null;
    S.smooth.reset();
  } else if (S.cal.ready && !(f.blink > 0.5) && !(gyro && S.motion > 80)) {
    const p = S.cal.predict(f.phi);
    if (p) {
      S.smooth.radius = clamp(sigmaPx(), 70, 160);
      const q = S.smooth.push(p[0], p[1], t);
      S.gaze = { x: clamp(q.x, 0, S.W), y: clamp(q.y, 0, S.H), t };
      if (S.selector) S.selector.update(S.gaze.x, S.gaze.y, sigmaPx(), t);
    }
  } else if (!S.cal.ready) {
    S.gaze = null;
  }
  if (S.onFrame) S.onFrame(f, t);
  drawCursor();
  if (S.onGaze) S.onGaze(t);
}

function drawCursor() {
  const c = $('#cursor');
  const show = S.settings.cursor && !!S.gaze && !IMMERSIVE[S.screen];
  c.hidden = !show;
  if (!show) return;
  c.style.transform = `translate3d(${S.gaze.x.toFixed(1)}px, ${S.gaze.y.toFixed(1)}px, 0)`;
  c.style.setProperty('--d', `${Math.round(2 * Math.min(sigmaPx(), 130))}px`);
}

// ---------- samples ----------
const r5 = (v) => Math.round(v * 1e5) / 1e5;

function b64(bytes) {
  let str = '';
  for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i]);
  return btoa(str);
}

// The eye cut-outs of the frames behind one sample, averaged into one picture for the log.
function meanEyes(frames) {
  const open = frames.filter((f) => f.face && f.hi && !(f.blink > 0.5));
  if (!open.length) return null;
  const acc = new Float32Array(open[0].hi.length);
  for (const f of open) for (let i = 0; i < acc.length; i++) acc[i] += f.hi[i];
  return b64(Uint8Array.from(acc, (v) => Math.round(v / open.length)));
}

function takeSample(target, src, tMs, baseW, { force = false, since = -Infinity } = {}) {
  const frames = S.ring.window(Math.max(tMs - 350, since), tMs + 50);
  const sum = summarizeWindow(frames);
  if (!sum.ok) {
    S.rejects[sum.reason] = (S.rejects[sum.reason] || 0) + 1;
    return { ok: false, reason: sum.reason };
  }
  const rec = S.cal.addSample({ phi: sum.phi, y: target, w: TOUCH_BOOST * baseW * windowWeight(sum, { eyeRef: EYE_REF }), src, t: Math.round(wall(tMs)) }, { force });
  S.anchor = { t: tMs, pose: S.cal.pose ? Float64Array.from(S.cal.pose) : null };
  rec.ok = true;
  rec.screen = S.screen;
  rec.frames = sum.n;
  rec.motion = Math.round(sum.motion);
  rec.w = +rec.w.toFixed(3);
  for (const k in rec.errs) if (rec.errs[k] != null) rec.errs[k] = Math.round(rec.errs[k] * 10) / 10;
  rec.phi = Array.from(sum.phi.subarray(0, N_BASE), r5);
  if (sum.phi.length >= OFF_PIX + N_PIX) rec.pix = packInt8(sum.phi.subarray(OFF_PIX, OFF_PIX + N_PIX));
  const eye = meanEyes(frames);
  if (eye) rec.eye = eye;
  // the head as the camera read it, before steadying (rec.phi holds the steadied numbers)
  const raws = frames.filter((fr) => fr.face && fr.raw && !(fr.blink > 0.5)).map((fr) => fr.raw);
  if (raws.length) rec.head = [0, 1, 2, 3, 4].map((j) => r5(median(raws.map((v) => v[j]))));
  if (performance.now() - S.motionT < 400) {
    rec.rot = S.rot.map((v) => Math.round(v * 1e4) / 1e4);
    if (S.grav) rec.grav = S.grav.map((v) => Math.round(v * 100) / 100);
  }
  if (rec.err != null) S.errMap.add(target[0] / S.W, target[1] / S.H, rec.err);
  S.log.push(rec);
  if (S.log.length > 3000) S.log.shift();
  saveSoon();
  if (S.onSample) S.onSample(rec);
  return rec;
}

// Error of every model on the same samples. Each sample was scored by all of them before any learned from it.
function modelBoard(recs) {
  const lead = S.cal.activeModel();
  return S.cal.models.map((m) => {
    const e = recs.map((r) => (r.errs ? r.errs[m.name] : null)).filter((v) => v != null);
    return { name: m.name, label: m.label, n: e.length, med: median(e), lead: lead === m, ready: m.ready };
  });
}
const boardLine = (board) => board.filter((b) => b.n).map((b) => `${b.name} ${cm(b.med)}`).join(' · ') || '—';
const leadLabel = () => {
  const m = S.cal.activeModel();
  return m ? `${m.name} · ${m.label}` : '—';
};

// Every touch on a small marked target is a free calibration point.
document.addEventListener('pointerdown', (e) => {
  if (!S.running || e.isPrimary === false) return;
  const t = eventTime(e);
  if (S.simUser) simLookAtTouch(e.clientX, e.clientY, t);
  const marked = e.target && e.target.closest ? e.target.closest('[data-gaze]') : null;
  const role = marked ? marked.dataset.gaze : null;
  if (role === 'ignore' || role === 'dot') return;
  if (role === 'target') {
    const r = marked.getBoundingClientRect();
    takeSample([r.left + r.width / 2, r.top + r.height / 2], 'touch-target', t, 0.9);
  } else if (S.screen === 'free') {
    takeSample([e.clientX, e.clientY], 'touch-free', t, 0.5);
  }
}, true);

// ---------- simulation mode ----------
function simLookAtTouch(x, y, t) {
  // The synthetic user had its eyes on the spot before touching it, as a person would.
  S.simUser.lookAt(x, y, t - 600);
  S.ring.frames = S.ring.frames.filter((f) => f.t < t - 380);
  for (let k = 0; k <= 11; k++) {
    const ft = t - 380 + k * 33;
    const f = S.simUser.frame(ft);
    S.ring.push({ t: ft, face: true, phi: f.phi, blink: f.blink, motion: 0, hi: null, raw: Array.from(f.phi.subarray(22, 27)) });
  }
}

function startSim() {
  if (S.simUser) return;
  S.simUser = new SimUser({ W: S.W, H: S.H, pxPerCm: S.pxPerCm, seed: 7, noise: 1.5, pixelJitter: 2.5 });
  window.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'mouse') S.simUser.lookAt(e.clientX, e.clientY, performance.now());
  });
  setInterval(() => {
    const t = performance.now();
    onFrame(S.simUser.frame(t), t);
  }, 33);
  begin();
}

// ---------- camera + face landmarks ----------
let visionLib = null;
let fileset = null;
let landmarker = null;
let lastVideoTime = -1;
let detectErrors = 0;
let importTries = 0;

async function loadLandmarker() {
  // A failed module import is remembered by the browser, so a retry needs a fresh URL.
  if (!visionLib) visionLib = await import(MP_BASE + '/vision_bundle.mjs' + (importTries++ ? `?retry=${importTries}` : ''));
  if (!fileset) fileset = await visionLib.FilesetResolver.forVisionTasks(MP_BASE + '/wasm');
  const make = (delegate) => visionLib.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  const old = landmarker;
  landmarker = null;
  if (old) {
    try { old.close(); } catch (e) { /* already closed */ }
  }
  if (S.settings.gpu) {
    try {
      landmarker = await make('GPU');
      S.delegate = 'GPU';
      return;
    } catch (e) {
      S.settings.gpu = false;
    }
  }
  landmarker = await make('CPU');
  S.delegate = 'CPU';
}

// ---------- eye images ----------
// Each eye is cut out of the camera frame, turned level with its corner line and scaled to one
// size, so the picture depends on where the eye looks and as little as possible on how the
// phone is held. Right eye in the left half of the canvas, left eye in the right half.
const HI_W = 48;
const HI_H = 24;
const CROP_SPAN = 1.5; // cut-out width, in corner-to-corner eye lengths
const cropCanvas = document.createElement('canvas');
cropCanvas.width = 2 * HI_W;
cropCanvas.height = HI_H;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
let cropFailures = 0;

function eyeImages(video, eyes) {
  if (!cropCtx || !eyes || cropFailures > 20) return null;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  try {
    cropCtx.setTransform(1, 0, 0, 1, 0, 0);
    cropCtx.fillStyle = '#808080';
    cropCtx.fillRect(0, 0, 2 * HI_W, HI_H);
    cropCtx.imageSmoothingEnabled = true;
    cropCtx.imageSmoothingQuality = 'high';
    for (let e = 0; e < 2; e++) {
      const { cx, cy, len, angle } = eyes[e];
      if (!(len > 4)) return null;
      const sx0 = Math.max(0, Math.floor(cx - len));
      const sy0 = Math.max(0, Math.floor(cy - len));
      const sw = Math.min(vw, Math.ceil(cx + len)) - sx0;
      const sh = Math.min(vh, Math.ceil(cy + len)) - sy0;
      if (sw < 8 || sh < 8) return null;
      const k = HI_W / (CROP_SPAN * len);
      const ca = k * Math.cos(-angle);
      const sa = k * Math.sin(-angle);
      cropCtx.save();
      cropCtx.beginPath();
      cropCtx.rect(e * HI_W, 0, HI_W, HI_H);
      cropCtx.clip();
      // maps the eye centre to the middle of its half and the corner line to horizontal
      cropCtx.setTransform(ca, sa, -sa, ca, e * HI_W + HI_W / 2 - (ca * cx - sa * cy), HI_H / 2 - (sa * cx + ca * cy));
      cropCtx.drawImage(video, sx0, sy0, sw, sh, sx0, sy0, sw, sh);
      cropCtx.restore();
    }
    const px = cropCtx.getImageData(0, 0, 2 * HI_W, HI_H).data;
    const hi = new Uint8Array(2 * HI_W * HI_H); // right eye then left eye, each row by row
    const pix = new Float64Array(2 * PIX_W * PIX_H);
    const pixl = new Float64Array(2 * PIX_W * PIX_H); // the same picture in log-brightness
    const bx = HI_W / PIX_W;
    const by = HI_H / PIX_H;
    for (let e = 0; e < 2; e++) {
      const lo = e * PIX_W * PIX_H;
      for (let y = 0; y < HI_H; y++) {
        for (let x = 0; x < HI_W; x++) {
          const i = (y * 2 * HI_W + e * HI_W + x) * 4;
          const g = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
          hi[e * HI_W * HI_H + y * HI_W + x] = g;
          const cell = lo + Math.floor(y / by) * PIX_W + Math.floor(x / bx);
          pix[cell] += g;
          pixl[cell] += Math.log(g + 8);
        }
      }
      // brightness and contrast of the cut-out are removed, so room lighting matters less
      for (const [arr, floor] of [[pix, bx * by], [pixl, 1e-3 * bx * by]]) {
        let mean = 0;
        for (let i = 0; i < PIX_W * PIX_H; i++) mean += arr[lo + i];
        mean /= PIX_W * PIX_H;
        let sd = 0;
        for (let i = 0; i < PIX_W * PIX_H; i++) sd += (arr[lo + i] - mean) * (arr[lo + i] - mean);
        sd = Math.sqrt(sd / (PIX_W * PIX_H));
        for (let i = 0; i < PIX_W * PIX_H; i++) arr[lo + i] = sd > floor ? (arr[lo + i] - mean) / sd : 0;
      }
    }
    cropFailures = 0;
    return { pix, pixl, hi };
  } catch (e) {
    cropFailures++;
    return null;
  }
}

function loop() {
  requestAnimationFrame(loop);
  const video = $('#cam');
  if (!landmarker || document.hidden || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const t = performance.now();
  let res;
  try {
    res = landmarker.detectForVideo(video, t);
    detectErrors = 0;
    S.lastResult = res;
  } catch (e) {
    if (++detectErrors === 30) toast('تعطّل تتبع الوجه. أعد تحميل الصفحة');
    return;
  }
  const f = extractFeatures(res, video.videoWidth, video.videoHeight);
  if (f.face) {
    // the full feature vector: the numbers from the face landmarks, then the eye pictures in both forms
    const img = eyeImages(video, f.eyes);
    const phi = new Float64Array(N_FULL);
    phi.set(f.phi);
    if (img) {
      phi.set(img.pix, OFF_PIX);
      phi.set(img.pixl, OFF_LOG);
    }
    f.phi = phi;
    f.hi = img ? img.hi : null;
  }
  S.video = { w: video.videoWidth, h: video.videoHeight };
  onFrame(f, t);
}

let motionOn = false;

function enableMotion() {
  // Must run inside a tap: iOS only shows the motion permission prompt then.
  try {
    const DM = window.DeviceMotionEvent;
    if (!DM) return Promise.resolve(false);
    const listen = () => {
      if (motionOn) return true; // one listener only: a second one would sum every turn twice
      motionOn = true;
      window.addEventListener('devicemotion', (e) => {
        const r = e.rotationRate;
        if (!r) return;
        const now = performance.now();
        const dt = Math.min(0.1, Math.max(0, (now - S.motionT) / 1000));
        S.motion = Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0);
        S.motionT = now;
        // Sum the turn: rates are in degrees per second around the phone's own x (beta), y (gamma), z (alpha).
        const k = Math.PI / 180;
        const wx = (r.beta || 0) * k;
        const wy = (r.gamma || 0) * k;
        const wz = (r.alpha || 0) * k;
        const w = Math.hypot(wx, wy, wz);
        if (w > 1e-6 && dt > 0) {
          const h = (w * dt) / 2;
          const sn = Math.sin(h) / w;
          const [qx, qy, qz, qw] = S.rot;
          const dx = wx * sn;
          const dy = wy * sn;
          const dz = wz * sn;
          const dw = Math.cos(h);
          const n = [qw * dx + qx * dw + qy * dz - qz * dy, qw * dy - qx * dz + qy * dw + qz * dx, qw * dz + qx * dy - qy * dx + qz * dw, qw * dw - qx * dx - qy * dy - qz * dz];
          const len = Math.hypot(n[0], n[1], n[2], n[3]) || 1;
          S.rot = n.map((v) => v / len);
        }
        const g = e.accelerationIncludingGravity;
        if (g && g.x != null) S.grav = [g.x, g.y, g.z];
      });
      return true;
    };
    if (typeof DM.requestPermission === 'function') {
      return DM.requestPermission().then((r) => (r === 'granted' ? listen() : false)).catch(() => false);
    }
    return Promise.resolve(listen());
  } catch (e) {
    return Promise.resolve(false);
  }
}

async function keepAwake() {
  try {
    if (navigator.wakeLock && !document.hidden) await navigator.wakeLock.request('screen');
  } catch (e) { /* optional */ }
}

// ---------- start overlay: idle, loading and error states ----------
const START_ERRORS = {
  insecure: 'الكاميرا تعمل على روابط https فقط. افتح الصفحة من رابط آمن في Safari.',
  unsupported: 'هذا المتصفح لا يتيح الكاميرا للصفحات. افتح الرابط في Safari نفسه.',
  denied: 'الكاميرا محجوبة هنا. داخل معاينة أي تطبيق لا تعمل مهما كانت الإعدادات: افتح الصفحة من رابط https في Safari نفسه. وفي Safari اسمح بالكاميرا لهذا الموقع ثم أعد المحاولة.',
  nocam: 'لم أجد كاميرا أمامية في هذا الجهاز.',
  camera: 'تعذّر تشغيل الكاميرا.',
  nomodel: 'تعذّر تحميل نموذج تتبع الوجه. تحقق من الاتصال ثم أعد المحاولة.',
};
const START_LOADING = {
  cam: 'في انتظار إذن الكاميرا…',
  model: 'تحميل نموذج تتبع الوجه… قد يستغرق نصف دقيقة في أول مرة',
};

function renderStart(state = 'idle', detail = '') {
  let ov = $('#start');
  if (!ov) {
    ov = el('<div id="start" class="overlay"></div>');
    document.body.appendChild(ov);
  }
  const intro = S.sim
    ? `<h2>معايرة النظر<br>وضع المحاكاة</h2>
       <ul><li>لا كاميرا هنا: مستخدم افتراضي ينظر حيث تلمس أو حيث تحرّك المؤشر.</li>
       <li>الأرقام في هذا الوضع لفحص التطبيق فقط وليست قياسًا حقيقيًا.</li></ul>`
    : `<h2>معايرة النظر<br>بالكاميرا الأمامية</h2>
       <ul><li>أمسك الهاتف على بعد 30 سم تقريبًا، في إضاءة جيدة على وجهك.</li>
       <li>الصورة تُعالج داخل المتصفح فقط. لا تُرفع ولا تُحفظ.</li>
       <li>النتائج تقريبية، وتتحسن كلما استخدمته.</li></ul>`;
  const err = START_ERRORS[state];
  // In a preview the camera cannot work, so the no-camera demo becomes the main action.
  const preview = EMBEDDED && !S.sim && !err;
  let body;
  if (START_LOADING[state]) {
    body = `<div class="loading"><div class="spinner"></div><span>${START_LOADING[state]}</span></div>`;
  } else if (S.sim) {
    body = '<button class="btn primary" id="start-go">بدء المحاكاة</button>';
  } else {
    body = `${err ? `<p class="err">${err}</p>` : ''}${detail ? `<p class="detail">${esc(detail)}</p>` : ''}
      ${preview ? '<p class="warn">هذه معاينة داخل تطبيق، والكاميرا محجوبة فيها. للقياس الحقيقي افتح الصفحة من رابط https في Safari.</p>' : ''}
      ${preview
        ? '<button class="btn primary" id="start-sim">تجربة الواجهة بلا كاميرا</button><button class="btn quiet" id="start-go">تشغيل الكاميرا على أي حال</button>'
        : `<button class="btn primary" id="start-go">${err ? 'إعادة المحاولة' : 'ابدأ'}</button><button class="btn quiet" id="start-sim">تجربة الواجهة بلا كاميرا</button>`}`;
  }
  ov.innerHTML = `<div class="inner">${intro}${body}<p class="note">GazeCal web ${VERSION}</p></div>`;
  const go = $('#start-go', ov);
  if (go) go.addEventListener('click', start);
  const sim = $('#start-sim', ov);
  if (sim) sim.addEventListener('click', enterSim);
}

function enterSim() {
  const video = $('#cam');
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
  S.sim = true;
  S.delegate = 'sim';
  startSim();
}

async function start() {
  if (S.sim) return startSim();
  if (!window.isSecureContext) return renderStart('insecure');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return renderStart('unsupported');
  renderStart('cam');
  // The gyroscope tells a shaken phone from a moving eye. iOS only offers it inside a tap.
  S.settings.motion = await enableMotion();
  const video = $('#cam');
  try {
    if (!video.srcObject) {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      });
    }
    await video.play();
  } catch (e) {
    const name = e && e.name;
    const kind = name === 'NotAllowedError' || name === 'SecurityError' ? 'denied'
      : name === 'NotFoundError' || name === 'OverconstrainedError' ? 'nocam' : 'camera';
    return renderStart(kind, kind === 'camera' ? String((e && e.message) || e) : '');
  }
  renderStart('model');
  try {
    load();
    await loadLandmarker();
  } catch (e) {
    return renderStart('nomodel', String((e && e.message) || e));
  }
  requestAnimationFrame(loop);
  begin();
}

function begin() {
  S.running = true;
  const ov = $('#start');
  if (ov) ov.remove();
  S.cal.learning = S.settings.learn;
  go('home');
  setInterval(tick, 300);
  keepAwake();
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden || !S.running) return;
  keepAwake();
  const video = $('#cam');
  if (video.srcObject && video.paused) video.play().catch(() => {});
});

// ---------- status ----------
function tick() {
  const chip = $('#chip');
  const stale = performance.now() - S.lastFrameT > 1500;
  let cls = 'ok';
  let text;
  if (S.sim) {
    text = 'محاكاة';
    cls = 'warn';
  } else if (stale) {
    text = 'الكاميرا متوقفة';
    cls = 'bad';
  } else if (!S.face) {
    text = 'لا يوجد وجه';
    cls = 'bad';
  } else {
    text = `${Math.round(S.fps)} إطار/ث`;
    if (S.fps < 12) cls = 'warn';
  }
  chip.textContent = text;
  chip.className = 'chip ' + cls;
  // slow upkeep of the models, kept away from the recording screen where a hiccup would cost frames
  if (S.screen !== 'pursue' && S.screen !== 'pursueMove') S.cal.tick(Date.now());

  const st = $('#st-state');
  if (!st) return;
  st.textContent = S.sim ? 'محاكاة' : stale ? 'الكاميرا متوقفة' : !S.face ? 'لا يوجد وجه' : S.cal.ready ? 'يتتبع النظر' : 'ينتظر المعايرة';
  $('#st-n').textContent = String(S.cal.samples.length);
  const e = S.errMap.recentMedian(10);
  $('#st-err').textContent = Number.isFinite(e) ? `${cm(e)} سم` : Number.isFinite(S.looPx) ? `≈ ${cm(S.looPx)} سم` : '—';
  const d = S.distCm;
  $('#st-dist').textContent = Number.isFinite(d) ? `≈ ${Math.round(d)} سم${d < 22 ? ' · قريب' : d > 45 ? ' · بعيد' : ''}` : '—';
  placeCam();
}

// ---------- navigation ----------
function go(name) {
  if (S.cleanup) {
    try { S.cleanup(); } catch (e) { /* screen already gone */ }
    S.cleanup = null;
  }
  S.onGaze = null;
  S.onSample = null;
  S.selector = null;
  S.screen = name;
  $('#app').classList.toggle('immersive', !!IMMERSIVE[name]);
  $('#back').hidden = name === 'home';
  $('#title').textContent = TITLES[name];
  const stage = $('#stage');
  stage.textContent = '';
  stage.appendChild(SCREENS[name]());
  placeCam();
  drawCursor();
  tick();
}

// The live camera picture sits exactly over its slot on the home screen and
// shrinks to a 2px speck everywhere else (it has to stay on the page to keep playing).
function placeCam() {
  const video = $('#cam');
  const slot = $('#cam-slot');
  const show = !!slot && !slot.hidden && S.screen === 'home';
  video.classList.toggle('thumb', show);
  if (!show) {
    video.removeAttribute('style');
    return;
  }
  const r = slot.getBoundingClientRect();
  video.style.left = `${r.left}px`;
  video.style.top = `${r.top}px`;
  video.style.width = `${r.width}px`;
  video.style.height = `${r.height}px`;
}

document.addEventListener('click', (e) => {
  const b = e.target.closest ? e.target.closest('[data-go]') : null;
  if (b && !b.disabled) go(b.dataset.go);
});

// ---------- home ----------
function home() {
  const ready = S.cal.ready;
  const root = el(`<section class="screen home">
    <div class="card status">
      <div class="rows">
        <div class="row"><span>الحالة</span><b id="st-state">—</b></div>
        <div class="row"><span>العينات</span><b id="st-n">0</b></div>
        <div class="row"><span>الخطأ الأخير</span><b id="st-err">—</b></div>
        <div class="row"><span>المسافة</span><b id="st-dist">—</b></div>
        ${S.sim ? '' : `<div class="row" id="eye-row" ${S.settings.preview ? '' : 'hidden'}><span>ما يراه النموذج</span><span class="eyes" id="eye-slot"></span></div>`}
      </div>
      <div class="slot" id="cam-slot" ${S.settings.preview && !S.sim ? '' : 'hidden'}></div>
    </div>
    <div class="scroll">
      <button class="btn ${ready ? '' : 'primary'}" data-go="pursue"><span>${ready ? 'معايرة بالتتبّع' : 'ابدأ: معايرة بالتتبّع'}</span><small>20 ثانية، الهاتف ثابت</small></button>
      <button class="btn" data-go="pursueMove"><span>معايرة متحركة</span><small>30 ثانية وأنت تحرّك الهاتف · تقيس أثر الحركة</small></button>
      <button class="btn" data-go="boot"><span>معايرة 9 نقاط</span><small>تضيف 9 عينات</small></button>
      <button class="btn" data-go="test" ${ready ? '' : 'disabled'}><span>قياس الدقة</span><small>20 نقطة</small></button>
      <button class="btn" data-go="free"><span>استخدام حر</span><small>يتعلّم من لمساتك</small></button>
      <button class="btn" data-go="pick" ${ready ? '' : 'disabled'}><span>نظر + لمس</span><small>اختيار بالعين</small></button>
      <button class="btn" data-go="stats"><span>النتائج والتصدير</span><small></small></button>
      ${ready ? '' : '<p class="note">«قياس الدقة» و«نظر + لمس» يُفتحان بعد المعايرة.</p>'}
      <div class="card settings">
        <div class="row"><span>إظهار مؤشر النظر</span><button class="switch" role="switch" data-set="cursor" aria-label="إظهار مؤشر النظر"></button></div>
        <div class="row"><span>التعلّم من اللمس</span><button class="switch" role="switch" data-set="learn" aria-label="التعلّم من اللمس"></button></div>
        ${S.sim ? '' : `<div class="row"><span>صورة الكاميرا</span><button class="switch" role="switch" data-set="preview" aria-label="صورة الكاميرا"></button></div>
        <div class="row"><span>حسّاس الحركة</span><button class="switch" role="switch" data-set="motion" aria-label="حسّاس الحركة"></button></div>
        <div class="row"><span>معالجة GPU (تجريبي)</span><button class="switch" role="switch" data-set="gpu" aria-label="معالجة GPU"></button></div>`}
      </div>
    </div>
  </section>`);

  const eyeSlot = $('#eye-slot', root);
  if (eyeSlot) eyeSlot.appendChild(cropCanvas);

  root.querySelectorAll('[data-set]').forEach((b) => {
    const k = b.dataset.set;
    b.setAttribute('aria-checked', String(!!S.settings[k]));
    b.addEventListener('click', async () => {
      const on = !S.settings[k];
      if (k === 'motion' && on) {
        const granted = await enableMotion();
        if (!granted) {
          toast('لم يُسمح بحسّاس الحركة');
          return;
        }
      }
      S.settings[k] = on;
      b.setAttribute('aria-checked', String(on));
      if (k === 'learn') S.cal.learning = on;
      if (k === 'preview') {
        $('#cam-slot', root).hidden = !on;
        const eyeRow = $('#eye-row', root);
        if (eyeRow) eyeRow.hidden = !on;
        placeCam();
      }
      if (k === 'cursor') drawCursor();
      if (k === 'gpu') {
        b.disabled = true;
        try {
          await loadLandmarker();
          toast(S.delegate === 'GPU' ? 'المعالجة على GPU' : on ? 'GPU غير متاح، عدت إلى CPU' : 'المعالجة على CPU');
        } catch (e) {
          toast('تعذّر تبديل المعالج');
        }
        b.disabled = false;
        b.setAttribute('aria-checked', String(!!S.settings.gpu));
      }
      save();
    });
  });
  return root;
}

// ---------- shared dot runner (calibration and accuracy test) ----------
function runDots(root, points, { src, force, onSample, onDone }) {
  let i = 0;
  let dot = null;
  let shownAt = 0;
  let stopped = false;
  const prog = $('.prog', root);

  function show() {
    if (stopped) return;
    if (i >= points.length) {
      prog.textContent = '';
      onDone();
      return;
    }
    const [x, y] = points[i];
    prog.textContent = `${i + 1} / ${points.length}`;
    dot = el('<div class="dot" data-gaze="dot"><i></i></div>');
    dot.style.left = `${x}px`;
    dot.style.top = `${y}px`;
    root.appendChild(dot);
    shownAt = performance.now();
    if (S.simUser) S.simUser.lookAt(x, y, shownAt);
    dot.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      const t = eventTime(e);
      if (t - shownAt < 450) {
        toast('انظر إلى النقطة أولًا ثم المسها');
        return;
      }
      const rec = takeSample([x, y], src, t, 1, { force, since: shownAt + 250 });
      if (!rec.ok) {
        toast(rec.reason === 'blink' ? 'رمشت لحظة اللمس. أعد المحاولة' : 'لا أرى وجهك بوضوح. عدّل وضعية الهاتف');
        return;
      }
      i++;
      dot.remove();
      dot = null;
      const wait = onSample ? onSample(rec, [x, y]) : 0;
      setTimeout(show, wait);
    });
  }

  show();
  return () => {
    stopped = true;
    if (dot) dot.remove();
  };
}

// ---------- 1. first calibration ----------
function boot() {
  const root = el(`<section class="screen full">
    <button class="close" data-gaze="ignore" data-go="home" aria-label="إغلاق">✕</button>
    <div class="hud hint">انظر إلى النقطة الخضراء ثم المسها</div>
    <div class="hud prog"></div>
  </section>`);
  const xs = [0.1, 0.5, 0.9];
  const ys = [0.09, 0.5, 0.91];
  const rest = [];
  for (const yn of ys) for (const xn of xs) if (!(xn === 0.5 && yn === 0.5)) rest.push([xn * S.W, yn * S.H]);
  const points = [[0.5 * S.W, 0.5 * S.H], ...shuffle(rest)];

  S.cleanup = runDots(root, points, {
    src: 'bootstrap',
    force: true,
    onSample: () => {
      $('.hint', root).hidden = true;
      return 120;
    },
    onDone: () => {
      S.cal.flush(Date.now());
      S.cal.freeze();
      const loo = S.cal.looErrors(40, Date.now()); // only meaningful while these dots are most of the memory
      if (loo.length) S.looPx = median(loo);
      save();
      root.appendChild(el(S.cal.ready
        ? `<div class="sheet">
            <h2>تمت معايرة 9 نقاط</h2>
            ${loo.length ? `<div class="big">${cm(S.looPx)} <small>سم خطأ متوقع، تقدير من 9 نقاط</small></div>` : `<div class="big">${S.cal.samples.length} <small>عينة في الذاكرة الآن</small></div>`}
            <p class="note">9 نقاط لا تكفي نموذج صورة العين. «معايرة بالتتبّع» تعطيه مئات العينات في 20 ثانية.</p>
            <button class="btn primary" data-go="pursue">معايرة بالتتبّع</button>
            <div class="btn-row"><button class="btn center" data-go="test">قياس الدقة</button><button class="btn center" data-go="home">الرئيسية</button></div>
          </div>`
        : `<div class="sheet">
            <h2>تعذّرت المعايرة</h2>
            <p class="note">لم تكفِ العينات لبناء النموذج. أعد المحاولة في إضاءة أفضل.</p>
            <button class="btn primary" data-go="boot">إعادة المحاولة</button>
            <button class="btn center" data-go="home">الرئيسية</button>
          </div>`));
    },
  });
  return root;
}

// ---------- 1b. follow-the-dot calibration ----------
// Twenty seconds of following one moving dot give several hundred labelled frames:
// enough for the models that learn from the eye pictures, which nine dots cannot feed.
// In the moving variant the user also keeps shifting the phone, so the frames cover
// many distances and tilts instead of one posture.
const spreadOf = (values) => {
  const a = values.filter(Number.isFinite);
  return a.length ? [percentile(a, 0.05), percentile(a, 0.95)] : [NaN, NaN];
};

// What the moving recordings in memory say about the head's share of the gaze point. Applied only
// when at least one part of the head really moved and the held-out error went down.
function measureGeometry() {
  const G = MODEL_SPECS.find((m) => m.name === 'G');
  const pool = S.cal.samples.filter((sm) => sm.src === 'pursuit-move' && sm.phi.length >= G.needs);
  const r = measureHeadShares(pool, { expand: G.expand, nFeat: G.nFeat, stdFloor: G.stdFloor, parts: headParts, prior: DEFAULT_SHARES });
  if (!r) return null;
  const applied = r.measured.some(Boolean) && r.after < r.before;
  if (applied) {
    setGeometry({ shares: r.shares });
    S.cal.refit({ cv: true, nowT: Date.now(), only: (m) => !!m.offset });
  }
  return { ...r, before: Math.round(r.before * 10) / 10, after: Math.round(r.after * 10) / 10, applied, at: new Date().toISOString() };
}

function learnPursuit(frames, path, tStart, { move, duration }) {
  const usable = frames.filter((f) => f.face && f.phi && !(f.blink > 0.5) && f.t > 800);
  const out = { ok: false, usable: usable.length, total: frames.length };
  if (usable.length < 200) return out;
  const lag = estimateLag(usable, path, MODEL_SPECS.filter((m) => m.name === 'B' || m.name === 'D'));
  const samples = pursuitSamples(frames, path, { lagMs: lag.lagMs, wallT0: wall(tStart), src: move ? 'pursuit-move' : 'pursuit' });
  if (samples.length < 80) return out;
  S.cal.addBatch(samples, { nowT: Date.now() });
  let geo = null;
  if (move) {
    try {
      geo = measureGeometry();
    } catch (e) {
      geo = null;
    }
    if (geo) S.geo = geo;
  }
  S.cal.freeze();
  S.smooth.reset();
  S.anchor = { t: performance.now(), pose: S.cal.pose ? Float64Array.from(S.cal.pose) : null };
  // how much of the space of postures this recording covered
  const dist = spreadOf(usable.map((f) => f.phi[24]));
  const deg = (v) => (Math.atan(v) * 180) / Math.PI;
  const tiltX = spreadOf(usable.map((f) => deg(f.phi[25])));
  const tiltY = spreadOf(usable.map((f) => deg(f.phi[26])));
  const spread = { distCm: dist.map((v) => Math.round(v * 10) / 10), tiltDeg: Math.round(Math.max(tiltX[1] - tiltX[0], tiltY[1] - tiltY[0]) * 10) / 10 };
  S.pursuitLog.push({
    at: new Date(wall(tStart)).toISOString(),
    mode: move ? 'moving' : 'still',
    durationMs: duration,
    path: { W: S.W, H: S.H, marginX: 0.1, marginY: 0.08, cyclesX: move ? 7 : 5, cyclesY: move ? 3 : 2, formula: 'x = W/2 + W(0.5-marginX) sin(2π cyclesX u), y = H/2 + H(0.5-marginY) cos(2π cyclesY u), u = t/durationMs' },
    lagMs: lag.lagMs,
    lagSure: lag.sure,
    lagModel: lag.model,
    lagCurve: lag.curve.map((v) => Math.round(v * 10) / 10),
    samples: samples.length,
    spread,
    geometry: geo, // the head shares measured from the moving recordings in memory (null for a still one)
    // every frame keeps its numbers and the plain eye picture; every second one also the larger picture
    frames: frames.map((f, i) => ({
      t: Math.round(f.t),
      k: Math.round(f.blink * 100) / 100,
      m: Math.round(f.motion || 0),
      g: f.device ? f.device.rot : null, // the phone's turn (quaternion x, y, z, w) from the gyroscope
      a: f.device ? f.device.grav : null, // gravity in the phone's own axes, m/s²
      b: f.phi ? Array.from(f.phi.subarray(0, N_BASE), r5) : null,
      r: f.raw ? f.raw.map(r5) : null, // tx, ty, tz, nxz, nyz as the camera read them; `b` holds them steadied
      q: f.phi && f.phi.length >= OFF_PIX + N_PIX ? packInt8(f.phi.subarray(OFF_PIX, OFF_PIX + N_PIX)) : null,
      e: f.hi && i % 2 === 0 ? b64(f.hi) : null,
    })),
  });
  if (S.pursuitLog.length > 3) S.pursuitLog.shift();
  save();
  return { ...out, ok: S.cal.ready, n: samples.length, lag, spread, geo };
}

// The head shares in effect, as a small table: position and direction, each across and down the
// screen. A share no moving calibration has settled yet is the default, shown dimmed with a star.
function geoRows(geo, { footnote = true } = {}) {
  const sh = getGeometry().shares;
  const set = (i) => !!(geo && geo.applied && geo.measured[i]);
  const cell = (i) => `<b dir="ltr"${set(i) ? '' : ' class="unset"'}>${sh[i].toFixed(2)}${set(i) ? '' : '*'}</b>`;
  const all = [0, 1, 2, 3].every(set);
  return `<div class="geo">
      <span>أثر حركة الرأس</span><span>أفقي</span><span>عمودي</span>
      <span>موضعه</span>${cell(0)}${cell(1)}
      <span>اتجاهه</span>${cell(2)}${cell(3)}
    </div>${footnote && !all ? '<p class="note">* قيمة افتراضية لم تُقَس بعد.</p>' : ''}${sh.some((v) => v < 0)
    ? '<p class="note">القيمة السالبة تعني أن الكاميرا تعطي ذلك المحور معكوسًا. قيس ذلك وصار محسوبًا.</p>' : ''}`;
}

function pursue(move = false) {
  const duration = move ? MOVE_MS : PURSUIT_MS;
  const root = el(`<section class="screen full">
    <button class="close" data-gaze="ignore" data-go="home" aria-label="إغلاق">✕</button>
    <div class="pbar"></div>
    <div class="hud hint">${move
      ? 'المس النقطة ثم تابعها بعينيك<br><small>وحرّك الهاتف بلا توقف طوال 30 ثانية: يمينًا ويسارًا، أعلى وأسفل، قرّبه وأبعده، وأمِله.</small>'
      : 'المس النقطة ثم تابعها بعينيك فقط<br><small>20 ثانية. أبقِ الهاتف ورأسك ثابتين.</small>'}</div>
    <div class="dot" data-gaze="dot"><i></i></div>
  </section>`);
  const dot = $('.dot', root);
  const bar = $('.pbar', root);
  const path = pursuitPath(S.W, S.H, move ? { durationMs: duration, cyclesX: 7, cyclesY: 3 } : { durationMs: duration });
  const frames = [];
  let tStart = 0;
  let raf = 0;
  let gone = false;
  const place = (pt) => {
    dot.style.left = `${pt[0]}px`;
    dot.style.top = `${pt[1]}px`;
  };
  place(path(0));

  function simRest() {
    if (!S.simUser) return;
    S.simUser.stopWobble();
    S.simUser.lookAt(S.W / 2, S.H / 2, performance.now());
  }

  function finish() {
    S.onFrame = null;
    dot.remove();
    bar.remove();
    simRest();
    const busy = el('<div class="sheet"><div class="loading"><div class="spinner"></div><span>جارٍ التعلّم من التسجيل…</span></div></div>');
    root.appendChild(busy);
    setTimeout(() => {
      if (gone) return;
      let out;
      try {
        out = learnPursuit(frames, path, tStart, { move, duration });
      } catch (e) {
        out = { ok: false, usable: 0, total: frames.length, error: String((e && e.message) || e) };
      }
      busy.remove();
      const geo = out.ok ? out.geo : null;
      const all = !!geo && geo.applied && geo.measured.every(Boolean);
      root.appendChild(el(out.ok
        ? `<div class="sheet">
            <h2>${move ? 'تمت المعايرة المتحركة' : 'تمت معايرة التتبّع'}</h2>
            <div class="big">${out.n} <small>عينة جديدة من ${out.total} إطارًا</small></div>
            <div class="row"><span>تأخر العين</span><b>${out.lag.lagMs} مللي ثانية${out.lag.sure ? '' : ' (افتراضي)'}</b></div>
            <div class="row"><span>المسافة</span><b><span dir="ltr">${out.spread.distCm[0]}–${out.spread.distCm[1]}</span> سم</b></div>
            <div class="row"><span>مدى الميل</span><b><span dir="ltr">${out.spread.tiltDeg}°</span></b></div>
            ${move ? geoRows(geo, { footnote: false }) : ''}
            ${move && !all ? '<p class="warn">القيم ذات النجمة لم تُقَس: الحركة لم تكفِ. أعدها وحرّك الهاتف في كل الاتجاهات، فكل تسجيل يُضاف إلى ما قبله.</p>' : ''}
            ${move
              ? '<p class="note">الرقم الحقيقي يأتي من «قياس الدقة»: غيّر مسكة الهاتف قبله لترى أثر الحركة.</p><button class="btn primary" data-go="test">قياس الدقة</button>'
              : '<p class="note">التالي: «معايرة متحركة» تقيس كيف يتحرك وجهك أمام الكاميرا حين يتحرك الهاتف.</p><button class="btn primary" data-go="pursueMove">معايرة متحركة</button><button class="btn center" data-go="test">قياس الدقة</button>'}
            <button class="btn center" data-go="home">الرئيسية</button>
          </div>`
        : `<div class="sheet">
            <h2>لم يكتمل التسجيل</h2>
            <p class="note">رأيت عينيك بوضوح في ${out.usable} من ${out.total} إطارًا فقط. أعد المحاولة في إضاءة أفضل${move ? ' وبحركة أهدأ' : ' والهاتف ثابت'}.</p>
            ${out.error ? `<p class="detail">${esc(out.error)}</p>` : ''}
            <button class="btn primary" data-go="${move ? 'pursueMove' : 'pursue'}">إعادة المحاولة</button>
            <button class="btn center" data-go="home">الرئيسية</button>
          </div>`));
    }, 60);
  }

  function step() {
    if (gone) return;
    const u = performance.now() - tStart;
    if (u >= duration) return finish();
    place(path(Math.max(0, u)));
    bar.style.width = `${(100 * Math.max(0, u)) / duration}%`;
    raf = requestAnimationFrame(step);
  }

  dot.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (tStart) return;
    // The finger is on the dot, so the eyes are too. It rests a moment, then starts to move.
    tStart = performance.now() + 700;
    $('.hint', root).hidden = true;
    dot.classList.add('moving');
    if (S.simUser) {
      S.simUser.follow((t) => path(t - tStart), 120);
      if (move) S.simUser.startWobble(tStart);
    }
    S.onFrame = (f, t) => {
      if (t >= tStart) frames.push({ t: t - tStart, face: !!f.face, phi: f.face ? f.phi : null, raw: f.face ? f.rawHead : null, blink: f.blink || 0, hi: f.hi || null, motion: t - S.motionT < 400 ? S.motion : 0, device: f.device });
    };
    raf = requestAnimationFrame(step);
  });

  S.cleanup = () => {
    gone = true;
    cancelAnimationFrame(raf);
    S.onFrame = null;
    if (tStart) simRest();
  };
  return root;
}

// ---------- 2. accuracy test ----------
function testPoints(n) {
  const rand = mulberry32((Date.now() & 0xffff) + 1);
  const cols = 4;
  const rows = 5;
  const mx = 0.07 * S.W;
  const my = 0.07 * S.H;
  const cw = (S.W - 2 * mx) / cols;
  const ch = (S.H - 2 * my) / rows;
  const pts = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let x = mx + (c + 0.5 + (rand() - 0.5) * 0.5) * cw;
      const y = my + (r + 0.5 + (rand() - 0.5) * 0.5) * ch;
      if (x < 84 && y < 84) x = 84; // keep clear of the close button
      pts.push([x, y]);
    }
  }
  return shuffle(pts, rand).slice(0, n);
}

function test() {
  const root = el(`<section class="screen full">
    <button class="close" data-gaze="ignore" data-go="home" aria-label="إغلاق">✕</button>
    <div class="hud hint">انظر إلى كل نقطة ثم المسها</div>
    <div class="hud prog"></div>
  </section>`);
  const trials = [];

  S.cleanup = runDots(root, testPoints(20), {
    src: 'test-dot',
    force: false,
    onSample: (rec, [x, y]) => {
      $('.hint', root).hidden = true;
      trials.push(rec);
      if (!rec.pred) return 150;
      // Show where the model thought the eyes were, then move on.
      const px = clamp(rec.pred[0], 0, S.W);
      const py = clamp(rec.pred[1], 0, S.H);
      const ghost = el('<div class="ghost"></div>');
      ghost.style.left = `${px}px`;
      ghost.style.top = `${py}px`;
      const line = el('<div class="ghost-line"></div>');
      line.style.left = `${x}px`;
      line.style.top = `${y}px`;
      line.style.width = `${Math.hypot(px - x, py - y)}px`;
      line.style.transform = `rotate(${Math.atan2(py - y, px - x)}rad)`;
      const mark = el('<div class="dot" style="pointer-events:none"><i></i></div>');
      mark.style.left = `${x}px`;
      mark.style.top = `${y}px`;
      mark.style.animation = 'none';
      root.append(line, ghost, mark);
      setTimeout(() => {
        line.remove();
        ghost.remove();
        mark.remove();
      }, 430);
      return 450;
    },
    onDone: () => {
      S.cal.flush(Date.now());
      const scored = trials.filter((r) => r.err != null);
      const med = median(scored.map((r) => r.err));
      const frozen = scored.filter((r) => r.frozenErr != null).map((r) => r.frozenErr);
      const dx = median(scored.map((r) => Math.abs(r.pred[0] - r.target[0])));
      const dy = median(scored.map((r) => Math.abs(r.pred[1] - r.target[1])));
      root.appendChild(el(`<div class="sheet">
        <h2>نتيجة القياس</h2>
        <div class="big">${cm(med)} <small>سم خطأ وسيط على ${scored.length} نقطة</small></div>
        <div class="row"><span>${S.cal.learning ? 'بالمعايرة وحدها' : 'التعلّم متوقف'}</span><b>${frozen.length ? cm(median(frozen)) + ' سم' : '—'}</b></div>
        <div class="row"><span>أفقي / عمودي</span><b>${cm(dx)} / ${cm(dy)} سم</b></div>
        <div class="row"><span>أسوأ ربع من النقاط فوق</span><b>${cm(percentile(scored.map((r) => r.err), 0.75))} سم</b></div>
        <div class="row"><span>كل نموذج على النقاط نفسها (سم)</span></div>
        <p class="board" dir="ltr">${boardLine(modelBoard(scored))}</p>
        <div class="row"><span>يقود الآن</span><b>${leadLabel()}</b></div>
        <button class="btn primary" data-go="test">إعادة القياس</button>
        <div class="btn-row"><button class="btn center" data-go="stats">النتائج الكاملة</button><button class="btn center" data-go="home">الرئيسية</button></div>
      </div>`));
    },
  });
  return root;
}

// ---------- 3. free use: ordinary UI, silent learning ----------
function free() {
  const digits = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  const root = el(`<section class="screen free">
    <div class="top">
      <button class="pill" data-gaze="target" data-act="new">رقم جديد</button>
      <button class="pill" data-gaze="target" data-act="clear">مسح</button>
    </div>
    <div class="mid">
      <p class="note" id="f-label">اكتب هذا الرقم ثم اضغط «تم»</p>
      <div class="code" id="f-code"></div>
      <div class="typed" id="f-typed"></div>
    </div>
    <div class="statline" id="f-stat"></div>
    <div class="keys">
      ${digits.map((k) => `<button class="key" data-gaze="target" data-k="${k}">${k}</button>`).join('')}
      <button class="key fn" data-gaze="target" data-k="del">حذف</button>
      <button class="key" data-gaze="target" data-k="0">0</button>
      <button class="key fn" data-gaze="target" data-k="ok">تم</button>
    </div>
  </section>`);
  const codeEl = $('#f-code', root);
  const typedEl = $('#f-typed', root);
  const labelEl = $('#f-label', root);
  const mid = $('.mid', root);
  let code = '';
  let typed = '';
  let done = 0;
  let nextBtn = null;

  function stat() {
    const scored = S.log.filter((r) => r.ok && r.err != null).slice(-12);
    const frozen = scored.filter((r) => r.frozenErr != null).map((r) => r.frozenErr);
    const parts = [`العينات ${S.cal.samples.length}`];
    if (scored.length) parts.push(`الخطأ ${cm(median(scored.map((r) => r.err)))} سم`);
    if (frozen.length && S.cal.learning) parts.push(`بلا تعلّم ${cm(median(frozen))} سم`);
    if (done) parts.push(`أرقام ${done}`);
    $('#f-stat', root).textContent = parts.join(' · ');
  }

  function fresh() {
    if (nextBtn) {
      nextBtn.remove();
      nextBtn = null;
    }
    code = Array.from({ length: 6 }, () => Math.floor(Math.random() * 10)).join('');
    typed = '';
    codeEl.hidden = false;
    codeEl.textContent = code;
    typedEl.textContent = '';
    typedEl.classList.remove('wrong');
    labelEl.hidden = false;
    labelEl.textContent = 'اكتب هذا الرقم ثم اضغط «تم»';
  }

  function success() {
    done++;
    codeEl.hidden = true;
    typedEl.textContent = '';
    labelEl.hidden = true;
    // The button lands somewhere new each time, so touches cover the upper screen too.
    nextBtn = el('<button class="next" data-gaze="target">التالي</button>');
    nextBtn.style.left = `${(0.2 + 0.6 * Math.random()) * mid.clientWidth}px`;
    nextBtn.style.top = `${(0.25 + 0.6 * Math.random()) * mid.clientHeight}px`;
    nextBtn.addEventListener('click', fresh);
    mid.appendChild(nextBtn);
  }

  root.addEventListener('click', (e) => {
    const key = e.target.closest('[data-k]');
    const act = e.target.closest('[data-act]');
    if (act) {
      if (act.dataset.act === 'new') fresh();
      else if (!nextBtn) {
        typed = '';
        typedEl.textContent = '';
        typedEl.classList.remove('wrong');
      }
      return;
    }
    if (!key || nextBtn) return;
    const k = key.dataset.k;
    typedEl.classList.remove('wrong');
    if (k === 'del') typed = typed.slice(0, -1);
    else if (k === 'ok') {
      if (typed === code) return success();
      typedEl.classList.add('wrong');
      labelEl.textContent = typed.length < 6 ? 'الرقم ناقص' : 'غير مطابق. صحّحه ثم اضغط «تم»';
    } else if (typed.length < 6) typed += k;
    typedEl.textContent = typed;
  });

  S.onSample = stat;
  fresh();
  stat();
  return root;
}

// ---------- 4. look + tap: the eyes choose, the thumb confirms ----------
// Nothing on this screen is touched where the eyes look, so the model gets no fresh labels here.
// Two things bring them back: one dot to touch when the posture has changed since the last
// labelled sample, and, after a wrong choice, the user's own correcting touch on the right square.
function pick() {
  const levels = [[2, 3], [3, 4], [3, 5]];
  const root = el(`<section class="screen pick">
    <button class="close" data-gaze="ignore" data-go="home" aria-label="إغلاق">✕</button>
    <div class="levels">${levels.map((l, i) => `<button class="seg" data-gaze="ignore" data-l="${i}">${targetsLabel(l[0] * l[1])}</button>`).join('')}</div>
    <div class="grid"></div>
    <div class="bottom">
      <div class="statline" id="p-stat"></div>
      <button class="btn primary" data-gaze="ignore" id="p-go">اختر</button>
    </div>
  </section>`);
  const grid = $('.grid', root);
  const statEl = $('#p-stat', root);
  let cells = [];
  let rects = [];
  let goal = -1;
  let t0 = 0;
  let busy = true;
  let alive = true;
  let fixing = false;
  let trace = [];
  let votes = [];
  let anchorDot = null;

  function stat() {
    const [c, r] = levels[S.pickLevel];
    const mine = S.picks.filter((p) => p.cols === c && p.rows === r);
    if (!mine.length) {
      statEl.textContent = 'انظر إلى المربع الذي فيه الدائرة ثم اضغط «اختر»';
      return;
    }
    const ok = mine.filter((p) => p.ok).length;
    const sec = median(mine.map((p) => p.ms)) / 1000;
    statEl.textContent = `صحيح ${ok} من ${mine.length} · ${sec.toFixed(1)} ث للاختيار`;
  }

  // Has the head moved, or has too long passed, since the model last saw a labelled sample?
  function needAnchor() {
    const a = S.anchor;
    if (!a || performance.now() - a.t > 25000) return true;
    if (!a.pose || !S.cal.pose) return false;
    let d2 = 0;
    for (let k = 0; k < a.pose.length; k++) d2 += ((S.cal.pose[k] - a.pose[k]) / POSE_SIGMA[k]) ** 2;
    return d2 > 0.6 * 0.6;
  }

  function anchor() {
    busy = true;
    cells.forEach((c) => (c.className = 'cell'));
    const box = root.getBoundingClientRect();
    const g = grid.getBoundingClientRect();
    const x = g.left + (0.2 + 0.6 * Math.random()) * g.width;
    const y = g.top + (0.2 + 0.6 * Math.random()) * g.height;
    statEl.textContent = 'انظر إلى النقطة والمسها لإعادة الضبط';
    anchorDot = el('<div class="dot anchor" data-gaze="dot"><i></i></div>');
    anchorDot.style.left = `${x - box.left}px`;
    anchorDot.style.top = `${y - box.top}px`;
    const shownAt = performance.now();
    if (S.simUser) S.simUser.lookAt(x, y, shownAt);
    anchorDot.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      const t = eventTime(e);
      if (t - shownAt < 450) return;
      const rec = takeSample([x, y], 'anchor', t, 1, { since: shownAt + 250 });
      if (!rec.ok) {
        toast(rec.reason === 'blink' ? 'رمشت لحظة اللمس. أعد المحاولة' : 'لا أرى وجهك بوضوح');
        return;
      }
      anchorDot.remove();
      anchorDot = null;
      setTimeout(trial, 200);
    });
    root.appendChild(anchorDot);
  }

  function trial() {
    if (!alive) return;
    if (needAnchor()) return anchor();
    cells.forEach((c) => (c.className = 'cell'));
    let g = goal;
    while (g === goal) g = Math.floor(Math.random() * cells.length);
    goal = g;
    cells[goal].classList.add('goal');
    S.selector.setTargets(rects);
    t0 = performance.now();
    trace = [];
    votes = [];
    if (S.simUser) S.simUser.lookAt(rects[goal].cx, rects[goal].cy, t0);
    fixing = false;
    busy = false;
    stat();
  }

  function build() {
    busy = true;
    if (anchorDot) {
      anchorDot.remove();
      anchorDot = null;
    }
    const [c, r] = levels[S.pickLevel];
    grid.style.gridTemplateColumns = `repeat(${c}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${r}, 1fr)`;
    grid.innerHTML = Array.from({ length: c * r }, () => '<div class="cell"></div>').join('');
    cells = Array.from(grid.children);
    root.querySelectorAll('.seg').forEach((seg, i) => seg.setAttribute('aria-pressed', String(i === S.pickLevel)));
    goal = -1;
    stat();
    requestAnimationFrame(() => {
      if (!alive) return;
      rects = cells.map((cell, i) => {
        const b = cell.getBoundingClientRect();
        return { id: i, cx: b.left + b.width / 2, cy: b.top + b.height / 2, w: b.width, h: b.height };
      });
      S.selector = new TargetSelector();
      trial();
    });
  }

  root.querySelectorAll('.seg').forEach((seg) => seg.addEventListener('click', () => {
    S.pickLevel = Number(seg.dataset.l);
    build();
  }));

  $('#p-go', root).addEventListener('click', () => {
    if (busy || fixing) return;
    const best = S.gaze && S.selector ? S.selector.best() : null;
    if (!best) {
      toast('لا أرى عينيك الآن');
      return;
    }
    busy = true;
    const now = performance.now();
    const [c, r] = levels[S.pickLevel];
    const ok = best.index === goal;
    // what the eyes had chosen 0.3 s before the tap, in case they leave for the button at the end
    let early = null;
    for (const v of votes) if (v[0] <= now - t0 - 300) early = v[1];
    const g = rects[goal];
    const lead = S.cal.activeModel();
    S.picks.push({
      t: Date.now(), cols: c, rows: r, goal, chosen: best.index, ok, early, okEarly: early === null ? null : early === goal,
      ms: Math.round(now - t0), p: +best.p.toFixed(2), sigma: Math.round(sigmaPx()),
      errPx: S.gaze ? Math.round(Math.hypot(S.gaze.x - g.cx, S.gaze.y - g.cy)) : null,
      lead: lead ? lead.name : null,
      sinceAnchorMs: S.anchor ? Math.round(now - S.anchor.t) : null,
      cell: [Math.round(g.w), Math.round(g.h)], goalAt: [Math.round(g.cx), Math.round(g.cy)],
      trace, // [ms since the squares appeared, gaze x, gaze y]
    });
    cells.forEach((cell) => cell.classList.remove('sel'));
    cells[best.index].classList.add(ok ? 'right' : 'wrong');
    stat();
    if (ok) {
      setTimeout(trial, 600);
      return;
    }
    fixing = true;
    statEl.textContent = 'اختيار خاطئ: المس المربع الصحيح';
  });

  grid.addEventListener('pointerdown', (e) => {
    if (!fixing) return;
    const cell = e.target.closest ? e.target.closest('.cell') : null;
    if (!cell || cells.indexOf(cell) !== goal) return;
    const g = rects[goal];
    const rec = takeSample([g.cx, g.cy], 'pick-fix', eventTime(e), 0.9);
    S.picks[S.picks.length - 1].fixed = !!rec.ok;
    fixing = false;
    setTimeout(trial, 300);
  });

  S.onGaze = (t) => {
    if (busy || !S.selector) return;
    const best = S.gaze ? S.selector.best() : null;
    for (let i = 0; i < cells.length; i++) cells[i].classList.toggle('sel', !!best && best.index === i);
    if (S.gaze && trace.length < 240) trace.push([Math.round(t - t0), Math.round(S.gaze.x), Math.round(S.gaze.y)]);
    if (best) votes.push([t - t0, best.index]);
  };
  S.cleanup = () => {
    alive = false;
  };
  build();
  return root;
}

// ---------- 5. results and export ----------
function summary() {
  const scored = S.log.filter((r) => r.ok && r.err != null && r.src !== 'bootstrap');
  const last = scored.slice(-20);
  const frozen = last.filter((r) => r.frozenErr != null).map((r) => r.frozenErr);
  const bySrc = {};
  for (const r of S.log) if (r.ok) bySrc[r.src] = (bySrc[r.src] || 0) + 1;
  return {
    samples: S.cal.samples.length,
    scored: scored.length,
    medErrPx: median(last.map((r) => r.err)),
    medFrozenPx: median(frozen),
    medDxPx: median(last.map((r) => Math.abs(r.pred[0] - r.target[0]))),
    medDyPx: median(last.map((r) => Math.abs(r.pred[1] - r.target[1]))),
    p75Px: percentile(last.map((r) => r.err), 0.75),
    looPx: S.looPx,
    bySrc,
    pursuit: S.cal.samples.filter((x) => x.src.startsWith('pursuit')).length,
    models: modelBoard(last).map((b) => ({ name: b.name, n: b.n, medPx: b.med, lead: b.lead, ready: b.ready })),
    lead: S.cal.activeModel() ? S.cal.activeModel().name : null,
    headShares: getGeometry().shares,
    headSharesMeasured: S.geo && S.geo.applied ? S.geo.measured : [false, false, false, false],
    rejects: { ...S.rejects },
    fps: Math.round(S.fps),
  };
}

function pickSummary() {
  const groups = {};
  for (const p of S.picks) {
    const k = targetsLabel(p.cols * p.rows);
    groups[k] = groups[k] || { n: 0, ok: 0 };
    groups[k].n++;
    if (p.ok) groups[k].ok++;
  }
  return groups;
}

function summaryText() {
  const s = summary();
  const picks = Object.entries(pickSummary()).map(([k, g]) => `${k}: ${g.ok}/${g.n}`).join(' · ');
  const srcNames = { bootstrap: '9 نقاط', 'test-dot': 'نقاط القياس', 'touch-target': 'لمس أهداف', 'touch-free': 'لمس حر', anchor: 'إعادة ضبط', 'pick-fix': 'تصحيح اختيار' };
  return [
    `GazeCal web ${VERSION}${S.sim ? ' (محاكاة)' : ''}`,
    `الشاشة ${S.W}×${S.H} · DPR ${window.devicePixelRatio || 1} · ${s.fps} إطار/ث · ${S.delegate}`,
    `الكاميرا ${S.video ? `${S.video.w}×${S.video.h}` : '—'}`,
    `عينات المعايرة: ${s.samples} (${[...(s.pursuit ? [`تتبّع ${s.pursuit}`] : []), ...Object.entries(s.bySrc).map(([k, v]) => `${srcNames[k] || k} ${v}`)].join('، ') || '—'})`,
    `مرفوضة: رمش ${s.rejects.blink || 0}، بلا وجه ${s.rejects['no-face'] || 0}`,
    `تقدير بعد 9 نقاط: ${cm(s.looPx)} سم`,
    `الخطأ الوسيط (آخر ${Math.min(20, s.scored)}): ${cm(s.medErrPx)} سم · بالمعايرة وحدها: ${cm(s.medFrozenPx)} سم`,
    `النماذج على العينات نفسها (سم): ${s.models.filter((m) => m.n).map((m) => `${m.name} ${cm(m.medPx)}`).join(' · ') || '—'} · يقود ${s.lead || '—'}`,
    `أفقي / عمودي: ${cm(s.medDxPx)} / ${cm(s.medDyPx)} سم · الربع الأسوأ فوق ${cm(s.p75Px)} سم`,
    `أثر الرأس (موضع أفقي، عمودي · اتجاه أفقي، عمودي): ${s.headShares.map((v, i) => `${v.toFixed(2)}${s.headSharesMeasured[i] ? '' : '*'}`).join('، ')}${s.headSharesMeasured.every(Boolean) ? '' : ' (* افتراضي لم يُقَس)'}`,
    `نظر + لمس: ${picks || '—'}`,
    `(${S.pxPerCm} px لكل سم، تقدير)`,
  ].join('\n');
}

function rolling(values, k = 8) {
  return values.map((_, i) => median(values.slice(Math.max(0, i - k + 1), i + 1)));
}

function curveSvg() {
  const scored = S.log.filter((r) => r.ok && r.err != null && r.src !== 'bootstrap');
  if (scored.length < 3) return '<p class="note">يظهر المنحنى بعد 3 عينات مقيسة على الأقل.</p>';
  const a = rolling(scored.map((r) => r.err / S.pxPerCm));
  const f = rolling(scored.map((r) => (r.frozenErr != null ? r.frozenErr : r.err) / S.pxPerCm));
  const top = Math.max(1, Math.ceil(Math.max(...a, ...f)));
  const Wd = 300;
  const Hd = 120;
  const pad = 6;
  const pts = (arr) => arr.map((v, i) => `${(pad + (i / (arr.length - 1)) * (Wd - 2 * pad)).toFixed(1)},${(Hd - pad - (v / top) * (Hd - 2 * pad)).toFixed(1)}`).join(' ');
  return `<svg class="curve" viewBox="0 0 ${Wd} ${Hd}" preserveAspectRatio="none" role="img" aria-label="منحنى الخطأ">
      <line x1="${pad}" y1="${Hd - pad}" x2="${Wd - pad}" y2="${Hd - pad}" stroke="#273140" stroke-width="1"/>
      <line x1="${pad}" y1="${pad}" x2="${Wd - pad}" y2="${pad}" stroke="#273140" stroke-width="1" stroke-dasharray="3 4"/>
      <polyline points="${pts(f)}" fill="none" stroke="#8b97a8" stroke-width="2" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>
      <polyline points="${pts(a)}" fill="none" stroke="#60a5fa" stroke-width="2.5" vector-effect="non-scaling-stroke"/>
    </svg>
    <div class="legend"><span><i style="background:#60a5fa"></i>مع التعلّم</span><span><i style="background:#8b97a8"></i>المعايرة وحدها</span><span>أعلى الرسم = ${top} سم</span></div>`;
}

function heatHtml() {
  const color = (c) => (c <= 1 ? '#4ade80' : c <= 2 ? '#fbbf24' : c <= 3 ? '#fb923c' : '#f87171');
  let html = `<div class="heat" style="grid-template-columns:repeat(${S.errMap.cols},1fr)">`;
  for (let r = 0; r < S.errMap.rows; r++) {
    for (let c = 0; c < S.errMap.cols; c++) {
      const cell = S.errMap.cell(c, r);
      html += cell.n ? `<div style="background:${color(cell.med / S.pxPerCm)}">${cm(cell.med)}</div>` : '<div class="empty">—</div>';
    }
  }
  return html + '</div>';
}

function stats() {
  const s = summary();
  const picks = pickSummary();
  const has = s.scored > 0;
  const root = el(`<section class="screen">
    <div class="card">
      <h2>الخطأ الحالي</h2>
      ${has ? `<div class="big">${cm(s.medErrPx)} <small>سم، وسيط آخر ${Math.min(20, s.scored)} عينة</small></div>
      <div class="row"><span>بالمعايرة وحدها</span><b>${cm(s.medFrozenPx)} سم</b></div>
      <div class="row"><span>أفقي / عمودي</span><b>${cm(s.medDxPx)} / ${cm(s.medDyPx)} سم</b></div>
      <div class="row"><span>أصغر هدف مريح</span><b>≈ ${cm(2 * s.p75Px)} سم</b></div>`
    : `<p class="note">لا قياسات بعد. أكمل المعايرة ثم «قياس الدقة» أو «استخدام حر».</p>`}
      <div class="row"><span>عينات المعايرة</span><b>${s.samples}</b></div>
      <div class="row"><span>مرفوضة (رمش / بلا وجه)</span><b>${s.rejects.blink || 0} / ${s.rejects['no-face'] || 0}</b></div>
    </div>
    <div class="card"><h2>بطولة النماذج</h2>
      ${s.models.map((m) => {
        const spec = S.cal.model(m.name);
        return `<div class="row"><span>${m.name} · ${spec.label}${m.lead ? ' <i class="tag">يقود</i>' : ''}</span><b>${m.n ? cm(m.medPx) + ' سم' : m.ready ? '—' : 'ينتظر عينات'}</b></div>`;
      }).join('')}
      <p class="note">كل عينة تُختبر على النماذج كلها قبل أن يتعلم منها أيٌّ منها. الأرقام وسيط آخر ${Math.min(20, s.scored)} عينة. النموذج الهندسي G يقود المؤشر لأنه يتحمّل حركة الهاتف، ولا يتقدّمه غيره إلا إذا كان خطؤه أقل بنحو الثلث.</p>
    </div>
    <div class="card"><h2>هندسة الرأس</h2>
      ${geoRows(S.geo)}
      <p class="note">كم يُصدَّق قياس الكاميرا لموضع رأسك واتجاهه حين يُحسب أثرهما على نقطة النظر. «معايرة متحركة» تقيسه، وكل تسجيل جديد يُضاف إلى ما قبله.</p>
    </div>
    <div class="card"><h2>الخطأ مع توالي اللمسات (سم)</h2>${curveSvg()}</div>
    <div class="card"><h2>الخطأ حسب المنطقة (سم)</h2>${has ? heatHtml() : '<p class="note">تظهر الخريطة بعد أول قياس.</p>'}</div>
    <div class="card"><h2>نظر + لمس</h2>${Object.keys(picks).length
      ? Object.entries(picks).map(([k, g]) => `<div class="row"><span>${k}</span><b>${g.ok} من ${g.n}</b></div>`).join('')
      : '<p class="note">لا محاولات بعد.</p>'}</div>
    <button class="btn center" id="s-copy">نسخ الملخص</button>
    <button class="btn center" id="s-dl">تنزيل السجل الكامل (JSON)</button>
    <button class="btn danger" id="s-reset">مسح المعايرة</button>
    <p class="note">السنتيمتر محسوب على ${S.pxPerCm} px لكل سم (تقدير لهذا الجهاز).</p>
  </section>`);

  $('#s-copy', root).addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(summaryText());
      toast('نُسخ الملخص');
    } catch (e) {
      toast('تعذّر النسخ من هذا المتصفح');
    }
  });

  $('#s-dl', root).addEventListener('click', () => {
    const data = {
      app: `GazeCal web ${VERSION}`,
      when: new Date().toISOString(),
      userAgent: navigator.userAgent,
      viewport: { w: S.W, h: S.H, dpr: window.devicePixelRatio || 1 },
      pxPerCm: S.pxPerCm,
      sim: S.sim,
      delegate: S.delegate,
      format: 4,
      video: S.video,
      features: FEATURE_NAMES,
      // tx, ty, tz, nxz, nyz: the head as the camera's face fit reports it, in cm (x toward image-right, y up,
      // tz = distance) and the slopes of the face direction. Model G computes the head's share from these.
      // shares: how much of the head's position (x, y) and direction (x, y) enters the head's part of
      // the gaze point; `measured` is the last moving calibration's verdict. rec.head / frame.r hold the
      // head numbers as the camera read them, rec.phi / frame.b the same after steadying.
      geometry: { pxPerCm: S.pxPerCm, touchBoost: TOUCH_BOOST, shares: getGeometry().shares, defaultShares: DEFAULT_SHARES, measured: S.geo, headFilter: { minCutoffHz: 1.5, beta: 0.3, hzPerDegPerSec: 0.4, needsGyroscope: true } },
      // rec.pix / frame.q: PIX_W×PIX_H values per eye (right then left), row by row, int8 = value×32 after
      // removing the cut-out's mean and dividing by its spread. rec.eye / frame.e: the larger grey
      // cut-outs, hiW×hiH bytes per eye. Each cut-out spans `span` eye lengths, levelled to the corner line.
      // In `calibration.samples` the packed part holds two pictures: plain brightness, then log-brightness.
      eyeImages: { w: PIX_W, h: PIX_H, hiW: HI_W, hiH: HI_H, span: CROP_SPAN, encoding: 'base64' },
      models: S.cal.leaderboard(),
      settings: S.settings,
      summary: summary(),
      picks: S.picks,
      log: S.log,
      pursuit: S.pursuitLog,
      calibration: S.cal.toJSON({ nBase: N_BASE }),
    };
    try {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `gazecal-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    } catch (e) {
      toast('تعذّر التنزيل. استخدم «نسخ الملخص»');
    }
  });

  let armed = false;
  $('#s-reset', root).addEventListener('click', (e) => {
    if (!armed) {
      armed = true;
      e.currentTarget.textContent = 'اضغط مرة ثانية لتأكيد المسح';
      return;
    }
    S.cal.reset();
    S.errMap.reset();
    S.log = [];
    S.picks = [];
    S.rejects = {};
    S.looPx = NaN;
    S.gaze = null;
    S.pursuitLog = [];
    S.smooth.reset();
    setGeometry({ shares: DEFAULT_SHARES });
    S.geo = null;
    save();
    toast('مُسحت المعايرة');
    go('home');
  });
  return root;
}

const SCREENS = { home, boot, pursue: () => pursue(false), pursueMove: () => pursue(true), test, free, pick, stats };

// ---------- rotation: each orientation keeps its own calibration ----------
let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const turned = w !== S.W || h >= w !== S.H >= S.W;
    if (!turned) {
      S.H = h; // browser bars showing or hiding: same screen, same calibration
      placeCam();
      return;
    }
    if (S.running) save();
    S.W = w;
    S.H = h;
    if (S.simUser) {
      S.simUser.W = w;
      S.simUser.H = h;
    }
    if (!S.running) return;
    S.cal.reset();
    S.errMap.reset();
    S.looPx = NaN;
    load();
    S.cal.learning = S.settings.learn;
    go('home');
    toast('تغيّر اتجاه الشاشة: لكل اتجاه معايرته');
  }, 350);
});

$('#back').addEventListener('click', () => go('home'));
window.__gazecal = { S, go, takeSample, summary, summaryText, getGeometry, setGeometry, save };
renderStart();
