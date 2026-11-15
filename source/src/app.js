import { Calibrator, ErrorMap, FrameRing, OneEuro, TargetSelector, summarizeWindow, windowWeight, median, percentile, clamp } from './engine.js';
import { FEATURE_NAMES, STD_FLOOR, extractFeatures } from './features.js';
import { SimUser, mulberry32 } from './sim.js';

const VERSION = '0.1';
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

const TITLES = { home: 'معايرة النظر', boot: 'معايرة أولية', test: 'قياس الدقة', free: 'استخدام حر', pick: 'نظر + لمس', stats: 'النتائج' };
const IMMERSIVE = { boot: true, test: true, pick: true };

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
  cal: new Calibrator({ nFeat: FEATURE_NAMES.length, stdFloor: STD_FLOOR }),
  ring: new FrameRing(),
  errMap: new ErrorMap(3, 4),
  fx: new OneEuro(),
  fy: new OneEuro(),
  gaze: null,
  settings: { cursor: true, learn: true, preview: true, gpu: false, motion: false },
  log: [],
  picks: [],
  rejects: {},
  looPx: NaN,
  motion: 0,
  motionT: -1e9,
  screen: null,
  cleanup: null,
  onGaze: null,
  onSample: null,
  selector: null,
  pickLevel: 0,
};

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
const profileKey = () => `gazecal:v1:${S.W}:${S.H >= S.W ? 'p' : 'l'}`;
let saveTimer = 0;

function save() {
  if (S.sim) return;
  try {
    localStorage.setItem('gazecal:settings', JSON.stringify(S.settings));
    localStorage.setItem(profileKey(), JSON.stringify({ cal: S.cal.toJSON(), loo: S.looPx }));
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
    if (s) Object.assign(S.settings, s);
    const p = JSON.parse(localStorage.getItem(profileKey()) || 'null');
    if (p && p.cal && S.cal.load(p.cal)) S.looPx = Number.isFinite(p.loo) ? p.loo : NaN;
  } catch (e) { /* ignore a corrupt or missing profile */ }
  S.settings.motion = false; // needs a fresh permission tap every visit
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
  S.ring.push({ t, face: S.face, phi: f.phi || null, blink: f.blink || 0, motion: t - S.motionT < 400 ? S.motion : 0 });

  if (!S.face) {
    S.gaze = null;
    S.fx.reset();
    S.fy.reset();
  } else if (S.cal.ready && !(f.blink > 0.5)) {
    const p = S.cal.predict(f.phi);
    S.gaze = { x: clamp(S.fx.filter(p[0], t / 1000), 0, S.W), y: clamp(S.fy.filter(p[1], t / 1000), 0, S.H), t };
    if (S.selector) S.selector.update(S.gaze.x, S.gaze.y, sigmaPx(), t);
  } else if (!S.cal.ready) {
    S.gaze = null;
  }
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
function takeSample(target, src, tMs, baseW, { force = false, since = -Infinity } = {}) {
  const sum = summarizeWindow(S.ring.window(Math.max(tMs - 350, since), tMs + 50));
  if (!sum.ok) {
    S.rejects[sum.reason] = (S.rejects[sum.reason] || 0) + 1;
    return { ok: false, reason: sum.reason };
  }
  const rec = S.cal.addSample({ phi: sum.phi, y: target, w: baseW * windowWeight(sum), src, t: Math.round(tMs) }, { force });
  rec.ok = true;
  rec.screen = S.screen;
  rec.frames = sum.n;
  rec.motion = Math.round(sum.motion);
  rec.w = +rec.w.toFixed(3);
  rec.phi = Array.from(sum.phi, (v) => +v.toFixed(5));
  if (rec.err != null) S.errMap.add(target[0] / S.W, target[1] / S.H, rec.err);
  S.log.push(rec);
  if (S.log.length > 3000) S.log.shift();
  saveSoon();
  if (S.onSample) S.onSample(rec);
  return rec;
}

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
    S.ring.push({ t: ft, face: true, phi: f.phi, blink: f.blink, motion: 0 });
  }
}

function startSim() {
  if (S.simUser) return;
  S.simUser = new SimUser({ W: S.W, H: S.H, pxPerCm: S.pxPerCm, seed: 7, noise: 3 });
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
  onFrame(extractFeatures(res, video.videoWidth, video.videoHeight), t);
}

function enableMotion() {
  // Must run inside a tap: iOS only shows the motion permission prompt then.
  try {
    const DM = window.DeviceMotionEvent;
    if (!DM) return Promise.resolve(false);
    const listen = () => {
      window.addEventListener('devicemotion', (e) => {
        const r = e.rotationRate;
        if (!r) return;
        S.motion = Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0);
        S.motionT = performance.now();
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
       <li>النتائج تقريبية: الكاميرا العادية أقل دقة من حسّاس Face ID.</li></ul>`;
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
  const video = $('#cam');
  try {
    if (!video.srcObject) {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
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
      </div>
      <div class="slot" id="cam-slot" ${S.settings.preview && !S.sim ? '' : 'hidden'}></div>
    </div>
    <div class="scroll">
      <button class="btn ${ready ? '' : 'primary'}" data-go="boot"><span>${ready ? 'إعادة المعايرة الأولية' : 'ابدأ بالمعايرة الأولية'}</span><small>9 نقاط</small></button>
      <button class="btn" data-go="test" ${ready ? '' : 'disabled'}><span>قياس الدقة</span><small>20 نقطة</small></button>
      <button class="btn" data-go="free"><span>استخدام حر</span><small>يتعلّم من لمساتك</small></button>
      <button class="btn" data-go="pick" ${ready ? '' : 'disabled'}><span>نظر + لمس</span><small>اختيار بالعين</small></button>
      <button class="btn" data-go="stats"><span>النتائج والتصدير</span><small></small></button>
      ${ready ? '' : '<p class="note">«قياس الدقة» و«نظر + لمس» يُفتحان بعد المعايرة الأولية.</p>'}
      <div class="card settings">
        <div class="row"><span>إظهار مؤشر النظر</span><button class="switch" role="switch" data-set="cursor" aria-label="إظهار مؤشر النظر"></button></div>
        <div class="row"><span>التعلّم من اللمس</span><button class="switch" role="switch" data-set="learn" aria-label="التعلّم من اللمس"></button></div>
        ${S.sim ? '' : `<div class="row"><span>صورة الكاميرا</span><button class="switch" role="switch" data-set="preview" aria-label="صورة الكاميرا"></button></div>
        <div class="row"><span>حسّاس الحركة</span><button class="switch" role="switch" data-set="motion" aria-label="حسّاس الحركة"></button></div>
        <div class="row"><span>معالجة GPU (تجريبي)</span><button class="switch" role="switch" data-set="gpu" aria-label="معالجة GPU"></button></div>`}
      </div>
    </div>
  </section>`);

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
  S.cal.reset();
  S.errMap.reset();
  S.looPx = NaN;
  S.fx.reset();
  S.fy.reset();

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
      S.cal.freeze();
      const loo = S.cal.looErrors();
      S.looPx = loo.length ? median(loo) : NaN;
      save();
      root.appendChild(el(S.cal.ready
        ? `<div class="sheet">
            <h2>تمت المعايرة الأولية</h2>
            <div class="big">${cm(S.looPx)} <small>سم خطأ متوقع، تقدير من 9 نقاط</small></div>
            <p class="note">من الآن كل لمسة على هدف صغير تحسّن المعايرة دون خطوة إضافية.</p>
            <button class="btn primary" data-go="test">قياس الدقة</button>
            <button class="btn center" data-go="home">الرئيسية</button>
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
      const scored = trials.filter((r) => r.err != null);
      const med = median(scored.map((r) => r.err));
      const frozen = scored.filter((r) => r.frozenErr != null).map((r) => r.frozenErr);
      const dx = median(scored.map((r) => Math.abs(r.pred[0] - r.target[0])));
      const dy = median(scored.map((r) => Math.abs(r.pred[1] - r.target[1])));
      root.appendChild(el(`<div class="sheet">
        <h2>نتيجة القياس</h2>
        <div class="big">${cm(med)} <small>سم خطأ وسيط على ${scored.length} نقطة</small></div>
        <div class="row"><span>${S.cal.learning ? 'بالمعايرة الأولية وحدها' : 'التعلّم متوقف'}</span><b>${frozen.length ? cm(median(frozen)) + ' سم' : '—'}</b></div>
        <div class="row"><span>أفقي / عمودي</span><b>${cm(dx)} / ${cm(dy)} سم</b></div>
        <div class="row"><span>أسوأ ربع من النقاط فوق</span><b>${cm(percentile(scored.map((r) => r.err), 0.75))} سم</b></div>
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
  let cells = [];
  let rects = [];
  let goal = -1;
  let t0 = 0;
  let busy = true;
  let alive = true;

  function stat() {
    const [c, r] = levels[S.pickLevel];
    const mine = S.picks.filter((p) => p.cols === c && p.rows === r);
    if (!mine.length) {
      $('#p-stat', root).textContent = 'انظر إلى المربع الذي فيه الدائرة ثم اضغط «اختر»';
      return;
    }
    const ok = mine.filter((p) => p.ok).length;
    const sec = median(mine.map((p) => p.ms)) / 1000;
    $('#p-stat', root).textContent = `صحيح ${ok} من ${mine.length} · ${sec.toFixed(1)} ث للاختيار`;
  }

  function trial() {
    if (!alive) return;
    cells.forEach((c) => (c.className = 'cell'));
    let g = goal;
    while (g === goal) g = Math.floor(Math.random() * cells.length);
    goal = g;
    cells[goal].classList.add('goal');
    S.selector.setTargets(rects);
    t0 = performance.now();
    if (S.simUser) S.simUser.lookAt(rects[goal].cx, rects[goal].cy, t0);
    busy = false;
  }

  function build() {
    busy = true;
    const [c, r] = levels[S.pickLevel];
    grid.style.gridTemplateColumns = `repeat(${c}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${r}, 1fr)`;
    grid.innerHTML = Array.from({ length: c * r }, () => '<div class="cell"></div>').join('');
    cells = Array.from(grid.children);
    root.querySelectorAll('.seg').forEach((s, i) => s.setAttribute('aria-pressed', String(i === S.pickLevel)));
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

  root.querySelectorAll('.seg').forEach((s) => s.addEventListener('click', () => {
    S.pickLevel = Number(s.dataset.l);
    build();
  }));

  $('#p-go', root).addEventListener('click', () => {
    if (busy) return;
    const best = S.gaze && S.selector ? S.selector.best() : null;
    if (!best) {
      toast('لا أرى عينيك الآن');
      return;
    }
    busy = true;
    const [c, r] = levels[S.pickLevel];
    const ok = best.index === goal;
    S.picks.push({ t: Date.now(), cols: c, rows: r, ok, ms: Math.round(performance.now() - t0), p: +best.p.toFixed(2), sigma: Math.round(sigmaPx()) });
    cells.forEach((cell) => cell.classList.remove('sel'));
    cells[best.index].classList.add(ok ? 'right' : 'wrong');
    stat();
    setTimeout(trial, 600);
  });

  S.onGaze = () => {
    if (busy || !S.selector) return;
    const best = S.gaze ? S.selector.best() : null;
    for (let i = 0; i < cells.length; i++) cells[i].classList.toggle('sel', !!best && best.index === i);
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
  const srcNames = { bootstrap: 'معايرة أولية', 'test-dot': 'نقاط القياس', 'touch-target': 'لمس أهداف', 'touch-free': 'لمس حر' };
  return [
    `GazeCal web ${VERSION}${S.sim ? ' (محاكاة)' : ''}`,
    `الشاشة ${S.W}×${S.H} · DPR ${window.devicePixelRatio || 1} · ${s.fps} إطار/ث · ${S.delegate}`,
    `عينات المعايرة: ${s.samples} (${Object.entries(s.bySrc).map(([k, v]) => `${srcNames[k] || k} ${v}`).join('، ') || '—'})`,
    `مرفوضة: رمش ${s.rejects.blink || 0}، بلا وجه ${s.rejects['no-face'] || 0}`,
    `تقدير بعد 9 نقاط: ${cm(s.looPx)} سم`,
    `الخطأ الوسيط (آخر ${Math.min(20, s.scored)}): ${cm(s.medErrPx)} سم · بالمعايرة الأولية وحدها: ${cm(s.medFrozenPx)} سم`,
    `أفقي / عمودي: ${cm(s.medDxPx)} / ${cm(s.medDyPx)} سم · الربع الأسوأ فوق ${cm(s.p75Px)} سم`,
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
    <div class="legend"><span><i style="background:#60a5fa"></i>مع التعلّم</span><span><i style="background:#8b97a8"></i>المعايرة الأولية وحدها</span><span>أعلى الرسم = ${top} سم</span></div>`;
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
      <div class="row"><span>بالمعايرة الأولية وحدها</span><b>${cm(s.medFrozenPx)} سم</b></div>
      <div class="row"><span>أفقي / عمودي</span><b>${cm(s.medDxPx)} / ${cm(s.medDyPx)} سم</b></div>
      <div class="row"><span>أصغر هدف مريح</span><b>≈ ${cm(2 * s.p75Px)} سم</b></div>`
    : `<p class="note">لا قياسات بعد. أكمل المعايرة الأولية ثم «قياس الدقة» أو «استخدام حر».</p>`}
      <div class="row"><span>عينات المعايرة</span><b>${s.samples}</b></div>
      <div class="row"><span>مرفوضة (رمش / بلا وجه)</span><b>${s.rejects.blink || 0} / ${s.rejects['no-face'] || 0}</b></div>
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
      features: FEATURE_NAMES,
      settings: S.settings,
      summary: summary(),
      picks: S.picks,
      log: S.log,
      calibration: S.cal.toJSON(),
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
    save();
    toast('مُسحت المعايرة');
    go('home');
  });
  return root;
}

const SCREENS = { home, boot, test, free, pick, stats };

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
window.__gazecal = { S, go, takeSample, summary, summaryText };
renderStart();
