// Turns one MediaPipe FaceLandmarker result into the feature vector the models learn from,
// and defines the competing models (which features each one uses).
// Nothing here knows about the screen: signs and scales are left for the regression to learn.

export const FEATURE_NAMES = [
  'hR', 'vR', 'hL', 'vL', // iris position inside each eye, along / across the corner line
  'upR', 'loR', 'upL', 'loL', // iris to upper lid, lower lid to iris (eyelids follow vertical gaze)
  'inL', 'outL', 'upLb', 'dnLb', 'inR', 'outR', 'upRb', 'dnRb', // gaze blendshapes
  'yaw', 'pitch', 'roll', 'cx', 'cy', 'scale', // head pose and where the face sits in the frame
  // The head in space, as the camera sees it: position in cm (x toward image-right, y up, z = distance)
  // and the direction the face points, as slopes against the viewing axis.
  'tx', 'ty', 'tz', 'nxz', 'nyz',
];
export const N_LM = 22; // the landmark features of version 0.2; models that compare with it read only these
export const N_BASE = FEATURE_NAMES.length;

// Two tiny pictures of the eyes follow the numbers above: each eye cut out, levelled to its corner
// line and shrunk to 12×6. First as plain brightness, then as log-brightness, which keeps the edges
// and gives up most of what a change of lighting or exposure does to the picture.
export const PIX_W = 12;
export const PIX_H = 6;
export const N_PIX = 2 * PIX_W * PIX_H;
export const OFF_PIX = N_BASE;
export const OFF_LOG = N_BASE + N_PIX;
export const N_FULL = N_BASE + 2 * N_PIX;

// Smallest spread each landmark feature is assumed to have. See RidgeMap for why this exists.
export const STD_FLOOR = [
  0.006, 0.008, 0.006, 0.008,
  0.01, 0.01, 0.01, 0.01,
  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
  0.03, 0.03, 0.03, 0.02, 0.02, 0.01,
];
const PIX_FLOOR = new Array(N_PIX).fill(0.08);

export const POSE_IDX = [16, 17, 18, 19, 20, 21];
export const POSE_SIGMA = [0.1, 0.1, 0.08, 0.06, 0.06, 0.03];
// Steadiness of these features in the frames before a touch decides how much the touch counts.
export const EYE_REF = [0.006, 0.008, 0.006, 0.008];

// ---- geometry: where the head itself points on the screen ----
const geometry = { pxPerCm: 60 };
export function setGeometry(g) {
  Object.assign(geometry, g);
}

// The point of the screen plane that the face points at, in px, up to a constant offset.
// A ray leaves the head at (tx, ty) along the face direction and travels tz to the screen.
// Screen x runs against image x (the camera faces the user), screen y against "up".
export function headRay(p) {
  const k = -geometry.pxPerCm;
  return [k * (p[22] + p[24] * p[25]), k * (p[23] + p[24] * p[26])];
}

// ---- model A: the ten averaged features of version 0.1, kept as the reference ----
const FLOOR_A = [0.006, 0.008, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01];
const lookH = (p) => (p[9] - p[8] + p[12] - p[13]) / 2;
const lookV = (p) => (p[10] - p[11] + p[14] - p[15]) / 2;

function expandA(p) {
  return Float64Array.of((p[0] + p[2]) / 2, (p[1] + p[3]) / 2, lookH(p), lookV(p), p[16], p[17], p[18], p[19], p[20], p[21]);
}

// ---- models D and E: the landmark features plus one of the two eye pictures ----
function withPicture(p, off) {
  const out = new Float64Array(N_LM + N_PIX);
  out.set(p.subarray(0, N_LM));
  out.set(p.subarray(off, off + N_PIX), N_LM);
  return out;
}

// ---- model G: geometry does the head, learning does only the eye ----
// What is learned is how the eye turns inside the head. On the screen that turn covers a distance
// proportional to how far away the head is, so every eye feature is multiplied by the distance.
// The head's own share (headRay) is taken out of the target before fitting and added back after.
const Z_REF = 30;
const N_EYE = 16;

function expandG(p, c) {
  const k = p[24] / Z_REF;
  const out = new Float64Array(N_EYE + N_PIX + 1);
  for (let j = 0; j < N_EYE; j++) out[j] = (p[j] - (c ? c[j] : 0)) * k;
  for (let j = 0; j < N_PIX; j++) out[N_EYE + j] = (p[OFF_LOG + j] - (c ? c[OFF_LOG + j] : 0)) * k;
  out[N_EYE + N_PIX] = k; // a constant turn of the eye (each person's own offset) also scales with distance
  return out;
}

const WIDE = { lambda: 30, lambdas: [3, 10, 30, 100, 300, 1000], autoLambda: true, robust: false, borrow: 'B', minFit: 30 };

// Ordered from simplest to richest. DEFAULT_MODEL leads until the running scores say otherwise:
// every richer model has to earn the lead on this user's own data.
export const DEFAULT_MODEL = 'B';

export const MODEL_SPECS = [
  { name: 'A', label: 'أساسي (0.1)', expand: expandA, nFeat: 10, stdFloor: FLOOR_A, lambda: 1, minFit: 5 },
  { name: 'B', label: 'نقاط الوجه', expand: (p) => p.subarray(0, N_LM), nFeat: N_LM, stdFloor: STD_FLOOR, lambda: 3, autoLambda: true, poseLocal: true, minFit: 8 },
  { name: 'D', label: 'صورة العين', expand: (p) => withPicture(p, OFF_PIX), nFeat: N_LM + N_PIX, needs: OFF_PIX + N_PIX, stdFloor: [...STD_FLOOR, ...PIX_FLOOR], poseLocal: true, ...WIDE },
  { name: 'E', label: 'عين مقاومة للإضاءة', expand: (p) => withPicture(p, OFF_LOG), nFeat: N_LM + N_PIX, needs: N_FULL, stdFloor: [...STD_FLOOR, ...PIX_FLOOR], poseLocal: true, ...WIDE },
  {
    name: 'G', label: 'هندسي', expand: expandG, nFeat: N_EYE + N_PIX + 1, needs: N_FULL,
    stdFloor: [...STD_FLOOR.slice(0, N_EYE), ...PIX_FLOOR, 0.03], offset: headRay, centered: true, poseLocal: false, ...WIDE,
  },
];

// Face mesh indices. "Right" and "left" are the subject's own, as in MediaPipe.
const R_OUTER = 33;
const R_INNER = 133;
const R_UPPER = 159;
const R_LOWER = 145;
const R_IRIS = 468;
const L_INNER = 362;
const L_OUTER = 263;
const L_UPPER = 386;
const L_LOWER = 374;
const L_IRIS = 473;
const NOSE_TIP = 1;

// Iris position inside the eye, measured along and across the corner-to-corner line.
// a → b always runs toward image-right, so both eyes share one sign convention
// and the measure does not change when the head rolls.
function eyeMeasure(lm, a, b, iris, upper, lower, W, H) {
  const ax = lm[a].x * W;
  const ay = lm[a].y * H;
  const ex = lm[b].x * W - ax;
  const ey = lm[b].y * H - ay;
  const len = Math.hypot(ex, ey);
  if (!(len > 1e-6)) return null;
  const ux = ex / len;
  const uy = ey / len;
  const across = (i) => ((lm[i].x * W - ax) * -uy + (lm[i].y * H - ay) * ux) / len; // + is image-down
  const ix = lm[iris].x * W - ax;
  const iy = lm[iris].y * H - ay;
  const v = (ix * -uy + iy * ux) / len;
  return {
    h: (ix * ux + iy * uy) / len,
    v,
    up: v - across(upper),
    lo: across(lower) - v,
    cx: ax + ex / 2,
    cy: ay + ey / 2,
    len,
    angle: Math.atan2(ey, ex),
  };
}

function blendshapeScores(result) {
  const cats = result.faceBlendshapes && result.faceBlendshapes[0] && result.faceBlendshapes[0].categories;
  if (!cats || !cats.length) return null;
  const m = {};
  for (const c of cats) m[c.categoryName] = c.score;
  return m;
}

// imgW / imgH: size of the frame that was analysed, in pixels.
export function extractFeatures(result, imgW, imgH) {
  const lm = result && result.faceLandmarks && result.faceLandmarks[0];
  if (!lm || lm.length < 478) return { face: false };
  const R = eyeMeasure(lm, R_OUTER, R_INNER, R_IRIS, R_UPPER, R_LOWER, imgW, imgH);
  const L = eyeMeasure(lm, L_INNER, L_OUTER, L_IRIS, L_UPPER, L_LOWER, imgW, imgH);
  if (!R || !L) return { face: false };

  const ipd = Math.hypot(L.cx - R.cx, L.cy - R.cy);
  const scale = ipd / imgW;
  const bs = blendshapeScores(result) || {};
  const g = (k) => bs[k] || 0;
  // crude blink measure from the eyelid gap when blendshapes are missing
  const blink = bs.eyeBlinkLeft !== undefined
    ? Math.max(g('eyeBlinkLeft'), g('eyeBlinkRight'))
    : Math.max(0, 1 - (R.up + R.lo + L.up + L.lo) / 0.5);

  // Head pose: from the fitted face transform when present, else from landmark proxies.
  let yaw;
  let pitch;
  let roll;
  const mat = result.facialTransformationMatrixes && result.facialTransformationMatrixes[0];
  const d = mat && mat.data;
  if (d && d.length === 16) {
    // column-major 4×4: element (row r, col c) is d[c * 4 + r]
    yaw = Math.atan2(d[8], d[10]);
    pitch = Math.asin(Math.max(-1, Math.min(1, -d[9])));
    roll = Math.atan2(d[1], d[5]);
  } else {
    roll = Math.atan2(L.cy - R.cy, L.cx - R.cx);
    yaw = (lm[NOSE_TIP].x * imgW - (L.cx + R.cx) / 2) / ipd;
    pitch = (lm[NOSE_TIP].y * imgH - (L.cy + R.cy) / 2) / ipd;
  }

  // The head in camera space. The face transform gives it in cm (x right, y up, the face at -z).
  // Without it, the same quantities are guessed from the picture: 63 mm between the eyes, a 58° lens.
  const cxn = (L.cx + R.cx) / 2 / imgW;
  const cyn = (L.cy + R.cy) / 2 / imgH;
  let tx;
  let ty;
  let tz;
  let nxz;
  let nyz;
  if (d && d.length === 16 && Math.abs(d[10]) > 0.2 && d[14] < 0) {
    tx = d[12];
    ty = d[13];
    tz = -d[14];
    nxz = d[8] / d[10];
    nyz = d[9] / d[10];
  } else {
    tz = scale > 0 ? 5.68 / scale : 30;
    tx = ((cxn - 0.5) * tz) / 0.902;
    ty = (-(cyn - 0.5) * (imgH / imgW) * tz) / 0.902;
    nxz = Math.tan(yaw);
    nyz = -Math.tan(pitch);
  }

  return {
    face: true,
    phi: Float64Array.of(
      R.h, R.v, L.h, L.v,
      R.up, R.lo, L.up, L.lo,
      g('eyeLookInLeft'), g('eyeLookOutLeft'), g('eyeLookUpLeft'), g('eyeLookDownLeft'),
      g('eyeLookInRight'), g('eyeLookOutRight'), g('eyeLookUpRight'), g('eyeLookDownRight'),
      yaw, pitch, roll,
      cxn, cyn, scale,
      tx, ty, tz, nxz, nyz
    ),
    blink,
    // Rough viewing distance, assuming a 63 mm eye spacing and a ~58° wide lens.
    distCm: scale > 0 ? 5.67 / scale : NaN,
    // Where each eye sits in the frame, for cutting out the eye images.
    eyes: [
      { cx: R.cx, cy: R.cy, len: R.len, angle: R.angle },
      { cx: L.cx, cy: L.cy, len: L.len, angle: L.angle },
    ],
  };
}
