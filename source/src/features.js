// Turns one MediaPipe FaceLandmarker result into the feature vector the models learn from,
// and defines the competing models (which features each one uses).
// Nothing here knows about the screen: signs and scales are left for the regression to learn.

export const FEATURE_NAMES = [
  'hR', 'vR', 'hL', 'vL', // iris position inside each eye, along / across the corner line
  'upR', 'loR', 'upL', 'loL', // iris to upper lid, lower lid to iris (eyelids follow vertical gaze)
  'inL', 'outL', 'upLb', 'dnLb', 'inR', 'outR', 'upRb', 'dnRb', // gaze blendshapes
  'yaw', 'pitch', 'roll', 'cx', 'cy', 'scale', // head pose and where the face sits in the frame
];
export const N_BASE = FEATURE_NAMES.length;

// Eye-image features: each eye cut out, levelled to the corner line, shrunk to 12×6 and
// contrast-normalised. They follow the 22 numbers above in the full feature vector.
export const PIX_W = 12;
export const PIX_H = 6;
export const N_PIX = 2 * PIX_W * PIX_H;
export const N_FULL = N_BASE + N_PIX;

// Smallest spread each feature is assumed to have. See RidgeMap for why this exists.
export const STD_FLOOR = [
  0.006, 0.008, 0.006, 0.008,
  0.01, 0.01, 0.01, 0.01,
  0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
  0.03, 0.03, 0.03, 0.02, 0.02, 0.01,
];

export const POSE_IDX = [16, 17, 18, 19, 20, 21];
export const POSE_SIGMA = [0.1, 0.1, 0.08, 0.06, 0.06, 0.03];
// Steadiness of these features in the frames before a touch decides how much the touch counts.
export const EYE_REF = [0.006, 0.008, 0.006, 0.008];

// ---- model A: the ten averaged features of version 0.1, kept as the reference ----
const FLOOR_A = [0.006, 0.008, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01];
const lookH = (p) => (p[9] - p[8] + p[12] - p[13]) / 2;
const lookV = (p) => (p[10] - p[11] + p[14] - p[15]) / 2;

function expandA(p) {
  return Float64Array.of((p[0] + p[2]) / 2, (p[1] + p[3]) / 2, lookH(p), lookV(p), p[16], p[17], p[18], p[19], p[20], p[21]);
}

// ---- model B: every feature on its own ----
const expandB = (p) => (p.length === N_BASE ? p : p.subarray(0, N_BASE));

// ---- model C: B plus curvature and "gain changes with pose and distance" terms ----
const FLOOR_Q = [2e-4, 3e-4, 2e-4, 3e-4, 3e-4, 3e-4, 3e-4];

function expandC(p, c) {
  const out = new Float64Array(N_BASE + 7);
  out.set(p.length === N_BASE ? p : p.subarray(0, N_BASE));
  // deviations from this user's own average, so the products are not copies of the linear terms
  const h = (p[0] + p[2]) / 2 - (c ? (c[0] + c[2]) / 2 : 0.5);
  const v = (p[1] + p[3]) / 2 - (c ? (c[1] + c[3]) / 2 : 0);
  const yaw = p[16] - (c ? c[16] : 0);
  const pitch = p[17] - (c ? c[17] : 0);
  const sc = p[21] - (c ? c[21] : 0.17);
  out[N_BASE] = h * h;
  out[N_BASE + 1] = v * v;
  out[N_BASE + 2] = h * v;
  out[N_BASE + 3] = h * yaw;
  out[N_BASE + 4] = v * pitch;
  out[N_BASE + 5] = h * sc;
  out[N_BASE + 6] = v * sc;
  return out;
}

// Ordered from simplest to richest. DEFAULT_MODEL leads until the running scores say otherwise:
// the eye-image model has to earn the lead on this user's own data.
export const DEFAULT_MODEL = 'B';

export const MODEL_SPECS = [
  { name: 'A', label: 'أساسي (0.1)', expand: expandA, nFeat: 10, stdFloor: FLOOR_A, lambda: 1, minFit: 5 },
  { name: 'B', label: 'خصائص غنية', expand: expandB, nFeat: N_BASE, stdFloor: STD_FLOOR, lambda: 3, autoLambda: true, poseLocal: true, minFit: 8 },
  { name: 'C', label: 'غنية + غير خطي', expand: expandC, nFeat: N_BASE + 7, stdFloor: [...STD_FLOOR, ...FLOOR_Q], lambda: 3, autoLambda: true, poseLocal: true, centered: true, minFit: 8 },
  {
    name: 'D', label: 'صورة العين + الخصائص', expand: (p) => p, nFeat: N_FULL, needs: N_FULL,
    stdFloor: [...STD_FLOOR, ...new Array(N_PIX).fill(0.08)], lambda: 30, lambdas: [3, 10, 30, 100, 300, 1000],
    autoLambda: true, poseLocal: true, robust: false, borrow: 'B', minFit: 30,
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

  return {
    face: true,
    phi: Float64Array.of(
      R.h, R.v, L.h, L.v,
      R.up, R.lo, L.up, L.lo,
      g('eyeLookInLeft'), g('eyeLookOutLeft'), g('eyeLookUpLeft'), g('eyeLookDownLeft'),
      g('eyeLookInRight'), g('eyeLookOutRight'), g('eyeLookUpRight'), g('eyeLookDownRight'),
      yaw, pitch, roll,
      (L.cx + R.cx) / 2 / imgW, (L.cy + R.cy) / 2 / imgH, scale
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
