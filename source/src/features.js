// Turns one MediaPipe FaceLandmarker result into the feature vector the calibrator learns from.
// Nothing here knows about the screen: signs and scales are left for the regression to learn.

export const FEATURE_NAMES = ['h', 'v', 'lookH', 'lookV', 'yaw', 'pitch', 'roll', 'cx', 'cy', 'scale'];

// Smallest spread each feature is assumed to have. See RidgeMap for why this exists.
export const STD_FLOOR = [0.006, 0.008, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01];

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
function eyeRatios(lm, a, b, iris, upper, lower, W, H) {
  const ax = lm[a].x * W;
  const ay = lm[a].y * H;
  const ex = lm[b].x * W - ax;
  const ey = lm[b].y * H - ay;
  const len = Math.hypot(ex, ey);
  if (!(len > 1e-6)) return null;
  const ux = ex / len;
  const uy = ey / len;
  const ix = lm[iris].x * W - ax;
  const iy = lm[iris].y * H - ay;
  return {
    h: (ix * ux + iy * uy) / len,
    v: (ix * -uy + iy * ux) / len,
    open: Math.hypot((lm[upper].x - lm[lower].x) * W, (lm[upper].y - lm[lower].y) * H) / len,
    cx: ax + ex / 2,
    cy: ay + ey / 2,
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
  const right = eyeRatios(lm, R_OUTER, R_INNER, R_IRIS, R_UPPER, R_LOWER, imgW, imgH);
  const left = eyeRatios(lm, L_INNER, L_OUTER, L_IRIS, L_UPPER, L_LOWER, imgW, imgH);
  if (!right || !left) return { face: false };

  const h = (right.h + left.h) / 2;
  const v = (right.v + left.v) / 2;
  const ipd = Math.hypot(left.cx - right.cx, left.cy - right.cy);
  const cx = (left.cx + right.cx) / 2 / imgW;
  const cy = (left.cy + right.cy) / 2 / imgH;
  const scale = ipd / imgW;

  let lookH = 0;
  let lookV = 0;
  let blink = Math.max(0, 1 - (right.open + left.open) / 0.5); // crude fallback from eyelid gap
  const bs = blendshapeScores(result);
  if (bs) {
    lookH = ((bs.eyeLookOutLeft || 0) - (bs.eyeLookInLeft || 0) + (bs.eyeLookInRight || 0) - (bs.eyeLookOutRight || 0)) / 2;
    lookV = ((bs.eyeLookUpLeft || 0) - (bs.eyeLookDownLeft || 0) + (bs.eyeLookUpRight || 0) - (bs.eyeLookDownRight || 0)) / 2;
    blink = Math.max(bs.eyeBlinkLeft || 0, bs.eyeBlinkRight || 0);
  }

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
    roll = Math.atan2(left.cy - right.cy, left.cx - right.cx);
    const midX = (left.cx + right.cx) / 2;
    const midY = (left.cy + right.cy) / 2;
    yaw = (lm[NOSE_TIP].x * imgW - midX) / ipd;
    pitch = (lm[NOSE_TIP].y * imgH - midY) / ipd;
  }

  return {
    face: true,
    phi: Float64Array.of(h, v, lookH, lookV, yaw, pitch, roll, cx, cy, scale),
    blink,
    // Rough viewing distance, assuming a 63 mm eye spacing and a ~58° wide lens.
    distCm: scale > 0 ? 5.67 / scale : NaN,
  };
}
