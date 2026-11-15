// GazeCal engine: calibration, filtering, error tracking. No DOM, no camera.
// Everything here is plain math so it can be ported to Swift line by line.

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function median(values) {
  if (!values || !values.length) return NaN;
  const a = Array.from(values).sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export function percentile(values, p) {
  if (!values || !values.length) return NaN;
  const a = Array.from(values).sort((x, y) => x - y);
  const idx = clamp((a.length - 1) * p, 0, a.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}

// Solves A·X = B (A is n×n, B is n×m) by Gauss-Jordan with partial pivoting.
// Returns null when A is singular.
export function solveLinear(A, B) {
  const n = A.length;
  const m = B[0].length;
  const M = A.map((row, i) => [...row, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    }
    if (Math.abs(M[p][c]) < 1e-12) return null;
    if (p !== c) [M[p], M[c]] = [M[c], M[p]];
    const d = M[c][c];
    for (let k = c; k < n + m; k++) M[c][k] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f === 0) continue;
      for (let k = c; k < n + m; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row) => row.slice(n));
}

// Solves A·X = B for a symmetric positive definite A by Cholesky factorisation.
// A: flat n×n, row-major, only the lower triangle is read. B: flat n×m.
// Both are overwritten (B ends up holding X). Returns false when A is not positive definite.
export function solveSPD(A, B, n, m) {
  for (let j = 0; j < n; j++) {
    const rj = j * n;
    let d = A[rj + j];
    for (let k = 0; k < j; k++) d -= A[rj + k] * A[rj + k];
    if (!(d > 1e-12)) return false;
    d = Math.sqrt(d);
    A[rj + j] = d;
    for (let i = j + 1; i < n; i++) {
      const ri = i * n;
      let v = A[ri + j];
      for (let k = 0; k < j; k++) v -= A[ri + k] * A[rj + k];
      A[ri + j] = v / d;
    }
  }
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) {
      const ri = i * n;
      let v = B[i * m + c];
      for (let k = 0; k < i; k++) v -= A[ri + k] * B[k * m + c];
      B[i * m + c] = v / A[ri + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let v = B[i * m + c];
      for (let k = i + 1; k < n; k++) v -= A[k * n + i] * B[k * m + c];
      B[i * m + c] = v / A[i * n + i];
    }
  }
  return true;
}

// Adds one weighted row z (with target y) to the normal equations. Lower triangle only.
function accumulate(A, B, z, w, y, m) {
  for (let r = 0; r < m; r++) {
    const wz = w * z[r];
    const row = r * m;
    for (let c = 0; c <= r; c++) A[row + c] += wz * z[c];
    B[r * 2] += wz * y[0];
    B[r * 2 + 1] += wz * y[1];
  }
}

// Small numbers (unit-variance pixel values) packed one byte each, for compact storage.
const PACK_SCALE = 32;
export function packInt8(values) {
  let s = '';
  for (let i = 0; i < values.length; i++) s += String.fromCharCode(clamp(Math.round(values[i] * PACK_SCALE), -127, 127) & 255);
  return btoa(s);
}

export function unpackInt8(b64, out, offset = 0) {
  const s = atob(b64);
  for (let i = 0; i < s.length; i++) {
    const v = s.charCodeAt(i);
    out[offset + i] = (v > 127 ? v - 256 : v) / PACK_SCALE;
  }
  return s.length;
}

// One Euro filter (Casiez et al.): low jitter while still, low lag while moving.
export class OneEuro {
  constructor(minCutoff = 0.8, beta = 0.004, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  reset() {
    this.x = null;
    this.dx = 0;
    this.t = null;
  }

  // boost (Hz) opens the filter from outside: something else already knows the value is about to change.
  filter(x, tSec, boost = 0) {
    if (this.t === null) {
      this.x = x;
      this.dx = 0;
      this.t = tSec;
      return x;
    }
    const dt = Math.max(1e-3, tSec - this.t);
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx) + boost;
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    this.t = tSec;
    return this.x;
  }
}

// Recent frames, so a touch can look back at what the eyes were doing just before it.
export class FrameRing {
  constructor(maxAgeMs = 2500) {
    this.maxAgeMs = maxAgeMs;
    this.frames = [];
  }

  push(frame) {
    this.frames.push(frame);
    const cut = frame.t - this.maxAgeMs;
    while (this.frames.length && this.frames[0].t < cut) this.frames.shift();
  }

  window(t0, t1) {
    return this.frames.filter((f) => f.t >= t0 && f.t <= t1);
  }
}

// Collapses the frames before a touch into one robust feature vector.
// Rejects the window when the face was missing or the eyes were closed.
export function summarizeWindow(frames, { minFrames = 3, blinkMax = 0.5 } = {}) {
  const seen = frames.filter((f) => f.face && f.phi);
  if (seen.length < minFrames) return { ok: false, reason: 'no-face' };
  const open = seen.filter((f) => !(f.blink > blinkMax));
  if (open.length < minFrames) return { ok: false, reason: 'blink' };
  const n = open[0].phi.length;
  const phi = new Float64Array(n);
  const spread = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const col = open.map((f) => f.phi[j]);
    const med = median(col);
    phi[j] = med;
    spread[j] = median(col.map((v) => Math.abs(v - med)));
  }
  let motion = 0;
  for (const f of open) if (f.motion > motion) motion = f.motion;
  return { ok: true, phi, spread, n: open.length, motion };
}

// How much to trust a window: steady eyes and a steady device count for more.
export function windowWeight(summary, { eyeRef = [0.006, 0.008], motionRef = 45 } = {}) {
  let q = 0;
  for (let j = 0; j < eyeRef.length; j++) q += summary.spread[j] / eyeRef[j];
  q /= eyeRef.length;
  const steady = 1 / (1 + q * q);
  const m = summary.motion / motionRef; // device rotation rate against the rate at which pictures start to smear
  const still = 1 / (1 + m * m);
  return clamp(steady * still, 0.05, 1);
}

// Linear map from features to a screen point, fitted by weighted ridge regression.
// Features are standardized with a per-feature floor on the spread, so a feature
// that barely moved (say, the head during a 15-second calibration) cannot be
// blown up into a large coefficient.
export class RidgeMap {
  constructor(nFeat, { lambda = 1, stdFloor = null } = {}) {
    this.n = nFeat;
    this.lambda = lambda;
    this.stdFloor = stdFloor ? Float64Array.from(stdFloor) : new Float64Array(nFeat).fill(1e-3);
    this.mu = new Float64Array(nFeat);
    this.sd = new Float64Array(nFeat).fill(1);
    this.beta = null; // (n + 1) rows of [bx, by]; row 0 is the bias
  }

  get ready() {
    return this.beta !== null;
  }

  // Weighted mean and (floored) spread of every feature.
  stats(samples, weights) {
    const n = this.n;
    const N = samples.length;
    const mu = new Float64Array(n);
    const sd = new Float64Array(n);
    let sw = 0;
    for (let i = 0; i < N; i++) {
      const w = weights[i];
      const p = samples[i].phi;
      sw += w;
      for (let j = 0; j < n; j++) mu[j] += w * p[j];
    }
    if (!(sw > 0)) return null;
    for (let j = 0; j < n; j++) mu[j] /= sw;
    for (let i = 0; i < N; i++) {
      const w = weights[i];
      const p = samples[i].phi;
      for (let j = 0; j < n; j++) {
        const d = p[j] - mu[j];
        sd[j] += w * d * d;
      }
    }
    for (let j = 0; j < n; j++) sd[j] = Math.max(Math.sqrt(sd[j] / sw), this.stdFloor[j]);
    return { mu, sd, sw };
  }

  fit(samples, weights) {
    const n = this.n;
    const N = samples.length;
    const m = n + 1;
    const st = N >= 3 ? this.stats(samples, weights) : null;
    if (!st) {
      this.beta = null;
      return false;
    }
    const { mu, sd, sw } = st;
    const A = new Float64Array(m * m);
    const B = new Float64Array(m * 2);
    const z = new Float64Array(m);
    z[0] = 1;
    const scale = N / sw; // mean weight becomes 1, so lambda keeps one meaning
    for (let i = 0; i < N; i++) {
      const w = weights[i] * scale;
      if (!(w > 0)) continue;
      const p = samples[i].phi;
      for (let j = 0; j < n; j++) z[j + 1] = (p[j] - mu[j]) / sd[j];
      accumulate(A, B, z, w, samples[i].y, m);
    }
    A[0] += 1e-9;
    for (let j = 1; j < m; j++) A[j * m + j] += this.lambda;
    if (!solveSPD(A, B, m, 2)) {
      this.beta = null;
      return false;
    }
    this.mu = mu;
    this.sd = sd;
    this.beta = Array.from({ length: m }, (_, r) => [B[r * 2], B[r * 2 + 1]]);
    return true;
  }

  // Huber reweighting: a touch the user was not looking at ends up far from the
  // fit and loses most of its vote. Kept soft on purpose (no hard cut-off), so
  // samples from a new posture can still pull the model toward them.
  fitRobust(samples, weights, { iters = 3, k = 1.5, scaleFloor = 20 } = {}) {
    this.trust = null;
    if (!this.fit(samples, weights)) return false;
    let w = weights;
    for (let it = 0; it < iters; it++) {
      const res = samples.map((s) => {
        const p = this.predict(s.phi);
        return Math.hypot(p[0] - s.y[0], p[1] - s.y[1]);
      });
      const s = Math.max(1.4826 * median(res), scaleFloor);
      w = weights.map((w0, i) => (res[i] <= k * s ? w0 : (w0 * k * s) / res[i]));
      if (!this.fit(samples, w)) return false;
    }
    // how much each sample was trusted, for models that are too wide to run this loop themselves
    this.trust = Float64Array.from(weights, (w0, i) => (w0 > 0 ? w[i] / w0 : 1));
    return true;
  }

  // Picks lambda by cross-validation over contiguous blocks of samples (blocks,
  // because neighbouring frames of a recording are near-copies of each other),
  // then fits on everything. Prefers the strongest regularisation that is
  // within 2% of the best score.
  fitCV(samples, weights, { lambdas = [0.1, 0.3, 1, 3, 10, 30, 100], folds = 6, cap = 400, robust = true } = {}) {
    const fitAll = () => (robust ? this.fitRobust(samples, weights) : this.fit(samples, weights));
    const n = this.n;
    const N = samples.length;
    const m = n + 1;
    const st = N >= 4 * folds ? this.stats(samples, weights) : null;
    if (!st) return fitAll();
    const { mu, sd, sw } = st;
    const scale = N / sw;
    const foldOf = (i) => Math.min(folds - 1, Math.floor((i * folds) / N));
    const G = Array.from({ length: folds }, () => new Float64Array(m * m));
    const B = Array.from({ length: folds }, () => new Float64Array(m * 2));
    const Z = new Float64Array(N * m);
    for (let i = 0; i < N; i++) {
      const z = Z.subarray(i * m, (i + 1) * m);
      const p = samples[i].phi;
      z[0] = 1;
      for (let j = 0; j < n; j++) z[j + 1] = (p[j] - mu[j]) / sd[j];
      const w = weights[i] * scale;
      if (w > 0) accumulate(G[foldOf(i)], B[foldOf(i)], z, w, samples[i].y, m);
    }
    const GT = new Float64Array(m * m);
    const BT = new Float64Array(m * 2);
    for (let k = 0; k < folds; k++) {
      for (let q = 0; q < m * m; q++) GT[q] += G[k][q];
      for (let q = 0; q < m * 2; q++) BT[q] += B[k][q];
    }
    const A = new Float64Array(m * m);
    const X = new Float64Array(m * 2);
    const scores = lambdas.map((lam) => {
      let err = 0;
      let wsum = 0;
      for (let k = 0; k < folds; k++) {
        for (let q = 0; q < m * m; q++) A[q] = GT[q] - G[k][q];
        for (let q = 0; q < m * 2; q++) X[q] = BT[q] - B[k][q];
        A[0] += 1e-9;
        for (let j = 1; j < m; j++) A[j * m + j] += lam;
        if (!solveSPD(A, X, m, 2)) return Infinity;
        for (let i = 0; i < N; i++) {
          if (foldOf(i) !== k) continue;
          const o = i * m;
          let px = 0;
          let py = 0;
          for (let r = 0; r < m; r++) {
            px += X[r * 2] * Z[o + r];
            py += X[r * 2 + 1] * Z[o + r];
          }
          err += weights[i] * Math.min(cap, Math.hypot(px - samples[i].y[0], py - samples[i].y[1]));
          wsum += weights[i];
        }
      }
      return wsum > 0 ? err / wsum : Infinity;
    });
    const best = Math.min(...scores);
    let pick = lambdas[scores.indexOf(best)];
    for (let i = 0; i < lambdas.length; i++) if (scores[i] <= best * 1.02) pick = lambdas[i];
    if (Number.isFinite(best)) {
      this.lambda = pick;
      this.cvScore = best;
    }
    return fitAll();
  }

  predict(phi) {
    if (!this.beta) return null;
    let x = this.beta[0][0];
    let y = this.beta[0][1];
    for (let j = 0; j < this.n; j++) {
      const zj = (phi[j] - this.mu[j]) / this.sd[j];
      x += this.beta[j + 1][0] * zj;
      y += this.beta[j + 1][1] * zj;
    }
    return [x, y];
  }

  clone() {
    const c = new RidgeMap(this.n, { lambda: this.lambda, stdFloor: this.stdFloor });
    c.mu = Float64Array.from(this.mu);
    c.sd = Float64Array.from(this.sd);
    c.beta = this.beta ? this.beta.map((r) => [r[0], r[1]]) : null;
    return c;
  }

  toJSON() {
    return { n: this.n, lambda: this.lambda, mu: Array.from(this.mu), sd: Array.from(this.sd), beta: this.beta };
  }

  static fromJSON(o, stdFloor) {
    const m = new RidgeMap(o.n, { lambda: o.lambda, stdFloor });
    m.mu = Float64Array.from(o.mu);
    m.sd = Float64Array.from(o.sd);
    m.beta = o.beta;
    return m;
  }
}

// One candidate mapping from the full feature vector to the screen.
// Candidates differ in which features they use (expand) and how they are tuned.
export class GazeModel {
  constructor({ name, label = name, expand, nFeat, stdFloor, lambda = 1, lambdas = undefined, autoLambda = false, poseLocal = false, centered = false, robust = true, borrow = null, minFit = 6, needs = 0, offset = null }) {
    this.name = name;
    // offset(phi) -> [x, y]: a part of the answer that is computed, not learned (the head's own share).
    // It is taken out of the targets before fitting and added back to every prediction.
    this.offset = offset;
    this.borrow = borrow; // name of a robust model whose outlier verdicts this one reuses
    this.robust = robust; // false = one pass, for wide models where refits must stay cheap
    this.lambdas = lambdas;
    this.needs = needs; // minimum length of the full feature vector this model can read
    this.label = label;
    this.expand = expand; // (phiFull, center) -> Float64Array(nFeat)
    this.autoLambda = autoLambda;
    this.poseLocal = poseLocal;
    this.centered = centered; // expansion uses the running mean of the base features
    this.minFit = minFit;
    this.center = null;
    this.map = new RidgeMap(nFeat, { lambda, stdFloor });
  }

  get ready() {
    return this.map.ready;
  }

  fit(samples, weights, { cv = false } = {}) {
    if (samples.length < this.minFit || samples[0].phi.length < this.needs) {
      this.map.beta = null;
      return false;
    }
    if (this.centered) {
      const n = samples[0].phi.length;
      const c = new Float64Array(n);
      let sw = 0;
      for (let i = 0; i < samples.length; i++) {
        sw += weights[i];
        for (let j = 0; j < n; j++) c[j] += weights[i] * samples[i].phi[j];
      }
      if (!(sw > 0)) return false;
      for (let j = 0; j < n; j++) c[j] /= sw;
      this.center = c;
    }
    const rows = samples.map((s) => {
      if (!this.offset) return { phi: this.expand(s.phi, this.center), y: s.y };
      const o = this.offset(s.phi);
      return { phi: this.expand(s.phi, this.center), y: [s.y[0] - o[0], s.y[1] - o[1]] };
    });
    if (cv && this.autoLambda) return this.map.fitCV(rows, weights, { lambdas: this.lambdas, robust: this.robust });
    return this.robust ? this.map.fitRobust(rows, weights) : this.map.fit(rows, weights);
  }

  predict(phi) {
    if (!this.map.ready || phi.length < this.needs) return null;
    const p = this.map.predict(this.expand(phi, this.center));
    if (!this.offset) return p;
    const o = this.offset(phi);
    return [p[0] + o[0], p[1] + o[1]];
  }

  // A frozen copy that keeps predicting the same way while the original learns on.
  snapshot() {
    const map = this.map.clone();
    const center = this.center ? Float64Array.from(this.center) : null;
    const expand = this.expand;
    const offset = this.offset;
    const raw = (phi) => {
      const q = map.predict(expand(phi, center));
      if (!offset) return q;
      const o = offset(phi);
      return [q[0] + o[0], q[1] + o[1]];
    };
    return {
      name: this.name,
      ready: map.ready,
      predict: (phi) => (map.ready && phi.length >= this.needs ? raw(phi) : null),
      toJSON: () => ({ name: this.name, map: map.toJSON(), center: center ? Array.from(center) : null }),
    };
  }
}

// Owns the sample memory and a set of competing models.
//  - Every incoming sample is scored by every model BEFORE anyone learns from it
//    (test-then-train), so the running errors are honest out-of-sample numbers.
//  - The model with the lowest recent error drives the prediction.
//  - Samples taken in a head pose close to the current one count for more, so the
//    memory of an old posture is kept and comes back when the user returns to it.
export class Calibrator {
  constructor({
    models, poseIdx = [], poseSigma = [], defaultModel = null,
    maxSamples = 900, halfLifeMs = 20 * 60000, recencyFloor = 0.25, poseFloor = 0.2,
    window = 30, minScored = 8, cvEvery = 12, deferCv = false, deferFit = false, poseRefitMs = 400,
    prefer = [], preferMargin = 0.7,
  }) {
    this.models = models;
    this.poseIdx = poseIdx;
    this.poseSigma = poseSigma;
    this.defaultModel = defaultModel || models[models.length - 1].name;
    this.maxSamples = maxSamples;
    this.halfLifeMs = halfLifeMs;
    this.recencyFloor = recencyFloor;
    this.poseFloor = poseFloor;
    this.window = window;
    this.minScored = minScored;
    this.cvEvery = cvEvery;
    this.deferCv = deferCv; // true: the lambda search runs from tick(), never inside addSample()
    this.deferFit = deferFit; // true: addSample() only stores; tick() or flush() does the refit
    this.poseRefitMs = poseRefitMs; // shortest pause between refits caused by head movement alone
    // Scores come from touches, and touches happen while the phone is held still. A model that falls
    // apart when the phone moves therefore looks as good as one that does not. The models named
    // here keep working when it moves; another model leads only if it is clearly better at rest:
    // its error has to be below preferMargin times theirs.
    this.prefer = prefer;
    this.preferMargin = preferMargin;
    this.learning = true;
    this.reset();
  }

  reset() {
    this.samples = [];
    this.scores = {};
    for (const m of this.models) {
      this.scores[m.name] = [];
      m.map.beta = null;
      m.center = null;
    }
    this.active = this.defaultModel;
    this.frozen = null;
    this.pose = null;
    this.fitPose = null;
    this.lastFitT = -Infinity;
    this.sinceCv = 0;
    this.dirty = false;
    this.cvCursor = 0;
  }

  model(name) {
    return this.models.find((m) => m.name === name) || null;
  }

  // The leading model, or the richest one that is ready while the leader is not.
  activeModel() {
    const m = this.model(this.active);
    if (m && m.ready) return m;
    for (let i = this.models.length - 1; i >= 0; i--) if (this.models[i].ready) return this.models[i];
    return null;
  }

  get ready() {
    return this.activeModel() !== null;
  }

  predict(phi) {
    const m = this.activeModel();
    return m ? m.predict(phi) : null;
  }

  // Smoothed current head pose, taken from the live feature vector.
  setPose(phi) {
    if (!this.poseIdx.length) return;
    if (!this.pose) {
      this.pose = Float64Array.from(this.poseIdx, (i) => phi[i]);
      return;
    }
    for (let k = 0; k < this.poseIdx.length; k++) this.pose[k] += 0.2 * (phi[this.poseIdx[k]] - this.pose[k]);
  }

  poseDistance2(phi, pose) {
    let d2 = 0;
    for (let k = 0; k < this.poseIdx.length; k++) {
      const d = (phi[this.poseIdx[k]] - pose[k]) / this.poseSigma[k];
      d2 += d * d;
    }
    return d2;
  }

  weightsFor(model, nowT) {
    const local = model.poseLocal && this.pose;
    return this.samples.map((s) => {
      let w = s.w * Math.max(this.recencyFloor, Math.pow(0.5, Math.max(0, nowT - s.t) / this.halfLifeMs));
      if (local) w *= this.poseFloor + (1 - this.poseFloor) * Math.exp(-0.5 * this.poseDistance2(s.phi, this.pose));
      return w;
    });
  }

  // Refit now if samples arrived since the last fit (only matters with deferFit).
  flush(nowT = Date.now()) {
    if (!this.dirty) return false;
    this.dirty = false;
    this.refit({ nowT });
    return true;
  }

  refit({ cv = false, nowT = Date.now(), only = null, countCv = true } = {}) {
    for (const m of this.models) {
      if (only && !only(m)) continue;
      let w = this.weightsFor(m, nowT);
      const trust = m.borrow && this.model(m.borrow) && this.model(m.borrow).map.trust;
      if (trust && trust.length === w.length) w = w.map((v, i) => v * trust[i]);
      m.fit(this.samples, w, { cv });
    }
    this.fitPose = this.pose ? Float64Array.from(this.pose) : null;
    this.lastFitT = nowT;
    if (cv && countCv) this.sinceCv = 0;
  }

  // Call a few times a second: runs a pending lambda search, and refits the pose-aware models once the head has moved.
  tick(nowT = Date.now()) {
    if (this.flush(nowT)) return true;
    if (this.deferCv && this.sinceCv >= this.cvEvery && this.samples.length) {
      // one model per call, so no single pause is long
      const auto = this.models.filter((m) => m.autoLambda);
      const m = auto[this.cvCursor];
      if (m) this.refit({ cv: true, nowT, only: (x) => x === m, countCv: false });
      if (++this.cvCursor >= auto.length) {
        this.cvCursor = 0;
        this.sinceCv = 0;
      }
      return true;
    }
    if (!this.pose || !this.fitPose || nowT - this.lastFitT < this.poseRefitMs || !this.samples.length) return false;
    let d2 = 0;
    for (let k = 0; k < this.pose.length; k++) {
      const d = (this.pose[k] - this.fitPose[k]) / this.poseSigma[k];
      d2 += d * d;
    }
    if (d2 < 0.35 * 0.35) return false;
    this.refit({ nowT, only: (m) => m.poseLocal });
    return true;
  }

  trim() {
    while (this.samples.length > this.maxSamples) {
      const count = {};
      for (const s of this.samples) count[s.src] = (count[s.src] || 0) + 1;
      let top = null;
      for (const k in count) if (top === null || count[k] > count[top]) top = k;
      this.samples.splice(this.samples.findIndex((s) => s.src === top), 1);
    }
  }

  // sample: { phi, y: [x, y], w, src, t (ms, wall clock) }. force = learn even when learning is off.
  addSample(sample, { force = false } = {}) {
    const dist = (p) => (p ? Math.hypot(p[0] - sample.y[0], p[1] - sample.y[1]) : null);
    const lead = this.activeModel();
    const pred = lead ? lead.predict(sample.phi) : null;
    const fpred = this.frozen ? this.frozen.predict(sample.phi) : null;
    const errs = {};
    const preds = {};
    for (const m of this.models) {
      const q = m.predict(sample.phi);
      const e = dist(q);
      errs[m.name] = e;
      preds[m.name] = q ? [Math.round(q[0]), Math.round(q[1])] : null;
      if (e != null) {
        this.scores[m.name].push(e);
        if (this.scores[m.name].length > 200) this.scores[m.name].shift();
      }
    }
    const rec = {
      t: sample.t,
      src: sample.src,
      target: [sample.y[0], sample.y[1]],
      w: sample.w,
      model: lead ? lead.name : null,
      pred,
      err: dist(pred),
      errs,
      preds,
      frozenPred: fpred,
      frozenErr: dist(fpred),
      learned: false,
    };
    this.pickActive();
    if (this.learning || force) {
      this.samples.push({ phi: Float64Array.from(sample.phi), y: [sample.y[0], sample.y[1]], w: sample.w, src: sample.src, t: sample.t });
      this.trim();
      this.sinceCv++;
      if (this.deferFit) this.dirty = true;
      else this.refit({ cv: !this.deferCv && this.sinceCv >= this.cvEvery, nowT: sample.t });
      rec.learned = true;
    }
    return rec;
  }

  // Many labelled frames at once (the follow-the-dot calibration). Not scored.
  addBatch(samples, { nowT = Date.now() } = {}) {
    for (const s of samples) {
      this.samples.push({ phi: Float64Array.from(s.phi), y: [s.y[0], s.y[1]], w: s.w, src: s.src, t: s.t });
    }
    this.trim();
    this.dirty = false;
    this.refit({ cv: true, nowT });
  }

  pickActive() {
    const cost = (m) => {
      const a = this.scores[m.name];
      if (!m.ready || a.length < this.minScored) return Infinity;
      return median(a.slice(-this.window)) * (this.prefer.includes(m.name) ? this.preferMargin : 1);
    };
    let best = null;
    for (const m of this.models) if (!best || cost(m) < cost(best)) best = m;
    if (!best || !Number.isFinite(cost(best)) || best.name === this.active) return;
    const cur = this.model(this.active);
    if (cost(best) < (cur ? cost(cur) : Infinity) * 0.93) this.active = best.name; // needs a clear lead to take over
  }

  leaderboard() {
    return this.models.map((m) => {
      const a = this.scores[m.name];
      return {
        name: m.name,
        label: m.label,
        active: this.activeModel() === m,
        ready: m.ready,
        n: a.length,
        med: a.length ? median(a.slice(-this.window)) : NaN,
        lambda: m.map.lambda,
      };
    });
  }

  // Keep a copy of the current leader as the "no further learning" baseline.
  freeze() {
    const m = this.activeModel();
    this.frozen = m ? m.snapshot() : null;
  }

  // Leave-one-out error of a small buffer: an honest first estimate right after
  // a 9-point calibration, before any test has been run.
  looErrors(maxN = 40, nowT = Date.now()) {
    const lead = this.activeModel();
    const N = this.samples.length;
    if (!lead || N < lead.minFit + 1 || N > maxN) return [];
    const weights = this.weightsFor(lead, nowT);
    const probe = new GazeModel({
      name: 'loo', expand: lead.expand, nFeat: lead.map.n, stdFloor: lead.map.stdFloor,
      lambda: lead.map.lambda, centered: lead.centered, minFit: lead.minFit, offset: lead.offset,
    });
    const out = [];
    for (let i = 0; i < N; i++) {
      const rest = this.samples.filter((_, k) => k !== i);
      const w = weights.filter((_, k) => k !== i);
      if (!probe.fit(rest, w)) continue;
      const p = probe.predict(this.samples[i].phi);
      out.push(Math.hypot(p[0] - this.samples[i].y[0], p[1] - this.samples[i].y[1]));
    }
    return out;
  }

  // nBase: the first nBase features are stored as numbers, the rest (pixels) one byte each.
  toJSON({ nBase = Infinity } = {}) {
    const r5 = (v) => Math.round(v * 1e5) / 1e5;
    const nFeat = this.samples.length ? this.samples[0].phi.length : 0;
    const nb = Math.min(nFeat, nBase);
    return {
      v: 3,
      nFeat,
      nBase: nb,
      samples: this.samples.map((s) => {
        const o = { p: Array.from(s.phi.subarray(0, nb), r5), y: [Math.round(s.y[0]), Math.round(s.y[1])], w: +s.w.toFixed(3), s: s.src, t: s.t };
        if (nFeat > nb) o.q = packInt8(s.phi.subarray(nb));
        return o;
      }),
      active: this.active,
      scores: Object.fromEntries(this.models.map((m) => [m.name, this.scores[m.name].slice(-60).map((e) => Math.round(e))])),
      frozen: this.frozen ? this.frozen.toJSON() : null,
    };
  }

  load(o, { nFeat, nowT = Date.now() } = {}) {
    if (!o || o.v !== 3 || !Array.isArray(o.samples) || (o.samples.length && o.nFeat !== nFeat)) return false;
    this.reset();
    this.samples = o.samples.map((s) => {
      const phi = new Float64Array(o.nFeat);
      phi.set(s.p);
      if (s.q) unpackInt8(s.q, phi, o.nBase);
      return { phi, y: [s.y[0], s.y[1]], w: s.w, src: s.s, t: s.t };
    });
    for (const m of this.models) if (o.scores && Array.isArray(o.scores[m.name])) this.scores[m.name] = o.scores[m.name].slice();
    if (o.active && this.model(o.active)) this.active = o.active;
    this.refit({ cv: true, nowT });
    if (o.frozen && o.frozen.map && o.frozen.map.beta) {
      const src = this.model(o.frozen.name);
      if (src) {
        const map = RidgeMap.fromJSON(o.frozen.map, src.map.stdFloor);
        const center = o.frozen.center ? Float64Array.from(o.frozen.center) : null;
        const frozenJson = o.frozen;
        const raw = (phi) => {
          const q = map.predict(src.expand(phi, center));
          if (!src.offset) return q;
          const o = src.offset(phi);
          return [q[0] + o[0], q[1] + o[1]];
        };
        this.frozen = { name: src.name, ready: true, predict: (phi) => (phi.length >= src.needs ? raw(phi) : null), toJSON: () => frozenJson };
      }
    }
    return true;
  }
}

// The camera's measurement of the head trembles from frame to frame, and every bit of that tremble
// lands on the screen once the head's share of the gaze point is computed from it. While the
// phone moves, though, the head's share and the eyes' share are large and opposite, and a head
// measurement that arrives late leaves the eyes' share uncovered: smoothing then costs more than
// the tremble did. So each number is held only while the gyroscope says the phone is at rest, and
// let go in step with how fast the phone turns. Without a gyroscope reading nothing is smoothed.
export class HeadFilter {
  constructor({ scale = [1, 1, 2, 0.02, 0.02], minCutoff = 1.5, beta = 0.3, gyroGain = 0.4 } = {}) {
    this.scale = scale; // one "ordinary step" of each number, so one setting fits cm and slopes alike
    this.gyroGain = gyroGain; // Hz of extra bandwidth per degree/second of phone rotation
    this.f = scale.map(() => new OneEuro(minCutoff, beta));
  }

  reset() {
    for (const f of this.f) f.reset();
  }

  // values: the head numbers of one frame; tSec: its time; turn: phone rotation rate in deg/s,
  // or null when no gyroscope reading is at hand
  push(values, tSec, turn = null) {
    if (turn == null || !(turn >= 0)) {
      this.reset();
      return Array.from(values);
    }
    const boost = this.gyroGain * turn;
    const out = new Array(values.length);
    for (let j = 0; j < values.length; j++) out[j] = this.f[j].filter(values[j] / this.scale[j], tSec, boost) * this.scale[j];
    return out;
  }
}

// How much of the measured head movement should be believed, and which way round? The camera may
// read a turn of the head too small or too large, its idea of distance rests on an assumed face
// size, and an axis may come out with the opposite sign to the one assumed here.
// Measured on a recording in which the gaze target is known while the head pose varies:
//  - The eyes' share is what a ridge fit of the eye features explains. That fit is linear in its
//    target, so the held-out error of "target minus share × head part" is
//        r(target) − share × r(head part),
//    with r(·) the held-out residual of each column. One pass of blocked cross-validation gives
//    every column's residual; the shares then solve a 2×2 least-squares problem per screen axis.
//  - The eyes run behind a moving target by an amount known only roughly. What that leaves in the
//    target follows the target's velocity, so the velocity sits among the regressors and takes it.
//  - A head part that hardly moved on its own (less than minMove px on the screen, once what
//    follows the target is set aside: people turn the head a little toward what they look at)
//    carries mostly the tremble of its measurement, which would drag its share toward zero. Its
//    share is left as it was. The others are held to the prior only loosely (v0, in px²), which
//    is enough to keep them in place when position and direction cancel each other, as they do
//    when the phone is turned on the spot.
//  - Several recordings may be passed together (in time order; a pause longer than gapMs starts a
//    new one). Each keeps its own level: only what varies inside a recording counts, so a change
//    of light or posture between two recordings is not mistaken for evidence.
//  - The ridge on the eye features is kept weak here (lambda 3, not the 30 the model itself
//    uses): whatever the ridge holds back of the eyes' share moves with the head and would be
//    booked against the head's share.
//  - Half a minute of movement settles the sign of a share and roughly its size, not its second
//    decimal. A camera misreads the head by tens of percent, not by multiples, so the size is
//    kept within `bands`: tighter for the position (the camera's idea of scale is rarely off by
//    more than a fifth; and of all four shares the up-down position is the one a short recording
//    pins down worst, because moving the phone up or down usually changes its distance too),
//    wider for the direction. A share that comes out near zero (below `unsure`), or that changes
//    a lot when a sixth of the recording is left out (above `maxSe`), is taken as not measured.
// parts(phi) -> [position x, position y, direction x, direction y], each in px per unit share.
// Returns { shares, raw, se, moved, measured, before, after, n, recordings }: shares, the estimates
// they come from and their uncertainty (block jackknife), in the order of parts; moved = rms of
// each head part's own movement, px; measured = which shares were settled; before / after = mean
// held-out error (px) with the prior and with the new shares.
export function measureHeadShares(all, { expand, nFeat, stdFloor, lambda = 3, parts, prior = [0.8, 0.8, 0.8, 0.8], folds = 6, v0 = 100, minMove = 25, bands = [[0.8, 1.2], [0.8, 1.2], [0.6, 1.4], [0.6, 1.4]], unsure = 0.3, maxSe = 0.25, gapMs = 2000, minRun = 60 }) {
  // split into recordings and drop the short ones
  const runs = [];
  for (let i = 0; i < all.length; i++) {
    const dt = i ? all[i].t - all[i - 1].t : NaN;
    if (!(dt >= 0 && dt < gapMs)) runs.push([]);
    runs[runs.length - 1].push(all[i]);
  }
  const samples = [];
  const runOf = [];
  let nRuns = 0;
  for (const run of runs) {
    if (run.length < minRun) continue;
    for (const sm of run) {
      samples.push(sm);
      runOf.push(nRuns);
    }
    nRuns++;
  }
  const N = samples.length;
  if (N < 20 * folds) return null;
  const n0 = samples[0].phi.length;
  const center = new Float64Array(n0);
  for (const sm of samples) for (let j = 0; j < n0; j++) center[j] += sm.phi[j] / N;
  const rows = samples.map((sm) => ({ phi: Float64Array.from(expand(sm.phi, center)) }));
  const weights = samples.map((sm) => sm.w);
  const P = samples.map((sm) => parts(sm.phi));
  const Y = samples.map((sm) => [sm.y[0], sm.y[1]]);
  // every recording around its own mean: features, head parts and targets alike
  {
    const wsum = new Float64Array(nRuns);
    const mf = Array.from({ length: nRuns }, () => new Float64Array(nFeat));
    const mp = Array.from({ length: nRuns }, () => new Float64Array(4));
    const my = Array.from({ length: nRuns }, () => new Float64Array(2));
    for (let i = 0; i < N; i++) {
      const r = runOf[i];
      const w = weights[i];
      wsum[r] += w;
      for (let j = 0; j < nFeat; j++) mf[r][j] += w * rows[i].phi[j];
      for (let c = 0; c < 4; c++) mp[r][c] += w * P[i][c];
      for (let c = 0; c < 2; c++) my[r][c] += w * Y[i][c];
    }
    for (let i = 0; i < N; i++) {
      const r = runOf[i];
      if (!(wsum[r] > 0)) continue;
      for (let j = 0; j < nFeat; j++) rows[i].phi[j] -= mf[r][j] / wsum[r];
      for (let c = 0; c < 4; c++) P[i][c] -= mp[r][c] / wsum[r];
      for (let c = 0; c < 2; c++) Y[i][c] -= my[r][c] / wsum[r];
    }
  }
  const st = new RidgeMap(nFeat, { lambda, stdFloor }).stats(rows, weights);
  if (!st) return null;
  const { mu, sd, sw } = st;
  const wScale = N / sw;

  // the target's position and velocity, standardised: [1, x, y, vx, vy]
  const Q = 5;
  const tv = samples.map((sm, i) => {
    const a = samples[Math.max(0, i - 1)];
    const b = samples[Math.min(N - 1, i + 1)];
    const dt = b.t - a.t;
    const ok = dt > 0 && dt < 1000;
    return [1, Y[i][0], Y[i][1], ok ? (b.y[0] - a.y[0]) / dt : 0, ok ? (b.y[1] - a.y[1]) / dt : 0];
  });
  for (let q = 1; q < Q; q++) {
    let mean = 0;
    let dev = 0;
    for (let i = 0; i < N; i++) mean += (weights[i] * tv[i][q]) / sw;
    for (let i = 0; i < N; i++) dev += (weights[i] * (tv[i][q] - mean) ** 2) / sw;
    dev = Math.sqrt(dev) || 1;
    for (let i = 0; i < N; i++) tv[i][q] = (tv[i][q] - mean) / dev;
  }

  // how much each head part moved on its own: what is left after a fit on the target
  const moved = [0, 0, 0, 0];
  {
    const A0 = new Float64Array(Q * Q);
    const B0 = new Float64Array(Q * 4);
    for (let i = 0; i < N; i++) {
      const z = tv[i];
      const w = weights[i];
      for (let r = 0; r < Q; r++) {
        for (let c = 0; c <= r; c++) A0[r * Q + c] += w * z[r] * z[c];
        for (let c = 0; c < 4; c++) B0[r * 4 + c] += w * z[r] * P[i][c];
      }
    }
    for (let r = 0; r < Q; r++) A0[r * Q + r] += 1e-9 * sw;
    if (!solveSPD(A0, B0, Q, 4)) return null;
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 4; c++) {
        let fit = 0;
        for (let r = 0; r < Q; r++) fit += B0[r * 4 + c] * tv[i][r];
        moved[c] += (weights[i] / sw) * (P[i][c] - fit) ** 2;
      }
    }
    for (let c = 0; c < 4; c++) moved[c] = Math.round(Math.sqrt(moved[c]) * 10) / 10;
  }
  const free = moved.map((v) => v >= minMove);

  // held-out residual of the target (2 columns) and of the head parts (4), by contiguous blocks;
  // regressors: the eye features, then the target's velocity
  const K = 6;
  const m = nFeat + 3;
  const foldOf = (i) => Math.min(folds - 1, Math.floor((i * folds) / N));
  const G = Array.from({ length: folds }, () => new Float64Array(m * m));
  const B = Array.from({ length: folds }, () => new Float64Array(m * K));
  const Z = new Float64Array(N * m);
  const T = new Float64Array(N * K);
  for (let i = 0; i < N; i++) {
    const z = Z.subarray(i * m, (i + 1) * m);
    const p = rows[i].phi;
    z[0] = 1;
    for (let j = 0; j < nFeat; j++) z[j + 1] = (p[j] - mu[j]) / sd[j];
    z[nFeat + 1] = tv[i][3];
    z[nFeat + 2] = tv[i][4];
    T[i * K] = Y[i][0];
    T[i * K + 1] = Y[i][1];
    for (let c = 0; c < 4; c++) T[i * K + 2 + c] = P[i][c];
    const w = weights[i] * wScale;
    if (!(w > 0)) continue;
    const g = G[foldOf(i)];
    const b = B[foldOf(i)];
    for (let r = 0; r < m; r++) {
      const wz = w * z[r];
      const row = r * m;
      for (let c = 0; c <= r; c++) g[row + c] += wz * z[c];
      for (let c = 0; c < K; c++) b[r * K + c] += wz * T[i * K + c];
    }
  }
  const GT = new Float64Array(m * m);
  const BT = new Float64Array(m * K);
  for (let k = 0; k < folds; k++) {
    for (let q = 0; q < m * m; q++) GT[q] += G[k][q];
    for (let q = 0; q < m * K; q++) BT[q] += B[k][q];
  }
  const R = new Float64Array(N * K);
  const A = new Float64Array(m * m);
  const X = new Float64Array(m * K);
  for (let k = 0; k < folds; k++) {
    for (let q = 0; q < m * m; q++) A[q] = GT[q] - G[k][q];
    for (let q = 0; q < m * K; q++) X[q] = BT[q] - B[k][q];
    A[0] += 1e-9;
    for (let j = 1; j <= nFeat; j++) A[j * m + j] += lambda;
    for (let j = nFeat + 1; j < m; j++) A[j * m + j] += 1e-6;
    if (!solveSPD(A, X, m, K)) return null;
    for (let i = 0; i < N; i++) {
      if (foldOf(i) !== k) continue;
      const o = i * m;
      for (let c = 0; c < K; c++) {
        let fit = 0;
        for (let r = 0; r < m; r++) fit += X[r * K + c] * Z[o + r];
        R[i * K + c] = T[i * K + c] - fit;
      }
    }
  }

  // per screen axis: r(target) ≈ sPos × r(position part) + sDir × r(direction part)
  const solveAxis = (a, skipFold, h) => {
    let spp = 0;
    let sdd = 0;
    let spd = 0;
    let spy = 0;
    let sdy = 0;
    let n = 0;
    for (let i = 0; i < N; i++) {
      if (foldOf(i) === skipFold) continue;
      const w = weights[i] * wScale * (h ? h[i] : 1);
      const ry = R[i * K + a];
      const rp = R[i * K + 2 + a];
      const rd = R[i * K + 4 + a];
      spp += w * rp * rp;
      sdd += w * rd * rd;
      spd += w * rp * rd;
      spy += w * rp * ry;
      sdy += w * rd * ry;
      n += w;
    }
    // a part that did not move enough is pinned to its prior
    const pen1 = n * (free[a] ? v0 : 1e12);
    const pen2 = n * (free[2 + a] ? v0 : 1e12);
    const a11 = spp + pen1;
    const a22 = sdd + pen2;
    const b1 = spy + pen1 * prior[a];
    const b2 = sdy + pen2 * prior[2 + a];
    const det = a11 * a22 - spd * spd;
    return [(b1 * a22 - b2 * spd) / det, (b2 * a11 - b1 * spd) / det];
  };
  const shares = prior.slice();
  const se = [0, 0, 0, 0];
  const raw = [null, null, null, null]; // the estimate before the band is applied
  const clip = (v, c) => Math.round(Math.sign(v) * clamp(Math.abs(v), bands[c][0], bands[c][1]) * 100) / 100;
  for (let a = 0; a < 2; a++) {
    // A share that comes out too uncertain or too close to zero is pinned to its prior, and the
    // other share of the same axis is then solved again on its own.
    for (let pass = 0; pass < 3 && (free[a] || free[2 + a]); pass++) {
      // two rounds of Huber weights: a blink or a glance away must not decide a share
      let h = null;
      let sol = solveAxis(a, -1, h);
      for (let it = 0; it < 2; it++) {
        const res = new Array(N);
        for (let i = 0; i < N; i++) res[i] = Math.abs(R[i * K + a] - sol[0] * R[i * K + 2 + a] - sol[1] * R[i * K + 4 + a]);
        const s = Math.max(1.4826 * median(res), 5);
        h = res.map((e) => (e <= 1.5 * s ? 1 : (1.5 * s) / e));
        sol = solveAxis(a, -1, h);
      }
      const jack = [];
      for (let k = 0; k < folds; k++) jack.push(solveAxis(a, k, h));
      let dropped = false;
      for (let j = 0; j < 2; j++) {
        const c = a + 2 * j;
        if (!free[c]) continue;
        const mean = jack.reduce((acc, v) => acc + v[j], 0) / folds;
        const seJ = Math.sqrt((jack.reduce((acc, v) => acc + (v[j] - mean) ** 2, 0) * (folds - 1)) / folds);
        raw[c] = Math.round(sol[j] * 100) / 100;
        se[c] = Math.round(seJ * 100) / 100;
        if (Math.abs(sol[j]) < unsure || !(seJ <= maxSe)) {
          free[c] = false;
          dropped = true;
        }
      }
      if (dropped) continue;
      for (let j = 0; j < 2; j++) if (free[a + 2 * j]) shares[a + 2 * j] = clip(sol[j], a + 2 * j);
      break;
    }
  }
  const errWith = (sh) => {
    let e = 0;
    let n = 0;
    for (let i = 0; i < N; i++) {
      const ex = R[i * K] - sh[0] * R[i * K + 2] - sh[2] * R[i * K + 4];
      const ey = R[i * K + 1] - sh[1] * R[i * K + 3] - sh[3] * R[i * K + 5];
      e += weights[i] * Math.min(400, Math.hypot(ex, ey));
      n += weights[i];
    }
    return e / n;
  };
  return { shares, raw, se, moved, measured: free, before: errWith(prior), after: errWith(shares), n: N, recordings: nRuns };
}

// Steady while the eyes rest, quick when they jump.
//  - While the eyes rest, every new frame joins the average, so the estimate gets calmer the
//    longer they stay (up to maxMs).
//  - A point far away has to be confirmed by the next one before the estimate moves there, so a
//    single bad frame cannot throw the cursor across the screen.
//  - A small move inside the radius (the next key, the next word) shows up as the newest few
//    frames disagreeing with the older ones; then the older ones are dropped.
//  - The result passes through a One Euro filter, which removes the last frame-to-frame tremor
//    and steps aside when the estimate really moves.
export class FixationSmoother {
  constructor({ radius = 90, maxMs = 1500, confirm = 2, drift = 0.5, ease = [2, 0.02] } = {}) {
    this.radius = radius;
    this.maxMs = maxMs;
    this.confirm = confirm;
    this.drift = drift; // share of the radius that counts as a small move
    this.fx = ease ? new OneEuro(ease[0], ease[1]) : null;
    this.fy = ease ? new OneEuro(ease[0], ease[1]) : null;
    this.reset();
  }

  reset() {
    this.pts = [];
    this.pending = [];
    this.x = NaN;
    this.y = NaN;
    if (this.fx) {
      this.fx.reset();
      this.fy.reset();
    }
  }

  static mean(pts) {
    let x = 0;
    let y = 0;
    for (const p of pts) {
      x += p.x;
      y += p.y;
    }
    return [x / pts.length, y / pts.length];
  }

  out(t, moved) {
    if (!this.fx) return { x: this.x, y: this.y, moved };
    return { x: this.fx.filter(this.x, t / 1000), y: this.fy.filter(this.y, t / 1000), moved };
  }

  push(x, y, t) {
    const p = { x, y, t };
    if (!this.pts.length) {
      this.pts = [p];
      this.x = x;
      this.y = y;
      return this.out(t, true);
    }
    let moved = false;
    if (Math.hypot(x - this.x, y - this.y) <= this.radius) {
      this.pts.push(p);
      this.pending.length = 0;
    } else {
      this.pending.push(p);
      if (this.pending.length >= this.confirm) {
        const [mx, my] = FixationSmoother.mean(this.pending);
        const tight = this.pending.every((q) => Math.hypot(q.x - mx, q.y - my) <= this.radius);
        if (tight || this.pending.length >= this.confirm + 2) {
          // a real jump (or sustained motion): restart from the new place
          this.pts = this.pending.slice(tight ? 0 : -this.confirm);
          this.pending = [];
          moved = true;
        }
      }
    }
    while (this.pts.length > 1 && t - this.pts[0].t > this.maxMs) this.pts.shift();
    if (this.pts.length >= 8) {
      const fresh = this.pts.slice(-4);
      const [nx, ny] = FixationSmoother.mean(fresh);
      const [ox, oy] = FixationSmoother.mean(this.pts.slice(0, -4));
      if (Math.hypot(nx - ox, ny - oy) > this.drift * this.radius) {
        this.pts = fresh;
        moved = true;
      }
    }
    [this.x, this.y] = FixationSmoother.mean(this.pts);
    return this.out(t, moved);
  }
}

// The moving target of the follow-the-dot calibration: one smooth closed curve
// that reaches every edge, slow enough for the eyes to follow without jumping.
export function pursuitPath(W, H, { durationMs = 20000, marginX = 0.1, marginY = 0.08, cyclesX = 5, cyclesY = 2 } = {}) {
  const ax = W * (0.5 - marginX);
  const ay = H * (0.5 - marginY);
  return (tMs) => {
    const u = clamp(tMs / durationMs, 0, 1);
    return [W / 2 + ax * Math.sin(2 * Math.PI * cyclesX * u), H / 2 + ay * Math.cos(2 * Math.PI * cyclesY * u)];
  };
}

// Eyes trail a moving target by a fraction of a second, and the camera adds its own delay.
// Tries several delays and keeps the one under which the features predict the target best on
// held-out stretches of the recording. When the recording cannot tell the delays apart, a
// typical value is used instead: a wrong delay costs little, an invented one could cost more.
// frames: [{ t (ms since the target started moving), phi }]
export function bestLag(frames, pathFn, {
  expand, nFeat, stdFloor, lambda = 3, lags = [0, 50, 100, 150, 200, 250, 300], fallbackMs = 100, minGain = 0.03,
}) {
  const weights = frames.map(() => 1);
  const feats = frames.map((f) => expand(f.phi, null));
  const curve = lags.map((lagMs) => {
    const rows = frames.map((f, i) => ({ phi: feats[i], y: pathFn(f.t - lagMs) }));
    const map = new RidgeMap(nFeat, { lambda, stdFloor });
    map.cvScore = Infinity;
    map.fitCV(rows, weights, { lambdas: [lambda], robust: false });
    return map.cvScore;
  });
  let k = 0;
  for (let i = 1; i < lags.length; i++) if (curve[i] < curve[k]) k = i;
  if (!Number.isFinite(curve[k])) return { lagMs: fallbackMs, residual: Infinity, sure: false, curve };
  let lagMs = lags[k];
  const inner = k > 0 && k < lags.length - 1;
  if (inner) {
    const bend = curve[k - 1] - 2 * curve[k] + curve[k + 1];
    if (bend > 0) lagMs += (0.5 * (curve[k - 1] - curve[k + 1]) / bend) * (lags[k + 1] - lags[k]);
  }
  const sure = inner && (Math.max(...curve) - curve[k]) / curve[k] >= minGain;
  return { lagMs: sure ? Math.round(lagMs) : fallbackMs, residual: curve[k], sure, curve };
}

// Runs bestLag with each candidate feature set and keeps the answer of the set that predicts best.
export function estimateLag(frames, pathFn, specs, opts = {}) {
  let best = null;
  for (const sp of specs) {
    if (!frames.length || frames[0].phi.length < (sp.needs || 0)) continue;
    const r = bestLag(frames, pathFn, { expand: sp.expand, nFeat: sp.nFeat, stdFloor: sp.stdFloor, lambda: sp.lambda, ...opts });
    r.model = sp.name;
    if (!best || r.residual < best.residual) best = r;
  }
  return best || { lagMs: opts.fallbackMs ?? 100, residual: Infinity, sure: false, curve: [], model: null };
}

// Turns the frames recorded while the eyes followed the moving target into training samples.
// Each sample averages a few neighbouring frames (the target barely moves in that time), which
// removes frame-to-frame noise that would otherwise make the fitted map too timid.
// frames: [{ t (ms since the target started moving), face, phi, blink }]
export function pursuitSamples(frames, pathFn, {
  lagMs = 0, half = 2, step = 2, skipMs = 800, blinkMax = 0.5, maxGapMs = 120, w = 0.35, src = 'pursuit', wallT0 = 0,
} = {}) {
  const out = [];
  const cnt = 2 * half + 1;
  for (let i = half; i < frames.length - half; i += step) {
    if (frames[i].t < skipMs) continue;
    let ok = true;
    for (let k = i - half; k <= i + half && ok; k++) {
      const f = frames[k];
      if (!f.face || !f.phi || f.blink > blinkMax) ok = false;
      else if (k > i - half && f.t - frames[k - 1].t > maxGapMs) ok = false;
    }
    if (!ok) continue;
    const n = frames[i].phi.length;
    const phi = new Float64Array(n);
    let tc = 0;
    for (let k = i - half; k <= i + half; k++) {
      const p = frames[k].phi;
      for (let j = 0; j < n; j++) phi[j] += p[j];
      tc += frames[k].t;
    }
    for (let j = 0; j < n; j++) phi[j] /= cnt;
    tc /= cnt;
    out.push({ phi, y: pathFn(tc - lagMs), w, src, t: wallT0 + tc });
  }
  return out;
}

// Error per screen region. Drives the heat map and the minimum target size.
export class ErrorMap {
  constructor(cols = 3, rows = 4, keep = 12) {
    this.cols = cols;
    this.rows = rows;
    this.keep = keep;
    this.cells = Array.from({ length: cols * rows }, () => []);
    this.recent = [];
  }

  index(xn, yn) {
    const c = clamp(Math.floor(xn * this.cols), 0, this.cols - 1);
    const r = clamp(Math.floor(yn * this.rows), 0, this.rows - 1);
    return r * this.cols + c;
  }

  add(xn, yn, err) {
    if (!(err >= 0)) return;
    const cell = this.cells[this.index(xn, yn)];
    cell.push(err);
    if (cell.length > this.keep) cell.shift();
    this.recent.push(err);
    if (this.recent.length > 200) this.recent.shift();
  }

  cell(c, r) {
    const a = this.cells[r * this.cols + c];
    return { n: a.length, med: a.length ? median(a) : NaN };
  }

  recentMedian(k = 10) {
    return this.recent.length ? median(this.recent.slice(-k)) : NaN;
  }

  reset() {
    this.cells.forEach((c) => (c.length = 0));
    this.recent.length = 0;
  }
}

// Picks the target the user is most likely looking at.
// Evidence accumulates over a short time window instead of trusting one frame,
// which is what makes selection workable when gaze error is larger than the gaps.
export class TargetSelector {
  constructor({ tauMs = 280 } = {}) {
    this.tauMs = tauMs;
    this.targets = [];
    this.score = [];
    this.lastT = null;
  }

  setTargets(targets) {
    this.targets = targets;
    this.score = new Array(targets.length).fill(0);
    this.lastT = null;
  }

  update(x, y, sigma, tMs) {
    const dt = this.lastT === null ? 33 : clamp(tMs - this.lastT, 1, 200);
    this.lastT = tMs;
    const decay = Math.exp(-dt / this.tauMs);
    const gain = dt / this.tauMs;
    for (let i = 0; i < this.targets.length; i++) {
      const t = this.targets[i];
      const dx = Math.max(Math.abs(x - t.cx) - t.w / 2, 0);
      const dy = Math.max(Math.abs(y - t.cy) - t.h / 2, 0);
      const ll = -(dx * dx + dy * dy) / (2 * sigma * sigma);
      this.score[i] = this.score[i] * decay + ll * gain;
    }
  }

  best() {
    if (!this.targets.length || this.lastT === null) return null;
    let mx = -Infinity;
    for (const s of this.score) if (s > mx) mx = s;
    let sum = 0;
    const p = this.score.map((s) => {
      const e = Math.exp(s - mx);
      sum += e;
      return e;
    });
    let bi = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i;
    return { index: bi, id: this.targets[bi].id, p: p[bi] / sum };
  }
}
