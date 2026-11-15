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

  filter(x, tSec) {
    if (this.t === null) {
      this.x = x;
      this.dx = 0;
      this.t = tSec;
      return x;
    }
    const dt = Math.max(1e-3, tSec - this.t);
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
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
export function windowWeight(summary, { eyeRef = [0.006, 0.008], motionRef = 60 } = {}) {
  let q = 0;
  for (let j = 0; j < eyeRef.length; j++) q += summary.spread[j] / eyeRef[j];
  q /= eyeRef.length;
  const steady = 1 / (1 + q * q);
  const still = summary.motion > motionRef ? 0.35 : 1;
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

  fit(samples, weights) {
    const n = this.n;
    const N = samples.length;
    let sw = 0;
    for (let i = 0; i < N; i++) sw += weights[i];
    if (N < 3 || !(sw > 0)) {
      this.beta = null;
      return false;
    }
    const mu = new Float64Array(n);
    const sd = new Float64Array(n);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < n; j++) mu[j] += weights[i] * samples[i].phi[j];
    }
    for (let j = 0; j < n; j++) mu[j] /= sw;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < n; j++) {
        const d = samples[i].phi[j] - mu[j];
        sd[j] += weights[i] * d * d;
      }
    }
    for (let j = 0; j < n; j++) sd[j] = Math.max(Math.sqrt(sd[j] / sw), this.stdFloor[j]);

    const m = n + 1;
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const B = Array.from({ length: m }, () => [0, 0]);
    const z = new Float64Array(m);
    const scale = N / sw; // mean weight becomes 1, so lambda keeps one meaning
    for (let i = 0; i < N; i++) {
      const w = weights[i] * scale;
      if (!(w > 0)) continue;
      z[0] = 1;
      for (let j = 0; j < n; j++) z[j + 1] = (samples[i].phi[j] - mu[j]) / sd[j];
      for (let r = 0; r < m; r++) {
        const wz = w * z[r];
        for (let c = r; c < m; c++) A[r][c] += wz * z[c];
        B[r][0] += wz * samples[i].y[0];
        B[r][1] += wz * samples[i].y[1];
      }
    }
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < r; c++) A[r][c] = A[c][r];
    }
    for (let j = 1; j < m; j++) A[j][j] += this.lambda;
    A[0][0] += 1e-9;
    const X = solveLinear(A, B);
    if (!X) {
      this.beta = null;
      return false;
    }
    this.mu = mu;
    this.sd = sd;
    this.beta = X;
    return true;
  }

  // Huber reweighting: a touch the user was not looking at ends up far from the
  // fit and loses most of its vote. Kept soft on purpose (no hard cut-off), so
  // samples from a new posture can still pull the model toward them.
  fitRobust(samples, weights, { iters = 3, k = 1.5, scaleFloor = 20 } = {}) {
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
    return true;
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

// Owns the sample buffer and the live model.
// Every incoming sample is scored BEFORE it is learned from (test-then-train),
// so the running error is an honest out-of-sample number.
export class Calibrator {
  constructor({ nFeat, stdFloor = null, lambda = 1, maxSamples = 300, halfLife = 120, minFit = 5 } = {}) {
    this.nFeat = nFeat;
    this.stdFloor = stdFloor;
    this.lambda = lambda;
    this.maxSamples = maxSamples;
    this.halfLife = halfLife;
    this.minFit = minFit;
    this.samples = [];
    this.model = new RidgeMap(nFeat, { lambda, stdFloor });
    this.frozen = null;
    this.learning = true;
  }

  get ready() {
    return this.model.ready;
  }

  predict(phi) {
    return this.model.predict(phi);
  }

  // sample: { phi, y: [x, y], w, src, t }. force = learn even when learning is off.
  addSample(sample, { force = false } = {}) {
    const pred = this.model.predict(sample.phi);
    const fpred = this.frozen ? this.frozen.predict(sample.phi) : null;
    const rec = {
      t: sample.t,
      src: sample.src,
      target: [sample.y[0], sample.y[1]],
      w: sample.w,
      pred,
      err: pred ? Math.hypot(pred[0] - sample.y[0], pred[1] - sample.y[1]) : null,
      frozenPred: fpred,
      frozenErr: fpred ? Math.hypot(fpred[0] - sample.y[0], fpred[1] - sample.y[1]) : null,
      learned: false,
    };
    if (this.learning || force) {
      this.samples.push({ phi: Float64Array.from(sample.phi), y: [sample.y[0], sample.y[1]], w: sample.w, src: sample.src, t: sample.t });
      if (this.samples.length > this.maxSamples) this.samples.splice(0, this.samples.length - this.maxSamples);
      this.refit();
      rec.learned = true;
    }
    return rec;
  }

  recencyWeights() {
    const N = this.samples.length;
    return this.samples.map((s, i) => s.w * Math.pow(0.5, (N - 1 - i) / this.halfLife));
  }

  refit() {
    if (this.samples.length < this.minFit) {
      this.model.beta = null;
      return false;
    }
    return this.model.fitRobust(this.samples, this.recencyWeights());
  }

  // Keep a copy of the current model as the "no further learning" baseline.
  freeze() {
    this.frozen = this.model.ready ? this.model.clone() : null;
  }

  // Leave-one-out error of the current buffer: an honest first error estimate
  // right after the 9-point calibration, before any test has been run.
  looErrors(maxN = 40) {
    const N = this.samples.length;
    if (N < this.minFit + 1 || N > maxN) return [];
    const weights = this.recencyWeights();
    const out = [];
    for (let i = 0; i < N; i++) {
      const rest = this.samples.filter((_, k) => k !== i);
      const w = weights.filter((_, k) => k !== i);
      const m = new RidgeMap(this.nFeat, { lambda: this.lambda, stdFloor: this.stdFloor });
      if (!m.fit(rest, w)) continue;
      const p = m.predict(this.samples[i].phi);
      out.push(Math.hypot(p[0] - this.samples[i].y[0], p[1] - this.samples[i].y[1]));
    }
    return out;
  }

  reset() {
    this.samples = [];
    this.model.beta = null;
    this.frozen = null;
  }

  toJSON() {
    return {
      v: 1,
      nFeat: this.nFeat,
      samples: this.samples.map((s) => ({ p: Array.from(s.phi), y: s.y, w: s.w, s: s.src, t: s.t })),
      frozen: this.frozen ? this.frozen.toJSON() : null,
    };
  }

  load(o) {
    if (!o || o.v !== 1 || o.nFeat !== this.nFeat || !Array.isArray(o.samples)) return false;
    this.samples = o.samples.map((s) => ({ phi: Float64Array.from(s.p), y: [s.y[0], s.y[1]], w: s.w, src: s.s, t: s.t }));
    this.frozen = o.frozen && o.frozen.beta ? RidgeMap.fromJSON(o.frozen, this.stdFloor) : null;
    this.refit();
    return true;
  }
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
