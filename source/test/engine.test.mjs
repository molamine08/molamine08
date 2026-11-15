import test from 'node:test';
import assert from 'node:assert/strict';
import {
  solveLinear, solveSPD, packInt8, unpackInt8, median, percentile, OneEuro, FrameRing, summarizeWindow, windowWeight,
  RidgeMap, GazeModel, Calibrator, FixationSmoother, pursuitPath, pursuitSamples, estimateLag, ErrorMap, TargetSelector,
} from '../src/engine.js';
import { MODEL_SPECS, DEFAULT_MODEL, POSE_IDX, POSE_SIGMA, EYE_REF, N_BASE, N_FULL } from '../src/features.js';
import { SimUser, mulberry32 } from '../src/sim.js';

const W = 390;
const H = 844;

test('solveLinear solves a 3x3 system and flags singular input', () => {
  const X = solveLinear([[2, 1, 0], [1, 3, 1], [0, 1, 4]], [[3], [5], [5]]);
  assert.ok(Math.abs(X[0][0] - 1) < 1e-9 && Math.abs(X[1][0] - 1) < 1e-9 && Math.abs(X[2][0] - 1) < 1e-9);
  assert.equal(solveLinear([[1, 2], [2, 4]], [[1], [2]]), null);
});

test('solveSPD agrees with solveLinear and rejects a matrix that is not positive definite', () => {
  const rand = mulberry32(9);
  const n = 12;
  const R = Array.from({ length: n }, () => Array.from({ length: n }, () => rand() - 0.5));
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => R[i].reduce((s, v, k) => s + v * R[j][k], 0) + (i === j ? 0.5 : 0)));
  const B = Array.from({ length: n }, () => [rand(), rand()]);
  const ref = solveLinear(A, B);
  const flatA = Float64Array.from(A.flat());
  const flatB = Float64Array.from(B.flat());
  assert.ok(solveSPD(flatA, flatB, n, 2));
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) assert.ok(Math.abs(flatB[i * 2 + c] - ref[i][c]) < 1e-8);
  assert.equal(solveSPD(Float64Array.of(1, 2, 2, 1), Float64Array.of(1, 1), 2, 1), false);
});

test('packInt8 keeps pixel values to within one step', () => {
  const v = Float64Array.of(-5, -1.234, 0, 0.017, 0.5, 2.9, 9);
  const out = new Float64Array(9);
  assert.equal(unpackInt8(packInt8(v), out, 2), 7);
  const want = [-127 / 32, -1.234, 0, 0.017, 0.5, 2.9, 127 / 32];
  for (let i = 0; i < 7; i++) assert.ok(Math.abs(out[i + 2] - want[i]) <= 1 / 64 + 1e-9, `${i}: ${out[i + 2]}`);
});

test('median and percentile', () => {
  assert.equal(median([5, 1, 3]), 3);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(percentile([0, 10], 0.25), 2.5);
  assert.ok(Number.isNaN(median([])));
});

test('RidgeMap recovers a known linear map', () => {
  const rand = mulberry32(1);
  const samples = [];
  for (let i = 0; i < 40; i++) {
    const a = rand() - 0.5;
    const b = rand() - 0.5;
    samples.push({ phi: Float64Array.of(a, b), y: [100 + 300 * a - 50 * b, 400 + 20 * a + 600 * b] });
  }
  const m = new RidgeMap(2, { lambda: 1e-6 });
  assert.ok(m.fit(samples, samples.map(() => 1)));
  const p = m.predict(Float64Array.of(0.2, -0.1));
  assert.ok(Math.abs(p[0] - (100 + 60 + 5)) < 0.01, `x=${p[0]}`);
  assert.ok(Math.abs(p[1] - (400 + 4 - 60)) < 0.01, `y=${p[1]}`);
});

test('fitRobust ignores touches the user was not looking at', () => {
  const rand = mulberry32(2);
  const mk = (outlier) => {
    const a = rand() - 0.5;
    const b = rand() - 0.5;
    const y = outlier ? [rand() * W, rand() * H] : [195 + 600 * a, 422 + 900 * b];
    return { phi: Float64Array.of(a, b), y, a, b };
  };
  const samples = [];
  for (let i = 0; i < 60; i++) samples.push(mk(i % 5 === 0)); // 20% outliers
  const w = samples.map(() => 1);
  const plain = new RidgeMap(2, { lambda: 1e-3 });
  plain.fit(samples, w);
  const robust = new RidgeMap(2, { lambda: 1e-3 });
  robust.fitRobust(samples, w);
  const errOf = (m) => {
    const e = [];
    for (let i = 0; i < 200; i++) {
      const a = rand() - 0.5;
      const b = rand() - 0.5;
      const p = m.predict(Float64Array.of(a, b));
      e.push(Math.hypot(p[0] - (195 + 600 * a), p[1] - (422 + 900 * b)));
    }
    return median(e);
  };
  const ePlain = errOf(plain);
  const eRobust = errOf(robust);
  assert.ok(eRobust < ePlain * 0.5, `robust ${eRobust.toFixed(1)} vs plain ${ePlain.toFixed(1)}`);
  assert.ok(eRobust < 15, `robust error ${eRobust.toFixed(1)}px`);
  // the outliers are the samples it stopped trusting
  const trustOut = median(samples.map((_, i) => (i % 5 === 0 ? robust.trust[i] : NaN)).filter((v) => !Number.isNaN(v)));
  const trustIn = median(samples.map((_, i) => (i % 5 !== 0 ? robust.trust[i] : NaN)).filter((v) => !Number.isNaN(v)));
  assert.ok(trustIn === 1 && trustOut < 0.4, `trust in ${trustIn} out ${trustOut}`);
});

test('std floor stops a barely-moving feature from getting a huge coefficient', () => {
  const rand = mulberry32(3);
  const samples = [];
  for (let i = 0; i < 9; i++) {
    const a = rand() - 0.5;
    const still = 0.3 + (rand() - 0.5) * 0.002; // e.g. head yaw during a short calibration
    samples.push({ phi: Float64Array.of(a, still), y: [195 + 600 * a + (rand() - 0.5) * 40, 400] });
  }
  const w = samples.map(() => 1);
  const floored = new RidgeMap(2, { lambda: 1, stdFloor: [0.006, 0.03] });
  floored.fit(samples, w);
  const free = new RidgeMap(2, { lambda: 1, stdFloor: [1e-6, 1e-6] });
  free.fit(samples, w);
  const shiftFloored = Math.abs(floored.predict(Float64Array.of(0, 0.4))[0] - floored.predict(Float64Array.of(0, 0.3))[0]);
  const shiftFree = Math.abs(free.predict(Float64Array.of(0, 0.4))[0] - free.predict(Float64Array.of(0, 0.3))[0]);
  assert.ok(shiftFloored < 60, `floored shift ${shiftFloored.toFixed(1)}px`);
  assert.ok(shiftFree > shiftFloored * 5, `free ${shiftFree.toFixed(0)} vs floored ${shiftFloored.toFixed(1)}`);
});

test('fitCV regularises harder when most features are noise', () => {
  const rand = mulberry32(6);
  const g = () => rand() + rand() + rand() - 1.5;
  const make = (nNoise, N) => {
    const rows = [];
    for (let i = 0; i < N; i++) {
      const a = g();
      const b = g();
      const phi = new Float64Array(2 + nNoise);
      phi[0] = a + 0.3 * g();
      phi[1] = b + 0.3 * g();
      for (let j = 0; j < nNoise; j++) phi[2 + j] = g();
      rows.push({ phi, y: [200 + 150 * a, 400 + 250 * b], a, b });
    }
    return rows;
  };
  const err = (m, rows) => median(rows.map((r) => { const p = m.predict(r.phi); return Math.hypot(p[0] - r.y[0], p[1] - r.y[1]); }));
  const train = make(60, 90);
  const testRows = make(60, 300);
  const w = train.map(() => 1);
  const cv = new RidgeMap(62, { lambda: 0.1 });
  assert.ok(cv.fitCV(train, w));
  const weak = new RidgeMap(62, { lambda: 0.1 });
  weak.fit(train, w);
  assert.ok(cv.lambda >= 10, `picked lambda ${cv.lambda}`);
  assert.ok(Number.isFinite(cv.cvScore));
  assert.ok(err(cv, testRows) < err(weak, testRows) * 0.9, `cv ${err(cv, testRows).toFixed(0)} vs weak ${err(weak, testRows).toFixed(0)}`);
  const cleanTrain = make(0, 90);
  const cvClean = new RidgeMap(2, { lambda: 30 });
  cvClean.fitCV(cleanTrain, cleanTrain.map(() => 1));
  assert.ok(cvClean.lambda < cv.lambda, `clean ${cvClean.lambda} vs noisy ${cv.lambda}`);
});

test('summarizeWindow takes medians and rejects blinks and missing faces', () => {
  const mk = (t, h, blink = 0.1, face = true) => ({ t, face, phi: Float64Array.of(h, 0), blink, motion: 5 });
  const ok = summarizeWindow([mk(0, 0.5), mk(33, 0.51), mk(66, 0.9), mk(99, 0.5), mk(132, 0.49)]);
  assert.ok(ok.ok);
  assert.equal(ok.phi[0], 0.5);
  assert.equal(summarizeWindow([mk(0, 0.5), mk(33, 0.5)]).reason, 'no-face');
  assert.equal(summarizeWindow([mk(0, 0.5, 0.9), mk(33, 0.5, 0.9), mk(66, 0.5, 0.9), mk(99, 0.5)]).reason, 'blink');
  assert.equal(summarizeWindow([mk(0, 0.5, 0.1, false), mk(33, 0.5, 0.1, false), mk(66, 0.5, 0.1, false)]).reason, 'no-face');
  const steady = windowWeight({ spread: Float64Array.of(0.001, 0.001), motion: 5 });
  const shaky = windowWeight({ spread: Float64Array.of(0.02, 0.03), motion: 5 });
  const moving = windowWeight({ spread: Float64Array.of(0.001, 0.001), motion: 200 });
  assert.ok(steady > 0.9 && shaky < 0.15 && moving < steady * 0.5);
});

test('FrameRing drops old frames', () => {
  const r = new FrameRing(1000);
  for (let t = 0; t <= 3000; t += 100) r.push({ t });
  assert.equal(r.frames[0].t, 2000);
  assert.equal(r.window(2500, 2700).length, 3);
});

test('OneEuro smooths jitter but still follows a jump', () => {
  const rand = mulberry32(4);
  const f = new OneEuro();
  const raw = [];
  const out = [];
  for (let i = 0; i < 120; i++) {
    const x = 200 + (rand() - 0.5) * 60;
    raw.push(x);
    out.push(f.filter(x, i / 30));
  }
  const sd = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  assert.ok(sd(out.slice(30)) < sd(raw.slice(30)) * 0.5);
  let y = 0;
  for (let i = 120; i < 135; i++) y = f.filter(800, i / 30);
  assert.ok(y > 700, `after 0.5s at the new point: ${y.toFixed(0)}`);
});

test('FixationSmoother: steady at rest, fast on a jump, deaf to one bad frame', () => {
  const rand = mulberry32(8);
  const fs = new FixationSmoother();
  const out = [];
  let t = 0;
  for (let i = 0; i < 45; i++) {
    t += 33;
    out.push(fs.push(200 + (rand() - 0.5) * 80, 300 + (rand() - 0.5) * 80, t));
  }
  const sd = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  assert.ok(sd(out.slice(15).map((o) => o.x)) < 80 / Math.sqrt(12) * 0.45, `rest jitter ${sd(out.slice(15).map((o) => o.x)).toFixed(1)}`);
  t += 33;
  const blip = fs.push(900, 900, t); // one wild frame
  assert.ok(Math.abs(blip.x - 200) < 20 && !blip.moved);
  t += 33;
  fs.push(205, 295, t);
  let o;
  for (let i = 0; i < 3; i++) {
    t += 33;
    o = fs.push(320 + i, 700, t); // a real jump: the eyes stay at the new place
  }
  assert.ok(Math.abs(o.x - 321) < 5 && Math.abs(o.y - 700) < 5, `after 100 ms: ${o.x.toFixed(0)}, ${o.y.toFixed(0)}`);
});

test('TargetSelector accumulates evidence under noise', () => {
  const rand = mulberry32(5);
  const targets = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) targets.push({ id: `${r}-${c}`, cx: 65 + c * 130, cy: 100 + r * 150, w: 120, h: 140 });
  const sel = new TargetSelector();
  sel.setTargets(targets);
  const goal = targets[7];
  let t = 0;
  for (let i = 0; i < 30; i++) {
    t += 33;
    sel.update(goal.cx + (rand() - 0.5) * 220, goal.cy + (rand() - 0.5) * 220, 80, t);
  }
  assert.equal(sel.best().id, goal.id);
});

test('ErrorMap bins by region', () => {
  const m = new ErrorMap(3, 4);
  m.add(0.1, 0.1, 30);
  m.add(0.1, 0.1, 50);
  m.add(0.9, 0.9, 200);
  assert.equal(m.cell(0, 0).med, 40);
  assert.equal(m.cell(2, 3).med, 200);
  assert.equal(m.cell(1, 1).n, 0);
});

// ---- whole sessions with the synthetic user ----

const makeCal = (specs = MODEL_SPECS, opts = {}) => new Calibrator({
  models: specs.map((s) => new GazeModel(s)), poseIdx: POSE_IDX, poseSigma: POSE_SIGMA, defaultModel: DEFAULT_MODEL, ...opts,
});
const weightOf = (s) => windowWeight(s, { eyeRef: EYE_REF });

// The user looks at a point (or elsewhere, for a careless touch) and taps the target.
function touch(sim, ring, cal, t0, target, { lookElsewhere = null } = {}) {
  const look = lookElsewhere || target;
  sim.lookAt(look[0], look[1], t0);
  for (let t = t0; t <= t0 + 700; t += 33) {
    const f = sim.frame(t);
    cal.setPose(f.phi);
    ring.push({ t, face: f.face, phi: f.phi, blink: f.blink, motion: 5 });
  }
  cal.tick(t0 + 700);
  const tTouch = t0 + 650;
  return summarizeWindow(ring.window(tTouch - 350, tTouch + 50));
}

function nineDots(sim, ring, cal, t) {
  for (const yn of [0.1, 0.5, 0.9]) for (const xn of [0.12, 0.5, 0.88]) {
    const target = [xn * W, yn * H];
    const s = touch(sim, ring, cal, t, target);
    t += 900;
    if (s.ok) cal.addSample({ phi: s.phi, y: target, w: weightOf(s), src: 'bootstrap', t }, { force: true });
  }
  return t;
}

// The user follows the moving dot for 20 s; the frames become training samples.
function pursuit(sim, cal, t0, { lagMs = 120, durationMs = 20000 } = {}) {
  const path = pursuitPath(W, H, { durationMs });
  sim.follow((t) => path(t - t0), lagMs);
  const frames = [];
  for (let t = t0; t <= t0 + durationMs; t += 33) {
    const f = sim.frame(t);
    cal.setPose(f.phi);
    frames.push({ t: t - t0, face: f.face, phi: f.phi, blink: f.blink });
  }
  const usable = frames.filter((f) => f.face && !(f.blink > 0.5) && f.t > 800);
  const lag = estimateLag(usable, path, MODEL_SPECS.filter((m) => m.name === 'B' || m.name === 'D'));
  const samples = pursuitSamples(frames, path, { lagMs: lag.lagMs, wallT0: t0 });
  cal.addBatch(samples, { nowT: t0 + durationMs });
  return { t: t0 + durationMs + 500, lag, n: samples.length };
}

// Error of every model on fresh points, without learning from them.
function evalDots(sim, ring, cal, t, n, rand) {
  const errs = { lead: [] };
  for (const m of cal.models) errs[m.name] = [];
  for (let i = 0; i < n; i++) {
    const target = [(0.06 + 0.88 * rand()) * W, (0.06 + 0.88 * rand()) * H];
    const s = touch(sim, ring, cal, t, target);
    t += 900;
    if (!s.ok) continue;
    const d = (p) => Math.hypot(p[0] - target[0], p[1] - target[1]);
    errs.lead.push(d(cal.predict(s.phi)));
    for (const m of cal.models) {
      const p = m.predict(s.phi);
      if (p) errs[m.name].push(d(p));
    }
  }
  return { t, errs };
}

function runSession({ seed = 11, learning = true, touches = 160, shiftAt = 60, outlierRate = 0.1, pixels = 'good', withPursuit = false, specs = MODEL_SPECS } = {}) {
  const sim = new SimUser({ W, H, seed, pixels });
  const rand = mulberry32(seed * 31 + 1);
  const ring = new FrameRing();
  const cal = makeCal(specs);
  let t = nineDots(sim, ring, cal, 1000);
  const loo = cal.looErrors(40, t);
  let pur = null;
  if (withPursuit) {
    pur = pursuit(sim, cal, t);
    t = pur.t;
  }
  cal.freeze();
  cal.learning = learning;
  const recs = [];
  for (let i = 0; i < touches; i++) {
    if (i === shiftAt) sim.shiftPosture(1);
    const target = [(0.06 + 0.88 * rand()) * W, (0.06 + 0.88 * rand()) * H];
    const outlier = rand() < outlierRate;
    const s = touch(sim, ring, cal, t, target, { lookElsewhere: outlier ? [rand() * W, rand() * H] : null });
    t += 900;
    if (!s.ok) continue;
    const rec = cal.addSample({ phi: s.phi, y: target, w: weightOf(s) * 0.9, src: 'touch', t });
    rec.outlier = outlier;
    rec.i = i;
    recs.push(rec);
  }
  return { cal, recs, loo, sim, ring, t, rand, pur };
}

const clean = (recs, from, to, key) => recs.filter((r) => !r.outlier && r.i >= from && r.i < to && r[key] != null).map((r) => r[key]);
const cleanModel = (recs, from, to, name) => recs.filter((r) => !r.outlier && r.i >= from && r.i < to && r.errs[name] != null).map((r) => r.errs[name]);
const f0 = (v) => (Number.isFinite(v) ? v.toFixed(0) : '—').padStart(4);

test('session: touch learning keeps error down after the user shifts posture', () => {
  const rows = [];
  let wins = 0;
  const seeds = [11, 12, 13, 14, 15, 16, 17, 18];
  for (const seed of seeds) {
    const { recs, loo, cal } = runSession({ seed });
    const before = { a: median(clean(recs, 0, 60, 'err')), f: median(clean(recs, 0, 60, 'frozenErr')) };
    const after = { a: median(clean(recs, 100, 160, 'err')), f: median(clean(recs, 100, 160, 'frozenErr')) };
    rows.push({ seed, loo: median(loo), before, after, active: cal.active });
    if (after.a < after.f) wins++;
    assert.ok(Number.isFinite(after.a) && Number.isFinite(after.f));
  }
  console.log('\nseed | LOO 9 dots | before shift: adaptive / frozen | after shift: adaptive / frozen | leader   (px, 60 px ≈ 1 cm)');
  for (const r of rows) console.log(`${String(r.seed).padStart(4)} | ${f0(r.loo)} | ${f0(r.before.a)} / ${f0(r.before.f)} | ${f0(r.after.a)} / ${f0(r.after.f)} | ${r.active}`);
  const med = (k, j) => median(rows.map((r) => r[k][j]));
  console.log(`median over seeds — before: ${f0(med('before', 'a'))} / ${f0(med('before', 'f'))}   after: ${f0(med('after', 'a'))} / ${f0(med('after', 'f'))}\n`);
  assert.ok(wins >= 6, `adaptive beat frozen after the shift in only ${wins}/${seeds.length} sessions`);
  assert.ok(med('after', 'a') < med('after', 'f') * 0.8, `after shift: adaptive ${f0(med('after', 'a'))} vs frozen ${f0(med('after', 'f'))}`);
  assert.ok(med('before', 'a') <= med('before', 'f') * 1.05, `before shift adaptive should not be worse`);
});

test('session: with learning off no touch is added to the memory', () => {
  const { recs, cal } = runSession({ seed: 21, learning: false, touches: 30 });
  assert.equal(cal.samples.length, 9);
  for (const r of recs) assert.ok(r.learned === false && r.err != null);
});

test('tournament: the eye-image model leads only when the images really carry the gaze', () => {
  const seeds = [41, 42, 43, 44, 45];
  const rows = [];
  for (const pixels of ['good', 'noise']) {
    let leadD = 0;
    const ratio = [];
    const cost = [];
    for (const seed of seeds) {
      const { cal, recs } = runSession({ seed, pixels, withPursuit: true, touches: 90, shiftAt: 1e9, outlierRate: 0.1 });
      if (cal.active === 'D') leadD++;
      const eB = median(cleanModel(recs, 20, 90, 'B'));
      const eD = median(cleanModel(recs, 20, 90, 'D'));
      const eLead = median(clean(recs, 20, 90, 'err'));
      ratio.push(eD / eB);
      cost.push(eLead / Math.min(eB, eD));
      rows.push(`${pixels.padEnd(5)} seed ${seed}: B ${f0(eB)}  D ${f0(eD)}  shown ${f0(eLead)}  leader ${cal.active}  λD ${cal.model('D').map.lambda}`);
    }
    console.log(rows.splice(0).join('\n'));
    if (pixels === 'good') {
      assert.ok(leadD >= 4, `D led in only ${leadD}/${seeds.length} sessions with informative images`);
      assert.ok(median(ratio) < 0.9, `D/B error ratio ${median(ratio).toFixed(2)}`);
    } else {
      assert.equal(leadD, 0, 'D must never lead when the images are noise');
    }
    // what the user saw is close to the better of the two, whichever that was
    assert.ok(median(cost) < 1.12, `${pixels}: shown error is ${median(cost).toFixed(2)}× the best model`);
  }
});

test('follow-the-dot calibration: finds the eye lag and beats nine dots', () => {
  const rows = [];
  for (const seed of [51, 52, 53, 54]) {
    const run = (mode) => {
      const sim = new SimUser({ W, H, seed });
      const ring = new FrameRing();
      const cal = makeCal();
      let t = 1000;
      let pur = null;
      if (mode === 'dots') t = nineDots(sim, ring, cal, t);
      else {
        pur = pursuit(sim, cal, t, { lagMs: 150 });
        t = pur.t;
      }
      const { errs } = evalDots(sim, ring, cal, t, 40, mulberry32(seed + 1000));
      return { errs, pur };
    };
    const dots = run('dots');
    const pur = run('pursuit');
    rows.push({ seed, dots: median(dots.errs.lead), purB: median(pur.errs.B), purD: median(pur.errs.D), lag: pur.pur.lag.lagMs, n: pur.pur.n });
  }
  console.log('\nseed | 9 dots | pursuit B | pursuit D | lag found (true 150) | samples');
  for (const r of rows) console.log(`${String(r.seed).padStart(4)} | ${f0(r.dots)} | ${f0(r.purB)} | ${f0(r.purD)} | ${String(r.lag).padStart(4)} | ${r.n}`);
  for (const r of rows) assert.ok(Math.abs(r.lag - 150) <= 50, `seed ${r.seed}: lag ${r.lag}`);
  // nine dots leave the eye-image model without data; twenty seconds of following feed it
  assert.ok(median(rows.map((r) => r.purD)) < median(rows.map((r) => r.dots)) * 0.6, 'pursuit should beat nine dots');
  assert.ok(median(rows.map((r) => r.purB)) < median(rows.map((r) => r.dots)) * 1.15, 'and must not cost the landmark model anything');
  assert.ok(rows.every((r) => r.n > 150), 'enough samples from 20 s');
});

test('posture memory: coming back to a known posture is cheaper than meeting a new one', () => {
  const flat = MODEL_SPECS.map((s) => ({ ...s, poseLocal: false }));
  const rows = [];
  for (const seed of [61, 62, 63, 64, 65, 66]) {
    const run = (specs) => {
      const sim = new SimUser({ W, H, seed });
      const rand = mulberry32(seed * 7 + 3);
      const ring = new FrameRing();
      const cal = makeCal(specs);
      let t = nineDots(sim, ring, cal, 1000);
      const phase = (n) => {
        const e = [];
        for (let i = 0; i < n; i++) {
          const target = [(0.06 + 0.88 * rand()) * W, (0.06 + 0.88 * rand()) * H];
          const s = touch(sim, ring, cal, t, target);
          t += 900;
          if (!s.ok) continue;
          e.push(cal.addSample({ phi: s.phi, y: target, w: weightOf(s) * 0.9, src: 'touch', t }).err);
        }
        return e;
      };
      phase(45); // posture 1
      const home = { ...sim.head };
      sim.shiftPosture(1.3);
      const fresh = phase(45); // posture 2, never seen
      sim.head = { ...home };
      const back = phase(12); // posture 1 again
      return { fresh: median(fresh.slice(0, 12)), settled: median(fresh.slice(25)), back: median(back) };
    };
    rows.push({ seed, local: run(MODEL_SPECS), flat: run(flat) });
  }
  console.log('\nseed | pose-aware: new posture first 12 / settled / back home | same without pose weighting');
  for (const r of rows) console.log(`${String(r.seed).padStart(4)} | ${f0(r.local.fresh)} / ${f0(r.local.settled)} / ${f0(r.local.back)} | ${f0(r.flat.fresh)} / ${f0(r.flat.settled)} / ${f0(r.flat.back)}`);
  const med = (k, j) => median(rows.map((r) => r[k][j]));
  console.log(`median — pose-aware ${f0(med('local', 'fresh'))} / ${f0(med('local', 'settled'))} / ${f0(med('local', 'back'))}   flat ${f0(med('flat', 'fresh'))} / ${f0(med('flat', 'settled'))} / ${f0(med('flat', 'back'))}\n`);
  assert.ok(med('local', 'back') < med('local', 'fresh'), 'returning home should cost less than a new posture');
  // The synthetic posture effect is nearly linear, so pose weighting has little to win here; the case for it
  // is the replay of a real session (1.39 cm against 1.56 cm). Here it only has to stay harmless.
  assert.ok(med('local', 'back') <= med('flat', 'back') * 1.15, 'pose weighting must not hurt the return');
  assert.ok(med('local', 'settled') <= med('flat', 'settled') * 1.15, 'pose weighting must not hurt the settled error');
});

test('calibrator survives a save / load round trip, eye images included', () => {
  const { cal, t } = runSession({ seed: 31, touches: 40, withPursuit: true });
  const json = JSON.stringify(cal.toJSON({ nBase: N_BASE }));
  const perSample = json.length / cal.samples.length;
  assert.ok(perSample < 520, `${perSample.toFixed(0)} characters per stored sample`);
  const copy = makeCal();
  assert.ok(copy.load(JSON.parse(json), { nFeat: N_FULL, nowT: t }));
  assert.equal(copy.samples.length, cal.samples.length);
  assert.equal(copy.active, cal.active);
  copy.pose = cal.pose ? Float64Array.from(cal.pose) : null;
  copy.refit({ cv: true, nowT: t });
  cal.refit({ cv: true, nowT: t });
  const phi = cal.samples[cal.samples.length - 3].phi;
  for (const m of cal.models) {
    const a = m.predict(phi);
    const b = copy.model(m.name).predict(phi);
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1]) < 6, `${m.name}: ${Math.hypot(a[0] - b[0], a[1] - b[1]).toFixed(2)} px apart`);
  }
  assert.ok(copy.frozen && copy.frozen.ready);
  assert.equal(copy.load({ v: 2, nBase: 22, samples: [] }, { nFeat: N_FULL }), false);
  assert.equal(makeCal().load(JSON.parse(json), { nFeat: N_BASE }), false);
});

test('a user without eye images still gets the landmark models', () => {
  const { cal, recs } = runSession({ seed: 71, pixels: 'none', touches: 40, shiftAt: 1e9 });
  assert.equal(cal.model('D').ready, false);
  assert.notEqual(cal.active, 'D');
  assert.ok(median(clean(recs, 10, 40, 'err')) < 150);
});

test('cost of the widest model stays small', () => {
  const { cal, t } = runSession({ seed: 81, touches: 10, withPursuit: true });
  const D = cal.model('D');
  const big = [];
  while (big.length < 900) big.push(...cal.samples);
  cal.samples = big.slice(0, 900);
  let t0 = performance.now();
  cal.refit({ nowT: t, only: (m) => m === D });
  const fitMs = performance.now() - t0;
  t0 = performance.now();
  cal.refit({ cv: true, nowT: t, only: (m) => m === D });
  const cvMs = performance.now() - t0;
  console.log(`\nmodel D on 900 samples: refit ${fitMs.toFixed(0)} ms, refit with λ search ${cvMs.toFixed(0)} ms (this machine)\n`);
  assert.ok(fitMs < 250 && cvMs < 1500);
});
