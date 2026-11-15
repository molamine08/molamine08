import test from 'node:test';
import assert from 'node:assert/strict';
import {
  solveLinear, median, percentile, OneEuro, FrameRing, summarizeWindow, windowWeight,
  RidgeMap, Calibrator, ErrorMap, TargetSelector,
} from '../src/engine.js';
import { STD_FLOOR, FEATURE_NAMES } from '../src/features.js';
import { SimUser, mulberry32 } from '../src/sim.js';

const W = 390;
const H = 844;

test('solveLinear solves a 3x3 system and flags singular input', () => {
  const X = solveLinear([[2, 1, 0], [1, 3, 1], [0, 1, 4]], [[3], [5], [5]]);
  assert.ok(Math.abs(X[0][0] - 1) < 1e-9 && Math.abs(X[1][0] - 1) < 1e-9 && Math.abs(X[2][0] - 1) < 1e-9);
  assert.equal(solveLinear([[1, 2], [2, 4]], [[1], [2]]), null);
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
  // Later the head really moves by 0.1 rad: the floored model must barely react.
  const shiftFloored = Math.abs(floored.predict(Float64Array.of(0, 0.4))[0] - floored.predict(Float64Array.of(0, 0.3))[0]);
  const shiftFree = Math.abs(free.predict(Float64Array.of(0, 0.4))[0] - free.predict(Float64Array.of(0, 0.3))[0]);
  assert.ok(shiftFloored < 60, `floored shift ${shiftFloored.toFixed(1)}px`);
  assert.ok(shiftFree > shiftFloored * 5, `free ${shiftFree.toFixed(0)} vs floored ${shiftFloored.toFixed(1)}`);
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

// ---- end-to-end session with the synthetic user ----

function touch(sim, ring, t0, target, { lookElsewhere = null } = {}) {
  const look = lookElsewhere || target;
  sim.lookAt(look[0], look[1], t0);
  for (let t = t0; t <= t0 + 700; t += 33) {
    const f = sim.frame(t);
    ring.push({ t, face: f.face, phi: f.phi, blink: f.blink, motion: 5 });
  }
  const tTouch = t0 + 650;
  return summarizeWindow(ring.window(tTouch - 350, tTouch + 50));
}

function runSession({ seed = 11, learning = true, touches = 160, shiftAt = 60, outlierRate = 0.1 } = {}) {
  const sim = new SimUser({ W, H, seed });
  const rand = mulberry32(seed * 31 + 1);
  const ring = new FrameRing();
  const cal = new Calibrator({ nFeat: FEATURE_NAMES.length, stdFloor: STD_FLOOR });
  let t = 1000;
  const xs = [0.12, 0.5, 0.88];
  const ys = [0.1, 0.5, 0.9];
  for (const yn of ys) for (const xn of xs) {
    const target = [xn * W, yn * H];
    const s = touch(sim, ring, t, target);
    t += 900;
    if (s.ok) cal.addSample({ phi: s.phi, y: target, w: windowWeight(s), src: 'bootstrap', t }, { force: true });
  }
  const loo = cal.looErrors();
  cal.freeze();
  cal.learning = learning;
  const recs = [];
  for (let i = 0; i < touches; i++) {
    if (i === shiftAt) sim.shiftPosture(1);
    const target = [(0.06 + 0.88 * rand()) * W, (0.06 + 0.88 * rand()) * H];
    const outlier = rand() < outlierRate;
    const s = touch(sim, ring, t, target, { lookElsewhere: outlier ? [rand() * W, rand() * H] : null });
    t += 900;
    if (!s.ok) continue;
    const rec = cal.addSample({ phi: s.phi, y: target, w: windowWeight(s) * 0.9, src: 'touch', t });
    rec.outlier = outlier;
    rec.i = i;
    recs.push(rec);
  }
  return { cal, recs, loo };
}

const clean = (recs, from, to, key) => recs.filter((r) => !r.outlier && r.i >= from && r.i < to && r[key] != null).map((r) => r[key]);

test('session: touch calibration keeps error down after the user shifts posture', () => {
  const rows = [];
  let wins = 0;
  const seeds = [11, 12, 13, 14, 15, 16, 17, 18];
  for (const seed of seeds) {
    const { recs, loo } = runSession({ seed });
    const before = { a: median(clean(recs, 0, 60, 'err')), f: median(clean(recs, 0, 60, 'frozenErr')) };
    const after = { a: median(clean(recs, 100, 160, 'err')), f: median(clean(recs, 100, 160, 'frozenErr')) };
    rows.push({ seed, loo: median(loo), before, after });
    if (after.a < after.f) wins++;
    assert.ok(Number.isFinite(after.a) && Number.isFinite(after.f));
  }
  console.log('\nseed | LOO after 9 dots | before shift: adaptive / frozen | after shift: adaptive / frozen  (px, 60 px ≈ 1 cm)');
  for (const r of rows) {
    console.log(`${String(r.seed).padStart(4)} | ${r.loo.toFixed(0).padStart(6)} | ${r.before.a.toFixed(0).padStart(6)} / ${r.before.f.toFixed(0).padStart(4)} | ${r.after.a.toFixed(0).padStart(6)} / ${r.after.f.toFixed(0).padStart(4)}`);
  }
  const medAfterA = median(rows.map((r) => r.after.a));
  const medAfterF = median(rows.map((r) => r.after.f));
  const medBeforeA = median(rows.map((r) => r.before.a));
  const medBeforeF = median(rows.map((r) => r.before.f));
  console.log(`median over seeds — before: ${medBeforeA.toFixed(0)} / ${medBeforeF.toFixed(0)}   after: ${medAfterA.toFixed(0)} / ${medAfterF.toFixed(0)}\n`);
  assert.ok(wins >= 6, `adaptive beat frozen after the shift in only ${wins}/${seeds.length} sessions`);
  assert.ok(medAfterA < medAfterF * 0.8, `after shift: adaptive ${medAfterA.toFixed(0)} vs frozen ${medAfterF.toFixed(0)}`);
  assert.ok(medBeforeA <= medBeforeF * 1.05, `before shift adaptive should not be worse: ${medBeforeA.toFixed(0)} vs ${medBeforeF.toFixed(0)}`);
});

test('session: with learning off the model stays exactly the frozen baseline', () => {
  const { recs, cal } = runSession({ seed: 21, learning: false, touches: 30 });
  assert.equal(cal.samples.length, 9);
  for (const r of recs) assert.ok(Math.abs(r.err - r.frozenErr) < 1e-9 && r.learned === false);
});

test('calibrator survives a save / load round trip', () => {
  const { cal } = runSession({ seed: 31, touches: 20 });
  const copy = new Calibrator({ nFeat: FEATURE_NAMES.length, stdFloor: STD_FLOOR });
  assert.ok(copy.load(JSON.parse(JSON.stringify(cal.toJSON()))));
  const phi = cal.samples[3].phi;
  const a = cal.predict(phi);
  const b = copy.predict(phi);
  assert.ok(Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6);
  assert.ok(copy.frozen && copy.frozen.ready);
  assert.equal(copy.load({ v: 1, nFeat: 3, samples: [] }), false);
});
