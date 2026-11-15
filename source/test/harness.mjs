// Whole sessions with the synthetic user: the pieces shared by the tests and by one-off experiments.
import {
  GazeModel, Calibrator, summarizeWindow, windowWeight, pursuitPath, pursuitSamples, estimateLag,
} from '../src/engine.js';
import { MODEL_SPECS, DEFAULT_MODEL, MOVE_SAFE, POSE_IDX, POSE_SIGMA, EYE_REF } from '../src/features.js';

export const W = 390;
export const H = 844;

export const makeCal = (specs = MODEL_SPECS, opts = {}) => new Calibrator({
  models: specs.map((s) => new GazeModel(s)), poseIdx: POSE_IDX, poseSigma: POSE_SIGMA, defaultModel: DEFAULT_MODEL, prefer: MOVE_SAFE, ...opts,
});
export const weightOf = (s) => windowWeight(s, { eyeRef: EYE_REF });

// The user looks at a point (or elsewhere, for a careless touch) and taps the target.
export function touch(sim, ring, cal, t0, target, { lookElsewhere = null } = {}) {
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

export function nineDots(sim, ring, cal, t) {
  for (const yn of [0.1, 0.5, 0.9]) for (const xn of [0.12, 0.5, 0.88]) {
    const target = [xn * W, yn * H];
    const s = touch(sim, ring, cal, t, target);
    t += 900;
    if (s.ok) cal.addSample({ phi: s.phi, y: target, w: weightOf(s), src: 'bootstrap', t }, { force: true });
  }
  return t;
}

// The user follows the moving dot; the frames become training samples.
// move: the user also keeps shifting the phone (30 s), so the head pose varies during the recording.
export function pursuit(sim, cal, t0, { lagMs = 120, move = false, periods = undefined } = {}) {
  const durationMs = move ? 30000 : 20000;
  const path = pursuitPath(W, H, move ? { durationMs, cyclesX: 7, cyclesY: 3 } : { durationMs });
  sim.follow((t) => path(t - t0), lagMs);
  if (move) sim.startWobble(t0, 1, periods);
  const frames = [];
  for (let t = t0; t <= t0 + durationMs; t += 33) {
    const f = sim.frame(t);
    cal.setPose(f.phi);
    frames.push({ t: t - t0, face: f.face, phi: f.phi, blink: f.blink });
  }
  sim.stopWobble();
  const usable = frames.filter((f) => f.face && !(f.blink > 0.5) && f.t > 800);
  const lag = estimateLag(usable, path, MODEL_SPECS.filter((m) => m.name === 'B' || m.name === 'D'));
  const samples = pursuitSamples(frames, path, { lagMs: lag.lagMs, wallT0: t0, src: move ? 'pursuit-move' : 'pursuit' });
  cal.addBatch(samples, { nowT: t0 + durationMs });
  return { t: t0 + durationMs + 500, lag, n: samples.length, samples };
}

// Error of every model on fresh points, without learning from them.
export function evalDots(sim, ring, cal, t, n, rand) {
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
