// A synthetic user, used for automated tests and for the no-camera demo mode.
// It produces the same features as the real pipeline (22 numbers plus two tiny eye images)
// from a "true" gaze point, with noise, blinks, a per-user eye offset and a head pose that can shift.
// It checks that the calibration code works; it is not a prediction of real accuracy.
// In particular, how much cleaner the eye images are than the landmarks is an assumption made here.

import { N_BASE, N_FULL, PIX_W, PIX_H, OFF_PIX, OFF_LOG } from './features.js';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A hand does not keep time with the dot on the screen: none of these periods matches the dot's.
const WOBBLE = { d: 8.3, x: 6.1, y: 7.3, yaw: 5.3, pitch: 6.7 };

export class SimUser {
  // pixels: 'good' (eye images carry the gaze), 'noise' (they carry nothing), 'none' (no eye images at all)
  // pixelJitter: how badly the eye cut-outs are aligned, relative to the default assumption
  // poseGain, depthGain: how the camera misreads the head (a turn read too small, a distance too large)
  constructor({ W, H, pxPerCm = 60, seed = 7, noise = 1, pixels = 'good', pixelJitter = 1, poseGain = 1, depthGain = 1 } = {}) {
    this.poseGain = poseGain;
    this.depthGain = depthGain;
    this.pixels = pixels;
    this.pixelJitter = pixelJitter;
    this.W = W;
    this.H = H;
    this.ppc = pxPerCm;
    this.noise = noise;
    this.rand = mulberry32(seed);
    this.gx = W / 2;
    this.gy = H / 2;
    this.tx = W / 2;
    this.ty = H / 2;
    this.tChange = -1e9;
    this.latencyMs = 180;
    this.pursuit = null; // { fn(tMs) -> [x, y], lagMs }
    this.head = { yaw: 0.02, pitch: -0.05, roll: 0.01, x: 0, y: 0, d: 32 };
    this.user = { kappaH: 0.03, kappaV: -0.02, gainH: 0.42, gainV: 0.34, headShare: 0.15 };
    this.blinkFrom = -1;
    this.blinkUntil = -1;
    this.nextBlink = 2500;
    // Slow tracking error that averaging over a few frames cannot remove; each eye has its own.
    this.slow = { hR: 0, hL: 0, vR: 0, vL: 0, t: null };
    // Error of the face-landmark fit, in radians of eye rotation. Both eyes and the gaze
    // blendshapes come out of the same fit, so they share it: averaging them does not remove it.
    this.lm = { h: 0, v: 0 };
    this.cropSlow = [0, 0, 0, 0]; // slow misalignment of each eye cut-out (x, y per eye)
    // Lighting falls across the eye as a gentle ramp, and changes when the user shifts posture.
    // Assumed here: it distorts the plain eye picture and leaves the log picture alone.
    this.light = [0, 0];
    this.wobble = null; // { t0 } while the user keeps moving the phone (the moving calibration)
  }

  gauss() {
    const u = Math.max(1e-12, this.rand());
    const v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // The eyes land on the new point after a normal saccade latency.
  lookAt(x, y, tMs) {
    this.pursuit = null;
    if (x === this.tx && y === this.ty) return;
    this.tx = x;
    this.ty = y;
    this.tChange = tMs;
  }

  // The eyes follow a moving target, a little behind it.
  follow(fn, lagMs = 120) {
    this.pursuit = { fn, lagMs };
  }

  // The user shifts how they hold the phone.
  shiftPosture(amount = 1) {
    const g = () => this.gauss() * amount;
    this.head.yaw += 0.06 * g();
    this.head.pitch += 0.06 * g();
    this.head.roll += 0.03 * g();
    this.head.x += 1.2 * g();
    this.head.y += 1.2 * g();
    this.head.d = Math.min(45, Math.max(24, this.head.d + 3 * g()));
    this.light[0] += 0.3 * g();
    this.light[1] += 0.2 * g();
  }

  // speed: how briskly the phone is moved (1 = the slow drift asked for in the moving calibration)
  // periods: seconds per swing of the distance, the sideways and up-down position, and the two tilts
  startWobble(tMs, speed = 1, periods = WOBBLE) {
    this.wobble = { t0: tMs, speed, periods };
  }

  stopWobble() {
    this.wobble = null;
  }

  frame(tMs) {
    if (this.pursuit) {
      [this.gx, this.gy] = this.pursuit.fn(tMs - this.pursuit.lagMs);
      this.tx = this.gx;
      this.ty = this.gy;
    } else if (tMs - this.tChange >= this.latencyMs) {
      this.gx = this.tx;
      this.gy = this.ty;
    }
    if (tMs >= this.nextBlink) {
      this.blinkFrom = tMs;
      this.blinkUntil = tMs + 150;
      this.nextBlink = tMs + 2000 + this.rand() * 4000;
    }
    const blinking = tMs >= this.blinkFrom && tMs < this.blinkUntil;
    const n = (s) => this.gauss() * s * this.noise;
    const relu = (v) => (v > 0 ? v : 0);
    const dt = this.slow.t === null ? 33 : Math.min(500, Math.max(1, tMs - this.slow.t));
    const rho = Math.exp(-dt / 1500);
    const kick = Math.sqrt(1 - rho * rho);
    for (const k of ['hR', 'hL']) this.slow[k] = rho * this.slow[k] + kick * n(0.002);
    for (const k of ['vR', 'vL']) this.slow[k] = rho * this.slow[k] + kick * n(0.003);
    const rhoLm = Math.exp(-dt / 800);
    const kickLm = Math.sqrt(1 - rhoLm * rhoLm);
    this.lm.h = rhoLm * this.lm.h + kickLm * n(0.017);
    this.lm.v = rhoLm * this.lm.v + kickLm * n(0.035);
    this.slow.t = tMs;
    let hd = this.head;
    let turn = 0; // how fast the phone turns, degrees per second: what its gyroscope would report
    if (this.wobble) {
      const u = ((tMs - this.wobble.t0) / 1000) * this.wobble.speed;
      const sw = (period, phase) => Math.sin((2 * Math.PI * u) / period + phase);
      const rate = (period, phase) => ((2 * Math.PI * this.wobble.speed) / period) * Math.cos((2 * Math.PI * u) / period + phase);
      const pr = this.wobble.periods;
      hd = { ...hd, d: hd.d + 3.5 * sw(pr.d, 0), x: hd.x + sw(pr.x, 1), y: hd.y + sw(pr.y, 2), yaw: hd.yaw + 0.05 * sw(pr.yaw, 3), pitch: hd.pitch + 0.05 * sw(pr.pitch, 4) };
      turn = (Math.hypot(0.05 * rate(pr.yaw, 3), 0.05 * rate(pr.pitch, 4)) * 180) / Math.PI;
    }
    const us = this.user;

    // Camera sits 0.8 cm above the top edge, centred.
    const dxCm = (this.gx - this.W / 2) / this.ppc - hd.x;
    const dyCm = this.gy / this.ppc + 0.8 - hd.y;
    const aH = Math.atan2(dxCm, hd.d);
    const aV = Math.atan2(dyCm, hd.d);
    const headYaw = hd.yaw + us.headShare * aH;
    const headPitch = hd.pitch + us.headShare * aV;
    const eH = aH - headYaw + us.kappaH; // true eye rotation: what the eye images show
    const eV = aV - headPitch + us.kappaV;
    const lH = eH + this.lm.h; // what the landmark fit believes
    const lV = eV + this.lm.v;
    const lid = blinking ? -0.12 : 0;

    const base = Float64Array.of(
      0.5 - us.gainH * Math.sin(lH) + this.slow.hR + n(0.008),
      0.03 + us.gainV * Math.sin(lV) + this.slow.vR + n(0.012) + (blinking ? 0.08 : 0),
      0.52 - us.gainH * Math.sin(lH + 0.01) + this.slow.hL + n(0.008),
      0.03 + us.gainV * Math.sin(lV) + this.slow.vL + n(0.012) + (blinking ? 0.08 : 0),
      0.16 - 0.1 * lV + lid + n(0.01),
      0.15 - 0.06 * lV + n(0.01),
      0.16 - 0.1 * lV + lid + n(0.01),
      0.15 - 0.06 * lV + n(0.01),
      relu(1.6 * lH) + Math.abs(n(0.03)),
      relu(-1.6 * lH) + Math.abs(n(0.03)),
      relu(0.1 - 1.2 * lV) + Math.abs(n(0.04)),
      relu(1.2 * lV - 0.1) + Math.abs(n(0.04)) + (blinking ? 0.3 : 0),
      relu(-1.6 * lH) + Math.abs(n(0.03)),
      relu(1.6 * lH) + Math.abs(n(0.03)),
      relu(0.1 - 1.2 * lV) + Math.abs(n(0.04)),
      relu(1.2 * lV - 0.1) + Math.abs(n(0.04)) + (blinking ? 0.3 : 0),
      headYaw + n(0.01),
      headPitch + n(0.012),
      hd.roll + n(0.008),
      0.5 - hd.x / (1.1 * hd.d) + n(0.003),
      0.45 + hd.y / (1.5 * hd.d) + n(0.003),
      5.67 / hd.d + n(0.002)
    );
    // the head in space as the camera reports it: position in cm, and the slopes of the face direction
    const phi = new Float64Array(this.pixels === 'none' ? N_BASE : N_FULL);
    phi.set(base);
    const dg = this.depthGain;
    const pg = this.poseGain;
    phi.set([-hd.x * dg + n(0.1), -hd.y * dg + n(0.1), hd.d * dg + n(0.3), -Math.tan(headYaw) * pg + n(0.01), -Math.tan(headPitch) * pg + n(0.012)], 22);
    if (this.pixels !== 'none') {
      for (let e = 0; e < 2; e++) {
        const pj = this.pixelJitter;
        for (let k = 0; k < 2; k++) this.cropSlow[e * 2 + k] = rho * this.cropSlow[e * 2 + k] + kick * n((k ? 0.0045 : 0.003) * pj);
        const x0 = -us.gainH * Math.sin(eH + (e ? 0.01 : 0)) + (e ? 0.02 : 0) + this.cropSlow[e * 2] + n(0.004 * pj);
        const y0 = us.gainV * Math.sin(eV) + this.cropSlow[e * 2 + 1] + n(0.004 * pj);
        const lid = blinking ? 0.3 : -0.2 + 0.25 * eV;
        this.eyeImage(phi, OFF_PIX + e * PIX_W * PIX_H, x0, y0, lid, n, this.light);
        this.eyeImage(phi, OFF_LOG + e * PIX_W * PIX_H, x0, y0, lid, n, null);
      }
    }
    return { face: true, phi, blink: blinking ? 0.9 : 0.05 + 0.05 * this.rand(), distCm: hd.d, eyes: null, turn };
  }

  // A tiny picture of one eye: bright sclera, dark iris at (x0, y0), darker above the upper lid.
  // Coordinates are in eye widths from the middle of the eye, like the real cut-out.
  eyeImage(out, o, x0, y0, lidY, n, light) {
    const N = PIX_W * PIX_H;
    let mean = 0;
    for (let r = 0; r < PIX_H; r++) {
      const y = ((r + 0.5) / PIX_H - 0.5) * 0.75;
      for (let c = 0; c < PIX_W; c++) {
        const x = ((c + 0.5) / PIX_W - 0.5) * 1.5;
        let v;
        if (this.pixels === 'noise') {
          v = n(1);
        } else {
          const d2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
          v = 1 - 0.75 * Math.exp(-d2 / (2 * 0.13 * 0.13));
          v *= 1 - 0.5 / (1 + Math.exp((y - lidY) / 0.04));
          if (light) v += light[0] * x + light[1] * y;
          v += n(0.03);
        }
        out[o + r * PIX_W + c] = v;
        mean += v;
      }
    }
    mean /= N;
    let sd = 0;
    for (let i = 0; i < N; i++) sd += (out[o + i] - mean) * (out[o + i] - mean);
    sd = Math.sqrt(sd / N) || 1;
    for (let i = 0; i < N; i++) out[o + i] = (out[o + i] - mean) / sd;
  }
}
