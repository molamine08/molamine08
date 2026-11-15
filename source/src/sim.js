// A synthetic user, used for automated tests and for the no-camera demo mode.
// It produces the same 10 features as the real pipeline from a "true" gaze point,
// with noise, blinks, a per-user eye offset and a head pose that can shift.
// It checks that the calibration code works; it is not a prediction of real accuracy.

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

export class SimUser {
  constructor({ W, H, pxPerCm = 60, seed = 7, noise = 1 } = {}) {
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
    this.head = { yaw: 0.02, pitch: -0.05, roll: 0.01, x: 0, y: 0, d: 32 };
    this.user = { kappaH: 0.03, kappaV: -0.02, gainH: 0.42, gainV: 0.34, headShare: 0.15 };
    this.blinkFrom = -1;
    this.blinkUntil = -1;
    this.nextBlink = 2500;
    // Slow tracking error that averaging over a few frames cannot remove.
    this.slow = { h: 0, v: 0, t: null };
  }

  gauss() {
    const u = Math.max(1e-12, this.rand());
    const v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // The eyes land on the new point after a normal saccade latency.
  lookAt(x, y, tMs) {
    if (x === this.tx && y === this.ty) return;
    this.tx = x;
    this.ty = y;
    this.tChange = tMs;
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
  }

  frame(tMs) {
    if (tMs - this.tChange >= this.latencyMs) {
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
    const dt = this.slow.t === null ? 33 : Math.min(500, Math.max(1, tMs - this.slow.t));
    const rho = Math.exp(-dt / 1500);
    const kick = Math.sqrt(1 - rho * rho);
    this.slow.h = rho * this.slow.h + kick * n(0.006);
    this.slow.v = rho * this.slow.v + kick * n(0.01);
    this.slow.t = tMs;
    const hd = this.head;
    const us = this.user;

    // Camera sits 0.8 cm above the top edge, centred.
    const dxCm = (this.gx - this.W / 2) / this.ppc - hd.x;
    const dyCm = this.gy / this.ppc + 0.8 - hd.y;
    const aH = Math.atan2(dxCm, hd.d);
    const aV = Math.atan2(dyCm, hd.d);
    const headYaw = hd.yaw + us.headShare * aH;
    const headPitch = hd.pitch + us.headShare * aV;
    const eH = aH - headYaw + us.kappaH;
    const eV = aV - headPitch + us.kappaV;

    const phi = Float64Array.of(
      0.5 - us.gainH * Math.sin(eH) + this.slow.h + n(0.008),
      0.03 + us.gainV * Math.sin(eV) + this.slow.v + n(0.012),
      -1.6 * eH + n(0.03),
      0.1 - 1.2 * eV + n(0.04),
      headYaw + n(0.01),
      headPitch + n(0.012),
      hd.roll + n(0.008),
      0.5 - hd.x / (1.1 * hd.d) + n(0.003),
      0.45 + hd.y / (1.5 * hd.d) + n(0.003),
      5.67 / hd.d + n(0.002)
    );
    if (blinking) {
      phi[1] += 0.08;
      phi[3] -= 0.3;
    }
    return { face: true, phi, blink: blinking ? 0.9 : 0.05 + 0.05 * this.rand(), distCm: hd.d };
  }
}
