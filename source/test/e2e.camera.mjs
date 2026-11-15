// Runs the real pipeline in headless Chrome: fake camera fed from a face video,
// MediaPipe served from local copies of the exact files the CDN URLs point to.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, writeFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('/opt/npm-tools/node_modules/playwright');

const SHOTS = 'test/shots';
mkdirSync(SHOTS, { recursive: true });
const MP_DIR = 'node_modules/@mediapipe/tasks-vision';

const server = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(readFileSync('dist/index.html'));
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: true,
  args: [
    '--no-sandbox',
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-video-capture=${resolve('test/fixtures/face.y4m')}`,
  ],
});
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
  locale: 'ar',
  acceptDownloads: true,
  permissions: ['camera'],
});
const page = await ctx.newPage();
const problems = [];
const requested = [];
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/^INFO: /.test(m.text())) problems.push('console: ' + m.text()); });

const cors = { 'access-control-allow-origin': '*' };
await page.route('https://cdn.jsdelivr.net/**', (route) => {
  const url = new URL(route.request().url());
  requested.push(url.pathname);
  const m = url.pathname.match(/^\/npm\/@mediapipe\/tasks-vision@1\.2\.3\/(.+)$/);
  if (!m) return route.fulfill({ status: 404, headers: cors, body: 'not found' });
  const file = `${MP_DIR}/${m[1]}`;
  const type = file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript';
  try {
    route.fulfill({ status: 200, headers: { ...cors, 'content-type': type }, body: readFileSync(file) });
  } catch (e) {
    route.fulfill({ status: 404, headers: cors, body: 'missing ' + file });
  }
});
await page.route('https://storage.googleapis.com/**', (route) => {
  requested.push(new URL(route.request().url()).pathname);
  route.fulfill({ status: 200, headers: { ...cors, 'content-type': 'application/octet-stream' }, body: readFileSync('vendor/face_landmarker.task') });
});

const sleep = (ms) => page.waitForTimeout(ms);
await page.goto(`http://localhost:${port}/`);
await page.waitForSelector('#start-go');
await page.screenshot({ path: `${SHOTS}/20-real-start.png` });
await page.locator('#start-go').tap();
await sleep(600);
await page.screenshot({ path: `${SHOTS}/21-real-loading.png` });
await page.waitForSelector('[data-go="pursue"]', { timeout: 60000 });
await sleep(2500);
await page.screenshot({ path: `${SHOTS}/22-real-home.png` });

// the eye cut-outs the model learns from, enlarged so they can be inspected
{
  const url = await page.evaluate(() => {
    const src = document.querySelector('#eye-slot canvas');
    const big = document.createElement('canvas');
    big.width = src.width * 8;
    big.height = src.height * 8;
    const c = big.getContext('2d');
    c.imageSmoothingEnabled = false;
    c.drawImage(src, 0, 0, big.width, big.height);
    return big.toDataURL('image/png');
  });
  writeFileSync(`${SHOTS}/24-eye-crop.png`, Buffer.from(url.split(',')[1], 'base64'));
  const pixInfo = await page.evaluate(() => {
    const { S } = window.__gazecal;
    const f = S.ring.frames[S.ring.frames.length - 1];
    const pix = Array.from(f.phi.slice(27, 171));
    const pixl = Array.from(f.phi.slice(171));
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const sd = (a) => Math.sqrt(mean(a.map((v) => (v - mean(a)) ** 2)));
    const grid = (o) => Array.from({ length: 6 }, (_, r) => pix.slice(o + r * 12, o + r * 12 + 12).map((v) => (v < -1 ? '#' : v < -0.3 ? '+' : v < 0.5 ? '.' : ' ')).join('')).join('\n');
    return { geo: Array.from(f.phi.slice(22, 27), (v) => +v.toFixed(3)), sdLog: +sd(pixl.slice(0, 72)).toFixed(3), len: f.phi.length, hi: f.hi ? f.hi.length : 0, meanR: +mean(pix.slice(0, 72)).toFixed(3), sdR: +sd(pix.slice(0, 72)).toFixed(3), sdL: +sd(pix.slice(72)).toFixed(3), right: grid(0), left: grid(72) };
  });
  console.log(`eye features: phi ${pixInfo.len}, hi ${pixInfo.hi} bytes, right mean ${pixInfo.meanR} sd ${pixInfo.sdR}, left sd ${pixInfo.sdL}, log picture sd ${pixInfo.sdLog}`);
  console.log(`head in space (tx, ty, tz cm; slopes of the face direction): ${JSON.stringify(pixInfo.geo)}`);
  console.log('right eye 12×6 (dark = #):\n' + pixInfo.right + '\nleft eye:\n' + pixInfo.left);
  if (pixInfo.len !== 315 || pixInfo.hi !== 2304 || Math.abs(pixInfo.sdR - 1) > 0.01 || Math.abs(pixInfo.sdLog - 1) > 0.01 || !(pixInfo.geo[2] > 15 && pixInfo.geo[2] < 90)) problems.push('eye image or head features look wrong');
}

const info = await page.evaluate(() => {
  const { S } = window.__gazecal;
  const last = S.ring.frames[S.ring.frames.length - 1];
  const lm = S.lastResult && S.lastResult.faceLandmarks && S.lastResult.faceLandmarks[0];
  const pick = (i) => (lm ? [+(lm[i].x).toFixed(4), +(lm[i].y).toFixed(4)] : null);
  const v = document.querySelector('#cam');
  const bs = S.lastResult && S.lastResult.faceBlendshapes && S.lastResult.faceBlendshapes[0];
  const mat = S.lastResult && S.lastResult.facialTransformationMatrixes && S.lastResult.facialTransformationMatrixes[0];
  return {
    face: S.face, fps: Math.round(S.fps), delegate: S.delegate, dist: Math.round(S.distCm),
    video: [v.videoWidth, v.videoHeight],
    frames: S.ring.frames.length,
    phi: last && last.phi ? Array.from(last.phi.slice(0, 22), (x) => +x.toFixed(4)) : null,
    blink: last ? +last.blink.toFixed(3) : null,
    nLandmarks: lm ? lm.length : 0,
    rightEye: { outer33: pick(33), inner133: pick(133), iris468: pick(468), up159: pick(159), low145: pick(145) },
    leftEye: { inner362: pick(362), outer263: pick(263), iris473: pick(473), up386: pick(386), low374: pick(374) },
    blendshapeNames: bs ? bs.categories.filter((c) => /^eye(Look|Blink)/.test(c.categoryName)).map((c) => `${c.categoryName}=${c.score.toFixed(2)}`) : null,
    matrix: mat ? { rows: mat.rows, cols: mat.columns, data: mat.data.map((x) => +x.toFixed(3)) } : null,
  };
});
console.log(JSON.stringify(info, null, 1));
console.log('requested:', [...new Set(requested)].join('\n           '));

// the whole calibration flow must survive real frames (the face is a still image, so features are constant)
await page.locator('[data-go="boot"]').tap();
for (let i = 0; i < 9; i++) {
  const dot = page.locator('.dot[data-gaze="dot"]');
  await dot.waitFor({ state: 'visible', timeout: 5000 });
  const b = await dot.boundingBox();
  await sleep(620);
  await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
  await sleep(150);
}
await page.waitForSelector('.sheet', { timeout: 5000 });
console.log('boot sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '));
// follow-the-dot calibration on real frames
await page.locator('.sheet [data-go="pursue"]').tap();
await sleep(300);
{
  const dot = page.locator('.dot[data-gaze="dot"]');
  const b = await dot.boundingBox();
  await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
  await sleep(5000);
  await page.screenshot({ path: `${SHOTS}/25-real-pursue.png` });
  const fpsDuring = await page.evaluate(() => Math.round(window.__gazecal.S.fps));
  await page.waitForSelector('.sheet h2', { timeout: 40000 });
  console.log('pursuit sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '), `(fps while recording ${fpsDuring})`);
}
await page.locator('.sheet [data-go="home"]').tap();
await sleep(600);
await page.screenshot({ path: `${SHOTS}/23-real-home-calibrated.png` });
const state = await page.evaluate(() => {
  const { S } = window.__gazecal;
  return { ready: S.cal.ready, samples: S.cal.samples.length, gaze: S.gaze && [Math.round(S.gaze.x), Math.round(S.gaze.y)], rejects: S.rejects };
});
console.log('after calibration:', JSON.stringify(state));

// export with eye pictures
await page.locator('[data-go="stats"]').tap();
await sleep(300);
{
  const [dl] = await Promise.all([page.waitForEvent('download'), page.locator('#s-dl').tap()]);
  const file = `${SHOTS}/export-camera.json`;
  await dl.saveAs(file);
  const d = JSON.parse(readFileSync(file, 'utf8'));
  const withEye = d.pursuit[0].frames.filter((f) => f.e).length;
  console.log(`export: ${(statSync(file).size / 1024).toFixed(0)} KB · video ${JSON.stringify(d.video)} · log ${d.log.length} (eye ${d.log[0].eye ? d.log[0].eye.length : 0} chars) · pursuit ${d.pursuit[0].frames.length} frames, ${withEye} with eye pictures · lag ${d.pursuit[0].lagMs} (${d.pursuit[0].lagSure ? 'found' : 'default'})`);
  if (!d.log[0].eye || d.log[0].eye.length !== 3072 || !withEye) problems.push('export has no eye pictures');
}

// the stored profile must come back after a reload
await sleep(800);
await page.reload();
await page.waitForSelector('#start-go');
await page.locator('#start-go').tap();
await page.waitForSelector('[data-go="pursue"]', { timeout: 60000 });
await sleep(1500);
const back = await page.evaluate(() => {
  const { S } = window.__gazecal;
  return { ready: S.cal.ready, samples: S.cal.samples.length, feat: S.cal.samples[0] ? S.cal.samples[0].phi.length : 0, lead: S.cal.active, stored: Math.round((localStorage.getItem(Object.keys(localStorage).find((k) => k.startsWith('gazecal:v3:'))) || '').length / 1024) };
});
console.log('after reload:', JSON.stringify(back));
if (!back.ready || back.samples !== state.samples) problems.push('profile did not survive a reload');

console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno console errors');
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
