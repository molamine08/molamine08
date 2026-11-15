// Runs the real pipeline in headless Chrome: fake camera fed from a face video,
// MediaPipe served from local copies of the exact files the CDN URLs point to.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync } from 'node:fs';
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
await page.waitForSelector('[data-go="boot"]', { timeout: 60000 });
await sleep(2500);
await page.screenshot({ path: `${SHOTS}/22-real-home.png` });

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
    phi: last && last.phi ? Array.from(last.phi, (x) => +x.toFixed(4)) : null,
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
await page.locator('.sheet [data-go="home"]').tap();
await sleep(600);
await page.screenshot({ path: `${SHOTS}/23-real-home-calibrated.png` });
const state = await page.evaluate(() => {
  const { S } = window.__gazecal;
  return { ready: S.cal.ready, samples: S.cal.samples.length, gaze: S.gaze && [Math.round(S.gaze.x), Math.round(S.gaze.y)], rejects: S.rejects };
});
console.log('after boot:', JSON.stringify(state));

console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno console errors');
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
