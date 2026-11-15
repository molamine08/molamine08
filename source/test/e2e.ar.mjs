// The AR view on the real pipeline: fake camera fed from a face video, MediaPipe from local files.
// Checks that the camera frame is painted mirrored, that the mesh, the head ray and the gaze are
// drawn where the landmarks say, and that the layer switches work and are remembered.
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
  args: ['--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-video-capture=${resolve('test/fixtures/face.y4m')}`],
});
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ar', permissions: ['camera'] });
const page = await ctx.newPage();
const problems = [];
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/^INFO: /.test(m.text())) problems.push('console: ' + m.text()); });

const cors = { 'access-control-allow-origin': '*' };
await page.route('https://cdn.jsdelivr.net/**', (route) => {
  const m = new URL(route.request().url()).pathname.match(/^\/npm\/@mediapipe\/tasks-vision@1\.2\.3\/(.+)$/);
  if (!m) return route.fulfill({ status: 404, headers: cors, body: 'not found' });
  const file = `${MP_DIR}/${m[1]}`;
  try {
    route.fulfill({ status: 200, headers: { ...cors, 'content-type': file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' }, body: readFileSync(file) });
  } catch (e) {
    route.fulfill({ status: 404, headers: cors, body: 'missing ' + file });
  }
});
await page.route('https://storage.googleapis.com/**', (route) => route.fulfill({ status: 200, headers: { ...cors, 'content-type': 'application/octet-stream' }, body: readFileSync('vendor/face_landmarker.task') }));

const sleep = (ms) => page.waitForTimeout(ms);
await page.addInitScript(() => { DeviceMotionEvent.requestPermission = () => Promise.resolve('granted'); });
await page.goto(`http://localhost:${port}/`);
await page.waitForSelector('#start-go');
await page.locator('#start-go').tap();
await page.waitForSelector('[data-go="ar"]', { timeout: 60000 });
await sleep(2000);

// What is on the canvas, read back in the page: colours around the places where things should be.
const probe = () => page.evaluate(() => {
  const { S } = window.__gazecal;
  const canvas = document.querySelector('#ar-view');
  const c = canvas.getContext('2d');
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const dpr = canvas.width / W;
  const v = document.querySelector('#cam');
  const k = Math.max(W / v.videoWidth, H / v.videoHeight);
  const sx = (v.videoWidth - W / k) / 2;
  const sy = (v.videoHeight - H / k) / 2;
  const lm = S.lastResult.faceLandmarks[0];
  const at = (i) => [W - (lm[i].x * v.videoWidth - sx) * k, (lm[i].y * v.videoHeight - sy) * k];
  // how many pixels in a small square are close to a colour
  const count = (x, y, half, [r, g, b], tol = 60) => {
    const x0 = Math.max(0, Math.round((x - half) * dpr));
    const y0 = Math.max(0, Math.round((y - half) * dpr));
    const n = Math.round(2 * half * dpr);
    const px = c.getImageData(x0, y0, n, n).data;
    let hit = 0;
    for (let i = 0; i < px.length; i += 4) if (Math.abs(px[i] - r) < tol && Math.abs(px[i + 1] - g) < tol && Math.abs(px[i + 2] - b) < tol) hit++;
    return hit;
  };
  const all = c.getImageData(0, 0, canvas.width, canvas.height).data;
  let sum = 0;
  let sq = 0;
  let n = 0;
  for (let i = 0; i < all.length; i += 4 * 97) {
    const g = 0.3 * all[i] + 0.59 * all[i + 1] + 0.11 * all[i + 2];
    sum += g;
    sq += g * g;
    n++;
  }
  const iris = at(468);
  const nose = at(1);
  const d = S.lastResult.facialTransformationMatrixes[0].data;
  const between = Math.hypot(at(473)[0] - iris[0], at(473)[1] - iris[1]);
  const tip = [nose[0] - d[8] * 3 * between, nose[1] - d[9] * 3 * between];
  return {
    size: [canvas.width, canvas.height, W, H],
    picture: { mean: +(sum / n).toFixed(1), sd: +Math.sqrt(sq / n - (sum / n) ** 2).toFixed(1) },
    irisBlue: count(iris[0], iris[1], 12, [96, 165, 250]),
    headAmber: count(tip[0], tip[1], 6, [251, 191, 36]),
    meshLight: count((at(10)[0] + at(152)[0]) / 2, (at(10)[1] + at(152)[1]) / 2, 40, [233, 238, 245], 70),
    gaze: S.gaze ? [Math.round(S.gaze.x), Math.round(S.gaze.y)] : null,
    gazeBlue: S.gaze ? count(S.gaze.x, S.gaze.y, 5, [96, 165, 250]) : 0,
    iris: iris.map(Math.round), nose: nose.map(Math.round), tip: tip.map(Math.round), eyesApart: Math.round(between),
    hint: document.querySelector('#ar-hint').hidden ? '' : document.querySelector('#ar-hint').textContent,
    stat: document.querySelector('#ar-stat').textContent,
    layers: [...document.querySelectorAll('[data-layer]')].map((b) => `${b.dataset.layer}:${b.getAttribute('aria-pressed')}`).join(' '),
    stored: S.settings.ar,
    barHidden: getComputedStyle(document.querySelector('#bar')).display === 'none',
    cursorHidden: document.querySelector('#cursor').hidden,
  };
});

// 1. before any calibration: picture, mesh and head ray, and a hint in place of the gaze
await page.locator('[data-go="ar"]').tap();
await sleep(1500);
const a = await probe();
console.log('before calibration:', JSON.stringify(a));
await page.screenshot({ path: `${SHOTS}/40-ar-before-calibration.png` });
if (a.size[0] !== 780 || a.size[1] !== 1688) problems.push('the canvas is not twice the screen size: ' + a.size);
if (!(a.picture.mean > 15 && a.picture.sd > 8)) problems.push('the camera frame is not on the canvas');
if (!(a.irisBlue > 20)) problems.push('no iris ring where the iris is');
if (!(a.headAmber > 20)) problems.push('no head ray where the face transform puts it');
if (!(a.meshLight > 30)) problems.push('no face mesh over the face');
if (a.hint !== 'عاير أولًا ليظهر اتجاه النظر') problems.push('wrong hint before calibration: ' + a.hint);
if (!/المسافة \d+ سم · دوران الرأس \d+° · ميله \d+°/.test(a.stat)) problems.push('status line: ' + a.stat);
if (!a.barHidden || !a.cursorHidden) problems.push('the AR view should own the whole screen');

// 2. the mirror: the frame on the canvas is the camera frame flipped left to right
const mirror = await page.evaluate(() => {
  const canvas = document.querySelector('#ar-view');
  const v = document.querySelector('#cam');
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const k = Math.max(W / v.videoWidth, H / v.videoHeight);
  const sw = W / k;
  const sh = H / k;
  const plain = document.createElement('canvas');
  plain.width = 39;
  plain.height = 84;
  const pc = plain.getContext('2d', { willReadFrequently: true });
  pc.drawImage(v, (v.videoWidth - sw) / 2, (v.videoHeight - sh) / 2, sw, sh, 0, 0, 39, 84);
  const small = document.createElement('canvas');
  small.width = 39;
  small.height = 84;
  const sc = small.getContext('2d', { willReadFrequently: true });
  sc.drawImage(canvas, 0, 0, 39, 84);
  const A = pc.getImageData(0, 0, 39, 84).data;
  const B = sc.getImageData(0, 0, 39, 84).data;
  const corr = (flip) => {
    let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, n = 0;
    for (let y = 0; y < 84; y++) for (let x = 0; x < 39; x++) {
      const a = A[(y * 39 + (flip ? 38 - x : x)) * 4 + 1];
      const b = B[(y * 39 + x) * 4 + 1];
      sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; n++;
    }
    return (sab / n - (sa / n) * (sb / n)) / Math.sqrt((saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2));
  };
  return { flipped: +corr(true).toFixed(3), plain: +corr(false).toFixed(3) };
});
console.log('canvas against the camera frame, correlation:', JSON.stringify(mirror));
if (!(mirror.flipped > 0.9 && mirror.flipped > mirror.plain)) problems.push('the picture is not the mirrored camera frame');

// 3. layers switch off one by one and the choice is stored
await page.locator('[data-layer="mesh"]').tap();
await sleep(400);
const b = await probe();
console.log('mesh off:', JSON.stringify({ meshLight: b.meshLight, irisBlue: b.irisBlue, headAmber: b.headAmber, layers: b.layers, stored: b.stored }));
await page.screenshot({ path: `${SHOTS}/41-ar-mesh-off.png` });
if (!(b.meshLight < a.meshLight / 3) || !/mesh:false/.test(b.layers) || b.stored.mesh !== false) problems.push('the mesh layer did not switch off');
await page.locator('[data-layer="head"]').tap();
await sleep(400);
const c = await probe();
if (c.headAmber > 2 || c.stored.head !== false) problems.push('the head layer did not switch off');
await page.locator('[data-layer="mesh"]').tap();
await page.locator('[data-layer="head"]').tap();

// 4. calibrated: lines from both eyes to the gaze point, and the ring around it
await page.locator('.close').tap();
await page.waitForSelector('[data-go="ar"]');
await page.evaluate(() => {
  const { takeSample } = window.__gazecal;
  for (const y of [120, 420, 720]) for (const x of [60, 195, 330]) takeSample([x, y], 'bootstrap', performance.now(), 1, { force: true });
});
await sleep(1500);
await page.locator('[data-go="ar"]').tap();
await sleep(1500);
const g = await probe();
console.log('calibrated:', JSON.stringify(g));
await page.screenshot({ path: `${SHOTS}/42-ar-gaze.png` });
if (!g.gaze) problems.push('no gaze estimate after calibration');
if (!(g.gazeBlue > 30)) problems.push('no gaze mark at the estimate');
if (g.hint !== '') problems.push('hint still shown after calibration: ' + g.hint);
if (!/يقود [A-G]/.test(g.stat)) problems.push('status line does not name the leading model: ' + g.stat);
if (!/mesh:true head:true gaze:true/.test(g.layers)) problems.push('layers were not restored: ' + g.layers);

// 5. leaving stops the drawing
await page.locator('.close').tap();
await page.waitForSelector('[data-go="ar"]');
const left = await page.evaluate(() => ({ onFrame: window.__gazecal.S.onFrame, canvas: !!document.querySelector('#ar-view') }));
if (left.onFrame || left.canvas) problems.push('the AR view kept running after it was closed');

console.log(problems.length ? 'PROBLEMS:\n' + problems.join('\n') : 'no console errors, AR view drawn as expected');
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
