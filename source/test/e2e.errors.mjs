// Error states of the start flow: camera refused, and face model unreachable.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require('/opt/npm-tools/node_modules/playwright');
const server = createServer((req, res) => { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(readFileSync('dist/index.html')); }).listen(0);
const url = `http://localhost:${server.address().port}/`;
const out = {};

// 1) camera refused
let browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--use-fake-device-for-media-stream'] });
let ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
let page = await ctx.newPage();
await page.goto(url);
await page.locator('#start-go').tap();
await page.waitForSelector('#start .err', { timeout: 15000 });
out.denied = { text: await page.locator('#start .err').innerText(), retry: await page.locator('#start-go').innerText() };
await page.screenshot({ path: 'test/shots/30-denied.png' });
await browser.close();

// 2) camera fine, model host unreachable, then retry succeeds
browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', `--use-file-for-fake-video-capture=${resolve('test/fixtures/face.y4m')}`] });
ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, permissions: ['camera'] });
page = await ctx.newPage();
let online = false;
const cors = { 'access-control-allow-origin': '*' };
await page.route('https://cdn.jsdelivr.net/**', (route) => {
  if (!online) return route.abort('internetdisconnected');
  const m = new URL(route.request().url()).pathname.match(/tasks-vision@1\.2\.3\/(.+)$/);
  const file = `node_modules/@mediapipe/tasks-vision/${m[1]}`;
  route.fulfill({ status: 200, headers: { ...cors, 'content-type': file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript' }, body: readFileSync(file) });
});
await page.route('https://storage.googleapis.com/**', (route) => online
  ? route.fulfill({ status: 200, headers: cors, body: readFileSync('vendor/face_landmarker.task') })
  : route.abort('internetdisconnected'));
await page.goto(url);
await page.locator('#start-go').tap();
await page.waitForSelector('#start .err', { timeout: 15000 });
out.model = { text: await page.locator('#start .err').innerText(), detail: await page.locator('#start .detail').innerText() };
await page.screenshot({ path: 'test/shots/31-model-error.png' });
online = true;
await page.locator('#start-go').tap();
await page.waitForSelector('[data-go="boot"]', { timeout: 60000 });
await page.waitForTimeout(1500);
out.afterRetry = await page.evaluate(() => ({ running: window.__gazecal.S.running, face: window.__gazecal.S.face }));

// 3) no face in front of the camera: status must say so, taps on dots must be refused politely
await page.evaluate(() => { const v = document.querySelector('#cam'); v.srcObject.getVideoTracks()[0].enabled = false; });
await page.waitForTimeout(1500);
out.noFace = await page.evaluate(() => ({ chip: document.querySelector('#chip').innerText, state: document.querySelector('#st-state').innerText }));
await page.locator('[data-go="boot"]').tap();
const dot = page.locator('.dot[data-gaze="dot"]');
await dot.waitFor();
await page.waitForTimeout(650);
await dot.tap();
await page.waitForTimeout(250);
out.noFaceTap = { toast: await page.locator('#toast').innerText(), progress: await page.locator('.prog').innerText() };
await page.screenshot({ path: 'test/shots/32-no-face.png' });
console.log(JSON.stringify(out, null, 1));
await browser.close();
server.close();
