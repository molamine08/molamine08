// Is the gaze cursor drawn where the estimate is? Renders it at a known point in the real page
// (which is right-to-left) and measures the painted ring and centre dot in a screenshot.
import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../dist/index.html', import.meta.url));
const server = createServer((req, res) => { res.setHeader('content-type', 'text/html; charset=utf-8'); res.end(html); }).listen(0);
const port = server.address().port;
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--no-sandbox'] });
const page = await (await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 })).newPage();
await page.goto(`http://localhost:${port}/?sim=1`);
await page.waitForSelector('#cursor', { state: 'attached' });
const AT = [200, 300];
const D = 60;
await page.evaluate(([x, y, d]) => {
  document.querySelector('.overlay')?.remove();
  document.querySelector('#app').style.visibility = 'hidden';
  document.body.style.background = '#000';
  const c = document.querySelector('#cursor');
  c.hidden = false;
  c.style.transition = 'none';
  c.style.setProperty('--d', `${d}px`);
  c.style.transform = `translate3d(${x}px, ${y}px, 0)`;
}, [...AT, D]);
await page.waitForTimeout(200);
const dir = await page.evaluate(() => getComputedStyle(document.documentElement).direction);
const png = await page.screenshot();
const box = await page.evaluate(async (b64) => {
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  await img.decode();
  const cv = document.createElement('canvas');
  cv.width = img.width; cv.height = img.height;
  const g = cv.getContext('2d');
  g.drawImage(img, 0, 0);
  const px = g.getImageData(0, 0, cv.width, cv.height).data;
  const ring = [1e9, 1e9, -1, -1]; const dot = [0, 0, 0];
  for (let y = 0; y < cv.height; y++) for (let x = 0; x < cv.width; x++) {
    const i = (y * cv.width + x) * 4;
    const r = px[i], gg = px[i + 1], b = px[i + 2];
    if (b > 40 && b > r + 15) { ring[0] = Math.min(ring[0], x); ring[1] = Math.min(ring[1], y); ring[2] = Math.max(ring[2], x); ring[3] = Math.max(ring[3], y); }
    if (b > 235 && r > 80 && gg > 150) { dot[0] += x; dot[1] += y; dot[2]++; } // the solid centre dot and the ring line
  }
  return { ring, scale: cv.width / innerWidth };
}, png.toString('base64'));
const s = box.scale;
const cx = (box.ring[0] + box.ring[2] + 1) / 2 / s;
const cy = (box.ring[1] + box.ring[3] + 1) / 2 / s;
const w = (box.ring[2] - box.ring[0] + 1) / s;
console.log(`page direction ${dir}; estimate at (${AT}); painted ring centre (${cx.toFixed(1)}, ${cy.toFixed(1)}), width ${w.toFixed(1)} px → offset (${(cx - AT[0]).toFixed(1)}, ${(cy - AT[1]).toFixed(1)}) px`);
const bad = Math.abs(cx - AT[0]) > 1 || Math.abs(cy - AT[1]) > 1 || Math.abs(w - D) > 2;
console.log(bad ? 'PROBLEM: the cursor is not drawn on the estimate' : 'cursor is drawn on the estimate');
await browser.close();
server.close();
process.exit(bad ? 1 : 0);
