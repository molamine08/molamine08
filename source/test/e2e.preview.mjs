// The page opened inside another app's preview (sandboxed iframe), and the
// "try without camera" escape hatch after the camera is refused.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
const require = createRequire(import.meta.url);
const { chromium } = require('/opt/npm-tools/node_modules/playwright');

const server = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  if (req.url.startsWith('/wrap')) {
    res.end('<!DOCTYPE html><html><body style="margin:0"><iframe src="/app" sandbox="allow-scripts" style="border:0;width:100vw;height:100vh"></iframe></body></html>');
  } else {
    res.end(readFileSync('dist/index.html'));
  }
}).listen(0);
const base = `http://localhost:${server.address().port}`;
const out = {};
const problems = [];

const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--use-fake-device-for-media-stream'] });
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });

async function bootIn(scope, page) {
  await scope.locator('[data-go="boot"]').tap();
  for (let i = 0; i < 9; i++) {
    const dot = scope.locator('.dot[data-gaze="dot"]');
    await dot.waitFor({ state: 'visible', timeout: 5000 });
    await page.waitForTimeout(600);
    await dot.tap();
    await page.waitForTimeout(160);
    if (await scope.locator('.sheet').count()) break;
    // a simulated blink can refuse a tap: try the same dot again
    const again = scope.locator('.dot[data-gaze="dot"]');
    if (await again.count() && (await scope.locator('.prog').innerText()).startsWith(`${i + 1} /`)) {
      await page.waitForTimeout(400);
      await again.tap();
      await page.waitForTimeout(160);
    }
  }
  await scope.locator('.sheet').waitFor({ timeout: 5000 });
  return (await scope.locator('.sheet').innerText()).replace(/\n+/g, ' | ');
}

// 1) inside a sandboxed iframe, like an in-app file preview
let page = await ctx.newPage();
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
await page.goto(`${base}/wrap`);
const frame = page.frameLocator('iframe');
await frame.locator('#start-sim').waitFor();
out.preview = {
  notice: await frame.locator('#start .warn').innerText(),
  primary: await frame.locator('#start .btn.primary').innerText(),
  secondary: await frame.locator('#start .btn.quiet').innerText(),
};
await page.screenshot({ path: 'test/shots/40-preview-start.png' });
await frame.locator('#start-sim').tap();
await frame.locator('[data-go="boot"]').waitFor();
out.previewChip = await frame.locator('#chip').innerText();
out.previewBoot = await bootIn(frame, page);
await page.close();

// 2) top level, camera refused, then the no-camera demo
page = await ctx.newPage();
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
await page.goto(base + '/');
out.topLevel = { primary: await page.locator('#start .btn.primary').innerText(), hasNotice: await page.locator('#start .warn').count() };
await page.locator('#start-go').tap();
await page.locator('#start .err').waitFor({ timeout: 15000 });
out.denied = await page.locator('#start .err').innerText();
await page.screenshot({ path: 'test/shots/41-denied.png' });
await page.locator('#start-sim').tap();
await page.locator('[data-go="boot"]').waitFor();
out.deniedThenSim = { chip: await page.locator('#chip').innerText(), boot: await bootIn(page, page) };

console.log(JSON.stringify(out, null, 1));
console.log(problems.length ? 'PROBLEMS:\n' + problems.join('\n') : 'no page errors');
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
