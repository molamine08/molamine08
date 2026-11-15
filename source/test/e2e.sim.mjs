// Drives the built page in headless Chrome, in simulation mode, through every screen.
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, statSync } from 'node:fs';

const require = createRequire(import.meta.url);
const { chromium } = require('/opt/npm-tools/node_modules/playwright');

const SHOTS = 'test/shots';
mkdirSync(SHOTS, { recursive: true });

const server = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(readFileSync('dist/index.html'));
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 3,
  isMobile: true,
  hasTouch: true,
  locale: 'ar',
  acceptDownloads: true,
  userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
});
const page = await ctx.newPage();
const problems = [];
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') problems.push('console: ' + m.text()); });

const shot = (name) => page.screenshot({ path: `${SHOTS}/${name}.png` });
const tapSel = async (sel) => { await page.locator(sel).first().tap(); };
const sleep = (ms) => page.waitForTimeout(ms);

async function tapDots(n) {
  for (let i = 0; i < n; i++) {
    const dot = page.locator('.dot[data-gaze="dot"]');
    await dot.waitFor({ state: 'visible', timeout: 5000 });
    const b = await dot.boundingBox();
    await sleep(560);
    await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
    await sleep(120);
    // a blink can reject a tap: retry on the same dot
    for (let k = 0; k < 4; k++) {
      const still = await dot.count() ? await dot.boundingBox().catch(() => null) : null;
      if (!still || Math.abs(still.x - b.x) > 1 || Math.abs(still.y - b.y) > 1) break;
      await sleep(350);
      await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
      await sleep(120);
    }
  }
}

await page.goto(`http://localhost:${port}/?sim=1`);
await page.waitForSelector('#start-go');
await shot('01-start');
await tapSel('#start-go');
await page.waitForSelector('[data-go="pursue"]');
await sleep(400);
await shot('02-home-empty');

// first calibration
await tapSel('[data-go="boot"]');
await sleep(300);
await shot('03-boot-dot');
await tapDots(9);
await page.waitForSelector('.sheet');
await shot('04-boot-done');
console.log('boot sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '));

// follow-the-dot calibration
await tapSel('.sheet [data-go="pursue"]');
await sleep(300);
await shot('04b-pursue-hint');
{
  const dot = page.locator('.dot[data-gaze="dot"]');
  const b = await dot.boundingBox();
  await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
  await sleep(4000);
  await shot('04c-pursue-moving');
  await page.waitForSelector('.sheet h2', { timeout: 30000 });
  await sleep(150);
  await shot('04d-pursue-done');
  console.log('pursuit sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '));
  const st = await page.evaluate(() => {
    const { S } = window.__gazecal;
    const run = S.pursuitLog[S.pursuitLog.length - 1];
    return { samples: S.cal.samples.length, lag: run.lagMs, sure: run.lagSure, via: run.lagModel, curve: run.lagCurve, frames: run.frames.length, board: S.cal.leaderboard().map((m) => `${m.name}:${m.ready ? 'ready' : 'no'} λ${m.lambda}`) };
  });
  console.log('pursuit state:', JSON.stringify(st));
}

// moving calibration: the synthetic user keeps shifting the phone while following the dot
await tapSel('.sheet [data-go="home"]');
await sleep(300);
await shot('04e-home-after-pursuit');
await tapSel('[data-go="pursueMove"]');
await sleep(300);
await shot('04f-move-hint');
{
  const dot = page.locator('.dot[data-gaze="dot"]');
  const b = await dot.boundingBox();
  await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
  await page.waitForSelector('.sheet h2', { timeout: 45000 });
  await sleep(150);
  await shot('04g-move-done');
  console.log('moving sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '));
}

// accuracy test
await tapSel('.sheet [data-go="test"]');
await sleep(300);
await tapDots(3);
await sleep(200);
await shot('05-test-ghost');
await tapDots(17);
await page.waitForSelector('.sheet');
await shot('06-test-done');
console.log('test sheet:', (await page.locator('.sheet').innerText()).replace(/\n+/g, ' | '));

// free use
await tapSel('.sheet [data-go="home"]');
await sleep(300);
await shot('07-home-ready');
await tapSel('[data-go="free"]');
await sleep(300);
for (let round = 0; round < 3; round++) {
  const code = (await page.locator('#f-code').innerText()).trim();
  for (const ch of code) {
    await tapSel(`.key[data-k="${ch}"]`);
    await sleep(140);
  }
  if (round === 0) await shot('08-free-typing');
  await tapSel('.key[data-k="ok"]');
  await sleep(200);
  if (round === 0) await shot('09-free-next');
  await tapSel('.next');
  await sleep(200);
}
// wrong code path
await tapSel('.key[data-k="1"]');
await tapSel('.key[data-k="ok"]');
await sleep(150);
await shot('10-free-wrong');
console.log('free stat:', await page.locator('#f-stat').innerText());

// look + tap
await tapSel('#back');
await sleep(250);
await page.evaluate(() => { window.__gazecal.S.anchor = null; }); // as if the posture had changed: the screen must ask for one touch first
await tapSel('[data-go="pick"]');
await sleep(500);
let anchors = 0;
let fixes = 0;
async function pickOnce(i, shots) {
  const a = page.locator('.dot.anchor');
  if (await a.count()) {
    anchors++;
    const b = await a.boundingBox();
    await sleep(560);
    if (shots) await shot('11a-pick-anchor');
    await page.touchscreen.tap(b.x + b.width / 2, b.y + b.height / 2);
    await sleep(450);
  }
  await sleep(750);
  if (shots && i === 2) await shot('11-pick-2x3');
  await tapSel('#p-go');
  await sleep(250);
  if (await page.locator('.cell.wrong').count()) {
    fixes++;
    if (fixes === 1) await shot('12b-pick-wrong');
    await tapSel('.cell.goal');
    await sleep(500);
  } else {
    if (shots && i === 2) await shot('12-pick-feedback');
    await sleep(500);
  }
}
for (let i = 0; i < 8; i++) await pickOnce(i, true);
console.log('pick 2x3:', await page.locator('#p-stat').innerText());
await tapSel('.seg[data-l="2"]');
await sleep(400);
for (let i = 0; i < 10; i++) await pickOnce(i, false);
await shot('13-pick-3x5');
console.log('pick 3x5:', await page.locator('#p-stat').innerText(), `| anchors ${anchors}, corrections ${fixes}`);

// results
await tapSel('.close');
await sleep(250);
await tapSel('[data-go="stats"]');
await sleep(300);
await shot('14-stats-top');
await page.evaluate(() => { document.querySelector('.screen').scrollTop = 520; });
await sleep(150);
await shot('15-stats-mid');
await page.evaluate(() => { document.querySelector('.screen').scrollTop = 5000; });
await sleep(150);
await shot('16-stats-bottom');
console.log('\n' + await page.evaluate(() => window.__gazecal.summaryText()));

// export
{
  const [dl] = await Promise.all([page.waitForEvent('download'), tapSel('#s-dl')]);
  const file = `${SHOTS}/export-sim.json`;
  await dl.saveAs(file);
  const d = JSON.parse(readFileSync(file, 'utf8'));
  const rec = d.log[d.log.length - 1];
  console.log(`\nexport: ${(statSync(file).size / 1024).toFixed(0)} KB · app ${d.app} · format ${d.format} · log ${d.log.length} · pursuit runs ${d.pursuit.length} (${d.pursuit[0].frames.length} frames)` +
    ` · calibration v${d.calibration.v} with ${d.calibration.samples.length} samples of ${d.calibration.nFeat} features · last rec errs ${JSON.stringify(rec.errs)} pix ${rec.pix ? rec.pix.length : 0} chars`);
  const pk = d.picks[d.picks.length - 1];
  console.log(`export 0.3: runs ${d.pursuit.map((r) => `${r.mode} ${r.samples} (dist ${r.spread.distCm.join('–')} cm, tilt ${r.spread.tiltDeg}°)`).join('; ')} · last pick ${JSON.stringify({ goal: pk.goal, chosen: pk.chosen, ok: pk.ok, okEarly: pk.okEarly, errPx: pk.errPx, trace: pk.trace.length })} · preds ${JSON.stringify(rec.preds)} · by src ${JSON.stringify(d.summary.bySrc)}`);
  if (d.format !== 3 || d.calibration.v !== 3 || d.calibration.nFeat !== 315 || d.pursuit.length !== 2 || !d.pursuit[1].frames[30].q || !rec.pix || !rec.preds || !pk.trace.length || rec.phi.length !== 27) problems.push('export is missing 0.3 fields');
}

// reset flow
await tapSel('#s-reset');
await sleep(100);
await tapSel('#s-reset');
await sleep(300);
const after = await page.evaluate(() => ({ n: window.__gazecal.S.cal.samples.length, screen: window.__gazecal.S.screen }));
console.log('\nafter reset:', JSON.stringify(after));

console.log(problems.length ? '\nPROBLEMS:\n' + problems.join('\n') : '\nno console errors');
await browser.close();
server.close();
process.exit(problems.length ? 1 : 0);
