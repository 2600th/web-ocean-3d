/**
 * What each part of the scene costs, by taking it away.
 *
 *   node scripts/perf-subtract.mjs <url> [tier]
 *
 * `<url>` is a served build (`npx vite preview --outDir <dir> --port 4179`).
 * One session, the benchmark's scene and stepping. Each case hides one scene
 * subtree by name (or empties the planar mirror), pins the world again and
 * reads GPU time from timestamp queries and the page's time per stepped frame
 * over 120 frames; the report is each case's difference from the first.
 *
 * Read with the usual caveat of subtraction: hiding a thing also removes what
 * it caused (the ocean triggers the mirror) and uncovers what it hid, so the
 * differences need not add up to the frame.
 *
 * Per-pass GPU timestamps were tried for this and smear: on this stack a small
 * pass that follows a heavy one is charged part of it. The frame total is
 * sound, which is why this subtracts whole frames.
 *
 * `scripts/profile-frame.mjs` does the same with more cases but was written
 * against an older scene and no longer runs; fix it or use this.
 */
import { chromium } from '@playwright/test';
const url = process.argv[2]; const tier = process.argv[3] ?? 'high';
const browser = await chromium.launch({ channel: 'chrome', headless: false, args: ['--enable-unsafe-webgpu', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--window-size=1616,1000', '--window-position=0,0'] });
const page = await (await browser.newContext({ viewport: { width: 1600, height: 900 } })).newPage();
const errs = []; page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text().slice(0, 200)); });
await page.goto(`${url}?gate=0&quality=${tier}`);
await page.waitForFunction(() => window.__ocean?.isReady?.() && window.__ocean.sceneContentLoaded() && window.__ocean.shadersReady(), null, { timeout: 240000 });
await page.evaluate(async () => { const o = window.__ocean; o.setState({ preset: 'skyPro', pixelRatio: 1, cameraMode: 'orbit' }); o.setCamera(0, 14, 48, 0, 2, 0); });
await page.waitForTimeout(4000);
const res = await page.evaluate(async () => {
  const o = window.__ocean; const r = o.renderer; r.backend.trackTimestamp = true;
  const measure = async () => {
    await o.resetDeterministic(0, 90); o.setPaused(true); o.loop.stats.fps = 1000;
    try { await r.resolveTimestampsAsync('render'); } catch {}
    const g = [], c = [];
    for (let i = 0; i < 150; i++) { await new Promise(requestAnimationFrame); const s = performance.now(); await o.loop.step(1 / 60, 1); const ms = performance.now() - s; const gpu = await r.resolveTimestampsAsync('render'); if (i >= 30) { g.push(gpu); c.push(ms); } }
    g.sort((a, b) => a - b); c.sort((a, b) => a - b);
    return { gpu: g[g.length >> 1], cpu: c[c.length >> 1], draws: r.info.render.drawCalls, calls: r.info.render.frameCalls };
  };
  const chain = o.post.chain;
  const cases = [
    ['base', () => {}, () => {}],
    ['island-canopy-field hidden', () => { const x = o.scene.getObjectByName('island-canopy-field'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['island-grass-field hidden', () => { const x = o.scene.getObjectByName('island-grass-field'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['props hidden', () => { const x = o.scene.getObjectByName('props'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['fish-schools hidden', () => { const x = o.scene.getObjectByName('fish-schools'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['spindrift hidden', () => { const x = o.scene.getObjectByName('spindrift'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['spray hidden', () => { const x = o.scene.getObjectByName('spray'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['splash hidden', () => { const x = o.scene.getObjectByName('splash'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['seabed-rocks hidden', () => { const x = o.scene.getObjectByName('seabed-rocks'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['reef-life hidden', () => { const x = o.scene.getObjectByName('reef-life'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['seabed-cover hidden', () => { const x = o.scene.getObjectByName('seabed-cover'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['sea-creatures hidden', () => { const x = o.scene.getObjectByName('sea-creatures'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['birds hidden', () => { const x = o.scene.getObjectByName('birds'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['kelp-forest hidden', () => { const x = o.scene.getObjectByName('kelp-forest'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['seafloor hidden', () => { const x = o.scene.getObjectByName('seafloor'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['ocean-surface hidden', () => { const x = o.scene.getObjectByName('ocean-surface'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['ship hidden', () => { const x = o.scene.getObjectByName('ship'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['sky-dome hidden', () => { const x = o.scene.getObjectByName('sky-dome'); o.__h = x; if (x) x.visible = false; else throw new Error('missing'); }, () => { if (o.__h) o.__h.visible = true; }],
    ['mirror layer empty', () => { const c = o.reflections.renderSetup(o.camera).camera; o.__m = c.layers.mask; c.layers.mask = 1 << 30; }, () => { o.reflections.renderSetup(o.camera).camera.layers.mask = o.__m; }],
    ['base again', () => {}, () => {}],
  ];
  const out = [];
  for (const [name, on, off] of cases) { let m; try { on(); await new Promise((r2) => setTimeout(r2, 300)); m = await measure(); } catch (e) { m = { error: String(e) }; } try { off(); } catch {} out.push([name, m]); }
  return out;
});
await browser.close();
const base = res[0][1];
console.log(`tier ${tier}; errors ${errs.length} ${errs[0] ?? ''}`);
for (const [name, m] of res) console.log(name.padEnd(24), m.error ? m.error : `gpu ${m.gpu.toFixed(2)} (${(m.gpu - base.gpu >= 0 ? '+' : '') + (m.gpu - base.gpu).toFixed(2)})  cpu ${m.cpu.toFixed(2)} (${(m.cpu - base.cpu >= 0 ? '+' : '') + (m.cpu - base.cpu).toFixed(2)})  draws ${m.draws} calls ${m.calls}`);
