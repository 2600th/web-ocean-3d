/**
 * Where the page's CPU time goes in a frame.
 *
 *   node scripts/perf-cpu.mjs <url> [tier] [frames] [out]
 *
 * `<url>` is a served build, e.g. `npx vite build --minify false --outDir <dir>`
 * then `npx vite preview --outDir <dir> --port 4178` (unminified, so the
 * functions have their names). The scene and the stepping are the benchmark's:
 * the canonical wide shot, pinned by a deterministic reset, frames stepped at
 * 1/60 s. A V8 sampling profile at 100 us covers `frames` of them; the report
 * is self time and inclusive time per function, in ms per frame, and the
 * profile is saved beside this script's working directory as `<out>.cpuprofile`
 * for DevTools.
 *
 * Take it with nothing else on the GPU and the window in front.
 */
import { chromium } from '@playwright/test';
import fs from 'node:fs';
const url = process.argv[2]; const tier = process.argv[3] ?? 'high'; const frames = Number(process.argv[4] ?? 600);
const out = process.argv[5] ?? 'cpuprof';
const browser = await chromium.launch({ channel: 'chrome', headless: false, args: ['--enable-unsafe-webgpu', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--window-size=1616,1000', '--window-position=0,0'] });
const page = await (await browser.newContext({ viewport: { width: 1600, height: 900 } })).newPage();
const errs = []; page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text().slice(0, 200)); });
await page.goto(`${url}?gate=0&quality=${tier}`);
await page.waitForFunction(() => window.__ocean?.isReady?.() && window.__ocean.sceneContentLoaded() && window.__ocean.shadersReady(), null, { timeout: 240000 });
await page.evaluate(async () => { const o = window.__ocean; o.setState({ preset: 'skyPro', pixelRatio: 1, cameraMode: 'orbit' }); o.setCamera(0, 14, 48, 0, 2, 0); });
await page.waitForTimeout(4000);
await page.evaluate(async () => { const o = window.__ocean; await o.resetDeterministic(0, 90); o.setPaused(true); o.loop.stats.fps = 1000; for (let i = 0; i < 120; i++) { await new Promise(requestAnimationFrame); await o.loop.step(1 / 60, 1); } });
const cdp = await page.context().newCDPSession(page);
await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 100 }); await cdp.send('Profiler.start');
const ms = await page.evaluate(async (n) => { const o = window.__ocean; const t = []; for (let i = 0; i < n; i++) { await new Promise(requestAnimationFrame); const s = performance.now(); await o.loop.step(1 / 60, 1); t.push(performance.now() - s); } t.sort((a, b) => a - b); return { p50: t[n >> 1], mean: t.reduce((a, b) => a + b) / n, info: { ...o.renderer.info.render } }; }, frames);
const { profile } = await cdp.send('Profiler.stop');
fs.writeFileSync(`${out}.cpuprofile`, JSON.stringify(profile));
await browser.close();
// aggregate
const nodes = new Map(profile.nodes.map((n) => [n.id, n]));
const parent = new Map(); for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
const self = new Map(); const dts = profile.timeDeltas; let total = 0;
profile.samples.forEach((id, i) => { const dt = dts[i] ?? 0; total += dt; self.set(id, (self.get(id) ?? 0) + dt); });
const name = (n) => `${n.callFrame.functionName || '(anon)'} ${n.callFrame.url.split('/').pop().split('?')[0]}:${n.callFrame.lineNumber + 1}`;
const bySelf = new Map(); const byIncl = new Map();
for (const [id, t] of self) {
  const n = nodes.get(id); const k = name(n); bySelf.set(k, (bySelf.get(k) ?? 0) + t);
  const seen = new Set(); for (let p = id; p !== undefined; p = parent.get(p)) { const kk = name(nodes.get(p)); if (seen.has(kk)) continue; seen.add(kk); byIncl.set(kk, (byIncl.get(kk) ?? 0) + t); }
}
const perFrame = (t) => (t / 1000 / frames).toFixed(3);
console.log(`step p50 ${ms.p50.toFixed(2)} mean ${ms.mean.toFixed(2)} ms; profile total ${(total / 1000 / frames).toFixed(2)} ms/frame; errors ${errs.length}`, JSON.stringify(ms.info));
console.log('--- self (ms/frame)'); for (const [k, t] of [...bySelf].sort((a, b) => b[1] - a[1]).slice(0, 60)) console.log(perFrame(t), k);
console.log('--- inclusive (ms/frame)'); for (const [k, t] of [...byIncl].sort((a, b) => b[1] - a[1]).slice(0, 110)) console.log(perFrame(t), k);
