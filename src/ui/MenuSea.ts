/**
 * The living backdrop behind the start gate and the loading chart.
 *
 * The gate exists *before* the renderer does — it is where the visitor picks
 * WebGPU or WebGL2 — so it cannot borrow the real sea. Instead it paints its
 * own: an illustrated dusk seascape in the interface's art direction, on a
 * plain 2D canvas. Parallax swell, drifting clouds, a silhouetted ship riding
 * the swell with spray off the bow, gulls, a lighthouse beam on a far headland
 * and sun glitter on the water.
 *
 * Budget, because it runs while shaders compile on the same thread:
 *   - no WebGL, no images, no filters; a few hundred path points a frame;
 *   - device pixel ratio capped at 1.5, and the backing store capped at ~2.4 MP;
 *   - 30 fps on machines reporting four or fewer cores, 60 elsewhere;
 *   - rAF stops while the tab is hidden, and the whole thing stops for good
 *     once the boot overlay has faded out;
 *   - `prefers-reduced-motion` gets one painted still, redrawn only on resize.
 *
 * Pure DOM/canvas, no three.js, like everything in `src/ui`.
 */

const TAU = Math.PI * 2;

/** Five swell bands, far to near: depth below the horizon (0..1 of the sea),
 *  amplitude and wavelength as fractions of the viewport, speed in rad/s. */
const SWELLS = [
  { depth: 0.03, amp: 0.0025, len: 0.09, speed: 0.35, top: '#2f6670', bottom: '#24525c', rim: 0.34 },
  { depth: 0.1, amp: 0.005, len: 0.16, speed: 0.45, top: '#22535d', bottom: '#1a414b', rim: 0.28 },
  { depth: 0.24, amp: 0.011, len: 0.27, speed: 0.55, top: '#183f49', bottom: '#11303a', rim: 0.24 },
  { depth: 0.46, amp: 0.02, len: 0.42, speed: 0.62, top: '#0f2c35', bottom: '#0a2129', rim: 0.18 },
  { depth: 0.76, amp: 0.032, len: 0.66, speed: 0.7, top: '#0a1f27', bottom: '#05131a', rim: 0.13 },
] as const;

/** The band the ship rides; bands after it are drawn over its hull. */
const SHIP_BAND = 2;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  max: number;
  r: number;
  /** The waterline it falls back into. */
  floor: number;
}

interface Gull {
  x: number;
  y: number;
  speed: number;
  size: number;
  phase: number;
  flap: number;
}

interface Cloud {
  sprite: HTMLCanvasElement;
  x: number;
  y: number;
  speed: number;
}

/** Deterministic, so the still frame and the first animated frame agree. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function reducedMotion(): boolean {
  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export class MenuSea {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly controller = new AbortController();
  private readonly observer: MutationObserver;
  private readonly frameInterval: number;
  private readonly still: boolean;

  private width = 0;
  private height = 0;
  private dpr = 1;
  private horizon = 0;
  private sunX = 0;
  private sunY = 0;
  private shipX = 0;
  private shipScale = 1;
  private headlandX = 0;
  private portrait = false;

  private sky: CanvasGradient | null = null;
  private seaFill: CanvasGradient | null = null;
  private vignette: CanvasGradient | null = null;
  private sunGlow: CanvasGradient | null = null;
  private sunPath: CanvasGradient | null = null;
  private roadR = 1;
  private bandFills: CanvasGradient[] = [];

  private clouds: Cloud[] = [];
  private gulls: Gull[] = [];
  private stars: { x: number; y: number; r: number; phase: number }[] = [];
  private glints: { u: number; v: number; phase: number; rate: number }[] = [];
  private readonly spray: Particle[] = [];
  private nextSpray = 0;

  private raf = 0;
  private last = 0;
  private lastPaint = 0;
  private time = 0;
  private running = false;
  private disposed = false;

  constructor(host: HTMLElement) {
    const canvas = document.createElement('canvas');
    canvas.className = 'boot__sea';
    canvas.setAttribute('aria-hidden', 'true');
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('MenuSea: 2D canvas unavailable');
    this.canvas = canvas;
    this.ctx = ctx;
    host.prepend(canvas);

    this.still = reducedMotion();
    this.frameInterval = (navigator.hardwareConcurrency || 8) <= 4 ? 1000 / 30 : 0;

    const signal = this.controller.signal;
    window.addEventListener('resize', () => this.resize(), { signal });
    document.addEventListener(
      'visibilitychange',
      () => (document.hidden ? this.pause() : this.resume()),
      { signal },
    );

    // Stop for good once the overlay has faded to the live sea. The fade is
    // 900 ms; a little slack lets the last frames of the backdrop cross-fade
    // rather than freezing under the transition.
    this.observer = new MutationObserver(() => {
      if (host.classList.contains('boot--hidden')) {
        window.setTimeout(() => this.dispose(), 1100);
      }
    });
    this.observer.observe(host, { attributes: true, attributeFilter: ['class'] });

    this.resize();
  }

  start(): void {
    if (this.disposed) return;
    if (this.still) {
      this.time = 7.5;
      this.paint();
      return;
    }
    this.running = true;
    this.resume();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.pause();
    this.running = false;
    this.controller.abort();
    this.observer.disconnect();
    // Release the backing store; a 2 MP canvas is 8 MB of RAM the live sea wants.
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.canvas.remove();
  }

  // ---------------------------------------------------------------- lifecycle

  private pause(): void {
    if (this.raf !== 0) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  private resume(): void {
    if (!this.running || this.disposed || this.raf !== 0 || document.hidden) return;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }

  private readonly tick = (now: number): void => {
    this.raf = requestAnimationFrame(this.tick);
    if (this.frameInterval > 0 && now - this.lastPaint < this.frameInterval - 1) return;
    // Clamp the step: a shader compile can stall the thread for a second, and
    // the sea should resume where it was rather than lurch forward.
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.lastPaint = now;
    this.time += dt;
    this.update(dt);
    this.paint();
  };

  private resize(): void {
    if (this.disposed) return;
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    let dpr = Math.min(1.5, window.devicePixelRatio || 1);
    const budget = 2_400_000;
    if (w * h * dpr * dpr > budget) dpr = Math.sqrt(budget / (w * h));
    this.width = w;
    this.height = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);

    this.portrait = h > w * 1.1;
    const short = h < 520;
    // The menu sheet sits low on a phone and to the left on a wide screen;
    // the horizon and the ship go where the sheet is not.
    this.horizon = h * (this.portrait ? 0.21 : short ? 0.42 : 0.56);
    this.sunX = w * (this.portrait ? 0.52 : short ? 0.93 : 0.77);
    this.sunY = this.horizon - h * (this.portrait ? 0.045 : 0.07);
    this.shipX = w * (this.portrait ? 0.68 : short ? 0.82 : 0.65);
    this.shipScale = Math.min(w, h) * (this.portrait ? 0.0019 : short ? 0.0024 : 0.0019);
    // The headland goes where neither the chart nor the ship is; on a short
    // landscape phone there is no such place, and it is left out.
    this.headlandX = w * (this.portrait ? 0.02 : short ? 2 : 0.85);

    this.buildGradients();
    this.buildScenery();
    if (this.still || !this.running) this.paint();
  }

  // ------------------------------------------------------------------ scenery

  private buildGradients(): void {
    const { ctx, width: w, height: h, horizon } = this;
    const sky = ctx.createLinearGradient(0, 0, 0, horizon);
    sky.addColorStop(0, '#081b25');
    sky.addColorStop(0.42, '#183f4c');
    sky.addColorStop(0.78, '#7e6b5c');
    sky.addColorStop(0.93, '#d99a62');
    sky.addColorStop(1, '#f3c27c');
    this.sky = sky;

    const sea = ctx.createLinearGradient(0, horizon, 0, h);
    sea.addColorStop(0, '#3a6b70');
    sea.addColorStop(0.08, '#24515a');
    sea.addColorStop(1, '#04111a');
    this.seaFill = sea;

    const r = Math.max(w, h);
    const vignette = ctx.createRadialGradient(w * 0.6, h * 0.5, r * 0.25, w * 0.6, h * 0.5, r * 0.85);
    vignette.addColorStop(0, 'rgba(2, 8, 11, 0)');
    vignette.addColorStop(1, 'rgba(2, 8, 11, 0.62)');
    this.vignette = vignette;

    const glowR = Math.min(w, h) * 0.42;
    const glow = ctx.createRadialGradient(this.sunX, this.sunY, 0, this.sunX, this.sunY, glowR);
    glow.addColorStop(0, 'rgba(255, 226, 170, 0.75)');
    glow.addColorStop(0.12, 'rgba(255, 196, 128, 0.38)');
    glow.addColorStop(0.45, 'rgba(240, 150, 90, 0.1)');
    glow.addColorStop(1, 'rgba(240, 150, 90, 0)');
    this.sunGlow = glow;

    // The sun's road on the water: brightest at the horizon, widening and
    // fading toward the viewer. A radial gradient about the origin, squashed
    // sideways into an ellipse when painted (see paintGlitter), so it has no
    // edge however narrow the screen.
    this.roadR = (h - horizon) * 0.9;
    const road = ctx.createRadialGradient(0, 0, 0, 0, 0, this.roadR);
    road.addColorStop(0, 'rgba(255, 200, 130, 0.42)');
    road.addColorStop(0.25, 'rgba(240, 160, 100, 0.14)');
    road.addColorStop(1, 'rgba(240, 160, 100, 0)');
    this.sunPath = road;

    this.bandFills = SWELLS.map((band) => {
      const y0 = horizon + band.depth * (h - horizon);
      const g = ctx.createLinearGradient(0, y0 - band.amp * h * 2, 0, h);
      g.addColorStop(0, band.top);
      g.addColorStop(0.35, band.bottom);
      g.addColorStop(1, '#04111a');
      return g;
    });
  }

  private buildScenery(): void {
    const rand = mulberry32(0x5eab0a7);
    const { width: w, height: h, horizon } = this;

    // Clouds: flat-bottomed, puffed, painted rather than blurred — dark and
    // cool on top, lit warm from underneath by the low sun. Pre-rendered once.
    this.clouds = [];
    const cloudCount = this.portrait ? 4 : 7;
    for (let i = 0; i < cloudCount; i++) {
      const cw = Math.min(w * 0.3, 160 + rand() * 260);
      const ch = cw * (0.3 + rand() * 0.12);
      const altitude = rand();
      const sprite = document.createElement('canvas');
      sprite.width = Math.ceil(cw * this.dpr);
      sprite.height = Math.ceil(ch * this.dpr);
      const c = sprite.getContext('2d');
      if (c) {
        c.scale(this.dpr, this.dpr);
        const baseY = ch * 0.86;
        const g = c.createLinearGradient(0, 0, 0, baseY);
        g.addColorStop(0, altitude > 0.5 ? 'rgba(36, 62, 74, 0.92)' : 'rgba(58, 82, 90, 0.92)');
        g.addColorStop(0.62, altitude > 0.5 ? 'rgba(92, 96, 102, 0.92)' : 'rgba(150, 118, 104, 0.92)');
        g.addColorStop(1, altitude > 0.5 ? 'rgba(176, 128, 104, 0.92)' : 'rgba(232, 164, 112, 0.95)');
        c.fillStyle = g;
        c.save();
        c.beginPath();
        c.rect(0, 0, cw, baseY);
        c.clip();
        c.beginPath();
        const puffs = 4 + Math.floor(rand() * 4);
        for (let j = 0; j < puffs; j++) {
          const t = (j + 0.5) / puffs;
          const r = ch * (0.22 + 0.34 * Math.sin(t * Math.PI) * (0.7 + rand() * 0.5));
          const px = cw * (0.1 + t * 0.8) + (rand() - 0.5) * cw * 0.06;
          c.moveTo(px + r, baseY - r * 0.55);
          c.arc(px, baseY - r * 0.55, r, 0, TAU);
        }
        // The rounded base, wound clockwise like the puffs: under the
        // non-zero rule an opposite winding would punch holes where they
        // overlap.
        const baseR = ch * 0.11;
        const left = cw * 0.1 + baseR;
        const right = cw * 0.9 - baseR;
        const cy = baseY - baseR;
        c.moveTo(left, cy - baseR);
        c.lineTo(right, cy - baseR);
        c.arc(right, cy, baseR, -Math.PI / 2, Math.PI / 2);
        c.lineTo(left, baseY);
        c.arc(left, cy, baseR, Math.PI / 2, Math.PI * 1.5);
        c.closePath();
        c.fill();
        c.restore();
        // A sunlit lip along the flat base.
        c.fillStyle = 'rgba(255, 196, 136, 0.55)';
        c.fillRect(cw * 0.1 + ch * 0.11, baseY - 1.5, cw * 0.8 - ch * 0.22, 1.5);
      }
      this.clouds.push({
        sprite,
        x: rand() * (w + cw) - cw,
        y: horizon * (0.08 + altitude * 0.62) - ch * 0.5,
        speed: 3 + rand() * 8,
      });
    }

    this.stars = [];
    for (let i = 0; i < 46; i++) {
      this.stars.push({ x: rand() * w, y: rand() * horizon * 0.45, r: 0.4 + rand() * 0.9, phase: rand() * TAU });
    }

    this.gulls = [];
    for (let i = 0; i < 5; i++) {
      this.gulls.push({
        x: rand() * w,
        y: horizon * (0.25 + rand() * 0.55),
        speed: 14 + rand() * 18,
        size: (4 + rand() * 4) * Math.max(0.7, Math.min(1.3, Math.min(w, h) / 800)),
        phase: rand() * TAU,
        flap: 4 + rand() * 3,
      });
    }

    // Glitter in the sun's path: u is across the path (-1..1), v down the sea.
    this.glints = [];
    const glintCount = Math.round(Math.min(320, (w * (h - horizon)) / 2600));
    for (let i = 0; i < glintCount; i++) {
      const v = Math.pow(rand(), 1.6);
      this.glints.push({ u: (rand() * 2 - 1) * (0.3 + v * 0.7), v, phase: rand() * TAU, rate: 1.5 + rand() * 3.5 });
    }
    this.spray.length = 0;
  }

  // ------------------------------------------------------------------- motion

  /** Height of swell band `i` at screen x, at the current time. */
  private bandY(i: number, x: number): number {
    const band = SWELLS[i];
    if (band === undefined) return this.height;
    const { width: w, height: h } = this;
    const base = this.horizon + band.depth * (h - this.horizon);
    const k = TAU / (band.len * w);
    const t = this.time * band.speed;
    const a = band.amp * h;
    return base
      + a * 0.62 * Math.sin(k * x - t * 1.0 + i * 1.7)
      + a * 0.28 * Math.sin(k * 1.73 * x + t * 1.31 + i * 0.6)
      + a * 0.1 * Math.sin(k * 3.1 * x - t * 0.7 + i * 2.9);
  }

  private update(dt: number): void {
    const { width: w } = this;
    for (const cloud of this.clouds) {
      cloud.x -= cloud.speed * dt;
      const cw = cloud.sprite.width / this.dpr;
      if (cloud.x < -cw) cloud.x = w + cw * 0.2;
    }
    for (const gull of this.gulls) {
      gull.x -= gull.speed * dt;
      gull.y += Math.sin(this.time * 0.6 + gull.phase) * 4 * dt;
      if (gull.x < -30) gull.x = w + 30;
    }

    // Spray off the bow, in bursts timed to the ship's pitch.
    if (this.time >= this.nextSpray) {
      this.nextSpray = this.time + 1.1 + Math.random() * 1.6;
      const bow = this.bowPoint();
      const s = this.shipScale;
      for (let i = 0; i < 10; i++) {
        this.spray.push({
          x: bow.x + (Math.random() - 0.5) * 8 * s,
          y: bow.y,
          vx: -(20 + Math.random() * 60) * s,
          vy: -(60 + Math.random() * 90) * s,
          life: 0,
          max: 0.7 + Math.random() * 0.6,
          r: (0.6 + Math.random() * 1.1) * Math.max(0.8, s),
          floor: bow.y + 2 * s,
        });
      }
    }
    for (let i = this.spray.length - 1; i >= 0; i--) {
      const p = this.spray[i]!;
      p.life += dt;
      if (p.life >= p.max || (p.vy > 0 && p.y > p.floor)) {
        this.spray.splice(i, 1);
        continue;
      }
      p.vy += 210 * this.shipScale * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
    }
  }

  private shipPose(): { x: number; y: number; angle: number } {
    const x = this.shipX;
    const y0 = this.bandY(SHIP_BAND, x - 30 * this.shipScale);
    const y1 = this.bandY(SHIP_BAND, x + 30 * this.shipScale);
    const y = (y0 + y1) * 0.5 + 3 * this.shipScale;
    // Sailing left: a slope rising to the left pitches the bow up.
    const angle = Math.atan2(y1 - y0, 60 * this.shipScale) * 0.85;
    return { x, y, angle };
  }

  /** The bow in screen space (the ship faces left). */
  private bowPoint(): { x: number; y: number } {
    const pose = this.shipPose();
    const lx = -50 * this.shipScale;
    const ly = -2 * this.shipScale;
    const c = Math.cos(pose.angle);
    const s = Math.sin(pose.angle);
    return { x: pose.x + lx * c - ly * s, y: pose.y + lx * s + ly * c };
  }

  // -------------------------------------------------------------------- paint

  private paint(): void {
    const { ctx, width: w, height: h, horizon, time } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.globalCompositeOperation = 'source-over';

    // Sky, stars, sun.
    ctx.fillStyle = this.sky!;
    ctx.fillRect(0, 0, w, horizon + 1);
    ctx.fillStyle = '#f6e7c6';
    for (const star of this.stars) {
      ctx.globalAlpha = (0.25 + 0.35 * Math.sin(time * 1.3 + star.phase)) * (1 - star.y / (horizon * 0.5));
      ctx.fillRect(star.x, star.y, star.r, star.r);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = this.sunGlow!;
    ctx.fillRect(0, 0, w, horizon + 1);
    const sunR = Math.min(w, h) * 0.034;
    ctx.fillStyle = '#ffe6ad';
    ctx.beginPath();
    ctx.arc(this.sunX, this.sunY, sunR, 0, TAU);
    ctx.fill();

    for (const cloud of this.clouds) {
      ctx.drawImage(cloud.sprite, cloud.x, cloud.y, cloud.sprite.width / this.dpr, cloud.sprite.height / this.dpr);
    }

    this.paintHeadland();

    // Gulls: two strokes, wings beating.
    ctx.strokeStyle = 'rgba(14, 22, 26, 0.85)';
    ctx.lineCap = 'round';
    for (const gull of this.gulls) {
      const flap = Math.sin(time * gull.flap + gull.phase);
      const s = gull.size;
      ctx.lineWidth = Math.max(1, s * 0.16);
      ctx.beginPath();
      ctx.moveTo(gull.x - s, gull.y - s * 0.35 * flap);
      ctx.quadraticCurveTo(gull.x - s * 0.45, gull.y - s * 0.45, gull.x, gull.y);
      ctx.quadraticCurveTo(gull.x + s * 0.45, gull.y - s * 0.45, gull.x + s, gull.y - s * 0.35 * flap);
      ctx.stroke();
    }

    // Sea body, then the glitter on it.
    ctx.fillStyle = this.seaFill!;
    ctx.fillRect(0, horizon, w, h - horizon);
    this.paintGlitter();

    // Swell bands far to near, with the ship between them.
    for (let i = 0; i < SWELLS.length; i++) {
      if (i === SHIP_BAND) this.paintShip();
      this.paintBand(i);
    }
    this.paintSpray();

    ctx.fillStyle = this.vignette!;
    ctx.fillRect(0, 0, w, h);
  }

  private paintBand(i: number): void {
    const { ctx, width: w, height: h } = this;
    const band = SWELLS[i]!;
    const step = Math.max(6, w / 140);
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w + step; x += step) ctx.lineTo(x, this.bandY(i, x));
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = this.bandFills[i]!;
    ctx.fill();

    // The crest catches the low sun: a warm rim, brighter toward the sun.
    ctx.beginPath();
    for (let x = 0; x <= w + step; x += step) {
      const y = this.bandY(i, x) + 0.5;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    const rim = ctx.createLinearGradient(0, 0, w, 0);
    const near = Math.max(0, Math.min(1, this.sunX / w));
    rim.addColorStop(0, `rgba(255, 205, 140, ${band.rim * 0.35})`);
    rim.addColorStop(near, `rgba(255, 214, 150, ${band.rim})`);
    rim.addColorStop(1, `rgba(255, 205, 140, ${band.rim * 0.45})`);
    ctx.strokeStyle = rim;
    ctx.lineWidth = 1 + i * 0.35;
    ctx.stroke();

    // Nearer swells break: broken foam streaks running along the crest.
    if (i >= 2) {
      const unit = band.len * w;
      ctx.setLineDash([unit * 0.05, unit * 0.11, unit * 0.02, unit * 0.16]);
      ctx.lineDashOffset = -this.time * band.speed * unit * 0.06;
      ctx.strokeStyle = `rgba(226, 238, 232, ${0.1 + (i - 2) * 0.05})`;
      ctx.lineWidth = 1.5 + i * 0.6;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  private paintGlitter(): void {
    const { ctx, height: h, horizon, time } = this;
    ctx.globalCompositeOperation = 'lighter';
    if (this.sunPath) {
      const r = this.roadR;
      const squash = Math.min(0.5, (this.width * 0.2) / r);
      ctx.save();
      ctx.translate(this.sunX, horizon);
      ctx.scale(squash, 1);
      ctx.fillStyle = this.sunPath;
      ctx.fillRect(-r, 0, r * 2, r);
      ctx.restore();
    }
    ctx.fillStyle = '#ffd9a0';
    const span = h - horizon;
    for (const g of this.glints) {
      const a = Math.sin(time * g.rate + g.phase);
      if (a <= 0.3) continue;
      const y = horizon + 2 + g.v * span;
      const spread = 12 + g.v * this.width * 0.14;
      const x = this.sunX + g.u * spread;
      const len = 3 + g.v * 16;
      ctx.globalAlpha = Math.min(1, (a - 0.3) * 1.6) * (1 - g.v * 0.5) * (1 - Math.abs(g.u) * 0.6);
      ctx.fillRect(x - len * 0.5, y, len, 1 + g.v * 1.6);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  /** A far headland with a lighthouse whose beam sweeps the dusk. */
  private paintHeadland(): void {
    const { ctx, width: w, height: h, horizon, time } = this;
    const baseX = this.headlandX;
    if (baseX > w) return;
    const hw = Math.min(w, h) * (this.portrait ? 0.34 : 0.3);
    const hh = hw * 0.16;
    ctx.fillStyle = '#16303a';
    ctx.beginPath();
    ctx.moveTo(baseX - hw * 0.2, horizon + 1);
    ctx.bezierCurveTo(baseX + hw * 0.1, horizon - hh * 0.6, baseX + hw * 0.25, horizon - hh * 1.1, baseX + hw * 0.45, horizon - hh);
    ctx.bezierCurveTo(baseX + hw * 0.7, horizon - hh * 0.9, baseX + hw * 0.9, horizon - hh * 0.3, baseX + hw * 1.15, horizon + 1);
    ctx.fill();

    const lx = baseX + hw * 0.42;
    const ly = horizon - hh;
    const tower = hh * 0.9;
    ctx.fillStyle = '#132a33';
    ctx.fillRect(lx - tower * 0.09, ly - tower, tower * 0.18, tower);
    const lampY = ly - tower - tower * 0.06;

    // The beam: a lamp turning in 3D, seen side on. It reads as a wedge that
    // swings across, and as a flash when it points at the viewer.
    const theta = time * 0.9;
    const facing = Math.max(0, Math.sin(theta));
    const sweep = Math.cos(theta);
    const reach = w * 0.38;
    ctx.globalCompositeOperation = 'lighter';
    const beam = ctx.createLinearGradient(lx, lampY, lx + sweep * reach, lampY);
    beam.addColorStop(0, `rgba(255, 228, 170, ${0.16 + facing * 0.14})`);
    beam.addColorStop(1, 'rgba(255, 228, 170, 0)');
    ctx.fillStyle = beam;
    ctx.beginPath();
    ctx.moveTo(lx, lampY);
    ctx.lineTo(lx + sweep * reach, lampY - reach * 0.05);
    ctx.lineTo(lx + sweep * reach, lampY + reach * 0.035);
    ctx.closePath();
    ctx.fill();
    const flash = ctx.createRadialGradient(lx, lampY, 0, lx, lampY, tower * (0.5 + facing * 1.6));
    flash.addColorStop(0, `rgba(255, 236, 190, ${0.5 + facing * 0.5})`);
    flash.addColorStop(1, 'rgba(255, 236, 190, 0)');
    ctx.fillStyle = flash;
    ctx.fillRect(lx - tower * 2.2, lampY - tower * 2.2, tower * 4.4, tower * 4.4);
    ctx.globalCompositeOperation = 'source-over';
  }

  /** The ship: an ink silhouette rimmed by the sun, facing left, 100 units long. */
  private paintShip(): void {
    const { ctx, time } = this;
    const pose = this.shipPose();
    const s = this.shipScale;
    ctx.save();
    ctx.translate(pose.x, pose.y);
    ctx.rotate(pose.angle);
    ctx.scale(-s, s);

    const hull = '#071118';
    const sail = '#1a2b31';
    const rim = 'rgba(255, 196, 128, 0.55)';

    // Masts and rigging.
    ctx.strokeStyle = hull;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(-24, -12); ctx.lineTo(-24, -60);
    ctx.moveTo(2, -10); ctx.lineTo(2, -80);
    ctx.moveTo(26, -11); ctx.lineTo(26, -66);
    ctx.moveTo(44, -13); ctx.lineTo(70, -26);
    ctx.moveTo(2, -80); ctx.lineTo(70, -26);
    ctx.moveTo(2, -80); ctx.lineTo(-46, -18);
    ctx.stroke();

    // Sails, bellied by a following breeze.
    const belly = 2.5 + Math.sin(time * 1.4) * 0.8;
    const squareSail = (x: number, top: number, bottom: number, half: number): void => {
      ctx.beginPath();
      ctx.moveTo(x - half, top);
      ctx.lineTo(x + half, top);
      ctx.quadraticCurveTo(x + half + belly, (top + bottom) / 2, x + half * 0.92, bottom);
      ctx.lineTo(x - half * 0.92, bottom);
      ctx.quadraticCurveTo(x - half + belly, (top + bottom) / 2, x - half, top);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    };
    ctx.fillStyle = sail;
    ctx.strokeStyle = rim;
    ctx.lineWidth = 0.8;
    squareSail(2, -76, -56, 13);
    squareSail(2, -53, -27, 17);
    squareSail(26, -62, -45, 11);
    squareSail(26, -42, -22, 14);
    // Lateen on the mizzen, jib on the stay.
    ctx.beginPath();
    ctx.moveTo(-24, -60); ctx.lineTo(-40, -20); ctx.lineTo(-22, -22); ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(30, -58); ctx.lineTo(62, -28); ctx.lineTo(36, -24); ctx.closePath();
    ctx.fill(); ctx.stroke();

    // Pennant at the main truck.
    ctx.fillStyle = '#8e1b17';
    ctx.beginPath();
    ctx.moveTo(2, -80);
    const wave = Math.sin(time * 6) * 1.5;
    ctx.quadraticCurveTo(-6, -79 + wave, -14, -77 - wave);
    ctx.lineTo(2, -76);
    ctx.closePath();
    ctx.fill();

    // Hull: high stern castle, sheer down to the bow.
    ctx.fillStyle = hull;
    ctx.beginPath();
    ctx.moveTo(-48, -20);
    ctx.lineTo(-36, -19);
    ctx.lineTo(-30, -13);
    ctx.quadraticCurveTo(8, -9, 36, -12);
    ctx.lineTo(48, -15);
    ctx.quadraticCurveTo(46, -2, 36, 5);
    ctx.lineTo(-34, 6);
    ctx.quadraticCurveTo(-46, 2, -48, -20);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = rim;
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(-48, -20); ctx.lineTo(-36, -19); ctx.lineTo(-30, -13);
    ctx.quadraticCurveTo(8, -9, 36, -12); ctx.lineTo(48, -15);
    ctx.stroke();

    // Stern lantern.
    const lantern = ctx.createRadialGradient(-46, -22, 0, -46, -22, 7);
    lantern.addColorStop(0, 'rgba(255, 210, 130, 0.95)');
    lantern.addColorStop(1, 'rgba(255, 170, 90, 0)');
    ctx.fillStyle = lantern;
    ctx.fillRect(-54, -30, 16, 16);
    ctx.restore();

    // Bow wave where the stem meets the band.
    const bow = this.bowPoint();
    ctx.fillStyle = 'rgba(232, 240, 236, 0.55)';
    ctx.beginPath();
    ctx.ellipse(bow.x + 4 * s, bow.y + 2 * s, 7 * s, 1.8 * s, 0, 0, TAU);
    ctx.fill();
  }

  private paintSpray(): void {
    const { ctx } = this;
    ctx.fillStyle = '#eef4f0';
    for (const p of this.spray) {
      ctx.globalAlpha = 0.85 * (1 - p.life / p.max);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, TAU);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}
