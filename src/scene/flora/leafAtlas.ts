/*
 * Procedural foliage atlas for the island.
 *
 * The coverage-preserving mip chain and the canvas spray-drawing approach are
 * adapted from mrdoob/slop-newyork `js/trees.js` (createLeafAtlas, coverageMips):
 *
 *   MIT License — Copyright (c) 2026 Mr.doob
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions: The above copyright
 *   notice and this permission notice shall be included in all copies or
 *   substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
 *   WITHOUT WARRANTY OF ANY KIND.
 *
 * Changes from the original: tropical species (pinnate palm fronds, glossy
 * broadleaf sprays, fern fronds, oblong banana-type leaves, flowers, grass,
 * hanging vines, beach wrack and shell litter), drawn in colour rather than as
 * data channels so the result works through three's ordinary `map` slot, and a
 * coverage pass generalised from a square grid of tiles to arbitrary rects.
 */
import * as THREE from 'three/webgpu';
import { mulberry32 } from '../../core/random';

export const ATLAS_SIZE = 1024;
const GUTTER = 6;

/** A tile in canvas pixels; y grows downward, which is also texture v. */
export interface AtlasRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Every rect is a multiple of 32 px so the per-tile coverage pass has whole
 * texels to work with down to the 32x reduction.
 */
export const TILES = {
  /** Palm frond, rachis along +x from the base at the left edge. */
  frondA: { x: 0, y: 0, w: 1024, h: 224 },
  /** An older, sun-yellowed frond with torn leaflets. */
  frondB: { x: 0, y: 224, w: 1024, h: 224 },
  /** Glossy tropical leaf spray, attached at the bottom centre. */
  broadleaf: { x: 0, y: 448, w: 320, h: 320 },
  /** A dense mass of leaves, centred: one card stands for a section of a far crown. */
  clump: { x: 320, y: 448, w: 320, h: 320 },
  /** Fern frond, base at the bottom centre. */
  fern: { x: 640, y: 448, w: 192, h: 320 },
  /** Grass tuft, base at the bottom centre. */
  grass: { x: 832, y: 448, w: 192, h: 320 },
  /** Oblong banana / heliconia leaf on its petiole, base at the bottom centre. */
  bigleaf: { x: 0, y: 768, w: 256, h: 256 },
  /** Leafy spray carrying flowers. */
  flower: { x: 256, y: 768, w: 256, h: 256 },
  /** Hanging vine, attached at the top centre. */
  vine: { x: 512, y: 768, w: 128, h: 256 },
  /** Beach wrack — the dried weed a tide leaves in a line. */
  wrack: { x: 640, y: 768, w: 384, h: 128 },
  /** Shell and pebble litter. */
  litter: { x: 640, y: 896, w: 384, h: 128 },
} as const satisfies Record<string, AtlasRect>;

export type TileName = keyof typeof TILES;

/** uv bounds of a tile, gutter excluded: `u0..u1` left to right, `vTop..vBot` canvas top to bottom. */
export function tileUv(name: TileName): { u0: number; u1: number; vTop: number; vBot: number } {
  const t = TILES[name];
  return {
    u0: (t.x + GUTTER) / ATLAS_SIZE,
    u1: (t.x + t.w - GUTTER) / ATLAS_SIZE,
    vTop: (t.y + GUTTER) / ATLAS_SIZE,
    vBot: (t.y + t.h - GUTTER) / ATLAS_SIZE,
  };
}

type Ctx = CanvasRenderingContext2D;
type Pt = [number, number];

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/** sRGB triple to css, with a brightness factor. */
function css(rgb: readonly number[], k = 1): string {
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${c(rgb[0])}, ${c(rgb[1])}, ${c(rgb[2])})`;
}

let cached: THREE.DataTexture | null = null;

/**
 * Builds (once) and returns the shared foliage atlas.
 *
 * Colour on one canvas and coverage on a second so the colour under transparent
 * texels can be the tile's mean leaf colour — filtering then never pulls a dark
 * fringe into a leaf edge — and the alpha channel is exactly the shapes drawn.
 */
export function getLeafAtlas(): THREE.DataTexture {
  if (cached) return cached;

  const make = (): Ctx => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = ATLAS_SIZE;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D canvas unavailable for the leaf atlas');
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    return ctx;
  };
  const col = make();
  const cov = make();
  const rng = mulberry32(0x7a1f3e);
  const R = (a: number, b: number): number => a + (b - a) * rng();

  // Background colour per tile = that tile's mean colour.
  const bg = (rect: AtlasRect, rgb: readonly number[]) => {
    col.fillStyle = css(rgb);
    col.fillRect(rect.x, rect.y, rect.w, rect.h);
  };

  const fill = (path: (c: Ctx) => void, style: string) => {
    for (const [c, s] of [
      [col, style],
      [cov, '#fff'],
    ] as const) {
      c.beginPath();
      path(c);
      c.fillStyle = s;
      c.fill();
    }
  };
  const line = (pts: Pt[], w0: number, w1: number, style: string) => {
    for (let i = 0; i < pts.length - 1; i++) {
      const w = Math.max(0.8, lerp(w0, w1, i / Math.max(1, pts.length - 1)));
      for (const [c, s] of [
        [col, style],
        [cov, '#fff'],
      ] as const) {
        c.beginPath();
        c.moveTo(pts[i][0], pts[i][1]);
        c.lineTo(pts[i + 1][0], pts[i + 1][1]);
        c.lineWidth = w;
        c.strokeStyle = s;
        c.stroke();
      }
    }
  };
  /** Leaf frame: point (u along, v across) of a blade from P pointing at angle a (canvas radians). */
  const frame = (P: Pt, a: number, L: number, W: number) => {
    const dx = Math.cos(a);
    const dy = Math.sin(a);
    return (u: number, v: number): Pt => [P[0] + dx * u * L - dy * v * W, P[1] + dy * u * L + dx * v * W];
  };
  const blade = (c: Ctx, f: (u: number, v: number) => Pt, pointy = 1, skew = 0) => {
    c.moveTo(...f(0, 0));
    c.bezierCurveTo(...f(0.12, 0.62 + skew), ...f(0.55 + 0.1 * pointy, 0.6), ...f(1, 0));
    c.bezierCurveTo(...f(0.55 + 0.1 * pointy, -0.6), ...f(0.12, -0.62 + skew), ...f(0, 0));
  };
  const halfBlade = (c: Ctx, f: (u: number, v: number) => Pt, pointy = 1) => {
    c.moveTo(...f(0, 0));
    c.bezierCurveTo(...f(0.12, 0.62), ...f(0.55 + 0.1 * pointy, 0.6), ...f(1, 0));
    c.closePath();
  };
  /** Filled blade with the far half shaded (folded on the midrib) and a midrib line. */
  const leaf = (P: Pt, a: number, L: number, W: number, rgb: readonly number[], k: number, pointy = 1) => {
    const f = frame(P, a, L, W);
    fill((c) => blade(c, f, pointy), css(rgb, k));
    if (L > 10) {
      col.beginPath();
      halfBlade(col, f, pointy);
      col.fillStyle = css(rgb, k * 0.8);
      col.fill();
      col.beginPath();
      col.moveTo(...f(0.02, 0));
      col.lineTo(...f(0.92, 0));
      col.lineWidth = Math.max(0.6, W * 0.06);
      col.strokeStyle = css(rgb, k * 1.18);
      col.stroke();
    }
  };

  // Palette, sRGB. Saturated and warm on purpose: the tone curve and the sky
  // fill both pull toward grey, so the albedo has to start further out.
  const LEAF = [74, 132, 38];
  const LEAF_DEEP = [46, 98, 30];
  const LEAF_YOUNG = [126, 168, 52];
  const FROND = [62, 124, 36];
  const FROND_OLD = [128, 132, 52];
  const RACHIS = [150, 140, 70];

  // ---------------------------------------------------------------- fronds
  const drawFrond = (rect: AtlasRect, old: boolean) => {
    bg(rect, old ? FROND_OLD : FROND);
    const y0 = rect.y + rect.h / 2;
    const x0 = rect.x + GUTTER + 2;
    const x1 = rect.x + rect.w - GUTTER - 4;
    const half = rect.h / 2 - GUTTER - 2;
    const spine = (t: number): Pt => [lerp(x0, x1, t), y0 + Math.sin(t * Math.PI) * 4];
    // Leaflets in pairs, angled toward the tip, longest a third of the way out.
    for (let t = 0.05; t < 0.985; t += R(0.009, 0.012)) {
      const P = spine(t);
      const env = Math.pow(Math.sin(Math.PI * Math.min(1, 0.12 + t * 0.95)), 0.55);
      for (const side of [-1, 1]) {
        if (rng() < (old ? 0.14 : 0.05)) continue; // torn out
        const a = side * R(0.6, 0.85); // toward the tip, off the rachis
        // Long enough to reach the tile edge at that angle.
        const L = ((half * env) / Math.sin(Math.abs(a))) * R(0.86, 1.0) * (old && rng() < 0.2 ? R(0.4, 0.75) : 1);
        const k = R(0.78, 1.12) * lerp(0.92, 1.05, t);
        const rgb = old ? (rng() < 0.4 ? [176, 150, 70] : FROND_OLD) : rng() < 0.15 ? LEAF_YOUNG : FROND;
        const f = frame(P, a, L, L * 0.13);
        fill((c) => blade(c, f, 1.2), css(rgb, k));
        // Midrib on the leaflet.
        col.beginPath();
        col.moveTo(...f(0.02, 0));
        col.lineTo(...f(0.95, 0));
        col.lineWidth = 0.9;
        col.strokeStyle = css(rgb, k * 1.2);
        col.stroke();
      }
    }
    const pts: Pt[] = [];
    for (let i = 0; i <= 24; i++) pts.push(spine(i / 24));
    line(pts, 5, 1.2, css(old ? [170, 140, 80] : RACHIS));
  };
  drawFrond(TILES.frondA, false);
  drawFrond(TILES.frondB, true);

  // ------------------------------------------------------------- broadleaf
  // Twigs fanning up from the attachment point, glossy oval leaves alternating
  // along them — a ficus / mango read rather than a temperate one.
  const vertSpray = (
    rect: AtlasRect,
    twigs: number,
    leafLen: [number, number],
    leafWid: number,
    colours: readonly (readonly number[])[],
    pointy: number,
  ) => {
    const bx = rect.x + rect.w / 2;
    const by = rect.y + rect.h - GUTTER;
    const H = rect.h - 2 * GUTTER;
    const Wd = rect.w / 2 - GUTTER;
    for (let i = 0; i < twigs; i++) {
      const f = (i + 0.5) / twigs - 0.5;
      let ang = -Math.PI / 2 + f * 1.7 + R(-0.12, 0.12);
      const len = H * R(0.7, 0.92) * (1 - Math.abs(f) * 0.3);
      const pts: Pt[] = [[bx + f * 6, by]];
      for (let s = 1; s <= 12; s++) {
        ang += f * 0.05 + R(-0.03, 0.03);
        const p = pts[s - 1];
        pts.push([p[0] + Math.cos(ang) * (len / 12), p[1] + Math.sin(ang) * (len / 12)]);
      }
      line(pts, 3.2, 1, css([92, 76, 50]));
      let side = rng() < 0.5 ? 1 : -1;
      for (let t = R(0.2, 0.3); t < 0.98; t += R(0.1, 0.14)) {
        const idx = Math.min(11, Math.floor(t * 12));
        const p = pts[idx];
        const q = pts[idx + 1];
        const a = Math.atan2(q[1] - p[1], q[0] - p[0]);
        side = -side;
        const L = R(leafLen[0], leafLen[1]) * (1 - t * 0.2);
        const la = a + side * R(0.5, 1.0);
        const rgb = colours[Math.floor(rng() * colours.length)];
        // Clamp so blades stay inside the tile.
        const tip: Pt = [p[0] + Math.cos(la) * L, p[1] + Math.sin(la) * L];
        if (Math.abs(tip[0] - bx) > Wd || tip[1] < rect.y + GUTTER) continue;
        leaf(p, la, L, L * leafWid, rgb, R(0.8, 1.15) * lerp(0.9, 1.1, t), pointy);
      }
      const tipA = Math.atan2(pts[12][1] - pts[11][1], pts[12][0] - pts[11][0]);
      leaf(pts[12], tipA + R(-0.2, 0.2), leafLen[0], leafLen[0] * leafWid, colours[0], 1.05, pointy);
    }
  };
  bg(TILES.broadleaf, LEAF);
  vertSpray(TILES.broadleaf, 5, [44, 62], 0.42, [LEAF, LEAF, LEAF_DEEP, LEAF_YOUNG], 1);

  // ----------------------------------------------------------------- clump
  {
    const rect = TILES.clump;
    bg(rect, LEAF_DEEP);
    const cx = rect.x + rect.w / 2;
    const cy = rect.y + rect.h / 2;
    const rad = rect.w / 2 - GUTTER - 18;
    const lobes = Array.from({ length: 8 }, () => ({ a: R(0, Math.PI * 2), r: R(0.72, 1.0), w: R(0.5, 0.95) }));
    const edge = (ang: number) =>
      lobes.reduce((m, l) => {
        const d = Math.abs(((ang - l.a + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
        return Math.max(m, l.r * Math.cos(Math.min(Math.PI / 2, d / l.w)));
      }, 0.62);
    for (let k = 0; k < 900; k++) {
      const ang = R(0, Math.PI * 2);
      const rr = Math.sqrt(rng()) * edge(ang) * rad;
      const P: Pt = [cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr];
      // Lit from above: the upper half of the mass is brighter.
      const up = 1 - (P[1] - (cy - rad)) / (2 * rad);
      const rgb = rng() < 0.18 ? LEAF_YOUNG : rng() < 0.5 ? LEAF : LEAF_DEEP;
      const L = R(14, 22);
      leaf(P, R(-Math.PI, Math.PI), L, L * 0.46, rgb, lerp(0.7, 1.15, up) * R(0.85, 1.1));
    }
  }

  // ------------------------------------------------------------------ fern
  {
    const rect = TILES.fern;
    bg(rect, LEAF);
    const bx = rect.x + rect.w / 2;
    const by = rect.y + rect.h - GUTTER;
    const top = rect.y + GUTTER + 2;
    const half = rect.w / 2 - GUTTER - 2;
    const spine = (t: number): Pt => [bx + Math.sin(t * 2.2) * 5, lerp(by, top, t)];
    for (let t = 0.1; t < 0.98; t += 0.032) {
      const P = spine(t);
      const env = Math.sin(Math.PI * Math.min(1, t * 1.02)) * (1 - t * 0.25);
      for (const side of [-1, 1]) {
        const L = half * env * R(0.9, 1.08);
        const a = -Math.PI / 2 + side * R(1.05, 1.25);
        const f = frame(P, a, L, L * 0.2);
        const rgb = rng() < 0.2 ? LEAF_YOUNG : LEAF;
        fill((c) => blade(c, f, 0.6), css(rgb, R(0.85, 1.12)));
        // Pinnule notches: darker dashes across the pinna.
        for (let u = 0.15; u < 0.95; u += 0.14) {
          col.beginPath();
          col.moveTo(...f(u, -0.55));
          col.lineTo(...f(u + 0.05, 0.55));
          col.lineWidth = 1;
          col.strokeStyle = css(rgb, 0.7);
          col.stroke();
        }
      }
    }
    const pts: Pt[] = [];
    for (let i = 0; i <= 20; i++) pts.push(spine(i / 20));
    line(pts, 3, 1, css([110, 120, 50]));
  }

  // ----------------------------------------------------------------- grass
  {
    const rect = TILES.grass;
    bg(rect, [110, 150, 50]);
    const bx = rect.x + rect.w / 2;
    const by = rect.y + rect.h - GUTTER;
    for (let i = 0; i < 46; i++) {
      const x = bx + R(-1, 1) * (rect.w * 0.22);
      const lean = R(-0.5, 0.5) + (x - bx) / rect.w;
      const len = (rect.h - 2 * GUTTER) * R(0.45, 0.98);
      const pts: Pt[] = [[x, by]];
      let a = -Math.PI / 2 + lean * 0.4;
      for (let s = 1; s <= 8; s++) {
        a += lean * 0.08;
        const p = pts[s - 1];
        const nx = Math.min(rect.x + rect.w - GUTTER - 2, Math.max(rect.x + GUTTER + 2, p[0] + Math.cos(a) * len / 8));
        pts.push([nx, Math.max(rect.y + GUTTER + 2, p[1] + Math.sin(a) * len / 8)]);
      }
      const k = R(0.75, 1.2);
      const rgb = rng() < 0.25 ? [150, 160, 70] : rng() < 0.5 ? [96, 150, 48] : [70, 124, 40];
      line(pts, R(4, 6.5), 0.8, css(rgb, k));
    }
  }

  // --------------------------------------------------------------- bigleaf
  {
    const rect = TILES.bigleaf;
    bg(rect, LEAF);
    const bx = rect.x + rect.w / 2;
    const by = rect.y + rect.h - GUTTER;
    const top = rect.y + GUTTER + 2;
    const L = by - top;
    // Petiole for the first fifth, blade above it.
    line([[bx, by], [bx, by - L * 0.2]], 6, 4, css([96, 128, 52]));
    const P: Pt = [bx, by - L * 0.18];
    const f = frame(P, -Math.PI / 2, L * 0.82, rect.w * 0.44);
    fill((c) => blade(c, f, 0.4, 0), css(LEAF, 1));
    col.beginPath();
    halfBlade(col, f, 0.4);
    col.fillStyle = css(LEAF, 0.82);
    col.fill();
    // Lateral veins, and a couple of wind tears.
    for (let u = 0.08; u < 0.95; u += 0.045) {
      for (const s of [-1, 1]) {
        col.beginPath();
        col.moveTo(...f(u, 0));
        col.lineTo(...f(u + 0.06, s * 0.58));
        col.lineWidth = 1.1;
        col.strokeStyle = css(LEAF, 1.18);
        col.stroke();
      }
    }
    for (let k = 0; k < 5; k++) {
      const u = R(0.25, 0.9);
      const s = rng() < 0.5 ? -1 : 1;
      // Coverage only: cutting the colour canvas would leave black under the
      // tear for the filter to pull into the leaf edge.
      cov.beginPath();
      cov.moveTo(...f(u, s * 0.62));
      cov.lineTo(...f(u + 0.03, s * 0.12));
      cov.lineWidth = 2;
      cov.globalCompositeOperation = 'destination-out';
      cov.stroke();
      cov.globalCompositeOperation = 'source-over';
    }
    col.beginPath();
    col.moveTo(...f(0, 0));
    col.lineTo(...f(0.97, 0));
    col.lineWidth = 3;
    col.strokeStyle = css([180, 200, 110]);
    col.stroke();
  }

  // ---------------------------------------------------------------- flower
  {
    const rect = TILES.flower;
    bg(rect, LEAF_DEEP);
    vertSpray(rect, 4, [34, 48], 0.45, [LEAF_DEEP, LEAF], 1);
    // Five-petal blooms, held at the ends of the sprays. Colour by quadrant so a
    // card mirrored or tinted differently still reads as one species.
    const blooms: Pt[] = [];
    for (let i = 0; i < 8; i++) {
      blooms.push([rect.x + rect.w / 2 + R(-0.3, 0.3) * rect.w, rect.y + R(0.14, 0.6) * rect.h]);
    }
    for (const [i, P] of blooms.entries()) {
      const hue = i % 3 === 0 ? [232, 52, 64] : i % 3 === 1 ? [246, 112, 150] : [250, 220, 96];
      const r = R(17, 24);
      for (let p = 0; p < 5; p++) {
        const a = (p / 5) * Math.PI * 2 + R(-0.2, 0.2);
        const f = frame(P, a, r, r * 0.75);
        fill((c) => blade(c, f, 0.2), css(hue, R(0.9, 1.1)));
      }
      fill((c) => c.arc(P[0], P[1], r * 0.22, 0, Math.PI * 2), css([250, 236, 160]));
    }
  }

  // ------------------------------------------------------------------ vine
  {
    const rect = TILES.vine;
    bg(rect, LEAF_DEEP);
    const top = rect.y + GUTTER;
    const bot = rect.y + rect.h - GUTTER;
    for (let s = 0; s < 3; s++) {
      const x0 = rect.x + rect.w * (0.3 + s * 0.2) + R(-6, 6);
      const len = (bot - top) * R(0.6, 0.98);
      const pts: Pt[] = [];
      for (let i = 0; i <= 16; i++) {
        const t = i / 16;
        pts.push([x0 + Math.sin(t * 5 + s) * 6, top + t * len]);
      }
      line(pts, 2.4, 1.2, css([86, 96, 44]));
      for (let t = 0.05; t < 1; t += R(0.06, 0.09)) {
        const p = pts[Math.min(15, Math.floor(t * 16))];
        const side = rng() < 0.5 ? -1 : 1;
        const L = R(12, 20);
        const a = Math.PI / 2 + side * R(0.6, 1.2);
        const tip: Pt = [p[0] + Math.cos(a) * L, p[1] + Math.sin(a) * L];
        if (tip[0] < rect.x + GUTTER || tip[0] > rect.x + rect.w - GUTTER) continue;
        leaf(p, a, L, L * 0.55, rng() < 0.3 ? LEAF : LEAF_DEEP, R(0.85, 1.15), 0.8);
      }
    }
  }

  // ----------------------------------------------------------------- wrack
  // Dried weed and eelgrass in a ragged line, lying flat on the sand.
  {
    const rect = TILES.wrack;
    bg(rect, [70, 60, 36]);
    const cy = rect.y + rect.h / 2;
    for (let i = 0; i < 260; i++) {
      const t = rng();
      const x = rect.x + GUTTER + 6 + t * (rect.w - 2 * GUTTER - 12);
      const spread = (rect.h / 2 - GUTTER - 8) * Math.pow(Math.sin(Math.PI * t), 0.4);
      const y = cy + R(-1, 1) * spread * R(0.2, 1);
      const a = R(-0.6, 0.6) + (rng() < 0.5 ? 0 : Math.PI);
      const L = R(8, 26);
      const rgb = rng() < 0.3 ? [96, 84, 46] : rng() < 0.5 ? [58, 52, 30] : [80, 72, 38];
      const f = frame([x, y], a, L, L * R(0.08, 0.3));
      fill((c) => blade(c, f, 1), css(rgb, R(0.8, 1.15)));
    }
  }

  // ---------------------------------------------------------------- litter
  // Shells, coral fragments and pebbles: discs and fans on a clear ground.
  {
    const rect = TILES.litter;
    bg(rect, [214, 200, 170]);
    for (let i = 0; i < 70; i++) {
      const P: Pt = [
        rect.x + GUTTER + 10 + rng() * (rect.w - 2 * GUTTER - 20),
        rect.y + GUTTER + 10 + rng() * (rect.h - 2 * GUTTER - 20),
      ];
      const kind = rng();
      if (kind < 0.45) {
        // pebble
        const r = R(3, 7);
        const shade = rng() < 0.5 ? [120, 112, 100] : [168, 156, 136];
        fill((c) => c.ellipse(P[0], P[1], r, r * R(0.6, 0.9), R(0, 3), 0, Math.PI * 2), css(shade, R(0.85, 1.1)));
      } else if (kind < 0.8) {
        // fan shell with ribs
        const r = R(5, 10);
        const a = R(0, Math.PI * 2);
        const shade = rng() < 0.5 ? [238, 214, 196] : [232, 190, 160];
        fill((c) => {
          c.moveTo(P[0], P[1]);
          c.arc(P[0], P[1], r, a - 0.9, a + 0.9);
          c.closePath();
        }, css(shade));
        for (let k = -3; k <= 3; k++) {
          col.beginPath();
          col.moveTo(P[0], P[1]);
          col.lineTo(P[0] + Math.cos(a + k * 0.25) * r, P[1] + Math.sin(a + k * 0.25) * r);
          col.lineWidth = 0.8;
          col.strokeStyle = css(shade, 0.78);
          col.stroke();
        }
      } else {
        // coral rubble
        const pts: Pt[] = [P];
        let a = R(0, Math.PI * 2);
        for (let k = 1; k < 5; k++) {
          a += R(-0.6, 0.6);
          pts.push([pts[k - 1][0] + Math.cos(a) * 3, pts[k - 1][1] + Math.sin(a) * 3]);
        }
        line(pts, 2.6, 1.6, css([236, 226, 210]));
      }
    }
  }

  const tileList = Object.values(TILES) as AtlasRect[];
  const data = col.getImageData(0, 0, ATLAS_SIZE, ATLAS_SIZE).data;
  const mask = cov.getImageData(0, 0, ATLAS_SIZE, ATLAS_SIZE).data;
  const level0 = new Uint8Array(ATLAS_SIZE * ATLAS_SIZE * 4);
  for (let i = 0; i < level0.length; i += 4) {
    level0[i] = data[i];
    level0[i + 1] = data[i + 1];
    level0[i + 2] = data[i + 2];
    level0[i + 3] = mask[i + 3];
  }

  const tex = new THREE.DataTexture(level0, ATLAS_SIZE, ATLAS_SIZE, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.mipmaps = coverageMips(level0, ATLAS_SIZE, tileList) as unknown as THREE.DataTexture['mipmaps'];
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  tex.name = 'island-leaf-atlas';
  tex.needsUpdate = true;
  cached = tex;
  return tex;
}

/**
 * Box-filtered mip chain whose alpha is rescaled per tile so the share of texels
 * above the alpha-test cutoff stays what it was at level 0 — otherwise
 * alpha-tested foliage thins out and vanishes with distance. See header.
 */
function coverageMips(
  level0: Uint8Array,
  size: number,
  tiles: readonly AtlasRect[],
  cutoff = 128,
): { data: Uint8Array; width: number; height: number }[] {
  const levels = [{ data: level0, width: size, height: size }];
  const target = tiles.map((t) => {
    let n = 0;
    for (let y = t.y; y < t.y + t.h; y++) {
      for (let x = t.x; x < t.x + t.w; x++) if (level0[(y * size + x) * 4 + 3] >= cutoff) n++;
    }
    return n / (t.w * t.h);
  });
  let src = level0;
  let w = size;
  let shift = 0;
  const hist = new Uint32Array(256);
  while (w > 1) {
    const h = w >> 1;
    shift++;
    const dst = new Uint8Array(h * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < h; x++) {
        const i0 = (2 * y * w + 2 * x) * 4;
        const i1 = i0 + w * 4;
        const o = (y * h + x) * 4;
        for (let c = 0; c < 4; c++) {
          dst[o + c] = (src[i0 + c] + src[i0 + 4 + c] + src[i1 + c] + src[i1 + 4 + c] + 2) >> 2;
        }
      }
    }
    tiles.forEach((t, k) => {
      const tx = t.x >> shift;
      const ty = t.y >> shift;
      const tw = t.w >> shift;
      const th = t.h >> shift;
      if (tw < 2 || th < 2) return;
      const want = target[k] * tw * th;
      hist.fill(0);
      for (let y = ty; y < ty + th; y++) for (let x = tx; x < tx + tw; x++) hist[dst[(y * h + x) * 4 + 3]]++;
      let acc = 0;
      let aStar = 255;
      for (let a = 255; a >= 1; a--) {
        acc += hist[a];
        if (acc >= want) {
          aStar = a;
          break;
        }
      }
      const scale = cutoff / Math.max(aStar, 1);
      if (Math.abs(scale - 1) < 0.01) return;
      for (let y = ty; y < ty + th; y++) {
        for (let x = tx; x < tx + tw; x++) {
          const i = (y * h + x) * 4 + 3;
          dst[i] = Math.min(255, Math.round(dst[i] * scale));
        }
      }
    });
    levels.push({ data: dst, width: h, height: h });
    src = dst;
    w = h;
  }
  return levels;
}
