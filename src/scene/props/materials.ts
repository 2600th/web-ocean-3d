import * as THREE from 'three/webgpu';
import {
  Fn,
  abs,
  cameraPosition,
  dot,
  float,
  floor,
  fract,
  fwidth,
  length,
  max,
  min,
  mix,
  mx_noise_float,
  mx_worley_noise_float,
  normalViewGeometry,
  normalWorldGeometry,
  positionGeometry,
  positionView,
  positionWorld,
  sin,
  smoothstep,
  uniform,
  uv,
  vec2,
  vec3,
  vertexColor,
} from 'three/tsl';
import type { Slot } from './kit';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;
const v3 = vec3 as unknown as (x: unknown, y?: unknown, z?: unknown) => Node;
const v2 = vec2 as unknown as (x: unknown, y?: unknown) => Node;
const mixN = mix as unknown as (a: unknown, b: unknown, t: unknown) => Node;

/**
 * TSL materials for the procedural props: zero textures, every surface drawn
 * from its metric UVs, its vertex tint and world position.
 *
 * The approach is mrdoob/slop-newyork's (MIT, Copyright (c) 2026 Mr.doob):
 * procedural meshes shaded by node graphs instead of downloaded maps. Four
 * helpers are ported from its `tsl-materials.js`: `bumpNormal` (Mikkelsen's
 * surface-gradient bump, here on the geometric normal as `islandDetail/rock`
 * does so it can sit in `normalNode`), `hash21`, `gridLine` and the
 * two-nearest-sites Voronoi behind `cells`, which also returns the cell id.
 *
 * Everything shared by the props lives here, so the whole library is eleven
 * materials: the boot compile pays per material, not per prop.
 *
 * Two treatments are written into every lit material rather than applied by a
 * pass, because both depend on where the prop *is*:
 *  - **Under water** (`marine`): from a metre down, silt settles on the upward
 *    faces and a brown-green film on the rest, and the surface goes matte. The
 *    reef chest and anything else on the bottom get it by being there.
 *  - **The swash line**: within half a metre of sea level a surface is wet,
 *    dark and glossy — floating casks and flotsam carry their own waterline.
 */

// ----------------------------------------------------------------- helpers

/** Surface-gradient bump on the geometric normal. Ported from slop-newyork `bumpNormal`. */
export const bumpNormal = Fn(([height]: [Node]) => {
  const h = float(height).toVar();
  const n = normalViewGeometry;
  const dpdx = positionView.dFdx();
  const dpdy = positionView.dFdy();
  const r1 = dpdy.cross(n);
  const r2 = n.cross(dpdx);
  const det = dpdx.dot(r1);
  const grad = det.sign().mul(h.dFdx().mul(r1).add(h.dFdy().mul(r2)));
  return det.abs().mul(n).sub(grad).normalize();
});

/** Ported from slop-newyork `hash2D`. */
const hash21 = Fn(([p]: [Node]) => fract(sin(dot(p, v2(127.1, 311.7))).mul(43758.5453)));

/** Anti-aliased repeating line, 1 on the line. Ported from slop-newyork `gridLine`. */
const gridLine = Fn(([coord, period, halfWidth]: [Node, Node, Node]) => {
  const p = float(period).toVar();
  const g = float(coord).div(p).toVar();
  const d = float(0.5).sub(abs(fract(g).sub(0.5))).toVar();
  const aa = fwidth(g).max(0.0001).toVar();
  const hw = float(halfWidth).div(p).toVar();
  const eff = max(hw, aa.mul(0.5)).toVar();
  return smoothstep(eff.add(aa), eff.sub(aa), d).mul(min(float(1), hw.div(eff)));
});

/**
 * Voronoi in 2D: x = half the F2-F1 gap (0 on a cell border), y = a hash of the
 * nearest site, for per-stone tone. After slop-newyork `voronoiBorder`, with
 * the id added.
 */
const cells = Fn(([p]: [Node]) => {
  const pv = v2(p).toVar();
  const ip = floor(pv).toVar();
  const fp = fract(pv).toVar();
  const f1 = float(8).toVar();
  const f2 = float(8).toVar();
  const id = float(0).toVar();
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cell = ip.add(v2(i, j));
      const site = v2(i, j).add(v2(hash21(cell), hash21(cell.add(v2(37.1, 11.7)))).mul(0.8).add(0.1));
      const d = length(fp.sub(site)).toVar();
      const nearer = d.lessThan(f1);
      f2.assign(min(f2, max(f1, d)));
      id.assign(nearer.select(hash21(cell.add(v2(5.3, 9.1))), id));
      f1.assign(min(f1, d));
    }
  }
  return v2(f2.sub(f1).mul(0.5), id);
});

/** 0..1 Perlin. */
const n01 = (p: Node): Node => mx_noise_float(p).mul(0.5).add(0.5);

/** Bump detail fades out with distance, so grain does not alias into shimmer. */
const detailFade = (near: number, far: number): Node =>
  smoothstep(float(far), float(near), length(positionWorld.sub(cameraPosition)));

/** 0 above water, 1 from three metres down. */
const submerged = (): Node => float(1).sub(positionWorld.y.smoothstep(-3, -0.8));
/** 1 in the swash band around sea level. */
const swash = (): Node => float(1).sub(abs(positionWorld.y).smoothstep(0.15, 0.6));

/** Silt on the upward faces and a film on the rest, from a metre or so down. */
function marine(albedo: Node, strength = 1): Node {
  const depth = submerged().mul(strength).toVar();
  const wp = positionWorld;
  const patch = n01(wp.mul(2.3)).add(n01(wp.mul(9)).mul(0.35)).toVar();
  const up = normalWorldGeometry.y;
  const silt = up.add(patch.mul(0.4)).smoothstep(0.35, 0.85).mul(depth);
  // A thin film: the wood and iron must still read as wood and iron at 16 m,
  // where the water has already taken most of the colour out of everything.
  const film: Node = mixN(albedo, albedo.mul(v3(0.82, 0.9, 0.76)).add(v3(0.01, 0.014, 0.006)), depth.mul(0.6));
  return mixN(film, v3(0.2, 0.19, 0.15).mul(patch.mul(0.3).add(0.8)), silt.mul(0.55));
}

/** Darkened and glossier in the swash band. */
function wetAlbedo(albedo: Node): Node {
  return mixN(albedo, albedo.mul(0.55), swash());
}
function wetRoughness(r: Node, marineRough = 0.95): Node {
  const wet = mixN(float(r), float(r).mul(0.4).max(0.12), swash());
  return mixN(wet, float(marineRough), submerged().mul(0.8));
}

function base(name: string, roughness: number, metalness = 0): THREE.MeshStandardNodeMaterial {
  const m = new THREE.MeshStandardNodeMaterial({ roughness, metalness });
  m.name = name;
  m.envMapIntensity = 0.8;
  return m;
}

// --------------------------------------------------------------- materials

/** Lamp level of the lighthouse lantern, 0 by day .. ~1 at night, flicker included. */
export const beaconLevel = uniform(0);

function wood(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-wood', 0.85);
  // u across the grain, v along it, both in metres.
  const grain = Fn(() => {
    const t = uv();
    const u = t.x.toVar();
    const v = t.y.toVar();
    const broad = n01(v3(u.mul(5), v.mul(0.6), 1.7)).toVar();
    const fine = n01(v3(u.mul(70), v.mul(2.2), 0)).toVar();
    const ring = sin(u.add(broad.mul(0.06)).mul(260)).mul(0.5).add(0.5).toVar();
    const knot = float(1).sub(mx_worley_noise_float(v3(u.mul(4), v.mul(1.1), 3.3)).smoothstep(0.0, 0.16)).toVar();
    return vec3(broad, fine.mul(0.7).add(ring.mul(0.3)), knot);
  });
  m.colorNode = Fn(() => {
    const g = v3(grain()).toVar();
    const tone = mixN(float(0.74), float(1.14), g.y.mul(0.55).add(g.x.mul(0.45)));
    let albedo: Node = vertexColor().mul(tone).mul(float(1).sub(g.z.mul(0.45)));
    // Sun-bleached where it faces the sky: the silver of old deck plank.
    const lum = dot(albedo, v3(0.33, 0.33, 0.33));
    const bleach = normalWorldGeometry.y.smoothstep(0.35, 0.95).mul(0.15).mul(float(1).sub(submerged()));
    albedo = mixN(albedo, v3(lum, lum, lum).mul(v3(1.12, 1.08, 1.0)), bleach);
    return marine(wetAlbedo(albedo));
  })();
  m.roughnessNode = Fn(() => wetRoughness(float(0.78).add(n01(v3(uv().mul(v2(40, 2)), 0)).mul(0.16))))();
  m.normalNode = Fn(() => {
    const g = v3(grain());
    const h = g.y.mul(0.0012).sub(g.z.mul(0.002)).add(g.x.mul(0.001));
    return bumpNormal(h.mul(detailFade(1.5, 22)));
  })();
  return m;
}

function iron(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-iron', 0.55, 0.7);
  const rust = Fn(() => {
    const p = positionGeometry;
    const r = n01(p.mul(7)).add(n01(p.mul(31)).mul(0.45)).toVar();
    return r.smoothstep(0.62, 1.05);
  });
  m.colorNode = Fn(() => {
    const r = float(rust()).toVar();
    const blacksmith: Node = v3(0.05, 0.047, 0.045).mul(vertexColor());
    const oxide: Node = mixN(v3(0.22, 0.09, 0.035), v3(0.12, 0.05, 0.025), n01(positionGeometry.mul(60)));
    const c: Node = mixN(blacksmith, oxide, r.max(submerged().mul(0.35)));
    // Lightly: concretion that greyed the straps to the wood's tone erased the
    // one contrast that makes a bound chest read as one.
    return marine(wetAlbedo(c), 0.45);
  })();
  m.metalnessNode = Fn(() => mixN(float(0.75), float(0.05), float(rust()).max(submerged().mul(0.5))))();
  m.roughnessNode = Fn(() => wetRoughness(mixN(float(0.48), float(0.9), rust())))();
  m.normalNode = Fn(() => {
    const pit = n01(positionGeometry.mul(90)).mul(0.0006).add(float(rust()).mul(0.0009));
    return bumpNormal(pit.mul(detailFade(1, 12)));
  })();
  return m;
}

/**
 * Rubble masonry: random-coursed stones in metric UV (u along the wall, v up),
 * whitewashed or bare. Whitewash was lime, renewed yearly on a light tower so
 * it read as a day-mark; it flakes off the stone faces first and stays in the
 * mortar, which is the pattern drawn here.
 */
function masonry(name: string, whitewash: number): THREE.MeshStandardNodeMaterial {
  const m = base(name, 0.92);
  const field = Fn(() => {
    const t = uv();
    // Courses of roughly 0.32 m, stones about half a metre long.
    const c = v2(cells(v2(t.x.div(0.52), t.y.div(0.3)))).toVar();
    const mortar = float(1).sub(c.x.smoothstep(0.015, 0.07)).toVar();
    return vec3(mortar, c.y, n01(v3(t.x.mul(3), t.y.mul(3), 0.5)));
  });
  m.colorNode = Fn(() => {
    const f = v3(field()).toVar();
    const t = uv();
    const stone: Node = mixN(v3(0.11, 0.1, 0.085), v3(0.26, 0.23, 0.18), f.y).mul(f.z.mul(0.3).add(0.85));
    const bare: Node = mixN(stone, v3(0.07, 0.065, 0.06), f.x.mul(0.6));
    // Lime: thick in the joints, worn off the proud faces in patches.
    const flake = n01(v3(t.x.mul(0.8), t.y.mul(1.1), 2)).add(n01(v3(t.x.mul(4), t.y.mul(4), 7)).mul(0.4)).toVar();
    // Mostly sound: at 0.98..0.78 the bare patches covered a third of the shaft
    // and read as a dappled animal, not a maintained day-mark.
    const cover = float(whitewash).mul(flake.smoothstep(1.15, 0.97).max(f.x.mul(0.8))).toVar();
    const lime: Node = v3(0.7, 0.69, 0.64).mul(float(1).sub(f.x.mul(0.12)));
    let c: Node = mixN(bare, lime, cover);
    // Rain runs off the gallery and stains the shaft in vertical streaks.
    const streak = n01(v3(t.x.mul(5), t.y.mul(0.12), 4)).smoothstep(0.58, 0.9).mul(0.28);
    c = c.mul(float(1).sub(streak));
    // Rising damp and lichen at the foot of the wall.
    const foot = float(1).sub(t.y.smoothstep(0.1, 1.4)).mul(n01(v3(t.x.mul(2), t.y.mul(2), 9)).mul(0.6).add(0.4));
    c = mixN(c, v3(0.08, 0.09, 0.05), foot.mul(0.55));
    return c.mul(vertexColor());
  })();
  m.roughnessNode = Fn(() => wetRoughness(float(0.9)))();
  m.normalNode = Fn(() => {
    const f = v3(field());
    const h = f.x.mul(-0.012).add(f.z.mul(0.006)).add(f.y.mul(0.004));
    return bumpNormal(h.mul(detailFade(2, 60)));
  })();
  return m;
}

function lead(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-lead', 0.5, 0.45);
  m.colorNode = Fn(() => {
    const p = positionGeometry;
    const patina = n01(p.mul(3)).add(n01(p.mul(17)).mul(0.4)).smoothstep(0.5, 1.1);
    return mixN(v3(0.13, 0.14, 0.15), v3(0.32, 0.36, 0.33), patina).mul(vertexColor());
  })();
  m.normalNode = Fn(() => {
    // Lead roofs are laid in sheets with standing seams every half metre.
    const seam = gridLine(uv().x, 0.5, 0.012);
    return bumpNormal(seam.mul(0.004).mul(detailFade(2, 40)));
  })();
  return m;
}

/** Riven cedar shingles in staggered courses; v runs down the slope. */
function shingle(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-shingle', 0.9);
  const field = Fn(() => {
    const t = uv();
    const course = t.y.div(0.17).toVar();
    const row = floor(course);
    const along = t.x.div(0.15).add(hash21(v2(row, 3.1)).mul(3)).toVar();
    const id = hash21(v2(floor(along), row));
    // Each shingle is thickest at its butt (the bottom of the course).
    const butt = fract(course).toVar();
    const gap = float(1).sub(abs(fract(along).sub(0.5)).smoothstep(0.44, 0.49));
    return vec3(id, butt, gap);
  });
  m.colorNode = Fn(() => {
    const f = v3(field()).toVar();
    const c: Node = mixN(v3(0.06, 0.05, 0.042), v3(0.15, 0.13, 0.1), f.x).mul(f.z.mul(0.25).add(0.75));
    return c.mul(float(1).sub(f.y.smoothstep(0.75, 1).mul(0.35))).mul(vertexColor());
  })();
  m.roughnessNode = Fn(() => wetRoughness(float(0.88)))();
  m.normalNode = Fn(() => {
    const f = v3(field());
    return bumpNormal(f.y.mul(0.012).sub(f.z.mul(0.004)).mul(detailFade(2, 50)));
  })();
  return m;
}

/** The lighthouse glazing: dark glass by day, lit from inside by the lamps at night. */
function beaconGlass(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-beacon-glass', 0.06);
  m.color.setRGB(0.03, 0.035, 0.035);
  m.envMapIntensity = 1.2;
  m.emissiveNode = Fn(() => {
    // Brightest at the height of the lamps, falling toward the sill and the
    // head of each pane: the glass is lit by a source, not self-luminous.
    const t = uv();
    const band = float(1).sub(abs(t.y.sub(0.95)).smoothstep(0.1, 1.0)).mul(0.7).add(0.3);
    return v3(1.0, 0.6, 0.26).mul(beaconLevel).mul(band).mul(3.2);
  })();
  m.userData.noGroundShading = true;
  return m;
}

/** Horn panes of a ship's lanthorn: amber, translucent-looking, lit by `LocalLights` (name matches /glass/). */
function lanternGlass(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-lantern-glass', 0.45);
  m.colorNode = Fn(() => mixN(v3(0.33, 0.22, 0.09), v3(0.45, 0.32, 0.14), n01(positionGeometry.mul(40))))();
  return m;
}

/** Candle and lamp flames: emission only, driven by `beaconLevel` with a little flutter. */
function flame(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-flame', 1);
  m.color.setRGB(0.02, 0.015, 0.01);
  m.emissiveNode = Fn(() => {
    const tip = uv().y.smoothstep(0, 1);
    const hot: Node = mixN(v3(6, 3.2, 1.1), v3(3, 1.1, 0.2), tip);
    return hot.mul(beaconLevel);
  })();
  m.userData.noGroundShading = true;
  return m;
}

function gold(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-gold', 0.28, 1);
  m.colorNode = Fn(() => {
    const p = positionGeometry;
    const tarnish = n01(p.mul(25)).smoothstep(0.55, 0.9).mul(0.35);
    // A tint of 0.5 grey marks silver (pieces of eight); any other tint
    // multiplies gold (escudos, white) or turns it to brass.
    const vc = vertexColor();
    const silver = float(1).sub(vc.x.smoothstep(0.55, 0.6));
    const goldC: Node = mixN(v3(1.0, 0.77, 0.34), v3(0.55, 0.4, 0.18), tarnish).mul(vc);
    const silverC: Node = mixN(v3(0.95, 0.93, 0.88), v3(0.3, 0.3, 0.32), tarnish.mul(1.6));
    return mixN(goldC, silverC, silver);
  })();
  m.roughnessNode = Fn(() => float(0.22).add(n01(positionGeometry.mul(60)).mul(0.18)).add(submerged().mul(0.1)))();
  // Gold does not tarnish, and on the bottom it is the one thing that catches
  // what light there is. A trace of self-colour keeps the find legible at 16 m
  // once the water has taken the red out of it; zero above water.
  m.emissiveNode = Fn(() => v3(0.55, 0.38, 0.12).mul(submerged()).mul(0.05))();
  // Coin-sized cells: on the heap they are the coins; on a single coin, the
  // hammered unevenness of a cob, which pieces of eight were.
  m.normalNode = Fn(() => {
    const d = mx_worley_noise_float(positionGeometry.mul(48));
    return bumpNormal(float(1).sub(d.smoothstep(0.25, 0.55)).mul(0.0025).mul(detailFade(1, 20)));
  })();
  return m;
}

function gem(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-gem', 0.04);
  m.colorNode = vertexColor();
  m.envMapIntensity = 1.5;
  // A little of their own colour, so a ruby in the gloom at 16 m still reads red.
  m.emissiveNode = vertexColor().mul(0.12);
  return m;
}

/** Salt-glazed stoneware: the mottled brown "tiger" glaze of Frechen Bartmann jugs. */
function stoneware(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-stoneware', 0.3);
  m.colorNode = Fn(() => {
    const p = positionGeometry;
    const mottle = n01(p.mul(38)).add(n01(p.mul(110)).mul(0.5)).smoothstep(0.55, 1.15);
    const c: Node = mixN(v3(0.19, 0.085, 0.026), v3(0.035, 0.018, 0.008), mottle);
    return marine(wetAlbedo(c.mul(vertexColor())));
  })();
  m.roughnessNode = Fn(() => wetRoughness(float(0.26).add(n01(positionGeometry.mul(70)).mul(0.1)), 0.7))();
  m.normalNode = Fn(() => bumpNormal(n01(positionGeometry.mul(240)).mul(0.00025).mul(detailFade(0.5, 6))))();
  return m;
}

/** Dark olive bottle glass, near-opaque at these thicknesses. */
function bottle(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-bottle', 0.05);
  m.colorNode = Fn(() => marine(v3(0.02, 0.045, 0.018).mul(vertexColor()), 0.6))();
  m.envMapIntensity = 1.4;
  return m;
}

/** Laid hemp. u around the rope in metres, v along it. */
function rope(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-rope', 0.92);
  const lay = Fn(() => {
    const t = uv();
    return sin(t.x.mul(37).add(t.y.mul(26)).mul(Math.PI * 2)).mul(0.5).add(0.5);
  });
  m.colorNode = Fn(() => {
    const s = float(lay()).toVar();
    const fibre = n01(v3(uv().mul(v2(200, 30)), 0));
    const c = vertexColor().mul(s.mul(0.35).add(0.7)).mul(fibre.mul(0.25).add(0.85));
    return marine(wetAlbedo(c));
  })();
  m.roughnessNode = Fn(() => wetRoughness(float(0.92)))();
  m.normalNode = Fn(() => bumpNormal(float(lay()).mul(0.0018).mul(detailFade(0.6, 10))))();
  return m;
}

/**
 * Shell: the vertex tint carries the cream outer whorl and the pink of the lip;
 * growth lines run around the whorl (uv.y is metres along the coil).
 */
function shell(): THREE.MeshStandardNodeMaterial {
  const m = base('prop-shell', 0.5);
  m.colorNode = Fn(() => {
    const growth = sin(uv().y.mul(260)).mul(0.5).add(0.5);
    const mottle = n01(positionGeometry.mul(30));
    const c = vertexColor().mul(growth.mul(0.12).add(0.88)).mul(mottle.mul(0.25).add(0.82));
    return marine(c, 0.5);
  })();
  m.normalNode = Fn(() => bumpNormal(sin(uv().y.mul(260)).mul(0.0003).mul(detailFade(0.5, 6))))();
  return m;
}

/** Every material the procedural props draw with, created once per `Props`. */
export class PropMaterials {
  private readonly bySlot = new Map<Slot, THREE.MeshStandardNodeMaterial>();

  get(slot: Slot): THREE.MeshStandardNodeMaterial {
    let m = this.bySlot.get(slot);
    if (m) return m;
    switch (slot) {
      case 'wood':
        m = wood();
        break;
      case 'iron':
        m = iron();
        break;
      case 'masonry':
        m = masonry('prop-masonry', 1);
        break;
      case 'lead':
        m = lead();
        break;
      case 'shingle':
        m = shingle();
        break;
      case 'beaconGlass':
        m = beaconGlass();
        break;
      case 'lanternGlass':
        m = lanternGlass();
        break;
      case 'flame':
        m = flame();
        break;
      case 'gold':
        m = gold();
        break;
      case 'gem':
        m = gem();
        break;
      case 'stoneware':
        m = stoneware();
        break;
      case 'bottle':
        m = bottle();
        break;
      case 'rope':
        m = rope();
        break;
      case 'shell':
        m = shell();
        break;
    }
    this.bySlot.set(slot, m);
    return m;
  }

  dispose(): void {
    for (const m of this.bySlot.values()) m.dispose();
    this.bySlot.clear();
  }
}
