import * as THREE from 'three/webgpu';
import {
  Fn,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  float,
  mix,
  perspectiveDepthToViewZ,
  positionGeometry,
  screenUV,
  smoothstep,
  texture,
  uniform,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  viewportDepthTexture,
} from 'three/tsl';
import { smoothstepDown } from '../core/tslMath';

/**
 * The look shared by every particle of thrown water: hull slams and floater
 * splashes (`Spray`), and the bow's running spray (`gameplay/effects.BowSpray`).
 * Each system owns its own trajectories; this owns what a particle *is* once it
 * is somewhere — its shape on screen, how it breaks up, how it is lit, and how
 * it meets the water and the hull.
 *
 * ### Why the old particles read as cotton
 *
 * The blind review called them "cotton-ball smoke puffs" and "paper blobs", and
 * the cause was in three places at once. The shape was a disc thresholded
 * through two octaves of value noise, which is a cauliflower: a puff, not
 * water. The droplets were 10–30 cm discs, which at 30 m are 3–9 px round dots
 * — the white ellipses scattered around the flotsam in the review frame. And
 * every particle met the sea and the planks with a hard quad edge, because
 * nothing compared it with the depth behind it.
 *
 * ### What replaced it
 *
 * **Breakup by alpha erosion over a generated atlas.** A water sheet thrown off
 * a hull does what a sheet off a drop impact does, only larger: its rim thickens
 * and goes unstable, fingers into radial ligaments, and the ligaments pinch off
 * into drops (Villermaux & Bossa, "Drop fragmentation on impact", JFM 668,
 * 2011; Villermaux, "Fragmentation", Annu. Rev. Fluid Mech. 39, 2007). That is a
 * flipbook a VFX artist would paint as sheet → torn sheet → strands → droplet
 * clusters. Here it is generated: one 256² tileable atlas built at boot, with a
 * channel per stage — a sheet field, a ligament field stretched along the
 * velocity, a droplet field — and the particle's age moves both *which* field
 * it shows and the threshold it is cut at. That threshold-over-life is the
 * standard "alpha erosion" technique (vfxdoc, "Alpha Erosion",
 * https://vfxdoc.readthedocs.io/en/latest/shaders/alpha-erosion/), with a
 * narrow feather so the edges stay crisp like water rather than soft like smoke.
 *
 * **Droplets stretched along their velocity.** Every quad's long axis is its
 * velocity projected into the view plane and its short axis the cross product
 * of that with the view ray — Niagara's "Velocity Aligned" sprite, and the axis
 * construction of three.quarks' stretched billboard
 * (`stretched_bb_particle_vert.glsl.ts`, MIT; see THIRD_PARTY_NOTICES.md).
 * The length differs from three.quarks', which scales `|v| * size`: here it is
 * `width + |v_rel| * streak`, with `streak` in seconds, so a streak is an
 * exposure time — the same at any droplet size — and `v_rel` is relative to
 * the camera, because motion blur is.
 * Droplets are 2–6 cm wide — the size class that streaks visibly; Ryerson
 * measured a 234 µm median on a cutter's slam spray, which is mist at any
 * range a camera sees it from — and never thinner than about a pixel, with
 * the opacity scaled down by what the clamp added so a distant streak keeps
 * the coverage it physically has instead of shimmering.
 *
 * **Soft particles.** Depth fade against the scene behind each fragment
 * (Lorach, "Soft Particles", NVIDIA whitepaper, 2007), so a sheet tearing up
 * the topsides or settling into the sea dissolves where it meets them instead
 * of showing a straight quad edge.
 *
 * **Lit like water, not like a cloud.** Sun through a Henyey–Greenstein phase
 * function (Henyey & Greenstein, ApJ 93, 1941) — g = 0.85 for mist, typical of
 * water droplets in the visible (Kokhanovsky, "Optical properties of
 * terrestrial clouds", Earth-Sci. Rev. 64, 2004) — blended with a diffuse term
 * for the thick, multiply-scattering white water; the sky from above; and the
 * hull's own shadow, tested analytically against an oriented box in the vertex
 * stage. Thin parts of a sheet are darker and more transparent than thick ones,
 * because a thin sheet of water transmits rather than scatters.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Particle classes, as stored per instance. */
export const CLASS_DROP = 0;
export const CLASS_SHEET = 1;
export const CLASS_MIST = 2;
export const CLASS_CLUSTER = 3;

/** Atlas resolution. 256² RGBA8 is 256 KB with mips; generated in ~10 ms. */
const ATLAS_SIZE = 256;

/**
 * Light and view state shared by every spray system. Module scope because the
 * bow spray lives inside gameplay and the slam spray inside physics, and both
 * must be lit by the same sun in the same frame; `Spray` writes them.
 */
export const sprayUniforms = {
  /** Near-surface wind velocity every particle's drag relaxes toward, m/s. */
  wind: uniform(new THREE.Vector3()),
  sunDirection: uniform(new THREE.Vector3(0, 1, 0)),
  /** Sun colour times its level. */
  sunColor: uniform(new THREE.Color(1, 1, 1)),
  /** Sky irradiance, for the ambient term. */
  skyColor: uniform(new THREE.Color(0.5, 0.6, 0.7)),
  /** World metres per pixel per metre of distance: 2 tan(fov/2) / height. */
  pixelAngle: uniform(0.0016),
  /**
   * The camera's world velocity, m/s. Streaks are motion blur, and blur is
   * motion relative to the lens: spray carried along with a ship is not
   * streaked in a camera riding with that ship.
   */
  cameraVelocity: uniform(new THREE.Vector3()),
  /** World → hull-local transform for the shadow test. */
  hullInverse: uniform(new THREE.Matrix4()),
  /** Hull box in hull-local space: centre and half-extents. Zero extents disable it. */
  hullCenter: uniform(new THREE.Vector3()),
  hullHalf: uniform(new THREE.Vector3()),
};

/** Radiance and opacity, quad shape, and the fragment's age/depth/softness. */
const vShade = /*@__PURE__*/ varyingProperty('vec4', 'sprayShade');
const vQuad = /*@__PURE__*/ varyingProperty('vec4', 'sprayQuad');
const vMisc = /*@__PURE__*/ varyingProperty('vec4', 'sprayMisc');

let atlas: THREE.DataTexture | null = null;
let depthNode: any = null;

/**
 * One depth-copy node for every spray material, on purpose.
 *
 * A viewport texture node copies the framebuffer the first time a draw that
 * uses it runs in a render call, and each node *instance* copies separately.
 * The spray systems draw their dense cores into depth (renderOrder 5) before
 * their colour (6) so the volumetric fog sees them; if the colour pass took its
 * own copy it would contain those cores, and every particle would soft-fade
 * against itself to nothing. Sharing one instance, used by the depth passes as
 * well, makes the copy happen before the first core is written.
 */
function sharedSceneDepth(): any {
  if (!depthNode) depthNode = viewportDepthTexture(screenUV);
  return depthNode;
}

export function sprayAtlas(): THREE.DataTexture {
  if (atlas) return atlas;
  const n = ATLAS_SIZE;
  const data = new Uint8Array(n * n * 4);
  // The sheet and mist fields are gradient noise, rank-matched to the value
  // noise they replace (`matchQuantiles`), so every erosion threshold tuned on
  // the old fields cuts the same share of a sheet at the same age.
  //
  // **Why.** Value noise is bilinear between lattice values, and a threshold
  // through it comes out as straight-sided, square-cornered contours aligned to
  // the lattice. Cut with a narrow feather and edged with a bright rim, those
  // were the "glassy, translucent, cube-like sprites ... ice, not spray" of
  // round 10's boat-sea-2, which every round since 8 had flagged. Gradient
  // noise has no lattice-aligned facets: its contours are rounded and ragged.
  const sheetOld = new Float32Array(n * n);
  const sheetNew = new Float32Array(n * n);
  const mistOld = new Float32Array(n * n);
  const mistNew = new Float32Array(n * n);
  const ligamentOld = new Float32Array(n * n);
  const ligamentNew = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const u = x / n;
      const v = y / n;
      const k = y * n + x;
      sheetOld[k] = fbm(u, v, 4, 4, 11);
      sheetNew[k] = gradientFbm(u, v, 4, 5, 11);
      mistOld[k] = billow(u, v, 3, 5, 53);
      mistNew[k] = gradientBillow(u, v, 3, 5, 53);
      ligamentOld[k] = ligaments(u, v, false);
      ligamentNew[k] = ligaments(u, v, true);
    }
  }
  matchQuantiles(sheetNew, sheetOld);
  matchQuantiles(mistNew, mistOld);
  matchQuantiles(ligamentNew, ligamentOld);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const u = x / n;
      const v = y / n;
      // R: the intact sheet — smooth fBm, so the first holes open as rounded
      // perforations, which is how a thinning sheet fails.
      const sheet = sheetNew[y * n + x];
      // B: ligaments — ridged noise three times finer across the velocity than
      // along it, so the strands run the way the water is going. (Six times read
      // as needles: ice, not water.) Gradient noise since round 10, like the
      // sheet: ridges of value noise ran along the lattice and broke at its
      // corners, which read as crystal facets in a sheet tearing into strands.
      const lig = ligamentNew[y * n + x];
      // G: droplets — jittered cells, one blob per cell with its own radius, so
      // a rising threshold shrinks drops unevenly and the small ones go first.
      const drops = dropField(u, v, 9, 37);
      // A: mist — billowed fBm with more octaves; soft, irregular, not round.
      const mist = mistNew[y * n + x];
      const i = (y * n + x) * 4;
      data[i] = Math.round(255 * clamp01(sheet));
      data[i + 1] = Math.round(255 * clamp01(drops));
      data[i + 2] = Math.round(255 * clamp01(lig * lig * 1.15));
      data[i + 3] = Math.round(255 * clamp01(mist));
    }
  }
  const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  // Data, not colour: erosion thresholds must be linear (the vfxdoc page makes
  // the same point about imported erosion maps).
  tex.colorSpace = THREE.NoColorSpace;
  tex.name = 'spray-atlas';
  tex.needsUpdate = true;
  atlas = tex;
  return tex;
}

/**
 * A spray material around a vertex builder that ends in `particleVertex`.
 *
 * `depthOnly` is the second pass every spray system draws: the dense cores
 * again, into depth only, so the post-process fog sees the spray at its own
 * distance rather than fogging it as whatever is behind it. Only the cores, by
 * alpha test, because a soft fringe written into depth occludes the particles
 * behind it as a hard-edged square.
 */
export function sprayMaterial(vertex: () => any, depthOnly: boolean): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial();
  material.side = THREE.DoubleSide;
  // Aerial perspective reaches this through the volumetric pass from scene
  // depth, like everything else in the scene; vertex fog would double it.
  material.fog = false;
  material.transparent = true;
  // One flow for the whole vertex stage, writing its varyings as it goes. A TSL
  // variable belongs to the flow that declared it, so building the graph twice
  // (once for position, once for a varying) is a trap with no upside.
  material.positionNode = Fn(() => vertex())() as any;
  material.colorNode = Fn(() => particleFragment())() as any;
  if (depthOnly) {
    material.depthWrite = true;
    material.colorWrite = false;
    material.alphaTest = 0.3;
  } else {
    material.depthWrite = false;
  }
  return material;
}

/** Per-particle inputs to `particleVertex`, all TSL nodes. */
export interface ParticleInputs {
  centre: any;
  velocity: any;
  /** 0..1 over the particle's life. */
  life: any;
  /** 1 while alive, 0 otherwise. */
  alive: any;
  /** Class weights; exactly one is 1. */
  isDrop: any;
  isSheet: any;
  isMist: any;
  isCluster: any;
  cls: any;
  seed: any;
  /** Quad width before the pixel clamp, metres. */
  width: any;
  /** Seconds of motion the streak spans (three.quarks' lengthFactor). */
  streak: any;
  /** Opacity before shape, softness and the pixel clamp. */
  opacity: any;
  /** Depth over which the particle fades into what is behind it, metres. */
  soft: any;
  /** Distance past which nothing is drawn, metres. */
  range: number;
}

/**
 * Camera-facing, velocity-stretched quad; lighting; varyings. Returns the
 * world position of this vertex.
 */
export function particleVertex(p: ParticleInputs): any {
  const u = sprayUniforms;
  const toCamera: any = cameraPosition.sub(p.centre).toVar('sprayToCam');
  const distance: any = toCamera.length().toVar('sprayDist');
  const forward: any = toCamera.div(distance.max(1e-3)).toVar('sprayFwd');
  const relative: any = p.velocity.sub(u.cameraVelocity).toVar('sprayVRel');
  const vPerp: any = relative.sub(forward.mul(relative.dot(forward))).toVar('sprayVPerp');
  const vPerpLen: any = vPerp.length().toVar('sprayVPerpLen');
  // Falls back to world up projected into the view plane, then to world X,
  // when the particle is moving straight at the eye or the eye is overhead.
  const upPerp: any = vec3(0, 1, 0).sub(forward.mul(forward.y)).toVar('sprayUpPerp');
  const upPerpLen: any = upPerp.length().toVar('sprayUpPerpLen');
  const fallback: any = upPerpLen
    .greaterThan(1e-3)
    .select(upPerp.div(upPerpLen.max(1e-3)), vec3(1, 0, 0))
    .toVar('sprayFallback');
  const axisU: any = vPerpLen
    .greaterThan(0.05)
    .select(vPerp.div(vPerpLen.max(1e-3)), fallback)
    .toVar('sprayAxisU');
  const axisR: any = axisU.cross(forward).normalize().toVar('sprayAxisR');

  // Never thinner than ~1.3 px. What the clamp adds in width it takes back in
  // opacity, so a streak at 200 m covers what it physically covers instead of
  // flickering in and out of the pixel grid under TAA jitter.
  const pixel: any = distance.mul(u.pixelAngle).toVar('sprayPixel');
  const widthClamped: any = p.width.max(pixel.mul(1.3)).toVar('sprayWidthC');
  const coverage: any = p.width.div(widthClamped.max(1e-5)).toVar('sprayCoverage');
  // Sheets elongate along their motion as well as streaking: a sheet thrown off
  // a hull is a curtain, taller than it is wide. Mist drawn out by the wind.
  const aspect: any = p.isSheet.mul(0.3).add(p.isCluster.mul(0.2)).add(p.isMist.mul(0.35)).add(1);
  const length: any = widthClamped.mul(aspect).add(vPerpLen.mul(p.streak)).toVar('sprayLength');

  const world: any = p.centre
    .add(axisR.mul(positionGeometry.x.mul(widthClamped).mul(p.alive)))
    .add(axisU.mul(positionGeometry.y.mul(length).mul(p.alive)))
    .toVar('sprayWorld');

  // --- lighting ------------------------------------------------------------
  // Scattering angle: light travels sun → particle → eye, so the forward peak
  // is the eye looking toward the sun through the spray.
  const cosTheta: any = forward.negate().dot(u.sunDirection).toVar('sprayCosT');
  const g: any = p.isMist.mul(0.85 - 0.6).add(0.6).toVar('sprayG');
  const g2: any = g.mul(g);
  // 4π-normalised HG, so 1 is isotropic; capped where bloom would take over.
  const phase: any = float(1)
    .sub(g2)
    .div(float(1).add(g2).sub(g.mul(2).mul(cosTheta)).max(1e-3).pow(1.5))
    .min(16)
    .toVar('sprayPhase');
  // How much of the sun reaches the eye by one forward scatter versus many.
  // Thick white water scatters many times and is near-Lambertian; mist and
  // fine drops scatter once and keep the phase function's forward lobe.
  const diffuse: any = p.isDrop
    .mul(0.55)
    .add(p.isSheet.mul(0.75))
    .add(p.isMist.mul(0.2))
    .add(p.isCluster.mul(0.6))
    .toVar('sprayDiffuse');
  const sunUp: any = u.sunDirection.y.smoothstep(-0.05, 0.12);
  const shadow: any = hullShadow(p.centre).toVar('sprayHullShadow');
  // 0.8 since round 10 (was 0.6): lit white water, not a grey-blue film, in
  // a side-lit chase view.
  const sunTerm: any = diffuse.mul(0.8).add(diffuse.oneMinus().mul(0.25).mul(phase));
  const radiance: any = (u.sunColor as any)
    .mul(sunTerm.mul(sunUp).mul(shadow))
    .add((u.skyColor as any).mul(0.6))
    // White water with a little of the sea's colour in it; albedo below one.
    .mul(vec3(0.84, 0.9, 0.93))
    .toVar('sprayRadiance');

  const rangeFade: any = smoothstepDown(distance, p.range * 0.55, p.range);
  const opacity: any = p.opacity.mul(coverage).mul(rangeFade).mul(p.alive).toVar('sprayOpacityV');

  const viewZ: any = cameraViewMatrix.mul(vec4(p.centre, 1)).z;
  vShade.assign(vec4(radiance, opacity));
  vQuad.assign(vec4(positionGeometry.x, positionGeometry.y, p.cls, p.seed));
  vMisc.assign(vec4(p.life, viewZ, p.soft, 0));
  return world;
}

/**
 * The particle's shape, breakup, softness and colour. One atlas tap per class
 * family, evaluated branch-free because a draw mixes all four classes.
 */
export function particleFragment(): any {
  const tex = sprayAtlas();
  const q: any = vec2(vQuad.x, vQuad.y).toVar('sprayQ');
  const cls: any = vQuad.z;
  const seed: any = vQuad.w;
  const life: any = vMisc.x.clamp(0, 1).toVar('sprayFLife');

  const isDrop: any = smoothstepDown(cls, 0.4, 0.5).toVar('sprayFDrop');
  const isSheet: any = smoothstep(0.5, 0.6, cls).mul(smoothstepDown(cls, 1.4, 1.5)).toVar('sprayFSheet');
  const isMist: any = smoothstep(1.5, 1.6, cls).mul(smoothstepDown(cls, 2.4, 2.5)).toVar('sprayFMist');
  const isCluster: any = smoothstep(2.5, 2.6, cls).toVar('sprayFCluster');

  // Radial distance in the quad, 0 at the centre and 1 at the inscribed circle.
  const r: any = q.length().mul(2).toVar('sprayR');
  const offset: any = vec2(seed.mul(7.31), seed.mul(3.17).add(0.37));
  // Broad tap: sheet, ligament and mist fields across the quad.
  const t1: any = texture(tex, q.mul(vec2(0.6, 0.45)).add(offset)).toVar('sprayT1');
  // Fine tap: droplet cells, about a dozen across a cluster.
  const t2: any = texture(tex, q.mul(1.7).add(offset.yx)).toVar('sprayT2');

  // Sheet: intact → torn (ligaments) → drops, the field cut at a threshold that
  // rises with age and is higher toward the rim, so the edge goes first.
  const envSheet: any = smoothstepDown(r, 0.35, 1.0).toVar('sprayEnvS');
  // The rim goes first: the outer half of the sheet is a quarter-life ahead.
  // Earlier than it was (0.12–0.55, 0.42–0.85): a sheet that held its shape
  // for half its flight was one of the ice cubes of round 10.
  const toLigament: any = smoothstep(0.05, 0.42, life.add(r.mul(0.25)));
  const toDrops: any = smoothstep(0.3, 0.72, life);
  // Streaked along the motion from the start, at a third strength, so even the
  // intact sheet has the grain of water being thrown rather than of a cloud.
  const intact: any = mix(t1.x, t1.z, 0.3);
  const sheetField: any = mix(mix(intact, t1.z, toLigament), t2.y, toDrops).toVar('sprayFieldS');
  const sheetCut: any = mix(float(0.3), float(0.66), life).add(envSheet.oneMinus().mul(0.6)).toVar('sprayCutS');
  // Mass collects in the rims: a sheet's holes and its torn edge are bordered by
  // thick bright rolls while the membrane between them is thin and translucent
  // (Villermaux & Bossa 2011). So opacity peaks just inside the cut and falls
  // to a third across the interior — lace, not crumpled paper.
  const inside: any = sheetField.sub(sheetCut).toVar('sprayInside');
  const rim: any = smoothstepDown(inside, 0.02, 0.2).toVar('sprayRim');
  // And inside the inscribed circle whatever the field says: a ligament that
  // reaches the quad's edge is cut by a straight line, which reads as paper.
  //
  // **Aerated, not glass.** The membrane was a third as opaque as its rims and
  // tinted as thin water, and that is exactly how it read in daylight at chase
  // distance: "a clear, glassy sprite with no whitening" (critic, boat-sea-2).
  // Spray torn off a bow is not a clean film; it is a sheet full of entrained
  // air, and air in water scatters white (the same physics as foam: Whitlock,
  // Bartlett & Gurney, "Sea foam reflectance", Nature 1982, ~0.5 albedo per
  // layer). So the membrane is mostly opaque and mostly white, the rims still a
  // little brighter, and as the sheet ages into drops it leaves a soft white
  // haze behind it — the mist a real sheet sheds as it disintegrates — rather
  // than vanishing to its ligaments.
  //
  // **Soft, not faceted** (round 10). A narrow feather and rims a third
  // brighter than the membrane are what made each sheet read as a cut block of
  // ice: hard outline, glassy middle, lit edges. Spray seen from a few tens of
  // metres has none of that; its edge is a fringe of drops too small to resolve,
  // which reads as a soft alpha edge, and it sheds mist from the moment it tears.
  // So the feather is half again wider, the rims only a touch brighter, the
  // envelope round and soft, and the mist it leaves starts earlier and thicker.
  const shedMist: any = envSheet.mul(envSheet).mul(smoothstep(0.2, 0.8, life)).mul(0.55);
  const aSheet: any = smoothstep(-0.11, 0.07, inside)
    .mul(rim.mul(0.12).add(0.88))
    .mul(smoothstepDown(r, 0.45, 0.97))
    .max(shedMist);
  const thickSheet: any = rim.mul(0.12).add(0.88);

  // Cluster: a handful of drops in one quad; they shrink as they evaporate.
  const envCluster: any = smoothstepDown(r, 0.45, 1.0);
  const clusterCut: any = mix(float(0.42), float(0.72), life).add(envCluster.oneMinus().mul(0.5));
  const aCluster: any = smoothstep(clusterCut.sub(0.05), clusterCut.add(0.03), t2.y).mul(smoothstepDown(r, 0.7, 1.0));

  // Mist: a soft envelope modulated, not cut, by billows — a hard cut is what
  // turned the old mist into cotton balls. It thins as it ages.
  const envMist: any = smoothstepDown(r, 0.0, 1.0);
  const mistCut: any = mix(float(0.2), float(0.6), life);
  const aMist: any = envMist.mul(envMist).mul(smoothstep(mistCut.sub(0.35), mistCut.add(0.35), t1.w));

  // Drop: a capsule (the quad is already stretched along the motion).
  const aDrop: any = smoothstepDown(r, 0.45, 1.0);

  const shape: any = isDrop
    .mul(aDrop)
    .add(isSheet.mul(aSheet))
    .add(isMist.mul(aMist))
    .add(isCluster.mul(aCluster))
    .toVar('sprayShape');
  const thick: any = isDrop
    .mul(aDrop)
    .add(isSheet.mul(thickSheet))
    .add(isCluster)
    // Mist is many fine drops, which scatter white; tinting it like a thin
    // sheet laid a blue-grey smear over the planks behind it.
    .add(isMist)
    .toVar('sprayThick');

  // Soft particles: fade over `soft` metres in front of the scene behind.
  const sceneZ: any = perspectiveDepthToViewZ(sharedSceneDepth().x, cameraNear, cameraFar);
  const depthFade: any = vMisc.y.sub(sceneZ).div(vMisc.z.max(1e-3)).clamp(0, 1);
  // And in front of the lens, so a sheet through the chase camera is not a wall.
  const nearFade: any = smoothstep(0.4, 2.0, vMisc.y.negate());

  // Thin water transmits: a little darker and bluer where it is thin. Only a
  // little: round 10's critic read the old 0.82 grey-blue membrane as glass.
  const tint: any = mix(vec3(0.92, 0.95, 0.96), vec3(1, 1, 1), thick);
  const alpha: any = vShade.w.mul(shape).mul(depthFade).mul(nearFade).mul(thick.mul(0.25).add(0.75)).clamp(0, 1);
  return vec4((vShade as any).xyz.mul(tint), alpha);
}

/**
 * Fraction of the sun reaching a point past the hull: a ray toward the sun
 * against the hull's oriented box, slab test in hull-local space. Soft with the
 * chord length, because the real shadow edge is the hull's curved topsides and
 * the box is only its envelope.
 */
function hullShadow(world: any): any {
  const u = sprayUniforms;
  const inv: any = u.hullInverse;
  const local: any = inv.mul(vec4(world, 1)).xyz.sub(u.hullCenter);
  const dir: any = inv.mul(vec4(u.sunDirection, 0)).xyz;
  const safe: any = vec3(
    dir.x.abs().max(1e-4).mul(dir.x.sign().add(0.5).sign()),
    dir.y.abs().max(1e-4).mul(dir.y.sign().add(0.5).sign()),
    dir.z.abs().max(1e-4).mul(dir.z.sign().add(0.5).sign()),
  );
  const t1: any = u.hullHalf.negate().sub(local).div(safe);
  const t2: any = (u.hullHalf as any).sub(local).div(safe);
  const tMin: any = t1.min(t2);
  const tMax: any = t1.max(t2);
  const enter: any = tMin.x.max(tMin.y).max(tMin.z).max(0);
  const exit: any = tMax.x.min(tMax.y).min(tMax.z);
  const chord: any = exit.sub(enter).max(0);
  return float(1).sub(smoothstep(0, 1.2, chord).mul(0.82));
}

// ------------------------------------------------------------------ atlas noise

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function hashCell(x: number, y: number, seed: number): number {
  let h = Math.imul(x ^ 0x27d4eb2d, 0x85ebca6b) ^ Math.imul(y + 0x165667b1, 0xc2b2ae35) ^ Math.imul(seed, 0x9e3779b9);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

/** Value noise on a lattice that wraps at (px, py) cells: tileable. */
function periodicNoise(x: number, y: number, px: number, py: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const sx = xf * xf * (3 - 2 * xf);
  const sy = yf * yf * (3 - 2 * yf);
  const x0 = ((xi % px) + px) % px;
  const y0 = ((yi % py) + py) % py;
  const x1 = (x0 + 1) % px;
  const y1 = (y0 + 1) % py;
  const a = hashCell(x0, y0, seed);
  const b = hashCell(x1, y0, seed);
  const c = hashCell(x0, y1, seed);
  const d = hashCell(x1, y1, seed);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}

function fbm(u: number, v: number, base: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let f = base;
  for (let o = 0; o < octaves; o++) {
    sum += amp * periodicNoise(u * f, v * f, f, f, seed + o);
    norm += amp;
    amp *= 0.5;
    f *= 2;
  }
  // Stretch the narrow fBm histogram so thresholds 0.3–0.7 span holes to strands.
  return clamp01((sum / norm - 0.5) * 1.9 + 0.5);
}

/** Gradient noise on a lattice that wraps at (px, py) cells, about [-0.7, 0.7]: tileable. */
function periodicGradient(x: number, y: number, px: number, py: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const sx = fade(xf);
  const sy = fade(yf);
  const corner = (ox: number, oy: number) => {
    const cx = (((xi + ox) % px) + px) % px;
    const cy = (((yi + oy) % py) + py) % py;
    const a = hashCell(cx, cy, seed) * Math.PI * 2;
    return Math.cos(a) * (xf - ox) + Math.sin(a) * (yf - oy);
  };
  const a = corner(0, 0);
  const b = corner(1, 0);
  const c = corner(0, 1);
  const d = corner(1, 1);
  const top = a + (b - a) * sx;
  const bottom = c + (d - c) * sx;
  return top + (bottom - top) * sy;
}

function gradientFbm(u: number, v: number, base: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let f = base;
  for (let o = 0; o < octaves; o++) {
    sum += amp * periodicGradient(u * f, v * f, f, f, seed + o);
    amp *= 0.5;
    f *= 2;
  }
  return sum;
}

function gradientBillow(u: number, v: number, base: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let f = base;
  for (let o = 0; o < octaves; o++) {
    sum += amp * (1 - Math.abs(periodicGradient(u * f, v * f, f, f, seed + o)) * 1.4);
    amp *= 0.55;
    f *= 2;
  }
  return sum;
}

/** Replaces `values` with `reference`'s values in `values`' rank order. */
function matchQuantiles(values: Float32Array, reference: Float32Array): void {
  const order = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const sorted = Float32Array.from(reference).sort();
  for (let i = 0; i < order.length; i++) values[order[i]] = sorted[i];
}

/** The ligament field: ridged noise, three times finer across than along. */
function ligaments(u: number, v: number, gradient: boolean): number {
  let lig = 0;
  let amp = 0.6;
  let fx = 6;
  let fy = 2;
  for (let o = 0; o < 3; o++) {
    const ridge = gradient
      ? 1 - Math.abs(periodicGradient(u * fx, v * fy, fx, fy, 23 + o)) * 1.4
      : 1 - Math.abs(2 * periodicNoise(u * fx, v * fy, fx, fy, 23 + o) - 1);
    lig += amp * ridge;
    amp *= 0.45;
    fx *= 2;
    fy *= 2;
  }
  return lig / (0.6 + 0.27 + 0.1215);
}

function billow(u: number, v: number, base: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let f = base;
  for (let o = 0; o < octaves; o++) {
    sum += amp * (1 - Math.abs(2 * periodicNoise(u * f, v * f, f, f, seed + o) - 1));
    norm += amp;
    amp *= 0.55;
    f *= 2;
  }
  return clamp01((sum / norm - 0.45) * 2.2 + 0.5);
}

/** Jittered blobs, one per cell with its own radius, tileable at `cells`. */
function dropField(u: number, v: number, cells: number, seed: number): number {
  const x = u * cells;
  const y = v * cells;
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  let best = 0;
  for (let j = -1; j <= 1; j++) {
    for (let i = -1; i <= 1; i++) {
      const cx = xi + i;
      const cy = yi + j;
      const wx = ((cx % cells) + cells) % cells;
      const wy = ((cy % cells) + cells) % cells;
      const px = cx + 0.15 + 0.7 * hashCell(wx, wy, seed);
      const py = cy + 0.15 + 0.7 * hashCell(wx, wy, seed + 1);
      const radius = 0.16 + 0.26 * hashCell(wx, wy, seed + 2);
      const d = Math.hypot(x - px, y - py) / radius;
      // A rounded profile: a threshold anywhere in 0.4–0.8 gives a round drop.
      const value = d < 1 ? 1 - d * d : 0;
      if (value > best) best = value;
    }
  }
  return best;
}
