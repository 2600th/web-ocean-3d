import * as THREE from 'three/webgpu';
import {
  Fn,
  attribute,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraViewMatrix,
  dFdx,
  dFdy,
  float,
  mix,
  modelWorldMatrixInverse,
  perspectiveDepthToViewZ,
  positionGeometry,
  positionWorld,
  smoothstep,
  texture,
  varyingProperty,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { uniform } from '../core/uniform';
import { smoothstepDown } from '../core/tslMath';
import { SPRAY_ALPHA_FLOOR, sharedSceneDepth, sheetRadiance, sprayAtlas } from '../physics/sprayLook';

/**
 * The bow-wave sheet: the thin, connected curtain of water a bluff bow peels off
 * at speed, attached to the hull at its root, rolling outward and down, its free
 * edge tearing into strands and drops.
 *
 * ### Why a surface and not particles
 *
 * Round 10's critic still saw the bow spray as "a cluster of separate white
 * chips": a particle system can only make a sheet out of many sheet-shaped
 * particles, and however soft their edges, a few dozen overlapping quads read as
 * a few dozen things. What a camera sees at a bow at six to eight knots is one
 * surface — the bow wave's crest thrown clear of the hull — that is continuous
 * where it leaves the planking and only breaks up toward its free edge. So the
 * sheet is a mesh, and the particles (`BowSpray`) are only what it sheds: drops
 * from its edge and mist downwind of it. Games draw it the same way (a hull-
 * attached bow-wave mesh with scrolling breakup, soft-faded where it meets the
 * hull and the sea, plus emitted spray; e.g. the Blood Stone water write-up,
 * Computer Graphics World, 2010).
 *
 * ### Its shape: ballistic, steady in the hull's frame
 *
 * In the hull's frame the flow past a ship at steady speed is steady, so the
 * sheet is a fixed surface swept by the water moving through it: the union of
 * the trajectories of water leaving the hull along the bow-wave crest. Each
 * station `s` along the root (stem to shoulder) launches water outward, upward
 * and aft; `tau` is the fraction of its flight back to the sea. Position and
 * velocity are closed-form (`sample`), evaluated identically here on the CPU,
 * where `BowSpray` emits drops from the sheet's edge, and in the vertex stage.
 *
 * The scale follows Noblesse et al., "Simple analytical relations for ship bow
 * waves" (J. Fluid Mech. 600, 2008): the bow wave's height is about
 * `2.2 (U²/g) tan(α) / (1 + F_D)` for a bow of half-entrance angle `α` and draft
 * Froude number `F_D`, which for a bluff bow (α ≈ 40°, 4 m draft) at seven
 * knots is about 1.5 m, and it scales with `U²`. The sheet's crest height is a
 * fraction of that, scaled by `intensity`.
 *
 * ### Rooted on the hull, flown level
 *
 * The root rides the planking: each station's point on the hull side, at the
 * waterline under the stem's centreline plus the crest. The flight does not.
 * Water leaving the hull falls under world gravity and lands on the sea, so the
 * trajectories are laid out in a frame that keeps only the ship's heading
 * (`levelQuaternion`). The whole sheet used to go through the hull's matrix,
 * so in a turn, when the ship heels, it heeled with it: the high side's free
 * edge, six or seven metres out, ended a metre above the water as a pale card,
 * and the low side cut under the sea. Upright, the two are the same surface.
 *
 * ### Intensity follows the bow
 *
 * `BowSpray` drives `intensity` from the ship's speed and the stem's immersion
 * and plunge, attacking fast and releasing slowly: calm while the bow lifts out
 * of a sea, a burst while it digs in.
 *
 * ### Its look
 *
 * Thin, translucent water near the root, aerated and white toward the edge,
 * eroded through the spray atlas's sheet and ligament fields advected along the
 * flow (the same alpha erosion as `sprayLook`), so the free edge tears into
 * strands running the way the water goes. Lit like the particles: the sun
 * through a Henyey–Greenstein forward lobe for the thin sheet, near-Lambertian
 * for the white water, sky from above, the hull's shadow. Soft-faded against
 * the scene's depth so it meets the planking and the sea without a seam.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

/** Grid resolution of one side's sheet: along the root, and along the flight. */
const ROOT_SEGMENTS = 64;
const FLIGHT_SEGMENTS = 12;

const GRAVITY = 9.81;

/**
 * Height the bow wave climbs the planking at full intensity, metres: most of
 * Noblesse's ~1.5 m bow wave, since the sheet leaves from its crest.
 */
const CREST_HEIGHT = 1.3;
/**
 * Launch speeds at full intensity, m/s: base plus the part that scales with
 * intensity.
 *
 * Roughly halved from 1.2 + 3.6 up and 2.6 + 4.2 out. At those speeds the sheet
 * flew about seven metres out from the planking and peaked two and a half
 * metres above the sea, and a turn, which buries the bow and drives the
 * intensity up, put that whole translucent curtain in front of the chase
 * camera: it read as a foam-textured card floating beside the bow. A bluff bow
 * at these speeds curls its crest over and drops it back within a few metres
 * of the hull, which is the shape these give.
 */
const UP_BASE = 0.7;
const UP_GAIN = 2.0;
const OUT_BASE = 1.4;
const OUT_GAIN = 2.3;
/** How far outside the waterline half-breadth the root sits, metres: the bow's flare above the water. */
const ROOT_OFFSET = 0.12;
/**
 * The stem at the waterline, as a fraction of the half-length (the water
 * material's hull band takes the waterline as 0.84 of the overall length: no
 * bowsprit), and the length of root the sheet leaves from, aft of it.
 */
const STEM = 0.84;
const ROOT_LENGTH = 0.24;

/** Depth over which the sheet fades into the hull or the sea behind it, metres. */
const SOFT_DEPTH = 0.35;

/**
 * The bow wave is not a steady crest: it surges with the pitch and with the
 * turbulence in the bow's boundary layer, so the launch speed along the root
 * pulses by about a third, in lumps a few stations long that run aft with the
 * water. Without it every trajectory's apex lies on one smooth envelope and the
 * sheet's top reads as a shell.
 */
function surge(s: number, side: number, time: number): number {
  return (
    1 +
    0.4 * Math.sin(s * 29 - time * 5.3 + side * 1.3) * Math.sin(s * 17 - time * 3.1 + side * 0.7) +
    0.2 * Math.sin(s * 53 - time * 8.9 + side * 2.1)
  );
}

/** Per-station launch, in hull-local metres and m/s. */
export interface SheetStation {
  x: number;
  y: number;
  z: number;
  up: number;
  out: number;
  aft: number;
  flight: number;
}

/**
 * Hull-local launch state at root station `s` (0 at the stem, 1 at the
 * shoulder), for one side (`side` +1 starboard, -1 port). Mirrors the vertex
 * stage's `stationNode` exactly.
 */
export function sheetStation(
  s: number,
  side: number,
  intensity: number,
  speed: number,
  halfLength: number,
  beam: number,
  time: number,
  out: SheetStation,
): SheetStation {
  const crest = Math.sqrt(Math.min(1, s / 0.12)) * Math.pow(1 - Math.min(1, s), 1.5);
  // A bluff, elliptic waterline: the half-breadth a distance q (fraction of
  // the waterline half-length) aft of the stem.
  const q = (s * ROOT_LENGTH) / STEM;
  const halfWidth = beam * 0.5 * Math.sqrt(Math.max(0, 2 * q - q * q));
  out.x = halfLength * (STEM - ROOT_LENGTH * s);
  out.y = -0.1 + CREST_HEIGHT * intensity * crest;
  out.z = side * (halfWidth + ROOT_OFFSET);
  out.up = UP_BASE + UP_GAIN * intensity * crest * surge(s, side, time);
  out.out = (OUT_BASE + OUT_GAIN * intensity) * (1 - 0.4 * s) * (0.6 + 0.4 * crest);
  out.aft = speed * (0.35 + 0.3 * s);
  out.flight = (out.up + Math.sqrt(out.up * out.up + 2 * GRAVITY * (out.y + 0.25))) / GRAVITY;
  return out;
}

/** Hull-local position and velocity at (s, tau) on the sheet. */
export function sheetPoint(
  station: SheetStation,
  side: number,
  tau: number,
  position: THREE.Vector3,
  velocity: THREE.Vector3,
): void {
  const t = tau * station.flight;
  const roll = 1 + 0.25 * tau;
  position.set(
    station.x - station.aft * t,
    station.y + station.up * t - 0.5 * GRAVITY * t * t,
    station.z + side * station.out * t * roll,
  );
  velocity.set(-station.aft, station.up - GRAVITY * t, side * station.out * (1 + 0.5 * tau));
}

const _forward = new THREE.Vector3();
const _levelBasis = new THREE.Matrix4();
const _up = new THREE.Vector3(0, 1, 0);
const _side = new THREE.Vector3();
const _hullQuat = new THREE.Quaternion();
const _sideWaterline = new THREE.Vector3();
const _centreWaterline = new THREE.Vector3();

/**
 * The hull's heading with its heel and pitch taken out: hull-local +x (the bow)
 * laid flat, world up, and +z (starboard) completing the frame. The frame the
 * sheet's trajectories fly in; see "Rooted on the hull, flown level".
 */
export function levelQuaternion(hull: THREE.Matrix4, out: THREE.Quaternion): THREE.Quaternion {
  hull.decompose(_centreWaterline, _hullQuat, _sideWaterline);
  _forward.set(1, 0, 0).applyQuaternion(_hullQuat);
  _forward.y = 0;
  if (_forward.lengthSq() < 1e-8) _forward.set(1, 0, 0);
  _forward.normalize();
  _side.crossVectors(_forward, _up);
  _levelBasis.makeBasis(_forward, _up, _side);
  return out.setFromRotationMatrix(_levelBasis);
}

/**
 * World position and velocity at (s, tau) on the sheet: `sheetPoint`'s flight,
 * flown level from the root on the hull. Mirrors the vertex stage. Returns the
 * height of the water the sheet lands on there (the centreline waterline under
 * the station), for the drops it sheds.
 */
export function sheetPointWorld(
  station: SheetStation,
  side: number,
  tau: number,
  hull: THREE.Matrix4,
  level: THREE.Quaternion,
  position: THREE.Vector3,
  velocity: THREE.Vector3,
): number {
  sheetPoint(station, side, tau, position, velocity);
  position.x -= station.x;
  position.y -= station.y;
  position.z -= station.z;
  position.applyQuaternion(level);
  velocity.applyQuaternion(level);
  _sideWaterline.set(station.x, 0, station.z).applyMatrix4(hull);
  const water = _centreWaterline.set(station.x, 0, 0).applyMatrix4(hull).y;
  position.x += _sideWaterline.x;
  position.y += water + station.y;
  position.z += _sideWaterline.z;
  return water;
}

const vSheet = /*@__PURE__*/ varyingProperty('vec4', 'bowSheetUv');
const vSheetMisc = /*@__PURE__*/ varyingProperty('vec4', 'bowSheetMisc');

export class BowSheet {
  readonly mesh: THREE.Mesh;

  readonly uIntensity = uniform(0);
  private readonly uSpeed = uniform(0);
  private readonly uHalfLength = uniform(10);
  private readonly uBeam = uniform(4);
  private readonly uClock = uniform(0);
  private readonly uHull = uniform(new THREE.Matrix4());
  /** The hull's heading alone; see `levelQuaternion`. */
  private readonly uLevel = uniform(new THREE.Matrix4());
  private readonly level = new THREE.Quaternion();

  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.MeshBasicNodeMaterial;

  constructor() {
    const positions: number[] = [];
    const index: number[] = [];
    for (let j = 0; j <= FLIGHT_SEGMENTS; j++) {
      for (let i = 0; i <= ROOT_SEGMENTS; i++) {
        positions.push(i / ROOT_SEGMENTS, j / FLIGHT_SEGMENTS, 0);
      }
    }
    const row = ROOT_SEGMENTS + 1;
    for (let j = 0; j < FLIGHT_SEGMENTS; j++) {
      for (let i = 0; i < ROOT_SEGMENTS; i++) {
        const a = j * row + i;
        index.push(a, a + 1, a + row, a + 1, a + row + 1, a + row);
      }
    }
    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.geometry.setIndex(index);
    this.geometry.setAttribute('sheetSide', new THREE.InstancedBufferAttribute(new Float32Array([1, -1]), 1));
    this.geometry.instanceCount = 2;
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Number.POSITIVE_INFINITY);

    const material = new THREE.MeshBasicNodeMaterial();
    material.side = THREE.DoubleSide;
    material.transparent = true;
    material.depthWrite = false;
    // Torn-away parts are discarded, not blended to nothing: alpha fades only
    // the colour target, and a zero-alpha fragment would still overwrite the
    // sea's velocity, normal and AO mask under the whole sheet (see
    // `sprayMaterial`).
    material.alphaTest = SPRAY_ALPHA_FLOOR;
    // No contact shadows or AO on the sheet; see `sceneMrt`'s surface flag.
    material.userData.airborneWater = true;
    // Aerial perspective comes from the volumetric pass, as for the spray.
    material.fog = false;
    material.positionNode = Fn(() => this.vertex())() as any;
    material.colorNode = Fn(() => this.fragment())() as any;
    this.material = material;

    this.mesh = new THREE.Mesh(this.geometry, material);
    this.mesh.name = 'gameplay-bow-sheet';
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    // Before the spray it sheds, so its drops draw over it.
    this.mesh.renderOrder = 5;
    this.mesh.matrixAutoUpdate = false;
  }

  update(time: number, hull: THREE.Object3D, intensity: number, speed: number, halfLength: number, beam: number): void {
    this.uClock.value = time;
    this.uIntensity.value = intensity;
    this.uSpeed.value = Math.max(0, speed);
    this.uHalfLength.value = halfLength;
    this.uBeam.value = beam;
    (this.uHull.value as THREE.Matrix4).copy(hull.matrixWorld);
    (this.uLevel.value as THREE.Matrix4).makeRotationFromQuaternion(levelQuaternion(hull.matrixWorld, this.level));
    this.mesh.visible = intensity > 0.01;
    this.mesh.userData.intensity = intensity;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }

  /** `sheetStation` and `sheetPoint` in TSL; returns the world position. */
  private vertex(): any {
    const s: any = (positionGeometry as any).x.toVar('bowSheetS');
    const tau: any = (positionGeometry as any).y.toVar('bowSheetTau');
    const side: any = attribute('sheetSide', 'float').toVar('bowSheetSide');
    const intensity: any = this.uIntensity;

    // The bow wave is highest just abaft the stem and dies away toward the
    // shoulder; the hull widens from the stem.
    const crest: any = s.div(0.12).min(1).sqrt().mul(s.oneMinus().clamp(0, 1).pow(1.5)).toVar('bowSheetCrest');
    const q: any = s.mul(ROOT_LENGTH / STEM);
    const halfWidth: any = this.uBeam.mul(0.5).mul(q.mul(2).sub(q.mul(q)).max(0).sqrt());
    const x0: any = this.uHalfLength.mul(s.mul(-ROOT_LENGTH).add(STEM));
    const y0: any = intensity.mul(crest).mul(CREST_HEIGHT).sub(0.1).toVar('bowSheetY0');
    const z0: any = side.mul(halfWidth.add(ROOT_OFFSET));
    const surgeNode: any = s
      .mul(29)
      .sub(this.uClock.mul(5.3))
      .add(side.mul(1.3))
      .sin()
      .mul(s.mul(17).sub(this.uClock.mul(3.1)).add(side.mul(0.7)).sin())
      .mul(0.4)
      .add(s.mul(53).sub(this.uClock.mul(8.9)).add(side.mul(2.1)).sin().mul(0.2))
      .add(1);
    const up: any = intensity.mul(crest).mul(surgeNode).mul(UP_GAIN).add(UP_BASE).toVar('bowSheetUp');
    const outward: any = intensity
      .mul(OUT_GAIN)
      .add(OUT_BASE)
      .mul(s.mul(-0.4).add(1))
      .mul(crest.mul(0.4).add(0.6));
    const aft: any = this.uSpeed.mul(s.mul(0.3).add(0.35));
    const flight: any = up
      .add(up.mul(up).add(y0.add(0.25).mul(2 * GRAVITY)).max(0).sqrt())
      .div(GRAVITY)
      .toVar('bowSheetFlight');
    const t: any = tau.mul(flight);
    const roll: any = tau.mul(0.25).add(1);
    const flown: any = vec3(
      aft.mul(t).negate(),
      up.mul(t).sub(t.mul(t).mul(0.5 * GRAVITY)),
      side.mul(outward).mul(t).mul(roll),
    );
    // Rooted on the hull side at the true waterline plus the crest, flown level:
    // see "Rooted on the hull, flown level" in the header.
    const sideWaterline: any = this.uHull.mul(vec4(x0, 0, z0, 1)).xyz;
    const centreWaterline: any = this.uHull.mul(vec4(x0, 0, 0, 1)).y;
    const root: any = vec3(sideWaterline.x, centreWaterline.add(y0), sideWaterline.z);
    const world: any = root.add(this.uLevel.mul(vec4(flown, 0)).xyz).toVar('bowSheetWorld');
    const viewZ: any = cameraViewMatrix.mul(vec4(world, 1)).z;
    // Root arc length in metres, for the breakup's scale across the flow.
    const rootMetres: any = s.mul(this.uHalfLength).mul(ROOT_LENGTH);
    vSheet.assign(vec4(rootMetres, t, tau, flight));
    vSheetMisc.assign(vec4(viewZ, crest, side, 0));
    // The mesh is drawn with an identity model matrix: the world position is
    // the hull's transform applied here, so undo the object's own.
    return modelWorldMatrixInverse.mul(vec4(world, 1)).xyz;
  }

  private fragment(): any {
    const tex = sprayAtlas();
    const rootMetres: any = vSheet.x;
    const age: any = vSheet.y;
    const tau: any = vSheet.z.clamp(0, 1).toVar('bowSheetFTau');
    const crest: any = vSheetMisc.y;
    const intensity: any = this.uIntensity;

    // Breakup fields advected with the water: across the flow by the root's arc
    // length, along it by time of flight, so the pattern moves out along the
    // sheet at the water's own speed.
    // A third of a metre of root and a third of a second of flight per tile of
    // the atlas: tears a decimetre or two across and drawn out along the flow
    // into fingers, as in footage of a bluff bow.
    const flow: any = vec2(rootMetres.mul(3), age.sub(this.uClock).mul(3));
    const t0: any = texture(tex, flow.mul(vec2(0.13, 0.45)).add(vec2(0.61, 0.13))).x;
    const t1: any = texture(tex, flow.mul(vec2(1, 0.5))).toVar('bowSheetT1');
    const t2: any = texture(tex, flow.mul(vec2(2.7, 1.4)).add(vec2(0.37, 0.71))).toVar('bowSheetT2');
    // Intact sheet near the root, strands past mid-flight, drops at the edge;
    // a finer octave on top so no edge is a smooth curve.
    const toStrands: any = smoothstep(0.3, 0.75, tau);
    const coarse: any = mix(t1.x, t1.z, toStrands);
    const fine: any = mix(t2.x, t2.y, smoothstep(0.6, 1, tau));
    // Summing two octaves narrows the spread; stretch it back so a threshold
    // still cuts holes rather than shaving the extremes.
    const field: any = coarse.mul(0.62).add(fine.mul(0.38)).sub(0.5).mul(1.7).add(0.5).toVar('bowSheetField');
    // Torn everywhere a little — the sheet leaves the stem already broken by
    // the turbulence of the bow wave's crest — and to strands past mid-flight.
    // The rising half is the intact curtain; the falling half, outside it and
    // in front of it from most views, is mostly strands so it never closes the
    // holes behind it.
    // A coarse tap moves the breakup front itself by a good part of the flight,
    // so the intact sheet ends in tongues and bays rather than along one line.
    const cut: any = mix(float(0.2), float(0.85), smoothstep(0.05, 0.9, tau))
      .add(t0.sub(0.5).mul(0.9))
      .toVar('bowSheetCut');
    const torn: any = smoothstep(cut.sub(0.06), cut.add(0.04), field);
    // Thin and translucent where it leaves the hull, aerated and white as it
    // thins and tears; the hollows between strands keep a faint veil.
    const membrane: any = mix(float(0.8), float(1), smoothstep(0.1, 0.5, tau));
    const veil: any = smoothstep(0.85, 0.3, tau).mul(0.18);
    // The free edge of each station and the two ends of the root fade out.
    // The stem end over the first sixth: past the stem the far side's sheet is
    // seen edge-on to its first trajectory, and a short fade drew it as a line.
    const ends: any = smoothstep(0, 0.16, vSheet.x.div(this.uHalfLength.mul(ROOT_LENGTH)))
      // And the shoulder end over its last two thirds, which the far side's
      // sheet shows edge-on past the stem.
      .mul(smoothstepDown(vSheet.x.div(this.uHalfLength.mul(ROOT_LENGTH)), 0.3, 0.92));
    const shape: any = torn.mul(membrane).max(veil).mul(ends).mul(crest.mul(0.6).add(0.4));

    // Soft against the hull and the sea behind it.
    const sceneZ: any = perspectiveDepthToViewZ(sharedSceneDepth().x, cameraNear, cameraFar);
    const soft: any = vSheetMisc.x.sub(sceneZ).div(SOFT_DEPTH).clamp(0, 1);
    const nearFade: any = smoothstep(0.4, 2.0, vSheetMisc.x.negate());

    // --- light ---------------------------------------------------------------
    const toEye: any = cameraPosition.sub(positionWorld).normalize();
    // Thin water transmits forward; aerated water scatters many times.
    // The root is the bow wave's own breaking crest, white with entrained air;
    // the sheet above it is clearer water for a moment (never glass: it is
    // full of bubbles from the crest), then white again as it thins and tears.
    const aerated: any = smoothstep(0.1, 0.5, tau).max(smoothstepDown(tau, 0.03, 0.14).mul(0.85)).max(0.35);
    // Mottled by its own breakup field so it reads as tumbling water rather
    // than a smooth shell. The lighting is shared with the splashes.
    const radiance: any = sheetRadiance(aerated, field.mul(0.4).add(0.75));

    // Seen edge-on, a torn sheet is its strands lined up along the eye ray, and
    // drawn as a surface it would be a hard, solid outline: the "glass shell".
    // So it thins out toward grazing and the outline is left to its strands,
    // the drops and the mist.
    const facetNormal: any = dFdx(positionWorld).cross(dFdy(positionWorld)).normalize();
    const facing: any = smoothstep(0.06, 0.45, facetNormal.dot(toEye).abs());
    const alpha: any = shape
      .mul(soft)
      .mul(nearFade)
      .mul(facing)
      .mul(intensity.mul(2).clamp(0, 1))
      .clamp(0, 1);
    return vec4(radiance, alpha);
  }
}
