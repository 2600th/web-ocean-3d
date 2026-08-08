import * as THREE from 'three/webgpu';
import {
  Fn,
  cameraPosition,
  float,
  materialColor,
  mix,
  positionGeometry,
  positionLocal,
  positionWorld,
  uniform,
  vec2,
  vec3,
} from 'three/tsl';
import { occludeLight } from '../core/lightOcclusion';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

/**
 * The three things every object standing on the seabed or the island was
 * missing.
 *
 * They are grouped because they are one omission with three faces: nothing
 * placed in this world was told anything about the ground it was placed on.
 *
 * 1. **Contact darkening.** Corals sat on sand with no shadow at their base and
 *    read as pasted on; the reef scatter and several island scatters clear
 *    `castShadow` for cost, which is defensible — a coral head is a few hundred
 *    triangles and there are hundreds of them — but the *replacement* was never
 *    added. iq's fish in `ldj3Dm` casts a soft shadow on the seabed and it is a
 *    large part of why it sits in the scene rather than over it. This is the
 *    cheap version of the same cue: darken by proximity to the heightfield,
 *    which is where an object's own occlusion of the sky is concentrated.
 *
 * 2. **Caustics.** `Caustics.intensityNode` was written to be shared — its own
 *    header says "a seafloor material, a rock material and a hull material" —
 *    and was wired to the seafloor alone. So the reef's corals and rocks stood
 *    in a caustic pattern that stopped at their feet. Every underwater reference
 *    in the set puts caustics on *every* lit surface.
 *
 * 3. **Key occlusion.** The island's own shadow and the cloud deck's, the same
 *    terms the terrain now receives. Without it a palm on a shaded hillside is
 *    lit as if it were on the sunlit one.
 *
 * All three are analytic functions of world position, so one treatment covers
 * land and seabed and the elevation ramps decide which parts apply where.
 */

export interface GroundShadingInputs {
  /** The key light, for the direct-occlusion hook. */
  light: THREE.Light;
  /** `(worldPosition) => 0..1` — terrain shadow times cloud shade. */
  keyShadow: (worldPosition: Node) => Node;
  /** `(worldPosition) => ~1` — the caustic pattern, centred on 1. */
  caustics: (worldPosition: Node) => Node;
  /** `(worldPosition) => metres` — seafloor elevation, for the contact term. */
  groundHeight: (worldPosition: Node) => Node;
  /**
   * Wind for the planting, and the test that decides what counts as planting.
   *
   * Omitted leaves every mesh static, which is what the scene did before.
   */
  foliage?: {
    wind: FoliageWind;
    /** Given a mesh name, is this foliage? */
    test: (meshName: string) => boolean;
    /** Direction toward the key light, world space, as a node. */
    sunDirection: Node;
    /** Its colour and strength, as a node. */
    sunColor: Node;
  };
}

/**
 * Exponent and strength of the leaf back-scatter.
 *
 * A leaf is a thin translucent sheet, so a good deal of the light that reaches
 * its back comes out of its front, and a canopy with the sun behind it glows.
 * `Props` records that the old procedural palm faked this through `emissiveNode`
 * and that it was removed for being "neither shadowed nor tone-mapped" — the
 * right call, and the wrong conclusion, because both objections are about *how*
 * it was wired rather than about the term.
 *
 * This one is shadowed by whatever occlusion its caller supplies, which for the
 * island's planting is the **cloud deck** — the props deliberately take the
 * cloud shade alone rather than the heightfield march, for the compile-time
 * reason recorded at that call site. So a frond under a cloud stops glowing and
 * a frond behind the hill does not; the emissive path also bypasses the ordinary
 * shadow map, which would otherwise have caught it. That is a known limit of
 * this term and not a claim it makes.
 *
 * It is tone-mapped for free, because emissive is added to the outgoing radiance
 * before the output transform rather than after it. And it is weighted by the
 * material's own albedo, so it is the leaf's colour coming through rather than a
 * wash.
 *
 * The lobe is the standard forward-scatter form — `dot(V, -L)` raised to a
 * power, which peaks when the viewer is looking straight down the sun's own
 * direction through the leaf. Deliberately narrow and weak: this is the last
 * fraction of a stop on a canopy, and a translucency term that reads at any
 * angle is not translucency, it is emission.
 */
export const TRANSLUCENCY_POWER = 4;
export const TRANSLUCENCY_GAIN = 0.55;

/**
 * Sway shared by the near trees and the distant canopy cards.
 *
 * Only the billboards moved. `Canopy`'s model is the right one — shear by the
 * square of the height so the trunk stays planted and the crown carries the
 * travel, and *one travelling gust* across the island rather than a per-instance
 * random phase, because a canopy moves in gusts and neighbours that move
 * independently read as static noise. It just was not applied to the meshes, so
 * a viewer walking up the beach watched the far forest breathe and the palms in
 * front of them stand perfectly still.
 *
 * The wavelength, speed and amplitude below are `Canopy`'s, deliberately, and
 * `Imposters` uses them too. There are real handovers now — a plant becomes a
 * card at 60 m if it is understorey and 120 m if it is a tree, and the canopy
 * field fades in over 90-180 m behind both — so the shared constants have
 * stopped being a hedge against a future fade and started doing their job.
 *
 * They do **not** share a phase, and no comment elsewhere should claim they do.
 * `Canopy` adds a per-instance offset from its card seed and `Imposters` adds
 * one from its yaw; a mesh has no equivalent constant available in the vertex
 * stage without new plumbing — `positionGeometry` varies across the tree and
 * `positionLocal` is post-instancing. What survives a handover is an amplitude
 * and frequency match, not a continuous phase: a plant swapping to a card keeps
 * the same sway *character* and may jump within its cycle. That is the honest
 * description, and it is why the swap is placed where the plant is small on
 * screen.
 */
const SWAY_WAVELENGTH = 90;
const SWAY_K = (Math.PI * 2) / SWAY_WAVELENGTH;
const SWAY_SPEED = 7;
const SWAY_OMEGA = SWAY_K * SWAY_SPEED;
/** Metres of crown travel at full wind. `Canopy`'s number. */
const SWAY_MAX = 0.9;
/**
 * Model height over which the shear reaches full strength, in model units.
 *
 * Model units rather than metres, because the value available in the vertex
 * stage before instancing is the raw attribute — and that is the right choice
 * anyway: it makes the lean a fraction of the *tree's own* height, so a scaled
 * instance bends over its own length rather than over an absolute distance.
 */
const SWAY_HEIGHT = 9;

const CLOCK_WRAP = 3600;

export class FoliageWind {
  private readonly uWind: any = uniform(new THREE.Vector2(1, 0));
  private readonly uStrength = uniform(0.5);
  private readonly uPhase = uniform(0);
  private phase = 0;

  /** `direction` is a unit vector in world xz; `strength` is 0..1. */
  setWind(direction: THREE.Vector2, strength: number): void {
    const wind = this.uWind.value as THREE.Vector2;
    wind.copy(direction);
    if (wind.lengthSq() < 1e-6) wind.set(1, 0);
    wind.normalize();
    this.uStrength.value = Math.max(0, Math.min(1, strength));
  }

  update(dt: number): void {
    this.phase = (this.phase + dt) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  /** Rewinds the gust, for reproducible captures. */
  resetClock(time = 0): void {
    this.phase = ((time % CLOCK_WRAP) + CLOCK_WRAP) % CLOCK_WRAP;
    this.uPhase.value = this.phase;
  }

  /**
   * Installs the shear on one material's vertex stage.
   *
   * `positionLocal` inside `positionNode` has **already been through the
   * instance matrix** — `NodeMaterial.setupPosition` applies instancing before
   * it reads `positionNode` — so it is world space for anything parented at the
   * scene root, which is what the gust's travelling term needs. The raw
   * `positionGeometry` is still the un-instanced attribute, which is what the
   * lean's height fraction needs. Both are available and they are different
   * things; using one where the other belongs gives either a forest that all
   * leans in lockstep or a forest that shears about the world origin.
   */
  applyTo(material: THREE.NodeMaterial): void {
    const existing = material.positionNode as Node;
    material.positionNode = Fn(() => {
      const p = (existing === null ? positionLocal : vec3(existing)).toVar('fwP');
      const lean = positionGeometry.y
        .max(0)
        .div(SWAY_HEIGHT)
        .clamp(0, 1)
        .toVar('fwLean');
      const along = vec2(p.x, p.z).dot(this.uWind).toVar('fwAlong');
      const gust = along.mul(SWAY_K).sub(this.uPhase.mul(SWAY_OMEGA)).sin().toVar('fwGust');
      // Squared, so the trunk is planted and the travel is all in the crown.
      const shear = gust
        .mul(lean)
        .mul(lean)
        .mul(SWAY_MAX)
        .mul(this.uStrength)
        .toVar('fwShear');
      return vec3(
        p.x.add(this.uWind.x.mul(shear)),
        p.y,
        p.z.add(this.uWind.y.mul(shear)),
      );
    })();
    material.needsUpdate = true;
  }
}

/**
 * Height above the ground over which contact darkening fades out, metres.
 *
 * Small on purpose. This is not ambient occlusion of the object by itself — it
 * is the wedge of sky a surface loses to the ground it is sitting on, and that
 * closes within about a metre for anything reef-sized. Wider and every coral
 * head goes uniformly dark, which trades one wrong read for another.
 */
const CONTACT_HEIGHT = 1.4;
/** Ambient reaching a surface in contact with the ground. */
const CONTACT_FLOOR = 0.34;

/** Where caustics stop, in metres of depth, and where they are strongest. */
const CAUSTIC_SHALLOW = 2;
const CAUSTIC_DEEP = 48;
/** Half-width of the band across the waterline where caustics fade in. */
const WATERLINE_BAND = 1.2;

/**
 * Applies the treatment to one material.
 *
 * **Nothing here touches `colorNode`**, and that is load-bearing rather than
 * tidy. `NodeMaterial.setupDiffuseColor` reads `this.colorNode ?? materialColor`,
 * and `materialColor` is where the diffuse *map* is folded in — so a wrapper that
 * set `colorNode` would silently untexture every prop in the scene. It would have
 * looked like it worked, too, because the models are close enough to their
 * average colour at a distance.
 *
 * So the two terms go where they belong instead:
 *
 * - The contact shadow is `aoNode`, which three applies to indirect light only.
 *   That is exactly right: a coral in contact with the sand has lost sky, not
 *   sun.
 * - The caustics ride the *key light* through the same hook the shadows use.
 *   Also exactly right, and arguably more so than multiplying albedo would have
 *   been: a caustic is not a property of the surface, it is the sun arriving
 *   focused by the water above it. Folding it into the light means it scales
 *   with the sun, disappears at night, and cannot brighten a surface the sun
 *   never reached.
 */
export function applyGroundShading(
  material: THREE.NodeMaterial,
  inputs: GroundShadingInputs,
): void {
  const contact = Fn(() => {
    const wp = positionWorld.toVar('gsWorld');
    const above = wp.y.sub(inputs.groundHeight(wp)).max(0).toVar('gsAbove');
    return mix(float(CONTACT_FLOOR), float(1), above.smoothstep(0, CONTACT_HEIGHT));
  })();

  material.aoNode =
    material.aoNode === null ? contact : (material.aoNode as Node).mul(contact);

  const keyFactor = Fn(() => {
    const wp = positionWorld.toVar('gsWorldK');

    // Underwater only, and fading with depth for the same reason the seafloor's
    // does: past a few tens of metres the surface pattern has diverged into
    // ambient light and there is no caustic left to project.
    const depth = wp.y.negate().toVar('gsDepth');
    const submerged = float(1).sub(wp.y.smoothstep(-WATERLINE_BAND, WATERLINE_BAND));
    const reach = float(1)
      .sub(depth.smoothstep(CAUSTIC_SHALLOW, CAUSTIC_DEEP))
      .mul(submerged)
      .toVar('gsReach');

    const caustic = mix(float(1), inputs.caustics(wp), reach).toVar('gsCaustic');
    return inputs.keyShadow(wp).mul(caustic);
  })();

  occludeLight(material, inputs.light, keyFactor);
  material.needsUpdate = true;
}

/**
 * Walks an object tree and treats every distinct material once.
 *
 * Once, and that matters: a kind is baked to one geometry per material and
 * placed as an `InstancedMesh` per part, so the same material object is reached
 * through many meshes. Wrapping `colorNode` twice would square the caustics and
 * wrapping `setupLighting` twice would square the shadow.
 */
export function applyGroundShadingTo(
  root: THREE.Object3D,
  inputs: GroundShadingInputs,
): void {
  /**
   * Gathered before anything is applied, because the wind is a property of a
   * *material* and foliage is a property of a *mesh*.
   *
   * A material reached through both a palm and a rock must not sway, so the
   * question is "is every mesh using this material foliage", which cannot be
   * answered until the whole tree has been walked. The loader keys its node
   * materials on the source glTF material, so in practice a shared one would
   * mean two kinds authored in the same file — but "in practice" is not a
   * guarantee, and a swaying rock is a memorable bug.
   */
  const users = new Map<THREE.Material, { all: boolean; any: boolean }>();

  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const foliage = inputs.foliage?.test(mesh.name) ?? false;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (!material) continue;
      const entry = users.get(material);
      if (entry === undefined) users.set(material, { all: foliage, any: foliage });
      else {
        entry.all = entry.all && foliage;
        entry.any = entry.any || foliage;
      }
    }
  });

  for (const [material, use] of users) {
    // Node materials only — the treatment is a node graph. Everything the asset
    // loader produces is one; anything else is left alone rather than silently
    // skipped in a way that looks like it worked.
    if ((material as THREE.NodeMaterial).isNodeMaterial !== true) continue;
    const node = material as THREE.NodeMaterial;
    applyGroundShading(node, inputs);
    if (use.all && inputs.foliage) {
      inputs.foliage.wind.applyTo(node);
      applyLeafTranslucency(node, inputs);
    }
  }
}

/**
 * Adds the back-scatter through a leaf. See `TRANSLUCENCY_POWER`.
 *
 * Multiplied by the same cloud shade the key light carries, which is the whole
 * difference between this and the term it replaces: an emissive that ignores
 * what is between the leaf and the sun goes on glowing under a cloud, at night,
 * and inside the hill's own shadow.
 */
function applyLeafTranslucency(
  material: THREE.NodeMaterial,
  inputs: GroundShadingInputs,
): void {
  const foliage = inputs.foliage;
  if (!foliage) return;

  // Loosely typed: `emissiveNode` is declared on the concrete standard/physical
  // node materials rather than on the `NodeMaterial` base, and this walks a tree
  // of whatever the asset loader produced.
  const target = material as unknown as { emissiveNode: Node };
  const existing = target.emissiveNode ?? null;
  const glow = Fn(() => {
    const wp = positionWorld.toVar('ltWorld');
    // Surface toward the eye.
    const toEye = cameraPosition.sub(wp).normalize().toVar('ltView');
    // Peaks when the eye is looking down the sun's own travel direction, which
    // is exactly when a leaf between the two is lit from behind.
    const back = toEye
      .dot(vec3(foliage.sunDirection).negate())
      .clamp(0, 1)
      .pow(TRANSLUCENCY_POWER)
      .toVar('ltBack');
    return vec3(materialColor)
      .mul(vec3(foliage.sunColor))
      .mul(back.mul(TRANSLUCENCY_GAIN))
      .mul(inputs.keyShadow(wp));
  })();

  target.emissiveNode = existing === null ? glow : vec3(existing).add(glow);
  material.needsUpdate = true;
}
