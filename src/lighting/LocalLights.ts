import * as THREE from 'three/webgpu';
import { color, materialEmissive, normalView, uniform, vec3 } from 'three/tsl';

/**
 * Fire, torch and lantern light after dark.
 *
 * The camp fire, the jetty torches and the lanterns had emissive meshes and
 * nothing else: at night a bright orange blob sat on a beach it did not light,
 * which reads as a sticker rather than a fire. These are the lights.
 *
 * ### The rules a scene like this lives by
 *
 * - **The set of lights never changes after boot.** A light's existence is part
 *   of every lit material's shader in three (it is in the lights node's cache
 *   key); adding one, removing one or flipping `visible` recompiles everything
 *   that is lit. So all of them are created up front, parented to a group that
 *   is always visible, and switched by *intensity*, which is a uniform. That
 *   includes the ship's lantern: the ship arrives seconds after boot and is
 *   hidden until compiled, and a light parented to it would join the list the
 *   moment it was revealed. It follows the hull by copying the lantern's world
 *   position each frame instead.
 * - **Culled by distance.** Each light fades out over the last 30% of a reach
 *   from the eye. Past that it is zero, and a light whose lit area is under a
 *   pixel from where the viewer is gives nothing for its cost.
 * - **Tier-gated.** Low creates none (`QualitySettings.localLights`); a later
 *   downgrade to Low zeroes them.
 * - **Only when it is dark enough.** Full strength from deep dusk on and under a
 *   storm deck, nothing at noon — a fire at noon lights nothing you can see.
 * - **No shadows.** A point light's shadow is six more passes of the scene, and
 *   at the radius these reach the occluders that matter are the fire's own logs.
 * - **Flicker is a function of the scene clock**, so a deterministic capture of
 *   the camp at night is the same frame every time.
 *
 * Intensities are in the units the sun uses (3.4 at noon): a fire puts about
 * twice the moon's light on the sand at three metres and matches it at ten.
 */

interface Source {
  light: THREE.PointLight;
  /** Candela-like peak, before the night ramp, the flicker and the cull. */
  peak: number;
  /** Distance from the eye at which this light is gone, metres. */
  reach: number;
  /** 0 steady (a lantern) .. 1 a fire. */
  flicker: number;
  seed: number;
  placed: boolean;
}

const FIRE_COLOR = new THREE.Color(1.0, 0.5, 0.2);
const TORCH_COLOR = new THREE.Color(1.0, 0.55, 0.24);
const LANTERN_COLOR = new THREE.Color(1.0, 0.64, 0.32);

/** Where the ship's stern lantern hangs, in the ship's own frame (metres). Measured off the hull mesh. */
export const SHIP_LANTERN_LOCAL = new THREE.Vector3(-13.2, 6.38, 0);
/** How far in front of the lantern's centre, toward the eye, its flame is drawn. */
const GLOW_STANDOFF = 0.3;

export class LocalLights {
  readonly group = new THREE.Group();
  private readonly fire: Source;
  private readonly torch: Source;
  private readonly lantern: Source;
  private readonly shipLantern: Source;
  /** In order of how much they matter; the tier lights the first `count`. */
  private readonly sources: Source[];
  private count = 0;
  /** Brightness of the ship lantern's glow (and the jetty lantern's glass), 0..1. */
  private readonly uGlow = uniform(0);
  private readonly glow: THREE.Mesh;
  private ship: THREE.Object3D | null = null;
  private readonly scratch = new THREE.Vector3();
  private readonly glowOffset = new THREE.Vector3();
  /** 0..1 of the fire's own strength after the night ramp and flicker, for the smoke. */
  fireLevel = 0;

  constructor(count: number) {
    this.group.name = 'local-lights';
    const make = (name: string, c: THREE.Color, peak: number, radius: number, reach: number, flicker: number, seed: number): Source => {
      const light = new THREE.PointLight(c, 0, radius, 2);
      light.name = name;
      light.castShadow = false;
      // Parked far below the world until an anchor arrives; zero intensity anyway.
      light.position.set(0, -10000, 0);
      return { light, peak, reach, flicker, seed, placed: false };
    };
    // In order of how much they matter: the fire lights the camp, the ship's
    // lantern is on the hero, and the jetty's pair are dressing.
    this.fire = make('camp-fire-light', FIRE_COLOR, 18, 26, 240, 1, 0.37);
    this.shipLantern = make('ship-lantern-light', LANTERN_COLOR, 3, 16, 280, 0.25, 0.71);
    this.torch = make('jetty-torch-light', TORCH_COLOR, 6, 16, 170, 0.8, 0.13);
    this.lantern = make('jetty-lantern-light', LANTERN_COLOR, 2.5, 10, 150, 0.2, 0.53);
    this.sources = [this.fire, this.shipLantern, this.torch, this.lantern];
    this.setCount(count);

    // The ship lantern's flame. Additive, so at zero it adds nothing — the hull
    // texture's lantern is opaque and a dark sphere inside it would show by day.
    const glowMaterial = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    glowMaterial.name = 'ship-lantern-glow';
    // HDR on purpose: past the bloom threshold at night, so the flame has a halo.
    // Soft-edged by the facing ratio, so the ball reads as a glow, not a bead.
    glowMaterial.colorNode = color(LANTERN_COLOR).mul(this.uGlow).mul(normalView.z.max(0).pow(2)).mul(7);
    this.glow = new THREE.Mesh(new THREE.SphereGeometry(0.17, 12, 8), glowMaterial);
    this.glow.name = 'ship-lantern-glow';
    this.glow.position.set(0, -10000, 0);
    this.glow.castShadow = false;
    this.glow.receiveShadow = false;
    this.group.add(this.glow);
  }

  /** Places the island's lights. `anchors` are world positions. */
  setIsland(anchors: { fire: THREE.Vector3; torches: THREE.Vector3[]; lanterns: THREE.Vector3[] }, props?: THREE.Object3D): void {
    this.fire.light.position.copy(anchors.fire);
    this.fire.placed = true;
    if (anchors.torches.length > 0) {
      // One light at the pair's midpoint stands in for both: they are six metres
      // apart and light the same stretch of jetty and sand.
      const mid = this.scratch.set(0, 0, 0);
      for (const t of anchors.torches) mid.add(t);
      this.torch.light.position.copy(mid.multiplyScalar(1 / anchors.torches.length));
      this.torch.placed = true;
    }
    if (anchors.lanterns.length > 0) {
      this.lantern.light.position.copy(anchors.lanterns[0]);
      this.lantern.placed = true;
    }
    // The jetty lantern's glass glows with its light. Set at load, before the
    // props are compiled, so this is not a gameplay recompile.
    props?.getObjectByName('cove-lantern')?.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        if (!/glass/i.test(m.name)) continue;
        const target = m as unknown as { emissiveNode: unknown; needsUpdate: boolean };
        target.emissiveNode = vec3((target.emissiveNode as never) ?? materialEmissive).add(
          color(LANTERN_COLOR).mul(this.uGlow).mul(2.2),
        );
        target.needsUpdate = true;
      }
    });
  }

  /** The ship whose stern lantern this follows. */
  setShip(ship: THREE.Object3D): void {
    this.ship = ship;
    this.shipLantern.placed = true;
  }

  /**
   * How many of the lights the tier pays for, most important first.
   *
   * Lights in the scene are part of every lit material's build, so a change
   * here recompiles them. It is only called at boot and from a tier change,
   * which recompiles everything with the loop paused anyway; anchors are kept,
   * so a session that boots on Low still gets its lights on an upgrade.
   */
  setCount(count: number): void {
    const n = Math.max(0, Math.min(this.sources.length, Math.round(count)));
    if (n === this.count && this.group.children.length > 0) return;
    this.count = n;
    this.sources.forEach((s, i) => {
      if (i < n) {
        if (s.light.parent !== this.group) this.group.add(s.light);
      } else {
        s.light.intensity = 0;
        s.light.removeFromParent();
      }
    });
  }

  /**
   * @param elapsed Scene clock, seconds — drives the flicker deterministically.
   * @param darkness 0 at day, 1 from deep dusk on (or under a storm deck).
   */
  update(elapsed: number, eye: THREE.Vector3, darkness: number): void {
    if (this.ship) {
      this.ship.updateWorldMatrix(true, false);
      const p = this.scratch.copy(SHIP_LANTERN_LOCAL).applyMatrix4(this.ship.matrixWorld);
      this.shipLantern.light.position.copy(p);
      // The model's lantern is opaque — its panes are painted on — so the flame
      // is drawn just in front of the face the viewer sees rather than inside it.
      const toEye = this.glowOffset.subVectors(eye, p);
      const d = toEye.length();
      this.glow.position.copy(p).addScaledVector(toEye, d > 1e-3 ? GLOW_STANDOFF / d : 0);
    }
    const night = Math.max(0, Math.min(1, darkness));
    // The glow meshes are emissive, not lights, so they are not tier-gated.
    this.uGlow.value = Math.max(0, Math.min(1, darkness)) * (this.ship ? 1 : 0);
    // The smoke's glow follows the fire whether or not this tier lights it.
    this.fireLevel = Math.max(0, Math.min(1, darkness));
    for (let i = 0; i < this.count; i++) {
      const s = this.sources[i];
      if (!s.placed || night <= 0) {
        s.light.intensity = 0;
        continue;
      }
      const d = s.light.position.distanceTo(eye);
      const cull = 1 - THREE.MathUtils.smoothstep(d, s.reach * 0.7, s.reach);
      const t = elapsed + s.seed * 100;
      // Three incommensurate sines: a fire's flicker has no period a viewer can
      // pick out, and a lantern's is the same shape at a fraction of the depth.
      const wobble = 0.13 * Math.sin(t * 11.3) + 0.08 * Math.sin(t * 23.9 + 1.7) + 0.06 * Math.sin(t * 5.3 + 0.4);
      const flicker = 1 + wobble * s.flicker;
      s.light.intensity = s.peak * night * cull * flicker;
      if (s === this.fire) this.fireLevel = night * flicker;
    }
  }

  dispose(): void {
    this.group.removeFromParent();
    this.glow.geometry.dispose();
    (this.glow.material as THREE.Material).dispose();
    for (const s of this.sources) s.light.dispose();
  }
}
