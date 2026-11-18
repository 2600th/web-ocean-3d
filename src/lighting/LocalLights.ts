import * as THREE from 'three/webgpu';
import { color, materialEmissive, normalView, vec3 } from 'three/tsl';
import { uniform } from '../core/uniform';
import { SHIP_DECK_LANTERN_PEAK, SHIP_DECK_LANTERN_RADIUS } from '../scene/ship/ShipLights';

/**
 * Fire, torch and lantern light after dark.
 *
 * The camp fire, the jetty torches and the lanterns had emissive meshes and
 * nothing else: at night a bright orange blob sat on a beach it did not light,
 * which reads as a sticker rather than a fire. These are the lights.
 *
 * ### The rules a scene like this lives by
 *
 * - **Lights cost what they light, not what exists.** The scene's lights are
 *   batched (`BatchedLighting`): every point light is one iteration of a loop
 *   over uniform arrays, and joining or leaving the scene is a uniform write,
 *   not a recompile. So all of them are created at boot on every tier, and a
 *   light at zero is hidden, which takes it out of the loop: by day the loop
 *   runs no iterations. That includes the ship's lantern, which follows the hull
 *   by copying the lantern's world position each frame rather than being
 *   parented to a ship that arrives seconds after boot.
 * - **Culled by distance.** Each light fades out over the last 30% of a reach
 *   from the eye. Past that it is zero, and a light whose lit area is under a
 *   pixel from where the viewer is gives nothing for its cost.
 * - **Tier-gated.** Low lights none (`QualitySettings.localLights`); a tier
 *   change only moves the count, and the count goes to the lights nearest the
 *   eye relative to their reach.
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
  /** Where it is when something else carries it (the deck lanterns); copied each update. */
  follow: THREE.Vector3 | null;
  /** Distance over reach this frame; Infinity when out of reach or unplaced. */
  score: number;
  /**
   * Metres the light wanders about `anchor` with its flicker, so the light on
   * the sand and the logs shifts as a flame's does. 0 for glass-cased lights.
   */
  sway: number;
  /** Where a swaying light is centred. */
  anchor: THREE.Vector3;
}

const FIRE_COLOR = new THREE.Color(1.0, 0.5, 0.2);
const TORCH_COLOR = new THREE.Color(1.0, 0.55, 0.24);
const LANTERN_COLOR = new THREE.Color(1.0, 0.64, 0.32);
/** Tallow candles behind crown glass: a little warmer than the jetty's lantern. */
const BEACON_COLOR = new THREE.Color(1.0, 0.6, 0.28);

/** Where the ship's stern lantern hangs, in the ship's own frame (metres). Measured off the hull mesh. */
export const SHIP_LANTERN_LOCAL = new THREE.Vector3(-13.2, 6.38, 0);
/** How far in front of the lantern's centre, toward the eye, its flame is drawn. */
const GLOW_STANDOFF = 0.3;

export class LocalLights {
  readonly group = new THREE.Group();
  private readonly fire: Source;
  /** One light per jetty torch. */
  private readonly torches: Source[];
  private readonly lantern: Source;
  private readonly shipLantern: Source;
  private readonly beacon: Source;
  /** The ship's swinging deck lanterns (`scene/ship/ShipLights`). */
  private readonly deckLanterns: Source[];
  /** Every light; the tier lights the `count` nearest, relative to their reach. */
  private readonly sources: Source[];
  private count = 0;
  /** Under reduced motion the flames' lights hold still. */
  private reducedMotion = false;
  /** Brightness of the ship lantern's glow (and the jetty lantern's glass), 0..1. */
  private readonly uGlow = uniform(0);
  private readonly glow: THREE.Mesh;
  private ship: THREE.Object3D | null = null;
  private readonly scratch = new THREE.Vector3();
  private readonly glowOffset = new THREE.Vector3();
  /** 0..1 of the fire's own strength after the night ramp and flicker, for the smoke. */
  fireLevel = 0;
  /**
   * 0..1, the lighthouse lamp after the night ramp and its candle flutter: the
   * level of its lit glass, flames and far halo. Not tier-gated, and not culled
   * by distance — the halo is the part of it that is seen from the play area.
   */
  beaconLevel = 0;

  constructor(count: number) {
    this.group.name = 'local-lights';
    const make = (name: string, c: THREE.Color, peak: number, radius: number, reach: number, flicker: number, seed: number): Source => {
      const light = new THREE.PointLight(c, 0, radius, 2);
      light.name = name;
      light.castShadow = false;
      // Parked far below the world until an anchor arrives; zero intensity anyway.
      light.position.set(0, -10000, 0);
      light.visible = false;
      this.group.add(light);
      return { light, peak, reach, flicker, seed, placed: false, follow: null, score: Infinity, sway: 0, anchor: new THREE.Vector3() };
    };
    // In order of how much they matter: the fire lights the camp, the ship's
    // lantern is on the hero, and the jetty's pair are dressing.
    this.fire = make('camp-fire-light', FIRE_COLOR, 18, 26, 240, 1, 0.37);
    this.shipLantern = make('ship-lantern-light', LANTERN_COLOR, 3, 16, 280, 0.25, 0.71);
    // A light each: the pair are six metres apart, and one light between them
    // lit the jetty from a point where there is no flame. Each carries a little
    // over half the old shared light, so the pair together are a touch brighter
    // where their pools overlap and the same at each torch's foot.
    this.torches = [0, 1].map((i) => make(`jetty-torch-light-${i}`, TORCH_COLOR, 3.6, 13, 170, 0.8, 0.13 + i * 0.29));
    this.lantern = make('jetty-lantern-light', LANTERN_COLOR, 2.5, 10, 150, 0.2, 0.53);
    // The lighthouse lamp: a hoop of candles in a glazed lantern 20 m up. Lights
    // its own gallery and lantern frame and the tower head; 30 m of radius
    // reaches the gallery deck and the cottage roof. Its long reach ranks it
    // behind anything nearer (see `setCount`).
    this.beacon = make('lighthouse-light', BEACON_COLOR, 14, 30, 420, 0.15, 0.91);
    // The ship's deck lanterns, reaching the deck, the rail and the foot of the
    // canvas above. At 5 and 12 m the planks two metres below barely took the
    // light and the courses not at all; a horn lantern is dim, but the eye at
    // night is adapted to the moon, and it is the warm pool on the deck that
    // says "lantern". Eight point lights in all, which is `BatchedLighting`'s
    // array size; at most `count` are ever lit. Zero by day, so day is unchanged.
    this.deckLanterns = [0, 1, 2].map((i) =>
      make(`ship-deck-lantern-light-${i}`, LANTERN_COLOR, SHIP_DECK_LANTERN_PEAK, SHIP_DECK_LANTERN_RADIUS, 200, 0.25, 0.19 + i * 0.23),
    );
    this.sources = [this.fire, this.shipLantern, ...this.torches, this.lantern, this.beacon, ...this.deckLanterns];
    // Open flames wander; lanterns behind glass do not.
    this.fire.sway = 0.14;
    for (const t of this.torches) t.sway = 0.05;
    try {
      const query = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)');
      this.reducedMotion = query?.matches ?? false;
      query?.addEventListener?.('change', (e) => (this.reducedMotion = e.matches));
    } catch {
      this.reducedMotion = false;
    }
    this.setCount(count);

    // The ship lantern's flame. Additive, so at zero it adds nothing — the hull
    // texture's lantern is opaque and a dark sphere inside it would show by day.
    const glowMaterial = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    glowMaterial.name = 'ship-lantern-glow';
    // Additive: fog would turn its unlit rim into a pale fog-coloured ball.
    glowMaterial.fog = false;
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
  setIsland(
    anchors: { fire: THREE.Vector3; torches: THREE.Vector3[]; lanterns: THREE.Vector3[]; beacon?: THREE.Vector3 | null },
    props?: THREE.Object3D,
  ): void {
    if (anchors.beacon) {
      this.beacon.light.position.copy(anchors.beacon);
      this.beacon.placed = true;
    }
    this.fire.light.position.copy(anchors.fire);
    this.fire.anchor.copy(anchors.fire);
    this.fire.placed = true;
    this.torches.forEach((s, i) => {
      const at = anchors.torches[i];
      if (!at) return;
      s.light.position.copy(at);
      s.anchor.copy(at);
      s.placed = true;
    });
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
   * The ship's deck lanterns, as world positions `ShipLights` moves in place.
   * Up to three; the lights read them on every update.
   */
  setShipLanterns(positions: readonly THREE.Vector3[]): void {
    this.deckLanterns.forEach((s, i) => {
      s.follow = positions[i] ?? null;
      s.placed = s.follow !== null;
    });
  }

  /**
   * How many of the lights the tier pays for. Only the count moves: the lights
   * beyond it go dark on the next update and stay in the group, so a tier change
   * compiles nothing (see `BatchedLighting`).
   *
   * The budget goes to the lights nearest the eye relative to their reach, not
   * to a fixed list: at sea the island's fire and torches are out of reach and
   * the ship's lanterns get the budget; at the cove the camp gets it back.
   */
  setCount(count: number): void {
    this.count = Math.max(0, Math.min(this.sources.length, Math.round(count)));
    if (this.count === 0) for (const s of this.sources) this.setLevel(s, 0);
  }

  /** A light at zero is hidden, which drops it from the batched loop. */
  private setLevel(s: Source, intensity: number): void {
    s.light.intensity = intensity;
    s.light.visible = intensity > 0;
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
    {
      // Candles gutter rather than flicker: a slow, shallow wander.
      const t = elapsed + this.beacon.seed * 100;
      const flutter = 1 + 0.06 * Math.sin(t * 2.3) + 0.04 * Math.sin(t * 5.7 + 1.1) + 0.03 * Math.sin(t * 13.1 + 0.3);
      this.beaconLevel = Math.max(0, Math.min(1, darkness)) * flutter;
    }
    for (const s of this.sources) {
      if (s.follow) s.light.position.copy(s.follow);
      const d = s.light.position.distanceTo(eye);
      s.score = s.placed && night > 0 && d < s.reach ? d / s.reach : Infinity;
    }
    for (let i = 0; i < this.sources.length; i++) {
      const s = this.sources[i];
      // Rank among the candidates, ties to list order: deterministic, no sort.
      let rank = 0;
      for (let j = 0; j < this.sources.length; j++) {
        const o = this.sources[j];
        if (o.score < s.score || (o.score === s.score && j < i)) rank++;
      }
      if (s.score === Infinity || rank >= this.count) {
        this.setLevel(s, 0);
        continue;
      }
      const d = s.light.position.distanceTo(eye);
      const cull = 1 - THREE.MathUtils.smoothstep(d, s.reach * 0.7, s.reach);
      const t = elapsed + s.seed * 100;
      // Three incommensurate sines: a fire's flicker has no period a viewer can
      // pick out, and a lantern's is the same shape at a fraction of the depth.
      const wobble = 0.13 * Math.sin(t * 11.3) + 0.08 * Math.sin(t * 23.9 + 1.7) + 0.06 * Math.sin(t * 5.3 + 0.4);
      const flicker = 1 + wobble * s.flicker;
      this.setLevel(s, s.peak * night * cull * flicker);
      if (s.sway > 0) {
        // The flame's centre of light moves with its brightness: up as it
        // flares, and side to side on the slower tongues. Same clock, so a
        // capture lands on the same pose.
        const k = this.reducedMotion ? 0 : s.sway;
        s.light.position.set(
          s.anchor.x + k * (0.7 * Math.sin(t * 7.1 + 0.3) + 0.3 * Math.sin(t * 17.3)),
          s.anchor.y + k * 0.6 * wobble * 4,
          s.anchor.z + k * (0.7 * Math.sin(t * 6.3 + 2.1) + 0.3 * Math.sin(t * 15.7 + 1.3)),
        );
      }
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
