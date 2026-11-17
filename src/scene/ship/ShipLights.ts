import * as THREE from 'three/webgpu';
import { color, float, max, normalView, smoothstep, uv, vec2 } from 'three/tsl';
import { uniform } from '../../core/uniform';

/**
 * Light aboard after dark: three deck lanterns that swing on their hooks, and
 * the stern windows of the great cabin.
 *
 * The hero used to be a black cut-out at night with one lantern on the
 * taffrail. This file owns what a viewer *sees* of the extra light — the
 * lantern bodies, their glass and halo, the window panes — and where the
 * lanterns are; `lighting/LocalLights` owns the point lights themselves and
 * copies `lanternPositions` into three of its batched sources each frame, so
 * the light on the deck and the sails moves with the swinging glass.
 *
 * ### Costs
 *
 * Five small materials compiled with the ship (lantern iron, glass, halo,
 * windows), none of them shadow casters, and nothing that changes with the
 * tier: the lights go through the same batched point loop the island's do, so a
 * tier change moves a count and compiles nothing. The panes and halos are
 * additive, so by day — glow at zero — they add nothing and the day looks are
 * unchanged. The pendulums are three tiny CPU integrations.
 *
 * ### The model has no windows and no lantern hooks
 *
 * The Poly Haven hull is one joined mesh with a painted transom, so the panes
 * are geometry laid on the measured transom face, and the hooks are measured
 * off the masts (fore at x = 7, main at x = -1.5, mizzen at x = -8.25 in the
 * ship's frame, from the rigging mesh).
 */

/** Warm tallow light, a touch warmer than the stern lantern's. */
const LANTERN_COLOR = new THREE.Color(1.0, 0.62, 0.3);
const WINDOW_COLOR = new THREE.Color(1.0, 0.55, 0.24);

/** Hook points in the ship's frame (metres, +X forward, +Z starboard). */
const HOOKS: readonly THREE.Vector3[] = [
  // Aft face of the mainmast, over the waist: lights the main deck and the main course above it.
  new THREE.Vector3(-1.85, 4.55, 0),
  // Aft face of the foremast, over the forecastle and under the fore course.
  new THREE.Vector3(6.65, 5.6, 0),
  // Forward face of the mizzen, over the quarterdeck.
  new THREE.Vector3(-7.9, 6.25, 0),
];

/** Rope from the hook to the lantern's cap, metres. */
const ROPE = 0.32;
const CAP_H = 0.1;
const GLASS_H = 0.24;
/** Hook to the centre of the glass: the pendulum's length. */
const PENDULUM = ROPE + CAP_H + GLASS_H * 0.5;

/** Damping of the swing relative to the hook, 1/s. A lantern rings for a few seconds. */
const SWING_DAMPING = 0.55;
/** Furthest a lantern swings from the vertical before it fetches up against the mast. */
const MAX_SWING = THREE.MathUtils.degToRad(38);
const GRAVITY = 9.81;
/** Integration step for the swing, seconds. */
const SUBSTEP = 1 / 120;

/**
 * Stern windows: four lights across the transom at the great cabin's deck, laid
 * on the measured face (x = -12.05 at y = 1.5 to -12.47 at y = 3.5, raked aft
 * as it rises) and a few centimetres proud of it.
 */
const WINDOW_Y = 3.0;
const WINDOW_RAKE = 0.21;
const WINDOW_FACE_X = -12.36 - 0.05;
const WINDOW_W = 0.34;
const WINDOW_H = 0.44;
const WINDOW_Z = [-0.93, -0.31, 0.31, 0.93];

interface Lantern {
  hook: THREE.Vector3;
  pivot: THREE.Group;
  bob: THREE.Vector3;
  velocity: THREE.Vector3;
  prevHook: THREE.Vector3;
  primed: boolean;
}

export class ShipLights {
  readonly group = new THREE.Group();
  /** World positions of the three lanterns' glass, for `LocalLights`. Updated in place. */
  readonly lanternPositions: THREE.Vector3[] = HOOKS.map(() => new THREE.Vector3(0, -10000, 0));

  private readonly ship: THREE.Object3D;
  private readonly lanterns: Lantern[] = [];
  private readonly uGlow = uniform(0);
  private readonly materials: THREE.Material[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private time = 0;
  private reducedMotion = false;
  private readonly motionQuery: MediaQueryList | null;
  private readonly onMotionChange = (e: MediaQueryListEvent): void => {
    this.reducedMotion = e.matches;
  };

  constructor(ship: THREE.Object3D) {
    this.ship = ship;
    this.group.name = 'ship-lights';
    this.motionQuery = typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
    this.reducedMotion = this.motionQuery?.matches ?? false;
    this.motionQuery?.addEventListener?.('change', this.onMotionChange);

    const iron = new THREE.MeshStandardNodeMaterial({ color: 0x1b1714, metalness: 0.55, roughness: 0.6 });
    iron.name = 'ship-lantern-iron';
    const glass = new THREE.MeshStandardNodeMaterial({ color: 0x3a2a18, roughness: 0.25 });
    glass.name = 'ship-lantern-glass';
    // HDR on purpose, like the stern lantern: past the bloom threshold at night.
    glass.emissiveNode = color(LANTERN_COLOR).mul(this.uGlow).mul(5);
    const halo = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    halo.name = 'ship-lantern-halo';
    halo.fog = false;
    // Soft-edged by the facing ratio, as `LocalLights` draws the stern lantern's flame.
    halo.colorNode = color(LANTERN_COLOR).mul(this.uGlow).mul(normalView.z.max(0).pow(3)).mul(1.6);
    this.materials.push(iron, glass, halo);

    const rope = new THREE.CylinderGeometry(0.012, 0.012, ROPE, 4, 1, true).translate(0, -ROPE / 2, 0);
    const cap = new THREE.ConeGeometry(0.11, CAP_H, 6).translate(0, -ROPE - CAP_H / 2, 0);
    const body = new THREE.CylinderGeometry(0.075, 0.085, GLASS_H, 6).translate(0, -ROPE - CAP_H - GLASS_H / 2, 0);
    const base = new THREE.CylinderGeometry(0.1, 0.1, 0.03, 6).translate(0, -ROPE - CAP_H - GLASS_H - 0.015, 0);
    const haloGeo = new THREE.SphereGeometry(0.26, 12, 8).translate(0, -PENDULUM, 0);
    this.geometries.push(rope, cap, body, base, haloGeo);

    for (const hook of HOOKS) {
      const pivot = new THREE.Group();
      pivot.name = 'ship-deck-lantern';
      pivot.position.copy(hook);
      for (const [geometry, material] of [[rope, iron], [cap, iron], [base, iron], [body, glass], [haloGeo, halo]] as const) {
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = false;
        mesh.receiveShadow = material !== halo;
        pivot.add(mesh);
      }
      this.group.add(pivot);
      this.lanterns.push({
        hook,
        pivot,
        bob: new THREE.Vector3(),
        velocity: new THREE.Vector3(),
        prevHook: new THREE.Vector3(),
        primed: false,
      });
    }

    this.group.add(this.buildWindows());
    ship.add(this.group);
  }

  /** The great cabin's lights: four leaded panes on the transom, one draw. */
  private buildWindows(): THREE.Mesh {
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    // Up the raked face, and across it.
    const up = new THREE.Vector3(-WINDOW_RAKE, 1, 0).normalize();
    const across = new THREE.Vector3(0, 0, 1);
    const corner = new THREE.Vector3();
    for (const z of WINDOW_Z) {
      const base = positions.length / 3;
      const centre = new THREE.Vector3(WINDOW_FACE_X, WINDOW_Y, z);
      for (const [u, v] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
        corner.copy(centre).addScaledVector(across, (u - 0.5) * WINDOW_W).addScaledVector(up, (v - 0.5) * WINDOW_H);
        positions.push(corner.x, corner.y, corner.z);
        uvs.push(u, v);
      }
      // Facing aft (-X): wound so the front face looks out of the transom.
      indices.push(base, base + 3, base + 2, base, base + 2, base + 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    this.geometries.push(geometry);

    const material = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    material.name = 'ship-stern-windows';
    material.fog = false;
    // Leaded lights: a frame and a cross of glazing bars, with the cabin's
    // lamplight brighter low in the pane where the lamp hangs.
    const p = uv();
    const edge = max(p.sub(0.5).abs().x.mul(2), p.sub(0.5).abs().y.mul(2));
    const frame = float(1).sub(smoothstep(0.82, 0.9, edge));
    const bars = vec2(
      smoothstep(0.02, 0.05, p.x.sub(0.5).abs()),
      smoothstep(0.02, 0.05, p.y.sub(0.5).abs()),
    );
    const lamp = float(1.25).sub(p.y.mul(0.55));
    material.colorNode = color(WINDOW_COLOR).mul(this.uGlow).mul(frame).mul(bars.x).mul(bars.y).mul(lamp).mul(2.2);
    this.materials.push(material);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'ship-stern-windows';
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    return mesh;
  }

  /** Back to hanging still, clock at `time`. For deterministic capture. */
  reset(time = 0): void {
    this.time = time;
    for (const l of this.lanterns) l.primed = false;
  }

  /**
   * @param dt Seconds since the last frame.
   * @param darkness 0 by day, 1 from deep dusk on, as `LightingRig.darkness`.
   */
  update(dt: number, darkness: number): void {
    this.time += Math.max(0, dt);
    const t = this.time;
    // A shallow, slow wander — tallow, not a fire.
    const flutter = 1 + 0.05 * Math.sin(t * 7.1) + 0.03 * Math.sin(t * 13.7 + 0.8);
    this.uGlow.value = THREE.MathUtils.clamp(darkness, 0, 1) * flutter;

    this.ship.updateWorldMatrix(true, false);
    const m = this.ship.matrixWorld;
    _inverseQuat.copy(this.ship.getWorldQuaternion(_quat)).invert();
    for (let i = 0; i < this.lanterns.length; i++) {
      const l = this.lanterns[i];
      const hook = _hook.copy(l.hook).applyMatrix4(m);
      // First frame, or the hull was put somewhere new (a reset, a respawn).
      if (!l.primed || hook.distanceToSquared(l.prevHook) > 4) {
        l.bob.copy(hook).y -= PENDULUM;
        l.velocity.set(0, 0, 0);
        l.prevHook.copy(hook);
        l.primed = true;
      } else if (dt > 0) {
        if (this.reducedMotion) {
          // Hangs plumb: no swing of its own beyond the hull's.
          l.bob.copy(hook).y -= PENDULUM;
          l.velocity.set(0, 0, 0);
        } else {
          this.swing(l, hook, dt);
        }
        l.prevHook.copy(hook);
      }
      this.lanternPositions[i].copy(l.bob);
      // Point the pivot's -Y down the rope, in the ship's frame.
      _dir.subVectors(l.bob, hook).normalize().applyQuaternion(_inverseQuat);
      l.pivot.quaternion.setFromUnitVectors(DOWN, _dir);
    }
  }

  /**
   * A pendulum on a rod whose top is carried by the hull: semi-implicit Euler
   * with the rod as a projection, substepped, damped relative to the hook so
   * that a ship under way does not trail her lanterns aft.
   */
  private swing(l: Lantern, hook: THREE.Vector3, dt: number): void {
    const steps = Math.min(16, Math.max(1, Math.ceil(dt / SUBSTEP)));
    const h = dt / steps;
    _hookVel.subVectors(hook, l.prevHook).divideScalar(dt);
    const keep = Math.exp(-SWING_DAMPING * h);
    for (let s = 1; s <= steps; s++) {
      _hk.lerpVectors(l.prevHook, hook, s / steps);
      l.velocity.y -= GRAVITY * h;
      _rel.subVectors(l.velocity, _hookVel).multiplyScalar(keep);
      l.velocity.addVectors(_hookVel, _rel);
      l.bob.addScaledVector(l.velocity, h);
      _d.subVectors(l.bob, _hk);
      const len = _d.length();
      if (len < 1e-6) _d.set(0, -1, 0);
      else _d.divideScalar(len);
      // Fetches up against the mast rather than swinging over the hook.
      if (_d.y > -Math.cos(MAX_SWING)) {
        const horiz = Math.hypot(_d.x, _d.z) || 1;
        const sinA = Math.sin(MAX_SWING);
        _d.set((_d.x / horiz) * sinA, -Math.cos(MAX_SWING), (_d.z / horiz) * sinA);
      }
      l.bob.copy(_hk).addScaledVector(_d, PENDULUM);
      // Only the swing survives the rod: drop the relative radial component.
      _rel.subVectors(l.velocity, _hookVel);
      l.velocity.addScaledVector(_d, -_rel.dot(_d));
    }
  }

  dispose(): void {
    this.motionQuery?.removeEventListener?.('change', this.onMotionChange);
    this.group.removeFromParent();
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
  }
}

const DOWN = /*@__PURE__*/ new THREE.Vector3(0, -1, 0);
const _hook = /*@__PURE__*/ new THREE.Vector3();
const _hk = /*@__PURE__*/ new THREE.Vector3();
const _hookVel = /*@__PURE__*/ new THREE.Vector3();
const _rel = /*@__PURE__*/ new THREE.Vector3();
const _d = /*@__PURE__*/ new THREE.Vector3();
const _dir = /*@__PURE__*/ new THREE.Vector3();
const _quat = /*@__PURE__*/ new THREE.Quaternion();
const _inverseQuat = /*@__PURE__*/ new THREE.Quaternion();
