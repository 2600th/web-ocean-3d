/**
 * Who the crew are, where they stand and what each group is doing.
 *
 * Positions are in the ship's frame: x metres forward of the hull's centre
 * (the hull runs from about -13.5 at the transom to +13.5 at the stem), z as a
 * fraction of the half-beam to starboard, so a different hull keeps them
 * inboard of the rails. The deck height under each is measured from the hull
 * at load (`Crew`'s `DeckProbe`). Measured off the hero hull with that probe
 * (half-beam 3.2 m): the waist runs from x = -7.5 to 3.5 about 1.5 m above
 * the water, with a hatch at -5 to -3.5 and the ship's boat at 1 to 2.5 on
 * the centreline; the forecastle from 4 to 8 at 3.1 m; the quarterdeck from
 * -10 to -7.5 at 3.2 m with the whipstaff at -9; masts at about 7.5 (fore)
 * and -0.8 (main). Forward of 8 there is no deck to stand on.
 *
 * `facing` is the direction a figure looks, radians in the deck plane: 0 is
 * forward (+X), +PI/2 is to starboard (+Z).
 *
 * ### Looks
 *
 * Every figure is the same mesh (`public/models/crew/crew_toy.glb`, built by
 * `scripts/modelkit/crew-toy.py`) wearing a different set of its parts, so a
 * look is a list of parts and seven colours. The parts are the KayKit
 * Adventurers' heads, bodies, arms and legs (CC0) and the pirate gear modelled
 * for this ship. The colours are sRGB; the slot each tints is fixed by the mesh
 * (`TINT_SLOTS`).
 */

/** Part bits in the mesh's vertex colour (G × 255). Keep in step with `crew-toy.py`. */
export const PART = {
  /** Bearded, square-jawed. */
  headBeard: 0,
  /** Clean-shaven, a short swept crop. */
  headCrop: 1,
  /** Clean-shaven, a dark mop. */
  headMop: 2,
  /** Long hair to the shoulders. */
  headLong: 3,
  /** A barrel chest in a shirt, belted, with a leather baldric. */
  bodyBurly: 4,
  /** A long buttoned coat over a waistcoat. */
  bodyCoat: 5,
  /** A short smock with a neckcloth. */
  bodyTunic: 6,
  armsBurly: 7,
  armsCoat: 8,
  armsTunic: 9,
  legsBurly: 10,
  legsBoots: 11,
  legsTunic: 12,
  tricorn: 13,
  /** The knitted Monmouth cap, the common sailor's hat of the period. */
  cap: 14,
  kerchief: 15,
  eyePatch: 16,
  sash: 17,
} as const;

/** Clips in the GLB. */
export type CrewClip = 'Idle' | '2H_Melee_Idle' | 'Walking_A' | 'Cheer' | 'PickUp' | 'Interact';

/**
 * What a group does on top of its clip (`Crew.pose`):
 * - `idle`: weight shifts from foot to foot and looks about.
 * - `helm`: both hands on the whipstaff, leaning into it as the ship answers.
 * - `haul`: hand over hand on a fall: reach up, heave down and back.
 * - `walk`: pacing a stretch of deck fore and aft, turning at each end.
 */
export type CrewMotion = 'idle' | 'helm' | 'haul' | 'walk';

export interface CrewLook {
  /** What the figure is aboard; for captions and tests. */
  role: string;
  parts: number[];
  shirt: number;
  coat?: number;
  breeches: number;
  hat?: number;
  sash?: number;
  skin: number;
  hair: number;
  /** A striped shirt: bands of the shirt colour and undyed linen. */
  stripes?: boolean;
  /** Breadth and height relative to the standard figure. */
  build?: [number, number];
}

export interface CrewStation {
  x: number;
  /** Athwartships, as a fraction of the half-beam, + to starboard. */
  zBeam: number;
  facing: number;
  look: CrewLook;
  /** Placed from the helm's node when the model has one. */
  atWheel?: boolean;
  /** Ray start height for the deck probe, metres; under a yard or a top it must start below it. */
  probeFrom?: number;
  /** Walks fore and aft between `x` and this x. */
  walkTo?: number;
}

export interface CrewGroupSpec {
  id: string;
  motion: CrewMotion;
  clip: CrewClip;
  /** Seconds into the clip (and the motion's own cycle) at clock zero. */
  phase: number;
  /** Playback rate. */
  rate: number;
  stations: CrewStation[];
}

const HALF_PI = Math.PI / 2;
const P = PART;

// Dyes a ship's slop chest could run to around 1700: undyed linen, indigo,
// madder red, weld yellow, walnut brown, and an officer's broadcloth.
const LINEN = 0xe9e0c8;
const INDIGO = 0x2d4a86;
const MADDER = 0xb8322a;
const WELD = 0xe0b23a;
const WALNUT = 0x6a4a32;
const SLATE = 0x4a5160;
const TAR = 0x2a2a30;

const CAPTAIN: CrewLook = {
  role: 'captain',
  parts: [P.headMop, P.tricorn, P.bodyCoat, P.armsCoat, P.legsBoots],
  shirt: WELD, coat: 0xa3201e, breeches: TAR, hat: 0x1e1a1c, skin: 0xf0c4a0, hair: 0x1d1612,
};
const HELMSMAN: CrewLook = {
  role: 'helmsman',
  parts: [P.headCrop, P.cap, P.bodyTunic, P.armsTunic, P.legsTunic],
  shirt: INDIGO, stripes: true, breeches: WALNUT, hat: MADDER, sash: LINEN, skin: 0xd9a07c, hair: 0xd8b060,
};
const BOSUN: CrewLook = {
  role: 'bosun',
  parts: [P.headBeard, P.kerchief, P.eyePatch, P.bodyBurly, P.armsBurly, P.legsBurly],
  shirt: 0x3f7a5a, breeches: SLATE, hat: MADDER, skin: 0x8a5636, hair: 0x231a16, build: [1.12, 1.03],
};
const HAND_KERCHIEF: CrewLook = {
  role: 'deckhand',
  parts: [P.headMop, P.kerchief, P.bodyTunic, P.armsTunic, P.legsTunic, P.sash],
  shirt: LINEN, breeches: INDIGO, hat: WELD, sash: MADDER, skin: 0x5a3824, hair: 0x141010, build: [1.02, 1.0],
};
const HAND_GINGER: CrewLook = {
  role: 'deckhand',
  parts: [P.headCrop, P.bodyBurly, P.armsBurly, P.legsBurly],
  shirt: MADDER, breeches: WALNUT, skin: 0xf2c8aa, hair: 0xb8602a, build: [0.96, 0.95],
};
const HAND_CAP: CrewLook = {
  role: 'deckhand',
  parts: [P.headLong, P.cap, P.bodyTunic, P.armsTunic, P.legsTunic],
  shirt: 0x7a8a4a, breeches: SLATE, hat: INDIGO, sash: WELD, skin: 0xc68a5e, hair: 0x4a2414, build: [0.94, 0.97],
};
const LOOKOUT: CrewLook = {
  role: 'lookout',
  parts: [P.headLong, P.bodyTunic, P.armsTunic, P.legsTunic, P.sash],
  shirt: MADDER, stripes: true, breeches: LINEN, sash: INDIGO, skin: 0xe6b48e, hair: 0x9a4a22, build: [0.92, 0.98],
};
const OLD_HAND: CrewLook = {
  role: 'deckhand',
  parts: [P.headBeard, P.cap, P.bodyBurly, P.armsBurly, P.legsBurly],
  shirt: WALNUT, breeches: LINEN, hat: 0x8c8a80, skin: 0xb07850, hair: 0x9a9890, build: [1.06, 0.98],
};
const MATE: CrewLook = {
  role: 'mate',
  parts: [P.headMop, P.kerchief, P.eyePatch, P.bodyTunic, P.armsTunic, P.legsBoots],
  shirt: 0xc9b48a, breeches: TAR, hat: INDIGO, sash: MADDER, skin: 0x6e4228, hair: 0x181210,
};

export const CREW_GROUPS: readonly CrewGroupSpec[] = [
  {
    id: 'helm',
    motion: 'helm',
    clip: '2H_Melee_Idle',
    phase: 0,
    rate: 0.8,
    stations: [{ x: -6.6, zBeam: 0, facing: 0, atWheel: true, look: HELMSMAN }],
  },
  {
    id: 'stand-a',
    motion: 'idle',
    clip: 'Idle',
    phase: 0.4,
    rate: 0.85,
    stations: [
      { x: -8.6, zBeam: 0.45, facing: 0.5, look: CAPTAIN },
      { x: 2.6, zBeam: -0.6, facing: -HALF_PI, look: OLD_HAND },
    ],
  },
  {
    id: 'stand-b',
    motion: 'idle',
    clip: 'Idle',
    phase: 2.3,
    rate: 0.7,
    stations: [
      { x: 5.0, zBeam: -0.45, facing: -0.4, look: LOOKOUT },
      { x: -2.6, zBeam: 0.6, facing: HALF_PI + 0.35, look: MATE },
    ],
  },
  {
    id: 'haul',
    motion: 'haul',
    clip: '2H_Melee_Idle',
    phase: 0.6,
    rate: 1,
    stations: [
      { x: -0.3, zBeam: -0.32, facing: Math.PI, look: HAND_KERCHIEF },
      { x: 6.5, zBeam: 0.3, facing: 0, look: HAND_GINGER },
      { x: -0.3, zBeam: 0.32, facing: Math.PI, look: HAND_CAP },
    ],
  },
  {
    id: 'walk',
    motion: 'walk',
    clip: 'Walking_A',
    phase: 1.1,
    rate: 0.9,
    stations: [{ x: -6.8, zBeam: -0.45, facing: 0, walkTo: -3.2, look: BOSUN }],
  },
];

/**
 * Order the tier adds figures in, as group indices: a tier with `n` figures
 * draws the first `n` of these. The helmsman, the captain, a hand at the
 * mainmast, the bosun and the lookout first, so Medium's five are five
 * different roles spread along the deck.
 */
export const CREW_ORDER: readonly number[] = [0, 1, 3, 4, 2, 3, 1, 3, 2];
