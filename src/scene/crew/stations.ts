/**
 * Where the crew stand, and what each group is doing.
 *
 * Positions are in the ship's frame: x metres forward of the hull's centre
 * (the hull runs from about -13.5 at the transom to +13.5 at the stem), z as a
 * fraction of the half-beam to starboard, so a different hull keeps them
 * inboard of the rails. The deck height under each is measured from the hull
 * at load (`Crew`'s `DeckProbe`). Measured off the Poly Haven hull: the
 * forecastle deck runs from x = 9.2 forward, the waist from 9.2 to -3.6, the
 * quarterdeck aft of that; masts at x = 7 (fore), -1.5 (main), -8.25 (mizzen).
 *
 * `facing` is the direction a figure looks, radians in the deck plane: 0 is
 * forward (+X), +PI/2 is to starboard (+Z).
 */

export type CrewModel = 'officer' | 'hand';

export interface CrewStation {
  x: number;
  /** Athwartships, as a fraction of the half-beam, + to starboard. */
  zBeam: number;
  facing: number;
  /** Placed from the wheel's node when the model has one. */
  atWheel?: boolean;
  /** Ray start height for the deck probe, metres; under a yard or a top it must start below it. */
  probeFrom?: number;
  /** sRGB hex. */
  shirt?: number;
  kerchief?: number;
}

export interface CrewGroupSpec {
  id: string;
  model: CrewModel;
  /** Clip name in the GLB: `Idle`, `PickUp` or `Walk`. */
  clip: string;
  /** Seconds into the clip at clock zero. */
  phase: number;
  /** Playback rate. */
  rate: number;
  stations: CrewStation[];
}

const HALF_PI = Math.PI / 2;

// Undyed and blued linen, a faded red check, and slop-chest grey.
const LINEN = 0xd8d0bc;
const BLUE = 0x5d6f8c;
const RED = 0x8c3a2c;
const GREY = 0x8f8a80;

export const CREW_GROUPS: readonly CrewGroupSpec[] = [
  {
    id: 'helm',
    model: 'officer',
    clip: 'Idle',
    phase: 0,
    rate: 0.8,
    stations: [{ x: -6.6, zBeam: 0, facing: 0, atWheel: true }],
  },
  {
    id: 'lookout',
    model: 'hand',
    clip: 'Idle',
    phase: 1.7,
    rate: 0.7,
    stations: [{ x: 10.9, zBeam: -0.18, facing: -0.3, shirt: LINEN, kerchief: RED }],
  },
  {
    id: 'watch',
    model: 'hand',
    clip: 'Idle',
    phase: 3.1,
    rate: 0.75,
    stations: [
      { x: 2.6, zBeam: -0.6, facing: -HALF_PI, shirt: BLUE, kerchief: LINEN },
      { x: -2.6, zBeam: 0.6, facing: HALF_PI + 0.35, shirt: LINEN, kerchief: BLUE },
      { x: 5.0, zBeam: 0.5, facing: 2.5, shirt: GREY, kerchief: RED },
    ],
  },
  {
    id: 'haul-main',
    model: 'hand',
    clip: 'PickUp',
    phase: 0.2,
    rate: 0.45,
    stations: [
      { x: -0.3, zBeam: -0.32, facing: Math.PI, shirt: RED, kerchief: LINEN },
      { x: -0.3, zBeam: 0.32, facing: Math.PI, shirt: LINEN, kerchief: GREY },
    ],
  },
  {
    id: 'haul-fore',
    model: 'hand',
    clip: 'PickUp',
    phase: 0.75,
    rate: 0.5,
    stations: [
      { x: 8.1, zBeam: -0.34, facing: Math.PI, shirt: GREY, kerchief: BLUE },
      { x: 8.1, zBeam: 0.36, facing: Math.PI, shirt: BLUE, kerchief: RED },
    ],
  },
];

/**
 * Order the tier adds figures in, as group indices: a tier with `n` figures
 * draws the first `n` of these. The helmsman and the lookout first, then one
 * from each working group, so a small crew is still spread along the deck.
 */
export const CREW_ORDER: readonly number[] = [0, 1, 2, 3, 2, 4, 3, 4, 2];
