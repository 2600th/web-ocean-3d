/**
 * The interface's drawn iconography.
 *
 * Authored SVG, one stroke weight (1.75 on a 24 grid) and round joins, so every
 * mark reads as having come off the same pen. Static strings, parsed through a
 * <template>, never interpolated with anything a caller supplies.
 */

const SVG_OPEN =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">';

const ICONS = {
  /** Orbit: a body circled by its path, with the arrow of travel. */
  orbit:
    '<ellipse cx="12" cy="12" rx="9.5" ry="4.6" transform="rotate(-18 12 12)"/>' +
    '<circle cx="12" cy="12" r="3" fill="currentColor" stroke="none"/>' +
    '<path d="M18.6 6.3l2.4.6-.9 2.3"/>',
  /** Fly: a gull's wings. */
  fly: '<path d="M2.5 11.5c2.6-3.4 6-3.6 9.5.6 3.5-4.2 6.9-4 9.5-.6"/><path d="M12 12.1v2.4"/>',
  /** Helm: a ship's wheel, eight spokes and their handles. */
  helm:
    '<circle cx="12" cy="12" r="6.2"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/>' +
    '<path d="M12 2.2v19.6M2.2 12h19.6M5.1 5.1l13.8 13.8M18.9 5.1L5.1 18.9"/>',
  /** Tour: a spyglass. */
  tour:
    '<path d="M3.2 15.4l11.6-6.7 2.1 3.6-11.6 6.7z"/><path d="M14.8 8.7l3.9-2.2 2.1 3.6-3.9 2.3"/>' +
    '<path d="M8.4 17.8l1.4 3.4M7.1 21.2h5"/>',
  /** Info toast: a compass needle. */
  info:
    '<circle cx="12" cy="12" r="9"/><path d="M12 5.5l2.4 6.5L12 18.5 9.6 12z" fill="currentColor" fill-opacity=".25"/>' +
    '<path d="M9.6 12h4.8"/>',
  /** Reward toast: a doubloon. */
  reward:
    '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="6.3"/>' +
    '<path d="M12 8.2l1.15 2.35 2.6.38-1.88 1.83.44 2.59L12 14.13l-2.32 1.22.44-2.59-1.88-1.83 2.6-.38z" fill="currentColor" stroke-width="1"/>',
  /** Warning toast: a hazard triangle. */
  warning:
    '<path d="M12 3.2l9.3 16.6H2.7z"/><path d="M12 9.4v4.6"/><circle cx="12" cy="17" r=".6" fill="currentColor"/>',
  /** Danger toast: a skull. */
  danger:
    '<path d="M12 3c-4.6 0-7.6 3-7.6 7.2 0 2.6 1.2 4.2 2.6 5.1V19h10v-3.7c1.4-.9 2.6-2.5 2.6-5.1C19.6 6 16.6 3 12 3z"/>' +
    '<circle cx="9" cy="11" r="1.7" fill="currentColor"/><circle cx="15" cy="11" r="1.7" fill="currentColor"/>' +
    '<path d="M12 13.6l-.9 1.8h1.8zM10 19v-2M14 19v-2"/>',
  /** Call to action: an anchor. */
  anchor:
    '<circle cx="12" cy="4.6" r="2"/><path d="M12 6.6v14.6M8 10h8"/>' +
    '<path d="M4 13.4c.4 4.3 3.7 7.8 8 7.8s7.6-3.5 8-7.8"/><path d="M2.6 15.2L4 13.4l1.9 1.5M21.4 15.2L20 13.4l-1.9 1.5"/>',
  /** Touch helm: set more sail — a full square sail with a rising arrow. */
  sailUp:
    '<path d="M7 4.5h9.5M7 4.5v13M7 6c3.6.6 5.6 2.8 5.6 5.6S10.6 16.6 7 17.2"/>' +
    '<path d="M18 19.5V9M15.2 11.8L18 9l2.8 2.8"/>',
  /** Touch helm: take in sail — a furled yard with a falling arrow. */
  sailDown:
    '<path d="M3.5 6h12M5 6c.8 2 2.6 3 4.5 3s3.7-1 4.5-3"/><path d="M9.5 9v11"/>' +
    '<path d="M18 6.5V17M15.2 14.2L18 17l2.8-2.8"/>',
  /** Collapse / expand chevron. */
  chevron: '<path d="M6.5 9.5l5.5 5.5 5.5-5.5"/>',
  /** Mobile ledger toggle: a bound ledger. */
  ledger:
    '<path d="M5 3.5h12.5a1.5 1.5 0 0 1 1.5 1.5v14a1.5 1.5 0 0 1-1.5 1.5H5z"/><path d="M8 3.5v17"/>' +
    '<path d="M11 8h5M11 11.5h5M11 15h3"/>',
  /** Objective marker: the cross that marks the spot. */
  mark: '<path d="M6 6l12 12M18 6L6 18" stroke-width="3"/>',
  /** Close. */
  close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
} as const;

export type IconName = keyof typeof ICONS;

/** A fresh SVG element for `name`, sized by CSS. */
export function icon(name: IconName, className = 'icon'): SVGSVGElement {
  const template = document.createElement('template');
  template.innerHTML = SVG_OPEN + ICONS[name] + '</svg>';
  const svg = template.content.firstElementChild as SVGSVGElement;
  svg.setAttribute('class', className);
  return svg;
}

/**
 * The ship that sails the boot route, and the rose on the wind dial.
 *
 * Drawn on a 100 grid. The rose is the classic sixteen-point card: cardinal
 * points split light and dark the way an engraved card shades them, so it
 * reads at a glance which way is north however the card is turned.
 */
export function compassRose(className = 'rose'): SVGSVGElement {
  const point = (angle: number, length: number, width: number, dark: string, light: string): string =>
    `<g transform="rotate(${angle} 50 50)">` +
    `<path d="M50 ${50 - length}L${50 - width} ${50 - width}L50 50z" fill="${dark}"/>` +
    `<path d="M50 ${50 - length}L${50 + width} ${50 - width}L50 50z" fill="${light}" stroke="${dark}" stroke-width=".6"/>` +
    '</g>';

  let ticks = '';
  for (let i = 0; i < 32; i++) {
    const major = i % 4 === 0;
    ticks +=
      `<path d="M50 ${major ? 3.5 : 4.5}V${major ? 9 : 7.2}" transform="rotate(${i * 11.25} 50 50)" ` +
      `stroke-width="${major ? 1.3 : 0.8}"/>`;
  }

  let points = '';
  for (let i = 0; i < 8; i++) points += point(22.5 + i * 45, 25, 3, 'var(--rose-ink)', 'var(--rose-paper)');
  for (let i = 0; i < 4; i++) points += point(45 + i * 90, 33, 5, 'var(--rose-ink)', 'var(--rose-paper)');
  for (let i = 0; i < 4; i++) {
    points += point(i * 90, 42, 6.2, i === 0 ? 'var(--rose-north)' : 'var(--rose-ink)', 'var(--rose-paper)');
  }

  const template = document.createElement('template');
  template.innerHTML =
    '<svg viewBox="0 0 100 100" aria-hidden="true" focusable="false">' +
    '<g fill="none" stroke="var(--rose-ink)">' +
    '<circle cx="50" cy="50" r="46.5" stroke-width="1.2"/><circle cx="50" cy="50" r="40" stroke-width=".6"/>' +
    ticks +
    '</g>' +
    points +
    '<circle cx="50" cy="50" r="2.6" fill="var(--rose-paper)" stroke="var(--rose-ink)" stroke-width=".8"/>' +
    '</svg>';
  const svg = template.content.firstElementChild as SVGSVGElement;
  svg.setAttribute('class', className);
  return svg;
}
