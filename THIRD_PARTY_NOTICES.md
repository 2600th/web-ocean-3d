# Third-party notices

This project includes code adapted from the following open-source projects.

## abyssal-garden

The underwater medium (analytic height-dependent in-scatter), the volumetric
light shafts (ray march, depth-aware blur and upsample), the chromatic caustics
and the submerged hemisphere ambient are adapted from abyssal-garden's WGSL
shaders and ported to three.js TSL. Affected files:

- `src/underwater/UnderwaterPass.ts`
- `src/underwater/Caustics.ts`

The instanced leaf renderer for kelp and seabed cover (leaf strips, sway, palette,
two-sided translucent lighting, glints) is adapted from abyssal-garden's `W_LEAF`
shader and its world layout:

- `src/scene/LeafField.ts`
- `src/scene/Kelp.ts`
- `src/scene/GroundCover.ts`

```
MIT License

Copyright (c) 2026 Maksymilian Dendura

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
