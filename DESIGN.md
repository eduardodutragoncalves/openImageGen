---
name: openImageGen
description: A GPU console built on a visible construction grid, where every value sits in a cell you can see.
colors:
  instrument-black: "#08090b"
  ground-sunk: "#050607"
  surface: "#0e1013"
  ink: "#f2f5fa"
  ink-muted: "#b4bdca"
  ink-faint: "#949dab"
  crouwel-blue: "#0057ff"
  accent-ink: "#6b9dff"
  caution-amber: "#ffb020"
  alarm-vermilion: "#ff4526"
  rule: "rgb(242 245 250 / 0.22)"
  rule-strong: "rgb(242 245 250 / 0.42)"
  grid-major: "rgb(242 245 250 / 0.17)"
  grid-fine: "rgb(242 245 250 / 0.07)"
typography:
  display:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "clamp(1.5rem, 4vw, 3.75rem)"
    fontWeight: 800
    lineHeight: 0.9
    letterSpacing: "-0.02em"
    fontVariation: "'wdth' 125"
    fontFeature: "'tnum' 1"
  headline:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Archivo, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.625rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.14em"
  mono:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, monospace"
    fontSize: "0.6875rem"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "normal"
    fontFeature: "'tnum' 1"
rounded:
  none: "0px"
spacing:
  cell: "8px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  major: "64px"
components:
  button-primary:
    backgroundColor: "{colors.crouwel-blue}"
    textColor: "#ffffff"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0 12px"
    height: "40px"
  button-primary-hover:
    backgroundColor: "#1a6bff"
    textColor: "#ffffff"
  button-primary-disabled:
    backgroundColor: "transparent"
    textColor: "{colors.ink-faint}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0 12px"
    height: "40px"
  button-secondary-hover:
    textColor: "{colors.accent-ink}"
  field:
    backgroundColor: "{colors.ground-sunk}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "8px 10px"
  toggle-active:
    backgroundColor: "{colors.crouwel-blue}"
    textColor: "#ffffff"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    height: "32px"
  state-plate-running:
    backgroundColor: "{colors.crouwel-blue}"
    textColor: "#ffffff"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    height: "20px"
    padding: "0 6px"
  state-plate-refused:
    backgroundColor: "transparent"
    textColor: "{colors.caution-amber}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    height: "20px"
    padding: "0 6px"
---

# Design System: openImageGen

## Overview

**Creative North Star: "The Instrument Sheet"**

A sheet of engineering graph paper that reports on a machine. The construction
grid is the default ground and everything on screen sits on it; readings hold
fixed positions so they can be found without looking for them, and a number
never changes width as it updates. The system descends from Wim Crouwel's
gridded type specimens by way of the instrument panel: the specimen supplies
the armature, the hairline, the single flat accent plane and the 45° mark, and
the console supplies what those elements are put to work saying.

Density is high and deliberate. This is a tool used for hours at a stretch by
someone who already knows what the controls do, so the surface spends its space
on values rather than on explanation, and separates regions with 1px rules
instead of containers. There are no cards, no corner radii, no shadows and no
elevation of any kind: depth is not part of this world, and a raised surface
would contradict the sheet it is drawn on.

Dark is the default because of the room, not the category — a workstation
beside a GPU rig, often dim, judging photographic output over long sessions. The
light register is the specimen sheet the world comes from and is fully
supported, not an afterthought. The build refuses the arrangement its category
ships: a centred prompt bar floating over rounded cards on a dark gradient.

**Key Characteristics:**

- The construction grid is the default ground and every region aligns to its
  8px cell — including in the flat grounds, where the armature is unpainted but
  still governs every dimension.
- Regions are divided by hairlines; boxes, radii and shadows do not exist.
- Colour is spent on state and navigation, never on decoration.
- Measurements are set in a grid-constructed mono with tabular figures.
- One recurring mark — a 45° diagonal — appears on the primary action, the
  wordmark and the active section.

## Colors

A near-neutral instrument ground carrying one saturated blue, with two further
hues reserved for the two things the operator must never misread.

### Primary
- **Crouwel Blue** (#0057ff): The live plane. It fills the primary action, the
  running job's progress, the selected control, and the slanted plate marking
  the active section. It is never a border alone and never a gradient.
- **Accent Ink** (#6b9dff): The same hue raised for legibility on the dark
  ground, used where the blue must be *text* or a hairline rather than a plane —
  links, the caret, focus rings, and the percentage readout on a running job.

### Secondary
- **Caution Amber** (#ffb020): Reserved for a job the content filter refused.
  This is not an error colour: a refusal is the system working, and the amber
  says "look and reword", not "something broke".
- **Alarm Vermilion** (#ff4526): Reserved for a job that actually failed, and
  for destructive intent on hover. Its rarity is what makes it readable.

### Neutral
- **Instrument Black** (#08090b): The page ground. Cool, near-neutral, and
  deliberately not blue-black slate.
- **Ground Sunk** (#050607): Inset surfaces only — the inside of an input.
- **Ink** (#f2f5fa): Primary text and monumental numerals.
- **Ink Muted** (#b4bdca): Secondary text, region labels, inactive controls.
- **Ink Faint** (#949dab, #5f6875 in the light register): Tertiary annotation,
  placeholder text, disabled controls. Both values clear 4.5:1 on the surfaces they are used on, including inside inputs. On the dark ground the ramp runs 18.2 : 10.5 : 7.3 against the page — three legible steps rather than one bright value and two grey ones.
- **Rule** (rgb(242 245 250 / 0.22)): Every divider in the system.

### Named Rules

**The Live Plane Rule.** A filled plane of Crouwel Blue means *this is the thing
happening now, or the thing you are about to do*: the primary action, the
running job, the current section, the selected option. Blue may also carry
navigation, links and emphasis, but it may never fill a region merely to make
it interesting, and two competing blue planes must never appear in one region.

**The Image Is The Subject Rule.** Generated images are the most saturated
thing on any screen. Chrome stays within the palette above so that no thumbnail
ever competes with the interface around it.

**The Two Alarms Rule.** Amber means refused, vermilion means failed, and
neither is ever used for anything else. Every other state is carried by the
neutral ramp and by typography.

## Typography

**Display Font:** Archivo Variable (with ui-sans-serif, system-ui fallback)
**Body Font:** Archivo Variable (same family, upright weights)
**Label/Mono Font:** Martian Mono Variable (with ui-monospace fallback)

**Character:** One engineered grotesque doing the structural work, pushed to its
width axis at 125% and weight 800 for monumental numerals, so a value reads as
constructed rather than merely enlarged. Martian Mono is grid-built by design
and carries anything that is a measurement. Both faces are self-hosted as
variable woff2 subsets; no webfont service is contacted at runtime.

**Scale in use:**

| Role | Size | Weight | Notes |
| --- | --- | --- | --- |
| Display / numeral | 1.5–3.75rem | 800 | `font-stretch: 125%`, tabular, `line-height: 0.9` |
| Headline | 0.875rem | 600 | Region and model names |
| Body | 0.875rem | 400 | Prompts, descriptions, `line-height: 1.5` |
| Label | 0.625rem | 600 | Uppercase, `letter-spacing: 0.14em` |
| Mono | 0.5625–0.6875rem | 400 | Seeds, sizes, VRAM, ids, counts |

### Named Rules

**The Measurement Rule.** Anything that is a measured quantity — a seed, a
pixel dimension, a VRAM figure, a job id, a step count, a timestamp — is set in
Martian Mono with tabular figures. Prose is never set in mono, and a label is
never set in mono to look technical.

**The Label Is The Heading Rule.** A region's small uppercase label *is* its
heading. There is no larger heading beneath it, because a small tracked label
sitting above a heading is the pattern this system exists to avoid.

## Layout

An 8px construction cell governs every dimension, with a 64px major line. The
grid is painted on the document ground as four repeating gradients and is
never covered: regions and cells are transparent, and their edges are drawn
with rules rather than fills, so the armature reads through the whole surface.

The ground is the operator's, through four swatches on the rail: the grid on
the dark ground or the light sheet, or a flat pure black or white with no grid
at all. The armature still governs every dimension in the flat grounds — it is
unpainted, not abandoned. The reason to allow it: the grid is a second image
behind the first, and a photograph judged against a graticule is judged against
the graticule too. Empty regions keep their dotted texture in every ground,
because that mark says *nothing here yet* rather than *ground*.

The application shell is a fixed-height column: a 56px status rail that never
scrolls, then the working area. On the studio route the working area is a
compose column and a fluid column holding the running job above the archive.
The compose column starts at `380px` and is dragged by the rule between the
two, from 288px to 760px, remembered across reloads; it never collapses,
because a form you cannot read is not a smaller form. Below 1280px the two
columns stack and the handle is withdrawn; the rail sheds the VRAM tapes at
1024px and the model plate at 640px so that navigation survives a narrow
window. The archive is a `repeat(auto-fill, minmax(196px, 1fr))` grid.

**Density:** tight inside a group (4–8px), generous between groups (16–24px),
and more space above a heading than below it.

**Breakpoints:** 640px, 768px, 1024px, 1280px. The product is desktop-first by
decision; narrower widths must remain functional and free of horizontal
overflow, but are not designed for.

## Elevation & Depth

**There is no elevation.** No shadows, no blurs used as depth, no layering of
lifted surfaces. Hierarchy is carried entirely by rules, by the visible grid,
and by the single accent plane. A shadow anywhere in this system is a defect,
not a variant.

## Shapes

**Radius is zero everywhere.** Buttons, inputs, plates, thumbnails, panels and
the focus ring are all square. There is no `sm`/`md`/`lg` radius scale to pick
from; `rounded.none` is the entire vocabulary.

Borders are 1px and one of two values: `rule` for ordinary division,
`rule-strong` for a control's own edge. The recurring form is the **45° stroke**,
drawn as a linear-gradient hairline and used as the mark on the primary action
and the wordmark; the active navigation item is the same angle taken
structurally, a plane clipped to a parallelogram.

Empty regions carry an 8px **dotted field** rather than an illustration or an
apology.

## Components

**Philosophy:** precise and unornamented. Every control is a rectangle with a
1px edge; what changes between states is the fill and the edge colour, never the
geometry.

- **Primary button** — filled Crouwel Blue, white label in uppercase label type,
  45° mark pinned right, 40px tall, full width where it closes a form. Disabled
  drops the fill entirely rather than dimming it.
- **Secondary button** — transparent with a `rule-strong` edge; hover moves the
  edge and the text to blue.
- **Input / textarea** — sunk ground, `rule` edge, blue edge on focus, blue
  caret. Placeholder sits at `ink-faint`.
- **Toggle group** — adjacent 32px cells separated by 2px; the selected cell
  becomes a blue plane.
- **Segment bar** — a quantity drawn as discrete 6px cells on the grid, filled
  from the left. This is the system's only progress form; a continuous rounded
  bar does not exist here.
- **State plate** — a 20px bordered plate carrying one word. Running is a blue
  plane, refused is an amber outline, failed is a vermilion outline, queued and
  done are neutral outlines.
- **Archive cell** — a square thumbnail over a label strip carrying prompt,
  seed, dimensions and time. The strip is designed to be legible at grid
  density so a wall of cells can be scanned without hovering anything.
- **Modal** — used only where the task genuinely needs protected focus, which
  in this system means choosing one model out of hundreds. Square, hairline
  edge in `rule-strong`, on the page ground and on the same 8px cell as
  everything else. The surface behind it dims to `ground-sunk` at 85% and is
  never blurred: this world has no glass, and a dialog is a region that has
  taken the foreground, not a pane floating above one. Escape and a click on
  the dimmed ground both close it.
- **Icons** — drawn on a 16-unit grid, 1.5 stroke, butt caps, mitre joins,
  corners chamfered at 45° rather than rounded. No glyph or emoji ever stands
  in for an icon.

**Browser surfaces are part of the system**: text selection, caret, scrollbars,
focus rings and underline offset are all themed from the palette.

**Motion** is one authored gesture — a 140ms colour and border transition on
`cubic-bezier(0.16, 1, 0.3, 1)` — applied to state changes only. Nothing
animates on entrance, nothing bounces, and `prefers-reduced-motion` disables
what little there is.

## Do's and Don'ts

**Do**

- Let the construction grid show through every region.
- Divide with a 1px rule and align to the 8px cell.
- Set every measured value in Martian Mono with tabular figures.
- Reserve the blue plane for what is live or what is about to be done.
- Show a state the machine is really in, with its phase and its numbers.
- Draw new icons in the existing grammar and keep the stroke at 1.5.

**Don't**

- Add a card, a corner radius, or a shadow of any kind.
- Introduce a second accent hue, or use amber and vermilion for anything but
  refused and failed.
- Use a gradient anywhere, least of all on text.
- Put a small tracked label above a heading.
- Substitute an emoji or a Unicode glyph for a drawn icon.
- Use monospace to make something look technical when it is not a measurement.
- Show an indeterminate spinner. This product always knows how far along it is,
  and where it genuinely does not, it says so in words.
