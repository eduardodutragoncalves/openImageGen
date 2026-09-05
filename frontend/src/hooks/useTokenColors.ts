import { useEffect, useState } from "react"

/**
 * Design-system tokens, read as RGB triplets a shader can take.
 *
 * The palette lives in CSS custom properties and changes under the operator —
 * four grounds, two palettes — so this reads the computed values rather than
 * hard-coding a copy that would drift the first time a token moves. A
 * MutationObserver on the root's attributes catches the switch, which is where
 * `data-theme` and `data-ground` are stamped.
 */
export type Rgb = [number, number, number]

const HEX = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i

/** Parse a token value into 0..1 sRGB. Handles `#rgb`, `#rrggbb` and `rgb()`. */
function parse(value: string): Rgb | null {
  const text = value.trim()
  if (HEX.test(text)) {
    const hex = text.slice(1)
    const full =
      hex.length === 3
        ? hex
            .split("")
            .map((c) => c + c)
            .join("")
        : hex
    return [
      parseInt(full.slice(0, 2), 16) / 255,
      parseInt(full.slice(2, 4), 16) / 255,
      parseInt(full.slice(4, 6), 16) / 255,
    ]
  }
  // rgb(242 245 250 / 0.07) and rgb(242, 245, 250). The alpha is dropped: a
  // shader colour stop has no compositing to do.
  const numbers = text.match(/-?[\d.]+/g)
  if (!numbers || numbers.length < 3) return null
  return [
    Number(numbers[0]) / 255,
    Number(numbers[1]) / 255,
    Number(numbers[2]) / 255,
  ]
}

export function readTokenColors(tokens: readonly string[]): Rgb[] {
  if (typeof window === "undefined") return []
  const style = getComputedStyle(document.documentElement)
  const out: Rgb[] = []
  for (const token of tokens) {
    const colour = parse(style.getPropertyValue(token))
    if (colour) out.push(colour)
  }
  return out
}

export function useTokenColors(tokens: readonly string[]): Rgb[] {
  const [colours, setColours] = useState<Rgb[]>(() => readTokenColors(tokens))

  useEffect(() => {
    const reread = () => setColours(readTokenColors(tokens))
    reread()
    const observer = new MutationObserver(reread)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme", "data-ground"],
    })
    return () => observer.disconnect()
    // The token list is a module constant at every call site; joining it keeps
    // the effect from resubscribing on each render over a fresh array literal.
  }, [tokens.join(",")])

  return colours
}
