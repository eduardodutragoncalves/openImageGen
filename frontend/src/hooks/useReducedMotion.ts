import { useEffect, useState } from "react"

/**
 * Whether the operator has asked the system for less movement.
 *
 * The stylesheet already flattens CSS animations under this query, but a WebGL
 * canvas draws outside CSS entirely: nothing in `prefers-reduced-motion` can
 * reach it, so it has to ask.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  )

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)")
    const update = () => setReduced(query.matches)
    update()
    query.addEventListener("change", update)
    return () => query.removeEventListener("change", update)
  }, [])

  return reduced
}
