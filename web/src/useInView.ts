import { useEffect, useRef, useState } from 'react'
import { prefersReducedMotion } from './components/hero/config'

// True once the element has entered the viewport (stays true). Without IntersectionObserver or with reduced motion it is true at once.
export function useInView<T extends HTMLElement>(threshold = 0.25): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined' || prefersReducedMotion())
  useEffect(() => {
    const el = ref.current
    if (seen || !el) return
    const io = new IntersectionObserver((hits) => { if (hits.some((h) => h.isIntersecting)) { setSeen(true); io.disconnect() } }, { threshold })
    io.observe(el)
    return () => io.disconnect()
  }, [seen, threshold])
  return [ref, seen]
}
