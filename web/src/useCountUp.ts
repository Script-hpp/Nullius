import { useEffect, useRef, useState } from 'react'
import { COUNT_MS, countAt } from './countUp'

const reduced = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

// Animates from the value shown now to `target` (ease-out, ~700 ms). Reduced motion: the target is shown directly.
export function useCountUp(target: number, duration = COUNT_MS): number {
  const [shown, setShown] = useState(() => (reduced() ? target : 0))
  const from = useRef(shown)
  useEffect(() => {
    if (reduced() || from.current === target) { from.current = target; setShown(target); return }
    const start = performance.now(), base = from.current
    let raf = 0
    const tick = (now: number) => {
      const v = base + countAt(target - base, now - start, duration)
      from.current = v
      setShown(v)
      if (now - start < duration) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])
  return shown
}
