import { useEffect, useMemo, useRef, useState } from 'react'
import { SCRIPT_TEXTS } from './scriptData'
import { HERO_TEXT, WRITE_MS, ease, prefersReducedMotion, heroSeen, markHeroSeen } from './config'

// Handwritten lettering that is drawn stroke by stroke (stroke-dashoffset), in writing order, at constant pen speed
// with an eased start and end. Reduced motion and repeat visits in one session: the finished lettering at once.
export function WrittenText({ variant = HERO_TEXT, onDone, className = '' }: { variant?: keyof typeof SCRIPT_TEXTS; onDone?: () => void; className?: string }) {
  const t = SCRIPT_TEXTS[variant]
  const reduce = useMemo(() => prefersReducedMotion() || heroSeen(), [])
  const ref = useRef<SVGSVGElement>(null)
  const [width, setWidth] = useState(0)
  const [settled, setSettled] = useState(reduce)
  const doneRef = useRef(onDone)
  useEffect(() => { doneRef.current = onDone }, [onDone])

  // pen width of about 3 px on screen, whatever the size of the drawing
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const measure = () => setWidth(el.getBoundingClientRect().width)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const vbWidth = Number(t.viewBox.split(' ')[2])
  const px = width > 0 ? vbWidth / width : 12
  const pen = Math.max(2.4, Math.min(3.4, width / 280)) * px

  const total = t.strokes.reduce((s, x) => s + x.len, 0)
  const timing = useMemo(() => {
    const ends = t.strokes.reduce<number[]>((acc, x) => [...acc, (acc.length ? acc[acc.length - 1] : 0) + x.len], [])
    return t.strokes.map((_, k) => {
      const start = ease((k ? ends[k - 1] : 0) / total) * WRITE_MS
      return { delay: start, duration: Math.max(60, ease(ends[k] / total) * WRITE_MS - start) }
    })
  }, [t, total])

  useEffect(() => {
    if (reduce) { doneRef.current?.(); return }
    markHeroSeen()
    const a = window.setTimeout(() => { setSettled(true); doneRef.current?.() }, WRITE_MS + 100)
    return () => window.clearTimeout(a)
  }, [reduce])

  return (
    <svg ref={ref} viewBox={t.viewBox} role="img" aria-label={t.text} className={className} fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round"
      style={{ strokeWidth: pen * (settled ? 1.18 : 1), transition: 'stroke-width 700ms ease-out' }}>
      {t.strokes.map((s, i) => (
        <path key={i} d={s.d} pathLength={s.len} strokeDasharray={s.len}
          style={reduce ? { strokeDashoffset: 0 } : { strokeDashoffset: s.len, animation: `nullius-write ${timing[i].duration}ms linear ${timing[i].delay}ms forwards` }} />
      ))}
    </svg>
  )
}
