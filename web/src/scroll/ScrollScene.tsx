import { Component, Suspense, lazy, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { prefersReducedMotion } from '../components/hero/config'
import { easeToward, sceneProgress, sceneTime, scrollLengthVh } from './progress'
import { FRAME_BG, SCENES, paletteFor, type SceneId } from './scenes'

// A film shot that plays on the page while you scroll. The section is tall; a 16:9 frame sticks to the screen and the scroll position inside the
// section sets the film time (scrub: scrolling back plays it back). The stage code and its data are a separate chunk, fetched when the scene
// comes within about half a screen of the viewport. Reduced motion and phones get the poster frame instead (a 1920x1080 frame on 358 px is too small to read moving).

const SceneStage = lazy(() => import('./SceneStage'))
const HEADER = 64   // px, the sticky site header (top-16)
const PHONE = '(max-width: 639px)'

function useMedia(query: string): boolean {
  const [on, setOn] = useState(() => typeof matchMedia === 'function' && matchMedia(query).matches)
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const mq = matchMedia(query)
    const f = () => setOn(mq.matches)
    f()
    mq.addEventListener('change', f)
    return () => mq.removeEventListener('change', f)
  }, [query])
  return on
}

function useDark(): boolean {
  const read = () => document.documentElement.dataset.theme !== 'light'
  const [dark, setDark] = useState(read)
  useEffect(() => {
    const mo = new MutationObserver(() => setDark(read()))
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => mo.disconnect()
  }, [])
  return dark
}

// a failed chunk or a rendering error leaves the empty frame, never a broken start page
class Quiet extends Component<{ children: ReactNode }, { bad: boolean }> {
  state = { bad: false }
  static getDerivedStateFromError() { return { bad: true } }
  render() { return this.state.bad ? <div className="absolute inset-0 grid place-items-center p-6 text-center text-[13px] text-muted">This animation could not be shown.</div> : this.props.children }
}

export function ScrollScene({ id }: { id: SceneId }) {
  const meta = SCENES[id]
  const phone = useMedia(PHONE)
  const still = phone || prefersReducedMotion()
  const dark = useDark()
  const palette = paletteFor(meta.theme, dark)

  const section = useRef<HTMLElement>(null)
  const pin = useRef<HTMLDivElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  const [near, setNear] = useState(false)       // within about half a screen of the viewport: scroll is tracked
  const [wanted, setWanted] = useState(false)   // has been near once: the stage chunk is requested
  const [scale, setScale] = useState(0.5)
  const [t, setT] = useState(() => sceneTime(0, meta.duration, { still, poster: meta.poster }))
  const live = useRef({ cur: t, target: t, raf: 0, last: 0 })

  // near the viewport
  useEffect(() => {
    const el = section.current
    if (!el || typeof IntersectionObserver === 'undefined') { setNear(true); setWanted(true); return }
    // the chunk is requested only if the scene stays near for a moment: while the page above is still laying out it can be near for a frame
    let timer = 0
    const io = new IntersectionObserver(([e]) => {
      setNear(e.isIntersecting)
      clearTimeout(timer)
      if (e.isIntersecting) timer = window.setTimeout(() => setWanted(true), 500)
    }, { rootMargin: '60% 0px 60% 0px' })
    io.observe(el)
    return () => { io.disconnect(); clearTimeout(timer) }
  }, [])

  // frame size -> canvas scale
  useEffect(() => {
    const el = frame.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setScale(el.clientWidth / 1920))
    ro.observe(el)
    setScale(el.clientWidth / 1920)
    return () => ro.disconnect()
  }, [])

  const target = useCallback(() => {
    const s = section.current, p = pin.current
    if (!s || !p) return 0
    const r = s.getBoundingClientRect()
    return sceneTime(sceneProgress(r.top - HEADER, r.height, p.clientHeight), meta.duration, { still: false })
  }, [meta.duration])

  // scrub: scroll sets the target time, a rAF loop eases the shown time towards it and stops once it has arrived
  useEffect(() => {
    const L = live.current
    if (still) { L.cur = L.target = sceneTime(0, meta.duration, { still: true, poster: meta.poster }); setT(L.cur); return }
    if (!near) return
    const frameTick = (now: number) => {
      L.raf = 0
      const dt = L.last ? Math.min(100, now - L.last) : 16
      L.last = now
      L.target = target()
      L.cur = easeToward(L.cur, L.target, dt)
      setT(L.cur)
      if (L.cur !== L.target) L.raf = requestAnimationFrame(frameTick)
      else L.last = 0
    }
    const kick = () => { if (!L.raf) L.raf = requestAnimationFrame(frameTick) }
    kick()
    addEventListener('scroll', kick, { passive: true })
    addEventListener('resize', kick)
    return () => { removeEventListener('scroll', kick); removeEventListener('resize', kick); if (L.raf) cancelAnimationFrame(L.raf); L.raf = 0; L.last = 0 }
  }, [still, near, target, meta.duration, meta.poster])

  const lenVh = scrollLengthVh(meta.duration)
  const sectionStyle = still ? undefined : { height: `calc(${(lenVh * 100).toFixed(0)}vh + 100vh - ${HEADER}px)` }
  const pinClass = still ? 'py-4' : 'sticky top-16 flex h-[calc(100vh-4rem)] flex-col justify-center'
  // the frame is as wide as the column allows, but never taller than the screen leaves room for (caption included)
  const frameWidth = still ? '100%' : 'min(100%, calc((100vh - 4rem - 6.5rem) * 16 / 9))'

  return (
    <section ref={section} data-scene={id} style={sectionStyle} className="relative mx-auto max-w-6xl px-6">
      <div ref={pin} className={pinClass}>
        <figure className="m-0 flex flex-col items-center">
          <div ref={frame} role="img" aria-label={meta.label} style={{ width: frameWidth, aspectRatio: '16 / 9', background: FRAME_BG[palette] }}
            className="relative overflow-hidden rounded-xl border border-line shadow-sm">
            <div aria-hidden>
              {wanted && (
                <Quiet>
                  <Suspense fallback={null}><SceneStage id={id} t={t} palette={palette} scale={scale} /></Suspense>
                </Quiet>
              )}
            </div>
          </div>
          <figcaption className="mt-3 max-w-2xl text-center text-[13px] leading-relaxed text-muted">{still ? '' : 'Scroll to play. '}{meta.caption}</figcaption>
        </figure>
      </div>
    </section>
  )
}
