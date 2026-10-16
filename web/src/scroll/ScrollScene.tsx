import { Component, Suspense, lazy, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { prefersReducedMotion } from '../components/hero/config'
import { advance, isFinished, isVisibleEnough } from './progress'
import { FRAME_BG, SCENES, paletteFor, stillSrc, type SceneId } from './scenes'

// A film shot that plays by itself once at least half of its frame is on screen, in real time, and then holds its last frame (a small control
// pauses and replays it). Before the stage code has loaded the frame shows a pre-rendered still of the first frame, so it is never blank; the live
// stage fades in over it after its first frame has painted. The stage chunk is requested well before the scene is reached.
// Phones and reduced motion get the poster still and no autoplay: the 1920x1080 canvas on 358 px is too small to read moving text and chart labels.

const SceneStage = lazy(() => import('./SceneStage'))
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

// a failed chunk or a rendering error leaves the still, never a broken start page
class Quiet extends Component<{ children: ReactNode; onBad: () => void }, { bad: boolean }> {
  state = { bad: false }
  static getDerivedStateFromError() { return { bad: true } }
  componentDidCatch() { this.props.onBad() }
  render() { return this.state.bad ? null : this.props.children }
}

export function ScrollScene({ id }: { id: SceneId }) {
  const meta = SCENES[id]
  const phone = useMedia(PHONE)
  const still = phone || prefersReducedMotion()
  const dark = useDark()
  const palette = paletteFor(meta.theme, dark)

  const frame = useRef<HTMLDivElement>(null)
  const [wanted, setWanted] = useState(false)     // the stage chunk is requested
  const [visible, setVisible] = useState(false)   // at least half of the frame is on screen
  const [ready, setReady] = useState(false)       // the stage has painted its first frame
  const [broken, setBroken] = useState(false)
  const [imgOk, setImgOk] = useState(false)
  const [userPaused, setUserPaused] = useState(false)
  const [scale, setScale] = useState(0.5)
  const [t, setT] = useState(meta.start)
  const clock = useRef({ raf: 0, last: 0, t: meta.start })
  const onReady = useCallback(() => setReady(true), [])
  const onBad = useCallback(() => setBroken(true), [])

  const ended = isFinished(t, meta.duration)
  const playing = !still && ready && visible && !userPaused && !ended

  // request the chunk well before the scene is reached (one screen ahead), but never during first paint: on a fresh load the page above is still
  // laying out and the scene can sit near the top for a moment. So: a moment after load, or at once when the reader has already scrolled.
  useEffect(() => {
    if (still) return
    const el = frame.current
    let idle = 0, settled = false, near = false
    const want = () => setWanted(true)
    if (!el || typeof IntersectionObserver === 'undefined') { want(); return }
    const io = new IntersectionObserver(([e]) => { near = e.isIntersecting; if (near && (settled || scrollY > 300)) want() }, { rootMargin: '100% 0px 100% 0px' })
    io.observe(el)
    const later = () => { idle = window.setTimeout(() => { settled = true; want() }, 1200) }
    if (document.readyState === 'complete') later(); else addEventListener('load', later, { once: true })
    return () => { io.disconnect(); clearTimeout(idle); removeEventListener('load', later) }
  }, [still])

  // visibility: play only while at least half of the frame is on screen
  useEffect(() => {
    const el = frame.current
    if (still || !el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([e]) => setVisible(isVisibleEnough(e.intersectionRatio)), { threshold: [0, 0.25, 0.5, 0.75, 1] })
    io.observe(el)
    return () => io.disconnect()
  }, [still])

  // frame size -> canvas scale
  useEffect(() => {
    const el = frame.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setScale(el.clientWidth / 1920))
    ro.observe(el)
    setScale(el.clientWidth / 1920)
    return () => ro.disconnect()
  }, [])

  // the play clock: real time, one rAF loop that exists only while playing
  useEffect(() => {
    if (!playing) return
    const C = clock.current
    const tick = (now: number) => {
      const dt = C.last ? now - C.last : 0
      C.last = now
      C.t = advance(C.t, dt, meta.duration, true)
      setT(C.t)
      if (!isFinished(C.t, meta.duration)) C.raf = requestAnimationFrame(tick)
    }
    C.last = 0
    C.raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(C.raf); C.last = 0 }
  }, [playing, meta.duration])

  const replay = () => { clock.current.t = meta.start; setT(meta.start); setUserPaused(false) }
  const control = ended ? { label: 'Replay the animation', icon: <RotateCcw size={16} aria-hidden />, act: replay }
    : userPaused ? { label: 'Play the animation', icon: <Play size={16} aria-hidden />, act: () => setUserPaused(false) }
    : { label: 'Pause the animation', icon: <Pause size={16} aria-hidden />, act: () => setUserPaused(true) }

  // the still that matches what the stage shows first: the start frame while it plays, the poster frame when it does not
  const kind = still ? 'poster' : 'start'
  const live = !still && wanted && !broken
  const showStage = live && ready

  return (
    <section data-scene={id} className="relative mx-auto max-w-6xl px-6 py-4">
      <figure className="m-0 flex flex-col items-center">
        <div ref={frame} role="img" aria-label={meta.label} style={{ width: '100%', aspectRatio: '16 / 9', background: FRAME_BG[palette], color: palette === 'night' ? '#f1eee6' : '#151310' }}
          className="relative overflow-hidden rounded-xl border border-line shadow-sm">
          {!imgOk && <div aria-hidden className="scene-shimmer absolute inset-0 overflow-hidden" />}
          <img key={`${kind}-${palette}`} src={stillSrc(id, kind, palette)} alt="" aria-hidden width={1200} height={675} decoding="async"
            onLoad={() => setImgOk(true)} onError={() => setImgOk(false)}
            className="absolute inset-0 h-full w-full select-none" draggable={false} />
          {live && (
            <div aria-hidden className="absolute inset-0 transition-opacity duration-200 motion-reduce:transition-none" style={{ opacity: showStage ? 1 : 0 }}>
              <Quiet onBad={onBad}>
                <Suspense fallback={null}><SceneStage id={id} t={t} palette={palette} scale={scale} onReady={onReady} /></Suspense>
              </Quiet>
            </div>
          )}
          {!still && showStage && (
            <button type="button" onClick={control.act} aria-label={control.label} title={control.label}
              className="absolute bottom-3 right-3 grid h-8 w-8 place-items-center rounded-full border border-line bg-surface/80 text-muted shadow-sm backdrop-blur transition-colors hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current motion-reduce:transition-none">
              {control.icon}
            </button>
          )}
        </div>
        <figcaption className="mt-3 max-w-2xl text-center text-[13px] leading-relaxed text-muted">{meta.caption}</figcaption>
      </figure>
    </section>
  )
}
