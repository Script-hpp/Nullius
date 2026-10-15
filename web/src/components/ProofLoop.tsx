import { useEffect, useRef, useState } from 'react'
import { prefersReducedMotion } from './hero/config'
import { loopShouldPlay } from '../media'

// Silent 10 s clip from the film: the same ten measurements against the true claim and a planted false claim.
// Not loaded until near the viewport, paused when out of view, poster only under reduced motion.
export function ProofLoop() {
  const wrap = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [reduced] = useState(prefersReducedMotion)

  useEffect(() => {
    const el = wrap.current
    if (!el || reduced || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([entry]) => {
      const visible = entry.isIntersecting
      const v = video.current
      if (!v) return
      if (visible && !v.getAttribute('src')) v.src = 'media/proof-loop.mp4'   // first time on screen: only now does the file load
      if (loopShouldPlay(visible, reduced)) void v.play().catch(() => { /* autoplay refused: poster stays */ })
      else v.pause()
    }, { threshold: 0.25 })
    io.observe(el)
    return () => io.disconnect()
  }, [reduced])

  return (
    <figure className="m-0 mt-10">
      <div ref={wrap} className="film-frame overflow-hidden rounded-xl border border-line bg-surface">
        {reduced ? (
          <img src="media/proof-loop-poster.webp" alt="Same ten measurements, planted claim of 88.0 percent: Not reproduced" className="block aspect-video w-full object-cover" loading="lazy" />
        ) : (
          <video ref={video} muted loop playsInline preload="none" poster="media/proof-loop-poster.webp" aria-label="Same ten measurements against the true claim, then against a planted false claim"
            className="block aspect-video w-full object-cover" />
        )}
      </div>
      <figcaption className="mt-3 text-[12px] leading-relaxed text-muted">From the film: the same ten measurements, checked against the real claim and against a false claim we planted. The false one is rejected.</figcaption>
    </figure>
  )
}
