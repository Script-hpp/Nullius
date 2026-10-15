import { useEffect, useRef, useState } from 'react'
import { Clapperboard, X } from 'lucide-react'
import { durationLabel, FILM_SECONDS, nextFocusIndex } from '../media'

// Quiet secondary hero action: opens the finished film in a modal. Sound only after the viewer presses play (no autoplay, preload none).
export function FilmButton() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)} aria-haspopup="dialog"
        className="inline-flex cursor-pointer items-center gap-2 border-0 bg-transparent p-1 text-sm text-fg-4 transition-colors hover:text-fg">
        <Clapperboard className="h-4 w-4" aria-hidden />Watch the film ({durationLabel(FILM_SECONDS)})
      </button>
      {open && <FilmModal onClose={() => setOpen(false)} />}
    </>
  )
}

function FilmModal({ onClose }: { onClose: () => void }) {
  const box = useRef<HTMLDivElement>(null)
  const closeBtn = useRef<HTMLButtonElement>(null)
  const video = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    closeBtn.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return }
      if (e.key !== 'Tab' || !box.current) return
      const items = Array.from(box.current.querySelectorAll<HTMLElement>('button, video[controls], [tabindex]:not([tabindex="-1"])'))
      const next = nextFocusIndex(items.length, items.indexOf(document.activeElement as HTMLElement), e.shiftKey)
      if (next !== null) { e.preventDefault(); items[next].focus() }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
      video.current?.pause()
      opener?.focus?.()
    }
  }, [onClose])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={box} role="dialog" aria-modal="true" aria-label="Nullius film" className="film-frame w-full max-w-5xl overflow-hidden rounded-xl border border-line bg-surface shadow-2xl">
        <div className="flex items-center justify-between px-4 py-2.5">
          <span className="text-[13px] text-fg-4">Nullius film · {durationLabel(FILM_SECONDS)} · music, no narration</span>
          <button ref={closeBtn} type="button" onClick={onClose} aria-label="Close film" className="cursor-pointer rounded-md border-0 bg-transparent p-1 text-fg-4 hover:text-fg">
            <X className="h-5 w-5" aria-hidden />
          </button>
        </div>
        <video ref={video} controls preload="none" playsInline poster="media/nullius-film-poster.webp" src="media/nullius-film.mp4" className="block aspect-video w-full bg-black" />
      </div>
    </div>
  )
}
