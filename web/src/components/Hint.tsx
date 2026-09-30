import { useRef, useState } from 'react'
import { Info } from 'lucide-react'

// Info icon with a short explanation: hover (mouse), tap (touch) and keyboard focus; Escape or a second tap closes it.
export function Hint({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ left: 0, width: 240 })
  const ref = useRef<HTMLButtonElement>(null)
  // Keep the popover inside the viewport (16 px margin), also on narrow phones.
  const show = (on: boolean) => {
    if (on && ref.current) {
      const r = ref.current.getBoundingClientRect()
      const width = Math.min(240, window.innerWidth - 32)
      setPos({ width, left: Math.min(Math.max(r.left, 16), window.innerWidth - 16 - width) - r.left })
    }
    setOpen(on)
  }
  return (
    <span className="relative inline-flex align-middle normal-case tracking-normal font-normal">
      <button ref={ref} type="button" aria-label={text} aria-expanded={open}
        onClick={(e) => { e.stopPropagation(); show(!open) }}
        onPointerEnter={(e) => { if (e.pointerType === 'mouse') show(true) }}
        onPointerLeave={(e) => { if (e.pointerType === 'mouse') show(false) }}
        onFocus={(e) => { if (e.currentTarget.matches(':focus-visible')) show(true) }}
        onBlur={() => show(false)}
        onKeyDown={(e) => { if (e.key === 'Escape') show(false) }}
        className="p-0.5 text-muted hover:text-fg-3 focus-visible:text-fg-2 focus-visible:outline focus-visible:outline-1 focus-visible:outline-zinc-500 rounded cursor-pointer">
        <Info className="w-3.5 h-3.5" />
      </button>
      {open && (
        <span role="tooltip" style={{ left: pos.left, width: pos.width }}
          className="absolute top-full z-20 mt-1.5 rounded-lg border border-line bg-raised p-3 text-[11px] leading-relaxed text-fg-3 shadow-lg">
          {text}
        </span>
      )}
    </span>
  )
}
