import { useRef, useState } from 'react'
import { GLOSSARY, splitTerms } from '../glossary'

// A word with a dotted underline and a plain-language explanation: hover (mouse), tap (touch), keyboard focus; Escape closes.
export function Term({ term, children }: { term: string; children: string }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ left: 0, width: 260 })
  const ref = useRef<HTMLButtonElement>(null)
  const show = (on: boolean) => {
    if (on && ref.current) {
      const r = ref.current.getBoundingClientRect()
      const width = Math.min(260, window.innerWidth - 32)
      setPos({ width, left: Math.min(Math.max(r.left, 16), window.innerWidth - 16 - width) - r.left })
    }
    setOpen(on)
  }
  return (
    <span className="relative inline">
      <button ref={ref} type="button" aria-expanded={open} aria-label={`${children}: ${GLOSSARY[term]}`}
        onClick={(e) => { e.stopPropagation(); show(!open) }}
        onPointerEnter={(e) => { if (e.pointerType === 'mouse') show(true) }}
        onPointerLeave={(e) => { if (e.pointerType === 'mouse') show(false) }}
        onFocus={(e) => { if (e.currentTarget.matches(':focus-visible')) show(true) }}
        onBlur={() => show(false)}
        onKeyDown={(e) => { if (e.key === 'Escape') show(false) }}
        className="p-0 m-0 bg-transparent border-0 font-inherit text-inherit cursor-help underline decoration-dotted decoration-fg-4 underline-offset-2 hover:text-fg focus-visible:outline focus-visible:outline-1 focus-visible:outline-line-stronger rounded-sm">
        {children}
      </button>
      {open && (
        <span role="tooltip" style={{ left: pos.left, width: pos.width }}
          className="absolute top-full z-20 mt-1.5 rounded-lg border border-line bg-raised p-3 text-[11px] leading-relaxed text-fg-3 shadow-lg normal-case tracking-normal font-normal">
          {GLOSSARY[term]}
        </span>
      )}
    </span>
  )
}

// Text in which every glossary term gets a tooltip.
export function Glossed({ text }: { text: string }) {
  return (
    <>
      {splitTerms(text).map((p, i) => (p.term ? <Term key={i} term={p.term}>{p.text}</Term> : <span key={i}>{p.text}</span>))}
    </>
  )
}
