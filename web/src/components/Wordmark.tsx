import { Mark } from './Mark'

// "nullius in verba": Inter Tight wordmark with the Instrument Serif italic accent (same on the landing page and in the app bar), with the mark in front.
export function Wordmark({ className = '' }: { className?: string }) {
  return (
    <a href="#/" className={`font-grotesk inline-flex items-center gap-[9px] text-[19px] font-semibold tracking-[-0.04em] text-fg no-underline ${className}`} aria-label="Nullius in verba, start page">
      <Mark size={16} className="shrink-0" />
      <span className="whitespace-nowrap">nullius <span className="font-serif-italic text-[22px] font-normal tracking-normal">in verba</span></span>
    </a>
  )
}
