import { Mark } from './Mark'

// "nullius in verba": Inter Tight wordmark with the Instrument Serif italic accent (same on the landing page and in the app bar), with the mark in front.
export function Wordmark({ className = '' }: { className?: string }) {
  return (
    <a href="#/" className={`font-grotesk inline-flex items-center gap-2 text-[19px] font-semibold tracking-[-0.04em] text-fg no-underline ${className}`} aria-label="Nullius in verba, start page">
      <Mark size={20} className="shrink-0" />
      <WordmarkText />
    </a>
  )
}

// The wordmark lettering alone, for use inside a sentence (inherits size and colour; not a link).
export function WordmarkText({ className = '' }: { className?: string }) {
  return <span className={`font-grotesk whitespace-nowrap font-semibold tracking-[-0.04em] ${className}`}>nullius <span className="font-serif-italic text-[1.16em] font-normal tracking-normal">in verba</span></span>
}
