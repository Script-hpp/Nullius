// Smooth-scroll to a section of the current page without touching the URL hash, so the same link works on every click
// (a hash link does nothing when the hash is already set, and routes through the top of the page first).
type Doc = { getElementById(id: string): { scrollIntoView(o?: { behavior?: 'auto' | 'smooth'; block?: 'start' }): void } | null }

const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

// Returns true when the section exists on this page and the scroll was started; false means: let the link navigate.
export function scrollToSection(id: string, doc: Doc = document, reduced: boolean = reducedMotion()): boolean {
  const el = doc.getElementById(id)
  if (!el) return false
  el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' })
  return true
}

// After a jump, mark the target for a moment (ring and tint, see .nx-flash in index.css). The class is removed again, so a second jump
// to the same target flashes again; under reduced motion the ring is static and disappears after the same time.
export function flashSection(el: { classList: { add(c: string): void; remove(c: string): void }; offsetWidth?: number } | null, ms = 1400): void {
  if (!el) return
  el.classList.remove('nx-flash')
  void el.offsetWidth   // restart the animation when the class is added again
  el.classList.add('nx-flash')
  window.setTimeout(() => el.classList.remove('nx-flash'), ms)
}
