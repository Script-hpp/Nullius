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
