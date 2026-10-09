import { SCRIPT_TEXTS } from './scriptData'

// Which lettering the hero writes: 'motto' = "Nullius in verba", 'word' = "take nobody's word".
export const HERO_TEXT: keyof typeof SCRIPT_TEXTS = 'motto'
export const WRITE_MS = 700   // one pass, well under a second; the page never waits for it

export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)   // soft start and end over the whole word

export const prefersReducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

// The lettering is written once per browser session; a later visit (or reload) shows it finished at once.
const SEEN_KEY = 'nullius-hero-seen'
export const heroSeen = () => { try { return sessionStorage.getItem(SEEN_KEY) === '1' } catch { return false } }
export const markHeroSeen = () => { try { sessionStorage.setItem(SEEN_KEY, '1') } catch { /* storage blocked: animate every time */ } }
