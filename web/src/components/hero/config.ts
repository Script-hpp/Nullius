import { SCRIPT_TEXTS } from './scriptData'

// Which lettering the hero writes: 'motto' = "Nullius in verba", 'word' = "take nobody's word".
export const HERO_TEXT: keyof typeof SCRIPT_TEXTS = 'motto'
export const WRITE_MS = 3200   // one pass, about three seconds

export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)   // soft start and end over the whole word

export const prefersReducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
