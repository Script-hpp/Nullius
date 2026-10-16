import type { CSSProperties, ReactNode } from 'react'
import type { Pal } from './palette'

// Building blocks for the video stage. Every value is a pure function of the shot time t (ms), so a frame can be rendered at any t.

export const W = 1920
export const H = 1080

export const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x))
// 0..1 while t runs from a to a + d
export const ramp = (t: number, a: number, d: number) => clamp((t - a) / d)
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p

export const easeOut = (p: number) => 1 - Math.pow(1 - p, 3)
export const easeInOut = (p: number) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2)
export const easeOutExpo = (p: number) => (p >= 1 ? 1 : 1 - Math.pow(2, -10 * p))
// a soft landing with one small overshoot, for dots that drop onto an axis
export const easeOutBack = (p: number) => { const c = 1.4; return 1 + (c + 1) * Math.pow(p - 1, 3) + c * Math.pow(p - 1, 2) }

// Keyframes [time, ...values] through a monotone cubic (Fritsch–Carlson): velocity is continuous across keys, a key that is a local
// extreme or the end of a hold stops the motion there, and nothing overshoots. Per-segment easing would stop at every key and jerk.
// ends: 'stop' comes to rest at the first and last key (a camera); 'through' keeps the end segments' speed (a time map).
export function spline(keys: number[][], t: number, ends: 'stop' | 'through' = 'stop'): number[] {
  const n = keys.length
  if (t <= keys[0][0]) return keys[0].slice(1)
  if (t >= keys[n - 1][0]) return keys[n - 1].slice(1)
  let i = 1
  while (keys[i][0] < t) i++
  const a = keys[i - 1], b = keys[i], h = b[0] - a[0], u = (t - a[0]) / h
  const slope = (k: number, j: number) => {   // tangent at key k for channel j
    const d0 = k > 0 ? (keys[k][j] - keys[k - 1][j]) / (keys[k][0] - keys[k - 1][0]) : null
    const d1 = k < n - 1 ? (keys[k + 1][j] - keys[k][j]) / (keys[k + 1][0] - keys[k][0]) : null
    if (d0 === null || d1 === null) return ends === 'stop' ? 0 : (d0 ?? d1 ?? 0)
    if (d0 * d1 <= 0) return 0
    return (2 * d0 * d1) / (d0 + d1)   // harmonic mean: monotone, no overshoot
  }
  const h00 = 2 * u ** 3 - 3 * u ** 2 + 1, h10 = u ** 3 - 2 * u ** 2 + u, h01 = -2 * u ** 3 + 3 * u ** 2, h11 = u ** 3 - u ** 2
  return a.slice(1).map((_, j0) => {
    const j = j0 + 1
    return h00 * a[j] + h10 * h * slope(i - 1, j) + h01 * b[j] + h11 * h * slope(i, j)
  })
}

// Fade in while rising by `dist` px.
export function rise(t: number, at: number, d = 700, dist = 24): CSSProperties {
  const p = easeOut(ramp(t, at, d))
  // its own compositor layer: the browser then moves the finished text smoothly instead of snapping it to whole pixels on every repaint
  return { opacity: p, transform: `translateY(${(1 - p) * dist}px)`, willChange: 'transform' }
}
export const fade = (t: number, at: number, d = 600): CSSProperties => ({ opacity: easeInOut(ramp(t, at, d)) })
// visible from `at`, gone after `until`
export const window_ = (t: number, at: number, until: number, d = 500) => easeInOut(ramp(t, at, d)) * (1 - easeInOut(ramp(t, until, d)))

// Characters of `text` shown after typing from `at` at `cps` characters per second.
export const typed = (text: string, t: number, at: number, cps = 40) => text.slice(0, Math.max(0, Math.floor(((t - at) / 1000) * cps)))

export { PAL } from './palette'
export type { Pal, Theme } from './palette'

export const GROTESK = "'Inter Tight Variable', 'Inter Tight', Inter, system-ui, sans-serif"
export const SERIF = "'Instrument Serif', Georgia, serif"
export const MONO = "'JetBrains Mono', ui-monospace, monospace"

// Small uppercase label in the style of the web app.
export function Kicker({ children, color, style }: { children: ReactNode; color: string; style?: CSSProperties }) {
  return <div style={{ fontFamily: GROTESK, fontSize: 20, fontWeight: 600, letterSpacing: '0.2em', textTransform: 'uppercase', color, ...style }}>{children}</div>
}

// The line at the bottom left that says where the data comes from. Every built shot carries one.
export function Source({ t, at = 600, pal, children }: { t: number; at?: number; pal: Pal; children: ReactNode }) {
  return (
    <div style={{ position: 'absolute', left: 96, bottom: 64, fontFamily: GROTESK, fontSize: 19, color: pal.muted, letterSpacing: '0.01em', ...fade(t, at, 800) }}>
      {children}
    </div>
  )
}

// The shot's framing. It is a fixed scale, not a slow push-in: scaling a layer that holds text by a fraction of a percent per frame makes the
// browser re-rasterise every glyph at a new sub-pixel position each frame, and the text visibly shakes. A fixed scale keeps glyphs still.
// `from`/`to` are kept so shots state their framing; the mid value is used.
export function Camera({ from = 1, to = 1.03, children }: { t?: number; dur?: number; from?: number; to?: number; children: ReactNode }) {
  return <div style={{ position: 'absolute', inset: 0, transform: `scale(${(from + to) / 2})`, transformOrigin: '50% 50%', backfaceVisibility: 'hidden' }}>{children}</div>
}

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`
