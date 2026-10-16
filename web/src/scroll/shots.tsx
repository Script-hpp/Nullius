import type { ReactNode } from 'react'
import { Check, X as Cross } from 'lucide-react'
import { translateClaim } from '../labels'
import { Camera, MONO, SERIF, Source, W, clamp, easeInOut, easeOut, easeOutBack, fade, lerp, pct, ramp, rise, typed, type Pal, type Theme } from './kit'
import type { StageData } from './data'
import { SCENES, type SceneId } from './scenes'

// The two film shots that play on the start page, scrubbed by scrolling. Ported from the film's stage (web/src/stage on branch Script-hpp/film,
// shots 01-opening, 04-placebo). Every shot is a pure function of its time t in ms, and every number and claim on it is read from
// the exported film run (data.ts). Changes against the film: the opening does not dissolve into the app window at its end (nothing follows it
// here), the placebo is the film's shorter 9 s version, duration and poster come from scenes.ts.

export type Shot = { id: SceneId; duration: number; theme: Theme; poster: number; render: (t: number, d: StageData, pal: Pal) => ReactNode }


const VERDICT: Record<string, { label: string; tone: 'green' | 'amber' | 'red' }> = {
  reproduziert: { label: 'Reproduced', tone: 'green' },
  teilweise: { label: 'Partially reproduced', tone: 'amber' },
  'nicht reproduziert': { label: 'Not reproduced', tone: 'red' },
}
const day = (iso: string) => (iso ? new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : '')
const quoteOf = (text: string) => `“${translateClaim(text).replace(/\s*\(.*\)\s*$/, '')}.”`

// ---------- 01 Opening: one claim, a wall of claims, back to the one claim, measured (21 s) ----------
// World coordinates: the featured claim sits at (0, 0); the camera scales the world around it.
const WALL_POS = [
  [-1350, -1060], [1250, -1000],
  [-1750, -700], [450, -680],
  [-1050, -340], [1500, -350],
  [-1650, 350], [1150, 330],
  [-650, 690], [1750, 700],
  [-1900, 1000], [400, 1040],
]
// the measurement under the claim: one dot per seed, each falls on its own beat (video/score.py plays a note where each dot stops)
const OPEN_DOTS = { at: 12200, every: 230, fall: 520 }
const opening: Shot = {
  id: 'opening', ...SCENES.opening,
  render: (t, d, pal) => {
    const [main, ...rest] = d.wall
    // camera: hold close, pull far back, keep drifting back under the riser (faster and faster), rush back in on the hit
    const back = easeInOut(ramp(t, 3000, 3800))
    const drift = Math.pow(ramp(t, 6800, 3600), 2)          // starts at rest, fastest right before the hit
    const rush = Math.pow(easeInOut(ramp(t, 10400, 1000)), 1.6)
    const far = lerp(0.36, 0.27, drift)
    const s = lerp(lerp(lerp(1, 1.035, easeInOut(ramp(t, 0, 3000))), far, back), 1, rush)
    const cy = 470   // screen y of the featured claim
    const othersOn = (k: number) => easeOut(ramp(t, 3600 + k * 140, 900)) * (1 - easeInOut(ramp(t, 10300, 700)))
    const blurOthers = 8 * easeInOut(ramp(t, 10300, 700))
    // measurement under the claim (after the rush)
    const ax = { w: 1300, y: cy + 150 }
    const lo = 0.76, hi = 0.86
    const xv = (v: number) => W / 2 - ax.w / 2 + ((v - lo) / (hi - lo)) * ax.w
    const axisP = easeInOut(ramp(t, 11300, 600))
    const spots = landing(d.seeds)
    const lastLand = OPEN_DOTS.at + (spots.length - 1) * OPEN_DOTS.every + OPEN_DOTS.fall
    const outro = easeInOut(ramp(t, 17300, 900))
    return (
      <>
        {/* the world: one layer the camera scales, so every claim moves on subpixels and nothing snaps */}
        <div style={{ position: 'absolute', left: 0, top: 0, width: 0, height: 0, transformOrigin: '0 0', transform: `translate(${W / 2}px, ${cy}px) scale(${s})`, willChange: 'transform' }}>
          {rest.slice(0, WALL_POS.length).map((c, k) => {   // one claim per place, never two on one spot
            const [x, y] = WALL_POS[k]
            const o = othersOn(k)
            if (o <= 0) return null
            return (
              <div key={c.text} style={{ position: 'absolute', left: x, top: y, transform: 'translate(-50%, -50%)', whiteSpace: 'nowrap', textAlign: 'center', opacity: o * 0.7, filter: blurOthers ? `blur(${blurOthers}px)` : undefined }}>
                <div style={{ fontFamily: SERIF, fontStyle: 'italic', fontSize: 96, color: pal.ink2 }}>{`\u201C${c.text}.\u201D`}</div>
                <div style={{ fontSize: 30, color: pal.muted, marginTop: 6 }}>{c.byline}</div>
              </div>
            )
          })}
          {/* the featured claim */}
          <div style={{ position: 'absolute', left: 0, top: 0, transform: `translate(-50%, -50%) translateY(${-outro * 60}px)`, whiteSpace: 'nowrap', textAlign: 'center', opacity: 1 - outro }}>
            <div style={{ fontFamily: SERIF, fontStyle: 'italic', fontSize: 96, color: pal.ink }}>{typed(`“${main.text}.”`, t, 250, 26)}</div>
            <div style={{ fontSize: 30, color: pal.muted, marginTop: 6, opacity: easeInOut(ramp(t, 1900, 600)) }}>{main.byline}</div>
          </div>
        </div>
        {/* measured: the axis, then one seed after the other, then the mean */}
        <div style={{ opacity: 1 - outro, transform: `translateY(${-outro * 60}px)` }}>
          <div style={{ position: 'absolute', left: xv(d.claim.target - d.claim.tolerance), width: xv(d.claim.target + d.claim.tolerance) - xv(d.claim.target - d.claim.tolerance), top: ax.y - 46, height: 60, background: pal.greenSoft, borderRadius: 5, opacity: easeInOut(ramp(t, 11800, 600)) }} />
          <div style={{ position: 'absolute', left: W / 2 - ax.w / 2, width: ax.w, top: ax.y, height: 2, background: pal.ink2, transformOrigin: '50% 50%', transform: `scaleX(${axisP})` }} />
          {spots.map((sp, k) => {
            const a = OPEN_DOTS.at + k * OPEN_DOTS.every
            const p = ramp(t, a, OPEN_DOTS.fall)
            if (p <= 0) return null
            const y = lerp(ax.y - 48, ax.y - 9 - sp.n * 20, easeInOut(p))   // falls (below the byline) and stops, no bounce
            return <div key={sp.seed} style={{ position: 'absolute', left: 0, top: 0, width: 18, height: 18, borderRadius: 99, background: pal.ink, opacity: 0.85 * clamp(p * 3), transform: `translate(${xv(sp.value) - 9}px, ${y}px)`, willChange: 'transform' }} />
          })}
          <div style={{ position: 'absolute', left: xv(d.verdict.measured_mean) - 1.5, top: ax.y - 56, width: 3, height: 70, background: pal.green, opacity: easeInOut(ramp(t, lastLand + 300, 400)) }} />
          <div style={{ position: 'absolute', left: 0, right: 0, top: ax.y + 34, textAlign: 'center', fontFamily: MONO, fontSize: 30, color: pal.green, ...rise(t, lastLand + 450, 600, 8) }}>
            measured {pct(d.verdict.measured_mean)} &plusmn; {(d.verdict.measured_std * 100).toFixed(1)}
          </div>
        </div>
        {/* captions carry the voice-over word for word (video/voice.py), at the time it is spoken (video/score.py, VO) */}
        <Caption t={t} at={3700} until={6200} pal={pal}>Papers are full of claims like this one.</Caption>
        <Caption t={t} at={6400} until={9800} pal={pal}>Checking a single one by hand takes days.</Caption>
        <Caption t={t} at={12000} until={17300} pal={pal}>Nullius rebuilds the experiment, runs it {word(spots.length).toLowerCase()} times, and measures.</Caption>
        {/* the product */}
        <div style={{ position: 'absolute', left: 0, right: 0, top: 380, display: 'flex', justifyContent: 'center', alignItems: 'baseline', gap: 22, ...rise(t, 17900, 900, 24) }}>
          <span style={{ fontSize: 150, fontWeight: 600, letterSpacing: '-0.05em' }}>nullius</span>
          <span style={{ fontFamily: SERIF, fontStyle: 'italic', fontSize: 158 }}>in verba</span>
        </div>
        <div style={{ position: 'absolute', left: 0, right: 0, top: 610, textAlign: 'center', fontSize: 48, fontWeight: 500, letterSpacing: '-0.03em', color: pal.ink2, ...rise(t, 18500, 900, 16) }}>
          Papers make claims. <span style={{ color: pal.ink }}>We measure them.</span>
        </div>
      </>
    )
  },
}

// ---------- 08 and 09: the measurement against the claim ----------
type MeasureProps = {
  d: StageData; pal: Pal
  lo: number; hi: number; target: number; tol: number
  dots: number            // how many seeds have landed (fractional: the last one is falling)
  axis: number; claimOn: number; bandOn: number; meanOn: number
  quote: string; byline: ReactNode; quoteOpacity: number
  verdict?: { label: string; tone: 'green' | 'amber' | 'red'; on: number; sub: string; subOn: number }
  oldVerdict?: { label: string; tone: 'green' | 'amber' | 'red'; on: number }
  axisCenter?: number      // the axis grows outward from this value instead of from the left
  claimLabelOn?: number    // the label can arrive later than the line (04-number flies it in)
  landedN?: number         // seeds counted in the running mean when the dots are drawn outside
  overlay?: ReactNode      // drawn between the band and the verdict (04-number: branches and their dots)
}

// Where each seed lands on the axis, in the order the seeds finished; equal values stack upward.
export function landing(seeds: StageData['seeds']) {
  const stack = new Map<string, number>()
  return seeds.map((s) => {
    const key = s.value.toFixed(3)
    const n = stack.get(key) ?? 0
    stack.set(key, n + 1)
    return { ...s, n }
  })
}
const AX = { x0: 170, x1: 1750, y: 600 }
const xOn = (v: number, lo: number, hi: number) => AX.x0 + ((v - lo) / (hi - lo)) * (AX.x1 - AX.x0)

function Measure({ d, pal, lo, hi, target, tol, dots, axis, claimOn, bandOn, meanOn, quote, byline, quoteOpacity, verdict, oldVerdict, axisCenter, claimLabelOn, landedN, overlay }: MeasureProps) {
  const { x0, x1, y } = AX
  const X = (v: number) => xOn(v, lo, hi)
  const order = landing(d.seeds)
  const landed = order.slice(0, landedN ?? Math.floor(dots))
  const mean = landed.length ? landed.reduce((s, x) => s + x.value, 0) / landed.length : null
  const fullMean = d.verdict.measured_mean
  const ticks: number[] = []
  for (let v = Math.ceil(lo * 100 - 1e-9); v <= Math.floor(hi * 100 + 1e-9); v++) ticks.push(v / 100)
  const tone = (k: 'green' | 'amber' | 'red') => (k === 'green' ? pal.green : k === 'red' ? pal.red : pal.amber)
  const axL = axisCenter === undefined ? x0 : lerp(X(axisCenter), x0, axis)
  const axR = axisCenter === undefined ? x0 + (x1 - x0) * axis : lerp(X(axisCenter), x1, axis)
  return (
    <>
      <div style={{ position: 'absolute', left: 120, top: 110, opacity: quoteOpacity }}>
        <div style={{ fontFamily: SERIF, fontStyle: 'italic', fontSize: 64, lineHeight: 1.05, color: pal.ink }}>{quote}</div>
        <div style={{ marginTop: 14, fontSize: 22, color: pal.muted }}>{byline}</div>
      </div>
      <div style={{ position: 'absolute', right: 120, top: 112, textAlign: 'right', opacity: landed.length ? 1 : 0 }}>
        <div style={{ fontSize: 20, fontWeight: 600, letterSpacing: '0.2em', textTransform: 'uppercase', color: pal.muted }}>Measured, mean of {landed.length} {landed.length === 1 ? 'seed' : 'seeds'}</div>
        <div style={{ fontFamily: MONO, fontSize: 92, color: pal.ink, marginTop: 6 }}>{mean === null ? '–' : pct(mean)}</div>
      </div>
      {/* band = claim ± tolerance */}
      <div style={{ position: 'absolute', left: X(target - tol), width: X(target + tol) - X(target - tol), top: 470, height: 160, background: pal.greenSoft, borderRadius: 6, opacity: bandOn }} />
      <div style={{ position: 'absolute', left: X(target + tol) + 18, top: 478, width: 240, fontSize: 18, lineHeight: 1.35, color: pal.green, opacity: bandOn }}>tolerance &plusmn; {(tol * 100).toFixed(1)} pp,<br />fixed before the run</div>
      {/* axis */}
      <div style={{ position: 'absolute', left: axL, top: y, height: 2, width: axR - axL, background: pal.ink2 }} />
      {ticks.map((v) => (
        <div key={v} style={{ position: 'absolute', left: X(v), top: y, opacity: X(v) >= axL - 1 && X(v) <= axR + 1 ? 1 : 0 }}>
          <div style={{ width: 2, height: 12, background: pal.ink2, marginLeft: -1 }} />
          <div style={{ position: 'absolute', top: 24, left: -40, width: 80, textAlign: 'center', fontFamily: MONO, fontSize: 18, color: pal.muted }}>{Math.round(v * 100)}%</div>
        </div>
      ))}
      {/* claim */}
      <div style={{ position: 'absolute', left: X(target) - 1, top: 440, height: 200, borderLeft: `2px dashed ${pal.ink}`, opacity: claimOn }} />
      <div style={{ position: 'absolute', left: X(target) - 150, width: 300, top: 400, textAlign: 'center', fontFamily: MONO, fontSize: 21, color: pal.ink, opacity: claimLabelOn ?? claimOn }}>claimed {pct(target)}</div>
      {/* seeds */}
      {order.map((s, k) => {
        if (k >= Math.ceil(dots)) return null
        const p = k < Math.floor(dots) ? 1 : dots - Math.floor(dots)
        const yy = lerp(250, y - s.n * 30, easeOutBack(clamp(p)))
        return <div key={s.seed} style={{ position: 'absolute', left: X(s.value) - 13, top: yy - 13, width: 26, height: 26, borderRadius: 99, background: pal.ink, opacity: 0.85 * clamp(p * 3) }} />
      })}
      {overlay}
      {/* mean */}
      <div style={{ position: 'absolute', left: X(fullMean) - 1.5, top: 500, height: 140, width: 3, background: pal.green, opacity: meanOn }} />
      <div style={{ position: 'absolute', left: X(fullMean) - 200, width: 400, top: 690, textAlign: 'center', fontFamily: MONO, fontSize: 22, color: pal.green, opacity: meanOn }}>mean {pct(fullMean)} &plusmn; {(d.verdict.measured_std * 100).toFixed(1)}</div>
      {oldVerdict && (
        <div style={{ position: 'absolute', left: 0, right: 0, top: 790, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 20, fontSize: 92, fontWeight: 650, letterSpacing: '-0.04em', color: tone(oldVerdict.tone), opacity: oldVerdict.on, transform: `translateY(${(1 - oldVerdict.on) * -20}px)` }}>
          <Check size={78} strokeWidth={2.6} /> {oldVerdict.label}
        </div>
      )}
      {verdict && (
        <>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 790, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 20, fontSize: 92, fontWeight: 650, letterSpacing: '-0.04em', color: tone(verdict.tone), opacity: verdict.on, transform: `translateY(${(1 - verdict.on) * 24}px)` }}>
            {verdict.tone === 'green' ? <Check size={78} strokeWidth={2.6} /> : <Cross size={78} strokeWidth={2.6} />} {verdict.label}
          </div>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 920, textAlign: 'center', fontSize: 26, color: pal.ink2, opacity: verdict.subOn }}>{verdict.sub}</div>
        </>
      )}
    </>
  )
}

const RANGE = { lo: 0.76, hi: 0.86 }
const verdictOf = (u: string) => VERDICT[u] ?? { label: u, tone: 'amber' as const }


// ---------- text as narrator (the film has music, no voice) ----------
// A line that comes into focus (blur to sharp), holds, and leaves. One idea per line, readable for at least 2.5 s.
function Beat({ t, at, until, children, size = 72, color, top, weight = 600 }: { t: number; at: number; until: number; children: ReactNode; size?: number; color: string; top: number; weight?: number }) {
  const i = easeOut(ramp(t, at, 900)), o = easeInOut(ramp(t, until, 600))
  if (i <= 0 || o >= 1) return null
  return (
    <div style={{ position: 'absolute', left: 120, right: 120, top, textAlign: 'center', fontSize: size, fontWeight: weight, letterSpacing: '-0.035em', lineHeight: 1.1, color, opacity: i * (1 - o), filter: `blur(${(1 - i) * 10 + o * 6}px)`, transform: `translateY(${(1 - i) * 12}px)` }}>
      {children}
    </div>
  )
}
const Caption = ({ t, at, until, children, pal }: { t: number; at: number; until: number; children: ReactNode; pal: Pal }) => (
  <Beat t={t} at={at} until={until} size={38} weight={500} color={pal.ink2} top={880}>{children}</Beat>
)
const WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve']
const word = (n: number) => WORDS[n] ?? String(n)


const PLACEBO = { verdictOut: 200, swap: 900, slide: 1100, slideDur: 1800, reject: 3300 }
const placebo: Shot = {
  id: 'placebo', ...SCENES.placebo,
  render: (film, d, pal) => {
    const t = film * (12000 / SCENES.placebo.duration)   // the shot is authored for 12 s; the film plays it in 9 s
    if (!d.placebo || !d.placeboVerdict) return <div style={{ padding: 96, fontSize: 40 }}>No placebo claim in the featured run.</div>
    const real = d.claim.target, fake = d.placebo.target
    const P = PLACEBO
    const slide = easeInOut(ramp(t, P.slide, P.slideDur))
    const target = lerp(real, fake, slide)
    const hi = lerp(RANGE.hi, Math.max(RANGE.hi, Math.ceil((fake + d.placebo.tolerance + 0.01) * 100) / 100), easeInOut(ramp(t, P.swap - 200, 1200)))
    const swap = ramp(t, P.swap, 700)
    const oldV = verdictOf(d.verdict.urteil), newV = verdictOf(d.placeboVerdict.urteil)
    const same = Math.abs(d.placeboVerdict.measured_mean - d.verdict.measured_mean) < 1e-9
    return (
      <Camera t={t} dur={12000} from={1.02} to={1.035}>
        <Measure d={d} pal={pal} lo={RANGE.lo} hi={hi} target={target} tol={d.placebo.tolerance}
          dots={d.seeds.length} axis={1} claimOn={1} bandOn={1} meanOn={1}
          quote={swap < 0.5 ? quoteOf(d.claim.text) : typed(quoteOf(d.placebo.text), t, P.swap + 350, 45)}
          byline={swap < 0.5 ? <>{d.byline} &middot; {d.claim.target_evidence?.table ?? d.claim.source}</> : <span style={{ color: pal.red }}>A wrong claim, planted on purpose.</span>}
          quoteOpacity={swap < 0.5 ? 1 - swap * 2 : 1}
          oldVerdict={{ ...oldV, on: 1 - easeInOut(ramp(t, P.verdictOut, 600)) }}
          verdict={{ label: newV.label, tone: newV.tone, on: easeOut(ramp(t, P.reject, 600)), sub: same ? 'Same measurements. Wrong claim. Rejected.' : 'Wrong claim. Rejected.', subOn: fade(t, P.reject + 800, 600).opacity as number }} />
        {/* the first frame equals the last frame of 04-number: its subtitle and source line fade out first */}
        <div style={{ position: 'absolute', left: 0, right: 0, top: 920, textAlign: 'center', fontSize: 26, color: pal.ink2, opacity: 1 - easeInOut(ramp(t, P.verdictOut, 600)) }}>with our setup &middot; target value reached &middot; method fidelity not checked</div>
        <div style={{ opacity: 1 - ramp(t, 300, 400) }}><Source t={1e9} pal={pal}>Real run, {day(d.retrievedAt)}: {d.seeds.length} seeds in Nebius Sandboxes, scored by our harness. Dots in the order the seeds finished.</Source></div>
      </Camera>
    )
  },
}

export const SHOTS: Record<SceneId, Shot> = { opening, placebo }
