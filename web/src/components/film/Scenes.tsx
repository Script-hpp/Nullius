import { useEffect, useState, type ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Check, Search, X } from 'lucide-react'
import { ResultBand } from '../ResultChart'
import { SENTENCE, addressBar, branchX, branches, claimEvidence, codeLines, diffFlash, dossierAt, measurements, planBullets, sourceDomains, tavilySentence, typed, type RepairScene, type SceneId } from '../../film'
import { TONE_TEXT_VAR, TONE_VAR, type Tone } from '../../labels'
import type { Event, Json } from '../../data'

export type SceneData = {
  events: Event[]; dossier: Json; claim: Json; title: string; tone: Tone; resultLabel: string; sentence: string; fidelityUnchecked: boolean; overall?: string | null
  fmt: (x: number) => string; fmtSpread: (x: number) => string; plan: string | null; firstCode: string | null; firstHash: string | null
  repair: RepairScene | null; codeLoader: (hash: string) => Promise<string | null>
}

// helpers: a value that rises from 0 to 1 while p runs from a to b
const ramp = (p: number, a: number, b: number) => Math.min(1, Math.max(0, (p - a) / (b - a)))

function Sentence({ children }: { children: ReactNode }) {
  return <p className="font-grotesk m-0 px-[4cqw] text-[clamp(1.1rem,3.3cqw,2.3rem)] font-semibold leading-[1.12] tracking-[-0.03em] text-fg @xl:px-[3cqw]">{children}</p>
}
const body = 'min-h-0 flex-1 px-[4cqw] pb-[3cqw] pt-[2.2cqw] @xl:px-[3cqw] @xl:pb-[2.4cqw]'
const small = 'text-[clamp(10px,1.45cqw,13px)]'
const mono = 'font-mono text-[clamp(9.5px,1.35cqw,12.5px)] leading-[1.5]'

function Pill({ children, on, tone }: { children: ReactNode; on: number; tone?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line-strong bg-raised px-3 py-1 text-[clamp(10px,1.45cqw,13px)] font-medium text-fg-3" style={{ opacity: on, transform: `scale(${0.7 + 0.3 * on})`, borderColor: tone }}>
      {children}
    </span>
  )
}

// ---- 1. Tavily reads the paper ----
function TavilyScene({ d, p }: { d: SceneData; p: number }) {
  const ev = claimEvidence(d.claim)
  const domains = sourceDomains(d.events, ev?.url ?? undefined)
  const glow = ramp(p, 0.3, 0.5)
  const fly = ramp(p, 0.52, 0.78)
  const widths = [94, 100, 88, 97, 72, 99, 90, 64]
  const quote = ev ? (ev.quote.length > 96 ? ev.quote.slice(0, 95) + '…' : ev.quote) : null
  return (
    <div className="flex h-full flex-col">
      <Sentence>{tavilySentence(!!ev)}</Sentence>
      <div className={`${body} grid gap-[2.5cqw] @xl:grid-cols-[1.5fr_1fr]`}>
        <div className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-line bg-raised">
          <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
            <span className="flex gap-1" aria-hidden>{[0, 1, 2].map((i) => <i key={i} className="block h-1.5 w-1.5 rounded-full bg-line-strong" />)}</span>
            {ev?.url && <span className={`min-w-0 truncate rounded-md bg-surface px-2 py-0.5 ${mono} text-muted`}>{addressBar(ev.url)}</span>}
          </div>
          <div className="flex min-h-0 flex-1 flex-col gap-[0.9cqw] overflow-hidden p-[2cqw]">
            <div className="font-grotesk text-[clamp(11px,1.8cqw,17px)] font-semibold leading-tight text-fg-2">{d.title}</div>
            {widths.map((w, i) => i === 3 && quote ? (
              <div key={i} className={`rounded px-1.5 py-1 ${mono} text-fg-2`} style={{ background: `color-mix(in srgb, var(--chart-target) ${Math.round(glow * 28)}%, transparent)`, boxShadow: glow > 0 ? `0 0 ${Math.round(glow * 14)}px color-mix(in srgb, var(--chart-target) ${Math.round(glow * 45)}%, transparent)` : undefined }}>{quote}</div>
            ) : <div key={i} className="h-[0.8cqw] min-h-[4px] rounded-full bg-line" style={{ width: `${w}%` }} />)}
          </div>
        </div>
        <div className="flex min-h-0 flex-col justify-between gap-3">
          {ev ? (
            <div className="rounded-xl border border-line-strong bg-surface-2 p-[2.2cqw] shadow-sm" style={{ opacity: fly, transform: `translateX(${(1 - fly) * -18}%) scale(${0.92 + 0.08 * fly})` }}>
              <div className="text-[clamp(9px,1.3cqw,11px)] font-semibold uppercase tracking-widest text-muted">Claimed by the paper</div>
              <div className="font-mono text-[clamp(1.5rem,5cqw,3rem)] font-light leading-none text-fg">{d.fmt(d.claim.target)}</div>
              <div className={`mt-1 ${small} text-fg-4`}>{[ev.table, ev.version && `arXiv ${ev.version}`].filter(Boolean).join(' · ')}</div>
            </div>
          ) : <div />}
          <div className="flex flex-wrap gap-2">{domains.map((dm, i) => <Pill key={dm} on={ramp(p, 0.62 + i * 0.07, 0.72 + i * 0.07)}>{dm}</Pill>)}</div>
        </div>
      </div>
    </div>
  )
}

// ---- 2. Nemotron plans and writes code ----
function CodeScene({ d, p }: { d: SceneData; p: number }) {
  const bullets = d.plan ? planBullets(d.plan, 3) : []
  const lines = d.firstCode ? codeLines(d.firstCode, 14) : []
  const text = lines.join('\n')
  return (
    <div className="flex h-full flex-col">
      <Sentence>{SENTENCE.code()}</Sentence>
      <div className={`${body} grid gap-[2.5cqw] ${bullets.length && lines.length ? '@xl:grid-cols-[1fr_1.5fr]' : ''}`}>
        {bullets.length > 0 && (
          <ol className="m-0 flex list-none flex-col gap-[1.2cqw] p-0">
            {bullets.map((b, i) => (
              <li key={i} className={`flex gap-2 text-[clamp(11px,1.65cqw,15px)] leading-snug text-fg-3`} style={{ opacity: ramp(p, i * 0.12, i * 0.12 + 0.12), transform: `translateY(${(1 - ramp(p, i * 0.12, i * 0.12 + 0.12)) * 8}px)` }}>
                <span className="font-mono text-muted">{i + 1}</span><span>{b}</span>
              </li>
            ))}
          </ol>
        )}
        {lines.length > 0 && (
          <div className="relative min-h-0 overflow-hidden rounded-xl border border-line bg-raised p-[1.8cqw]">
            <span className={`absolute right-3 top-2 rounded-full border border-line-strong px-2 py-0.5 ${mono} text-fg-4`}>Version 1</span>
            <pre className={`m-0 overflow-hidden whitespace-pre text-fg-3 ${mono}`}><code>{typed(text, ramp(p, 0.25, 0.92))}<span className="text-fg-4">▌</span></code></pre>
          </div>
        )}
      </div>
    </div>
  )
}

// ---- 3. Repair ----
function RepairSceneView({ d, p }: { d: SceneData; p: number }) {
  const r = d.repair!
  const [flash, setFlash] = useState<{ t: 'add' | 'del' | 'ctx'; text: string }[]>([])
  useEffect(() => {
    let live = true
    if (r.card.toHash) Promise.all([d.codeLoader(r.card.fromHash), d.codeLoader(r.card.toHash)]).then(([a, b]) => { if (live && a != null && b != null) setFlash(diffFlash(a, b, 4)) })
    return () => { live = false }
  }, [r, d])
  const err = ramp(p, 0.04, 0.2)
  const search = ramp(p, 0.28, 0.42)
  const diff = ramp(p, 0.5, 0.7)
  const done = ramp(p, 0.8, 0.9)
  const ok = r.card.outcome.tone === 'ok'
  return (
    <div className="flex h-full flex-col">
      <Sentence>{r.sentence}</Sentence>
      <div className={`${body} flex flex-col gap-[1.6cqw] overflow-hidden`}>
        <div className="flex items-start gap-2 rounded-lg border px-3 py-2" style={{ opacity: err, transform: `translateX(${(1 - err) * -12}px)`, borderColor: 'color-mix(in srgb, var(--c-tone-red) 55%, transparent)', background: 'color-mix(in srgb, var(--c-tone-red) 10%, transparent)' }}>
          <X className="mt-0.5 h-4 w-4 shrink-0 text-tone-red" /><span className={`${mono} min-w-0 break-words text-fg-2`}>{r.card.problem}</span>
        </div>
        {r.card.search && (
          <div className="flex flex-wrap items-center gap-2" style={{ opacity: search, transform: `translateY(${(1 - search) * 8}px)` }}>
            <Pill on={1}><Search className="h-3.5 w-3.5" />Tavily</Pill>
            <span className={`${mono} min-w-0 break-words text-fg-4`}>{r.card.search.query.length > 90 ? r.card.search.query.slice(0, 89) + '…' : r.card.search.query}</span>
            {r.card.search.links.slice(0, 3).map((l) => <Pill key={l.url} on={1}>{l.domain}</Pill>)}
          </div>
        )}
        {flash.length > 0 && (
          <div className="overflow-hidden rounded-lg border border-line bg-raised py-1.5" style={{ opacity: diff }}>
            {flash.map((l, i) => (
              <div key={i} className={`flex whitespace-pre px-3 ${mono}`} style={{ background: l.t === 'add' ? 'color-mix(in srgb, var(--c-tone-green) 14%, transparent)' : 'color-mix(in srgb, var(--c-tone-red) 14%, transparent)', opacity: ramp(p, 0.5 + i * 0.05, 0.56 + i * 0.05) }}>
                <span className={`w-4 shrink-0 ${l.t === 'add' ? 'text-tone-green' : 'text-tone-red'}`}>{l.t === 'add' ? '+' : '-'}</span><span className="min-w-0 truncate text-fg-3">{l.text}</span>
              </div>
            ))}
          </div>
        )}
        <div className="mt-auto flex items-center gap-3">
          <span className={`rounded-full border border-line-strong px-2.5 py-1 ${mono} text-fg-3`}>{r.version.to != null && done > 0.5 ? `Version ${r.version.to}` : `Version ${r.version.from}`}</span>
          <span className={`${small} flex items-center gap-1.5`} style={{ opacity: done, color: ok ? 'var(--c-tone-green)' : 'var(--c-tone-red)' }}>{ok ? <Check className="h-4 w-4" /> : <X className="h-4 w-4" />}{r.card.outcome.text}</span>
        </div>
      </div>
    </div>
  )
}

// ---- 4. Nebius sandbox branches ----
function SandboxScene({ d, p }: { d: SceneData; p: number }) {
  const br = branches(d.events)
  const n = br.length
  const r = n > 14 ? 1.7 : n > 8 ? 2.2 : 2.8
  return (
    <div className="flex h-full flex-col">
      <Sentence>{SENTENCE.sandbox(n)}</Sentence>
      <div className={`${body}`}>
        <svg viewBox="0 0 100 46" className="h-full w-full" role="img" aria-label={`A snapshot and ${n} sandbox branches, one per seed`} preserveAspectRatio="xMidYMid meet">
          <motion.g initial={false} style={{ opacity: ramp(p, 0, 0.12) }}>
            <rect x="38" y="2" width="24" height="8" rx="2" fill="var(--c-raised)" stroke="var(--c-line-strong)" strokeWidth="0.4" />
            <text className="hidden @xl:block" x="50" y="7" textAnchor="middle" fontSize="2.6" fontWeight="500" fill="var(--c-fg-3)">Snapshot</text>
            <text className="@xl:hidden" x="50" y="7.6" textAnchor="middle" fontSize="4" fontWeight="500" fill="var(--c-fg-3)">Snapshot</text>
          </motion.g>
          {br.map((b, i) => {
            const x = branchX(i, n) * 100
            const t0 = 0.12 + (0.6 * i) / Math.max(1, n)
            const line = ramp(p, t0, t0 + 0.18)
            const node = ramp(p, t0 + 0.14, t0 + 0.24)
            const color = b.ok ? TONE_VAR.green : TONE_VAR.red
            return (
              <g key={b.seed}>
                <path d={`M50 10 C 50 24, ${x} 22, ${x} 34`} fill="none" stroke="var(--c-line-strong)" strokeWidth="0.35" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - line} />
                <circle cx={x} cy={34 + r} r={r * node} fill={color} opacity={0.9} />
                {n <= 12 && <text className="hidden font-mono @xl:block" x={x} y={34 + 2 * r + 4} textAnchor="middle" fontSize="1.6" fill="var(--c-muted)" opacity={node}>seed {b.seed}</text>}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

// ---- 5. Harness: measurements fall into the band ----
function HarnessScene({ d, p }: { d: SceneData; p: number }) {
  const vals = measurements(d.events, d.claim.id)
  const k = Math.min(vals.length, Math.floor(ramp(p, 0.08, 0.82) * vals.length + 1e-9))
  const verdict = p > 0.9
  // fixed axis from the first frame: it covers the band and every measurement
  const lo = d.claim.target - d.claim.tolerance, hi = d.claim.target + d.claim.tolerance
  const all = [...vals, lo, hi]
  const pad = (Math.max(...all) - Math.min(...all)) * 0.3 || 0.02
  const domain: [number, number] = [Math.min(...all) - pad, Math.max(...all) + pad]
  const part = dossierAt(d.dossier, d.claim.id, d.events, k, verdict)
  return (
    <div className="flex h-full flex-col">
      <Sentence>{SENTENCE.harness()}</Sentence>
      <div className={`${body} flex flex-col justify-center`}>
        <ResultBand dossier={part} claimId={d.claim.id} tone={verdict ? d.tone : 'gray'} fmt={d.fmt} fmtSpread={d.fmtSpread} showKey={false} domain={domain} animateDots
          pending="Each run is scored by our harness." />
        <div className={`mt-[1.4cqw] text-center ${small} text-muted`}>{k} of {vals.length} runs scored · the band is the tolerance</div>
      </div>
    </div>
  )
}

// ---- 6. Verdict stamp ----
function VerdictScene({ d, p }: { d: SceneData; p: number }) {
  const t = ramp(p, 0.1, 0.4)
  const note = d.fidelityUnchecked ? 'Method fidelity not checked.' : d.overall ?? null
  return (
    <div className="flex h-full flex-col">
      <Sentence>{d.sentence}</Sentence>
      <div className={`${body} flex flex-col items-center justify-center gap-[1.6cqw]`}>
        <div className="font-grotesk select-none rounded-xl border-[0.5cqw] px-[3.2cqw] py-[1cqw] text-[clamp(1.6rem,6.5cqw,4.2rem)] font-bold uppercase tracking-[0.04em]"
          style={{ color: TONE_TEXT_VAR[d.tone], borderColor: TONE_VAR[d.tone], opacity: t, transform: `rotate(${-7 + 4 * t}deg) scale(${1.9 - 0.9 * t})` }}>
          {d.resultLabel}
        </div>
        {note && <div className={`${small} text-muted`} style={{ opacity: ramp(p, 0.5, 0.7) }}>{note}</div>}
      </div>
    </div>
  )
}

export function Scene({ id, p, data }: { id: SceneId; p: number; data: SceneData }) {
  switch (id) {
    case 'tavily': return <TavilyScene d={data} p={p} />
    case 'code': return <CodeScene d={data} p={p} />
    case 'repair': return data.repair ? <RepairSceneView d={data} p={p} /> : null
    case 'sandbox': return <SandboxScene d={data} p={p} />
    case 'harness': return <HarnessScene d={data} p={p} />
    case 'verdict': return <VerdictScene d={data} p={p} />
  }
}
