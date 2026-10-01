import { AlertCircle, CheckCircle2, HelpCircle, Info, XCircle } from 'lucide-react'
import { toPercent } from './components/ui'
import { NOT_ASSESSABLE, type Tone } from './labels'
import type { Json } from './data'

export function parseDate(ts: string) {
  const dateStr = ts.split('-')[0]
  if (dateStr.length !== 8) return ts
  const d = new Date(parseInt(dateStr.slice(0,4)), parseInt(dateStr.slice(4,6))-1, parseInt(dateStr.slice(6,8)))
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function colorForTone(tone: Tone) {
  if (tone === 'green') return 'text-tone-green'
  if (tone === 'amber') return 'text-tone-amber'
  if (tone === 'red') return 'text-tone-red'
  if (tone === 'violet') return 'text-tone-violet'
  return 'text-fg-4'
}

export function iconForTone(tone: Tone) {
  if (tone === 'green') return <CheckCircle2 className="w-5 h-5 text-tone-green mt-0.5 shrink-0" />
  if (tone === 'amber') return <AlertCircle className="w-5 h-5 text-tone-amber mt-0.5 shrink-0" />
  if (tone === 'red') return <XCircle className="w-5 h-5 text-tone-red mt-0.5 shrink-0" />
  if (tone === 'violet') return <HelpCircle className="w-5 h-5 text-tone-violet mt-0.5 shrink-0" />
  return <Info className="w-5 h-5 text-fg-4 mt-0.5 shrink-0" />
}

// MSE/MAE are plain decimals, never percent; accuracy is a share shown as percent.
export const showValue = (metric: string | undefined, x: any) => (metric && metric !== 'accuracy' ? (x == null ? '–' : Number(x).toFixed(3)) : toPercent(x))

export function getDelta(measured: any, target: any, metric?: string) {
  if (measured === null || measured === undefined || target === null || target === undefined) return null
  const m = Number(measured)
  let t = Number(target)
  if (Number.isNaN(m) || Number.isNaN(t)) return null
  
  const plain = !!metric && metric !== 'accuracy'
  const mVal = plain ? m : (m <= 2.0 && m > 0) ? m * 100 : m
  const tVal = plain ? t : (t <= 2.0 && t > 0) ? t * 100 : t
  
  const diff = mVal - tVal
  if (plain) {
    if (Math.abs(diff) < 0.0005) return { text: '±0.000 vs. claimed', color: 'text-muted' }
    return { text: `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(3)} vs. claimed`, color: diff > 0 ? 'text-tone-green' : 'text-tone-red' }
  }
  // percentage points, same wording as the catalog
  if (Math.abs(diff) < 0.05) return { text: '±0.0 pp vs. claimed', color: 'text-muted' }
  if (diff > 0) return { text: `+${diff.toFixed(1)} pp vs. claimed`, color: 'text-tone-green' }
  return { text: `−${Math.abs(diff).toFixed(1)} pp vs. claimed`, color: 'text-tone-red' }
}

// All runs: always "with our setup", never a verdict on the paper (method fidelity is not checked).
export function getCatalogStatement(claimed: string, measured: string, urteil: string, gesamt?: string) {
  if (gesamt === NOT_ASSESSABLE) return `The code did not implement the method, so the measurement (${measured}) says nothing about the reported ${claimed}.`
  if (urteil === 'reproduziert') return `Reproduced with our setup: measured ${measured} against the reported ${claimed}.`
  if (urteil === 'teilweise') return `Partially reproduced with our setup: measured ${measured} against the reported ${claimed}.`
  if (urteil === 'nicht reproduziert') return `Not reproduced with our setup: measured ${measured} against the reported ${claimed}. This is not a finding about the paper.`
  return `No valid measurement with our setup against the reported ${claimed}.`
}

// Our own verdict words; never "verified": method fidelity is not checked.
const RESULT_WORDS: Record<string, string> = {
  'reproduziert': 'reproduced with our setup (target value reached)',
  'teilweise': 'partially reproduced with our setup',
  'nicht reproduziert': 'not reproduced with our setup',
  'Agenten-Fehler vermutet': 'agent error suspected (says nothing about the paper)',
  'nicht reproduzierbar (Setup)': 'not reproducible with our setup',
}

// One sentence from the data: what we measured against what was claimed, with the verdict label.
export function resultSentence(claim: Json, v: Json | undefined, gesamt?: string): string {
  const target = showValue(claim.metric, claim.target)
  if (!v || v.measured_mean == null || gesamt === NOT_ASSESSABLE) {
    return v ? getCatalogStatement(target, showValue(claim.metric, v.measured_mean), v.urteil, gesamt) : `No verdict yet for the claimed ${target}.`
  }
  const spread = v.measured_std == null ? '' : ` (± ${claim.metric === 'accuracy' ? (v.measured_std * 100).toFixed(1) : Number(v.measured_std).toFixed(3)})`
  const verdict = RESULT_WORDS[String(v.urteil)] ?? 'no valid verdict with our setup'
  return `Our rebuild reached ${showValue(claim.metric, v.measured_mean)}${spread} against the claimed ${target}: ${verdict}.`
}

