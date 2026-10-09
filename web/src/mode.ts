import type { Event, Json } from './data'

// Which way a check rebuilt the paper. "nachbau_autorencode" is the product default; "nachbau" is the paper-only rebuild, also the
// fallback when no authors' code could be used. The label never claims more than the run did: the dossier (or the live `mode` event)
// is the source, the requested mode in run_start is only shown as "planned".
export type RunMode = {
  key: 'nachbau' | 'nachbau_autorencode'
  label: string
  explain: string
  planned: boolean          // live run whose mode is not confirmed yet (repository not read)
  fallback: string | null   // why a requested authors' code rebuild ran from the paper alone
}

export const MODE_AUTHOR = "Rebuild from paper and authors' code"
export const MODE_PAPER = 'Rebuild from paper'
const EXPLAIN_AUTHOR = "The agent reads the authors' public code as text and writes its own version. The authors' code is never run; the number is measured by our harness."
const EXPLAIN_PAPER = 'The agent writes its own code from the paper and the details quoted from it. The number is measured by our harness.'

const FALLBACK_TEXT: [RegExp, string][] = [
  [/no repository found or clone failed/, "no usable repository was found for this paper"],
  [/no model code found/, "the repository holds no code for this model"],
  [/setup details switched off/, "the paper text was not read"],
]
export const fallbackText = (reason: string): string => FALLBACK_TEXT.find(([re]) => re.test(reason))?.[1] ?? reason

const build = (key: RunMode['key'], planned: boolean, fallback: string | null): RunMode => ({
  key, planned, fallback,
  label: key === 'nachbau_autorencode' ? MODE_AUTHOR : MODE_PAPER,
  explain: key === 'nachbau_autorencode' ? EXPLAIN_AUTHOR : EXPLAIN_PAPER + (fallback ? ` No authors' code was used because ${fallbackText(fallback)}.` : ''),
})

const WARNING_RE = /^Autorencode nicht verfügbar \((.*)\):/

export function runMode(dossier: Json | null | undefined, events: Event[]): RunMode | null {
  const resolved = events.find((e) => e.type === 'mode')
  if (resolved) return build(resolved.mode === 'nachbau_autorencode' ? 'nachbau_autorencode' : 'nachbau', false, resolved.fallback_reason ? String(resolved.fallback_reason) : null)
  const dm = dossier?.mode
  if (dm === 'nachlauf') return null
  if (dm === 'nachbau_autorencode' || dm === 'nachbau') {
    const w = ((dossier?.warnings ?? []) as string[]).map((t) => t.match(WARNING_RE)).find(Boolean)
    return build(dm, false, dm === 'nachbau' && w ? w[1] : null)
  }
  const start = events.find((e) => e.type === 'run_start' || e.type === 'queued')   // queued: the API job event, before the loop starts
  if (start?.mode_requested === 'nachbau_autorencode') return build('nachbau_autorencode', true, null)
  return start || dossier?.paper_id ? build('nachbau', false, null) : null
}
