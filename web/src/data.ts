// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Json = any

export type RunRef = { id: string; paper_id: string; title: string; timestamp: string; note?: string }
export type Event = { ts: number; type: string; [k: string]: Json }
export type RunData = { ref: RunRef; dossier: Json; events: Event[]; code: string | null; plan: string | null; zeugnis: string | null }

const base = import.meta.env.BASE_URL + 'data/'

async function getText(path: string): Promise<string | null> {
  try {
    const r = await fetch(base + path)
    if (!r.ok) return null
    const t = await r.text()
    // Statische Server liefern für fehlende Dateien oft index.html (SPA-Fallback): als "nicht vorhanden" behandeln.
    return /^\s*<!doctype html/i.test(t) ? null : t
  } catch {
    return null
  }
}

export async function loadIndex(): Promise<RunRef[]> {
  const t = await getText('index.json')
  return t ? (JSON.parse(t) as RunRef[]) : []
}

// Ohne events.jsonl (ältere Läufe): Ereignisse aus dem Dossier ableiten, damit die Wiedergabe trotzdem funktioniert.
function eventsFromDossier(d: Json): Event[] {
  const ev: Event[] = []
  let t = 0
  const push = (type: string, data: Json) => ev.push({ ts: t++, type, ...data })
  for (const c of d.claims ?? []) {
    if (c.target_evidence) push('target_resolved', { claim_id: c.id, value: c.target, quote: c.target_evidence.quote, table: c.target_evidence.table, url: c.target_evidence.url, arxiv_version: c.target_evidence.arxiv_version, column_check: c.target_evidence.column_check })
  }
  for (const s of d.sources ?? []) push('source', s)
  if ((d.setup_details ?? []).length) push('setup_details', { n: d.setup_details.length, details: d.setup_details })
  for (const c of d.claims ?? []) push('claim_start', { claim_id: c.id, text: c.text, target: c.target, tolerance: c.tolerance })
  for (const a of d.attempts ?? []) push('attempt', { claim_id: a.claim_id, seed: a.seed, branch_id: a.branch_id, parent_snapshot: a.parent_snapshot, code_hash: a.code_hash, status: a.status, measured: a.measured, error: a.error_summary })
  for (const [cid, ab] of Object.entries(d.ablations ?? {})) push('ablation', { claim_id: cid, ...(ab as object) })
  for (const v of d.verdicts ?? []) push('verdict', { claim_id: v.claim_id, urteil: v.urteil, methode: v.methode, mean: v.measured_mean, std: v.measured_std, explanation: v.explanation })
  return ev
}

export async function loadRun(ref: RunRef): Promise<RunData> {
  const dir = `${ref.paper_id}/${ref.timestamp}/`
  const dossier = JSON.parse((await getText(dir + 'dossier.json')) ?? '{}')
  const evText = await getText(dir + 'events.jsonl')
  const events = evText
    ? evText.split('\n').filter(Boolean).map((l) => JSON.parse(l) as Event)
    : eventsFromDossier(dossier)
  return { ref, dossier, events, code: await getText(dir + 'code.py'), plan: await getText(dir + 'plan.md'), zeugnis: await getText(dir + 'zeugnis.md') }
}
