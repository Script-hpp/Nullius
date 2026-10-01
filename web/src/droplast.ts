// Text of the drop_last hint. Same wording as droplast_text() in src/repro_agent/droplast_note.py (the Markdown certificate); numbers come only from the hint object.
export type DropLastNote = {
  metric: string; claimed: number; tolerance: number; full: number; droplast: number
  batch_size: number; windows_dropped: number; windows_total: number
  kind: 'within_tolerance' | 'outside_tolerance'; batch_context?: string | null
  sources: { label: string; url: string }[]
}

const f4 = (x: number) => x.toFixed(4)

export function droplastParagraphs(n: DropLastNote): string[] {
  const ctx = n.batch_context ? ` (${n.batch_context})` : ''
  const lead = `The paper's number matches our measurement almost exactly if the last ${n.windows_dropped} of ${n.windows_total} test windows are skipped, as a data loader with drop_last=True does at batch size ${n.batch_size}${ctx}.`
  return n.kind === 'within_tolerance'
    ? [lead, `With all ${n.windows_total} windows we measure ${f4(n.full)}, which is still within tolerance of the claimed ${f4(n.claimed)}. With the last windows skipped we measure ${f4(n.droplast)}.`]
    : [lead, `With all ${n.windows_total} windows we measure ${f4(n.full)}; with the last windows skipped, ${f4(n.droplast)}, against a claimed ${f4(n.claimed)}. This may explain the difference.`]
}
