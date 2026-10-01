import { ExternalLink, Info } from 'lucide-react'
import { droplastParagraphs, type DropLastNote } from '../droplast'
import type { Json } from '../data'

// Own card after "Why you can trust this number": only shown when the dossier holds a drop_last hint for this claim (harness/droplast.py,
// droplast_note.py). An observation about the paper's number; the verdict above is not changed by it.
export function DropLast({ dossier, claimId }: { dossier: Json; claimId: string }) {
  const note: DropLastNote | undefined = dossier?.droplast_notes?.[claimId]
  if (!note) return null
  const [lead, rest] = droplastParagraphs(note)
  return (
    <section id="sec-droplast" className="scroll-mt-4 min-w-0 rounded-xl border border-line bg-surface p-6 md:p-8" aria-label="A known evaluation setting">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted"><Info className="h-3.5 w-3.5" />A known evaluation setting</div>
      <p className="mb-0 mt-4 max-w-3xl text-[15px] leading-relaxed text-fg-3">{lead}</p>
      <p className="mb-0 mt-3 max-w-3xl text-[15px] leading-relaxed text-fg-4">{rest}</p>
      <dl className="m-0 mt-6 grid max-w-xl grid-cols-3 gap-4 border-t border-line pt-5">
        {([['Claimed', note.claimed], ['All windows', note.full], [`Skipping ${note.windows_dropped}`, note.droplast]] as const).map(([k, x]) => (
          <div key={k}><dt className="text-[10px] font-semibold uppercase tracking-widest text-muted">{k}</dt><dd className="m-0 mt-1 font-mono text-lg text-fg-3">{x.toFixed(4)}</dd></div>
        ))}
      </dl>
      <p className="mb-0 mt-5 max-w-3xl text-[13px] leading-relaxed text-muted">
        This is an observation, not a criticism. The verdict above still uses all {note.windows_total} windows.
      </p>
      <ul className="m-0 mt-4 flex list-none flex-wrap gap-x-6 gap-y-2 p-0 text-[13px]">
        {note.sources.map((s) => (
          <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-fg-4 underline decoration-line-strong underline-offset-4 hover:text-fg">{s.label}<ExternalLink className="h-3 w-3" /></a></li>
        ))}
      </ul>
    </section>
  )
}
