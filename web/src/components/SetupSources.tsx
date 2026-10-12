import type { Json } from '../data'
import { setupSourceUrl } from '../targetSource'

export function SetupSources({ dossier, claim }: { dossier: Json; claim: Json }) {
  const details: Json[] = dossier.setup_details ?? []
  const included = details.filter((d) => !d.excluded)
  const excluded = details.filter((d) => d.excluded)
  const row = (detail: Json, i: number) => {
    const href = setupSourceUrl(String(detail.source ?? ''), dossier, claim)
    return <li key={i} className="min-w-0 border-t border-line py-4">
      <p className="m-0 text-sm font-medium text-fg">{String(detail.topic ?? 'Setup reference').replaceAll('_', ' ')} · {String(detail.value ?? 'Not recorded')}</p>
      <p className="mb-0 mt-2 break-words text-[12px] text-muted">{href ? <a href={href} target="_blank" rel="noreferrer" className="text-tone-blue underline underline-offset-4">{detail.source}</a> : detail.source}{detail.source_rank && ` · ${detail.source_rank}`}</p>
      {detail.applies_to && <p className="mb-0 mt-2 text-[12px] leading-relaxed text-muted">Recorded context: {[detail.applies_to.model, detail.applies_to.dataset, detail.applies_to.setting].filter(Boolean).join(' · ')}{detail.optional ? ' · optional' : ''}</p>}
      {detail.quote && <blockquote className="m-0 mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words border-l-2 border-line-strong pl-3 font-mono text-[12px] leading-relaxed text-fg-4">{detail.quote}</blockquote>}
      {detail.excluded && <p className="mb-0 mt-2 text-[12px] text-muted">Excluded from setup references: {detail.excluded}</p>}
    </li>
  }
  return <div>
    <p className="m-0 mb-4 text-sm leading-relaxed text-fg-4">References collected from the paper and authors’ code. These include defaults and alternatives; they are not a verified list of the final program’s settings. Inspect the generated code for what it used.</p>
    {included.length ? <ul className="m-0 list-none p-0">{included.map(row)}</ul> : <p className="text-sm text-muted">No quoted setup references were recorded. Check the original paper and repository, then inspect the generated code.</p>}
    {(dossier.setup_missing ?? []).length > 0 && <p className="text-sm text-fg-4">Unverified setup topics: {dossier.setup_missing.join(', ')}.</p>}
    {excluded.length > 0 && <details className="mt-4"><summary className="cursor-pointer text-[12px] text-muted">Excluded references ({excluded.length})</summary><ul className="m-0 list-none p-0">{excluded.map(row)}</ul></details>}
  </div>
}
