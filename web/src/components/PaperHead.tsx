import { GraduationCap } from 'lucide-react'
import { Glossed } from './Glossed'
import { authorLine, firstSentence, monogram, whenLine } from '../paperhead'
import type { PaperSummary } from '../summaries'

type Head = PaperSummary & { arxiv_id: string }

// Affiliation chip: a monogram square (initials of the name, no logo) and the name in the serif italic of the wordmark.
function Chip({ name, small = false }: { name: string; small?: boolean }) {
  return (
    <li className="flex min-w-0 items-center gap-2.5" title={name}>
      <span aria-hidden className={`flex shrink-0 items-center justify-center rounded-md border border-line-strong bg-raised font-mono font-medium tracking-tight text-fg-3 ${small ? 'h-5 min-w-5 px-1 text-[9px]' : 'h-7 min-w-7 px-1.5 text-[10px]'}`}>{monogram(name)}</span>
      <span className={`font-serif-italic min-w-0 leading-tight text-fg-3 ${small ? 'text-[14px]' : 'text-[18px]'}`}>{name}</span>
    </li>
  )
}

function Authors({ head, small = false }: { head: Head; small?: boolean }) {
  if (!head.authors?.length) return null
  const { shown, more } = authorLine(head.authors, small ? 3 : 6)
  const when = whenLine(head)
  return (
    <p className={`m-0 leading-relaxed text-fg-4 ${small ? 'text-[12px]' : 'text-[15px]'}`}>
      {shown.join(', ')}{more > 0 && <span className="text-muted"> and {more} more</span>}
      {when && <span className="ml-3 font-mono text-[11px] text-muted">{when}</span>}
    </p>
  )
}

// Two lines for the verdict card: what the paper is about (at most two sentences) and "Wu et al., ICML 2019". Nothing is shown for a missing part.
export function PaperByline({ head }: { head: Head | undefined }) {
  if (!head) return null
  const text = (head.summary ?? '').trim().split(/(?<=[.!?])\s+/).slice(0, 2).join(' ')   // decimals like 81.5 have no space after the dot
  const sur = (a: string) => a.trim().split(/\s+/).pop()
  const who = head.authors?.length ? (head.authors.length > 2 ? `${sur(head.authors[0])} et al.` : head.authors.map(sur).join(' & ')) : ''
  const when = whenLine(head)
  const line = [who, when].filter(Boolean).join(', ')
  if (!text && !line) return null
  return (
    <div className="mt-4 max-w-3xl" aria-label="What this paper is about">
      {text && <p className="m-0 text-[15px] leading-relaxed text-fg-3"><span className="text-muted">What this paper is about · </span><Glossed text={text} /></p>}
      {line && <p className="m-0 mt-1.5 font-mono text-[11.5px] text-muted">{line}</p>}
    </div>
  )
}

// The head of a paper: what it is about in plain words, who wrote it, where they work. Compact: for catalog cards.
export function PaperHead({ head, compact = false }: { head: Head | undefined; compact?: boolean }) {
  if (!head || (!head.summary && !head.authors?.length && !head.institutions?.length)) return null
  const inst = head.institutions ?? []
  if (compact) {
    return (
      <div className="mt-3 flex min-w-0 flex-col gap-2.5" aria-label="About the paper">
        {head.summary && <p className="m-0 text-[13px] leading-relaxed text-fg-4"><Glossed text={firstSentence(head.summary)} /></p>}
        <Authors head={head} small />
        {inst.length > 0 && (
          <ul className="m-0 flex list-none flex-wrap gap-x-4 gap-y-1.5 p-0">
            {inst.slice(0, 3).map((i) => <Chip key={i.name} name={i.name} small />)}
            {inst.length > 3 && <li className="self-center text-[11px] text-muted">+{inst.length - 3}</li>}
          </ul>
        )}
      </div>
    )
  }
  const sources = [
    head.summary && 'Summary: written by Nemotron from the arXiv abstract',
    head.authors?.length && `Authors and year: ${head.authors_source ?? 'arXiv'}`,
    inst.length > 0 && `Affiliations: ${head.institutions_source ?? 'the paper text'}, each checked against a quote`,
  ].filter(Boolean) as string[]
  return (
    <section className="mt-8 min-w-0 max-w-3xl" aria-label="About the paper">
      {head.summary && (
        <>
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">In plain words</div>
          <p className="font-grotesk m-0 mt-3 text-[1.3rem] font-medium leading-snug tracking-[-0.02em] text-fg-2 sm:text-[1.55rem]"><Glossed text={head.summary} /></p>
        </>
      )}
      <div className="mt-6 flex flex-col gap-6">
        <Authors head={head} />
        {inst.length > 0 && (
          <div>
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted"><GraduationCap className="h-3.5 w-3.5" />Affiliations</div>
            <ul className="m-0 mt-3 flex list-none flex-wrap gap-x-7 gap-y-3 p-0">{inst.map((i) => <Chip key={i.name} name={i.name} />)}</ul>
          </div>
        )}
      </div>
      {sources.length > 0 && <p className="mb-0 mt-6 text-[11px] leading-relaxed text-muted">{sources.join('. ')}.</p>}
    </section>
  )
}
