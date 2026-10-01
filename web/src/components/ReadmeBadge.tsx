import { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { badgeMarkdown } from '../badge'
import { SITE_BASE_IS_PLACEHOLDER, SITE_BASE_URL } from '../config'
import { runDir, type RunRef } from '../data'

// "Show this result in your README": preview of the badge file of this run and the Markdown to copy. Shown only when the badge exists (older exports have none).
export function ReadmeBadge({ ref_ }: { ref_: RunRef }) {
  const [copied, setCopied] = useState(false)
  const [missing, setMissing] = useState(false)
  if (missing) return null
  const md = badgeMarkdown(SITE_BASE_URL, ref_.paper_id, ref_.timestamp)
  const copy = async () => {
    try { await navigator.clipboard.writeText(md); setCopied(true); window.setTimeout(() => setCopied(false), 1800) } catch { /* clipboard blocked: the text stays selectable */ }
  }
  return (
    <section id="sec-badge" className="scroll-mt-4 min-w-0 rounded-xl border border-line bg-surface p-6 md:p-8" aria-label="Show this result in your README">
      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Show this result in your README</div>
      <p className="mb-0 mt-4 max-w-3xl text-[15px] leading-relaxed text-fg-3">
        Authors can add this badge to a README. It states our verdict with our setup, the measured and the claimed value, and that the method was not checked. A click leads to this certificate.
      </p>
      <div className="mt-6 inline-block rounded-lg border border-line bg-white p-3">
        <img src={`${import.meta.env.BASE_URL}data/${runDir(ref_)}badge.svg`} alt="Nullius badge for this run" onError={() => setMissing(true)} className="block h-auto max-w-full" />
      </div>
      <div className="mt-5 flex items-stretch gap-2">
        <code className="min-w-0 flex-1 select-all overflow-x-auto whitespace-nowrap rounded-lg border border-line bg-raised px-3 py-2.5 font-mono text-[12px] text-fg-3">{md}</code>
        <button type="button" onClick={copy} className="inline-flex shrink-0 cursor-pointer items-center gap-2 rounded-lg border border-line-strong bg-transparent px-4 text-xs font-medium text-fg hover:bg-raised-2">
          {copied ? <><Check className="h-3.5 w-3.5" />Copied</> : <><Copy className="h-3.5 w-3.5" />Copy</>}
        </button>
      </div>
      {SITE_BASE_IS_PLACEHOLDER && (
        <p className="mb-0 mt-4 text-[12px] leading-relaxed text-muted">The address in the snippet is a placeholder until the demo URL is set.</p>
      )}
    </section>
  )
}
