import { Search } from 'lucide-react'
import { REPO, REPO_BLOB, SUBMISSION_LINKS } from '../data/benchmark'

// The one place for everything that is not on the main path: Discover, Runs, README and the submission links.
export function SiteFooter({ className = '' }: { className?: string }) {
  const a = 'text-fg-4 no-underline transition-colors hover:text-fg'
  return (
    <footer className={`mx-auto flex max-w-5xl flex-col gap-6 border-t border-line px-6 py-10 text-sm text-muted sm:flex-row sm:items-start sm:justify-between ${className}`}>
      <div className="flex flex-col gap-2">
        <span>Nullius in verba: take nobody&rsquo;s word for it.</span>
        <a href="#/tavily" className="inline-flex items-center gap-1.5 text-fg-4 no-underline transition-colors hover:text-fg" title="How Nullius uses Tavily">
          <Search className="h-3.5 w-3.5" aria-hidden />Live research by Tavily
        </a>
      </div>
      <ul className="m-0 flex list-none flex-wrap gap-x-6 gap-y-2 p-0">
        <li><a href="#discover" className={a}>Discover</a></li>
        <li><a href="#overview" className={a}>Runs</a></li>
        <li><a href="#benchmark" className={a}>Benchmark</a></li>
        <li><a href={REPO} target="_blank" rel="noreferrer" className={a}>GitHub</a></li>
        <li><a href={`${REPO_BLOB}/README.md`} target="_blank" rel="noreferrer" className={a}>README</a></li>
        {SUBMISSION_LINKS.filter((l) => l.href).map((l) => (
          <li key={l.label}>
            {l.href
              ? <a href={l.href} target="_blank" rel="noreferrer" className={a}>{l.label}</a>
              : <span>{l.label}: <span className="font-mono text-[13px]">TBD</span></span>}
          </li>
        ))}
      </ul>
    </footer>
  )
}
