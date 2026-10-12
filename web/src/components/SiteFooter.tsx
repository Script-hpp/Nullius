import { REPO, REPO_BLOB, SUBMISSION_LINKS } from '../data/benchmark'
import { scrollToSection } from '../sectionScroll'
import { WordmarkText } from './Wordmark'
import { ExtLink, GoLink } from './ui'

// The place for everything that is not on the main path: Tavily, README, contact and the submission links.
// Catalog, Discover, Benchmark and Runs live in the header only.
export function SiteFooter({ className = '' }: { className?: string }) {
  const a = 'text-fg-4 no-underline transition-colors hover:text-fg'
  return (
    <footer className={`mx-auto flex max-w-5xl flex-col gap-6 border-t border-line px-6 py-10 text-sm text-muted sm:flex-row sm:items-start sm:justify-between ${className}`}>
      <div className="flex flex-col gap-2">
        <span><WordmarkText className="text-fg" />: take nobody&rsquo;s word for it.</span>
        {/* On the start page this scrolls in place (works on every click); elsewhere it opens #/tavily. */}
        <GoLink href="#/tavily" className={a} title="How Nullius uses Tavily" onClick={(e) => { if (scrollToSection('tavily')) e.preventDefault() }}>Live research by Tavily</GoLink>
      </div>
      <ul className="m-0 flex list-none flex-wrap gap-x-6 gap-y-2 p-0">
        <li><a href="#/contact" className={a}>Contact</a></li>
        <li><ExtLink href={REPO} className={a}>GitHub</ExtLink></li>
        <li><ExtLink href={`${REPO_BLOB}/README.md`} className={a}>README</ExtLink></li>
        {SUBMISSION_LINKS.filter((l) => l.href).map((l) => (
          <li key={l.label}><ExtLink href={l.href!} className={a}>{l.label}</ExtLink></li>
        ))}
      </ul>
    </footer>
  )
}
