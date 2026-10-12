import { useState, type ReactNode } from 'react'
import { Menu, X } from 'lucide-react'
import { ThemeToggle } from './ThemeSwitch'
import { SiteFooter } from './SiteFooter'
import { Wordmark } from './Wordmark'
import { ExtLink } from './ui'
import { REPO } from '../data/benchmark'

const DISCOVER_TIP = 'Papers found live by Tavily'

// Header items are text-only; the one external item (GitHub) carries the external arrow. Shared with the start page's top bar.
export const NAV_LINKS: { label: string; href: string; hash: string; title?: string }[] = [
  { label: 'Catalog', href: '#catalog', hash: 'catalog' },
  { label: 'Discover', href: '#discover', hash: 'discover', title: DISCOVER_TIP },
  { label: 'Benchmark', href: '#benchmark', hash: 'benchmark' },
  { label: 'Runs', href: '#overview', hash: 'overview' },
]

// The one site header: used by the start page and by every app page, so wordmark, nav position, underline and colors never differ.
// currentHash is the hash without "#/" prefix of the active page ('' on the start page: no link is active).
export function SiteHeader({ currentHash, className = '' }: { currentHash: string; className?: string }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const links = NAV_LINKS.map((l) => ({ ...l, active: currentHash.split('?')[0] === l.hash }))
  const link = (active: boolean) => `text-sm no-underline transition-colors ${active ? 'text-fg underline decoration-fg-4 decoration-1 underline-offset-[6px]' : 'text-fg-4 hover:text-fg'}`

  return (
    <header className={`shrink-0 border-b border-line-faint bg-bg ${className}`}>
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-5 px-5 md:px-8">
        <Wordmark />
        <nav className="hidden items-center gap-8 md:flex" aria-label="Main">
          {links.map((l) => <a key={l.label} href={l.href} className={link(l.active)} aria-current={l.active ? 'page' : undefined} title={l.title}>{l.label}</a>)}
          <ExtLink href={REPO} className={link(false)}>GitHub</ExtLink>
        </nav>
        <div className="flex items-center gap-4">
          <a href="#catalog" className="hidden shrink-0 whitespace-nowrap rounded-full border border-fg px-4 py-1.5 text-[11px] font-medium uppercase tracking-[0.2em] text-fg no-underline transition-colors hover:bg-fg hover:text-bg sm:inline">Try it</a>
          <ThemeToggle />
          <button onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}
            className="-mr-2 cursor-pointer border-0 bg-transparent p-2 text-fg-3 hover:text-fg-max md:hidden">
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {menuOpen && (
        <div className="border-t border-line-faint px-5 pb-5 pt-3 md:hidden" onClick={() => setMenuOpen(false)}>
          <nav className="flex flex-col gap-1" aria-label="Main">
            {links.map((l) => <a key={l.label} href={l.href} className={`py-2 ${link(l.active)}`}>{l.label}{l.label === 'Discover' && <span className="ml-2 text-[11px] text-muted">papers found live by Tavily</span>}</a>)}
            <ExtLink href={REPO} className={`py-2 ${link(false)}`}>GitHub</ExtLink>
          </nav>
        </div>
      )}
    </header>
  )
}

export function Layout({ children, currentHash }: { children: ReactNode; currentHash: string }) {
  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-bg font-sans text-fg">
      <SiteHeader currentHash={currentHash} />

      <main id="app-scroll" className="relative min-w-0 flex-1 overflow-y-auto overflow-x-hidden bg-bg">
        {children}
        <SiteFooter className="mt-10 max-w-6xl px-5 md:px-8" />
      </main>
    </div>
  )
}
