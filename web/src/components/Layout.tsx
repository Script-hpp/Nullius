import { useState, type ReactNode } from 'react'
import { ArrowRight, Menu, X } from 'lucide-react'
import { ThemeSwitch } from './ThemeSwitch'
import { Wordmark } from './Wordmark'

const REPO = 'https://github.com/Script-hpp/Nullius'

// App bar in the style of the landing page (wordmark, quiet links, one outlined "Try it"). The page scrolls inside <main>.
export function Layout({ children, currentHash }: { children: ReactNode; currentHash: string }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const runs = currentHash === 'overview' || currentHash.startsWith('/run/') || currentHash.startsWith('/live/')
  const links: { label: string; href: string; active: boolean }[] = [
    { label: 'Discover', href: '#discover', active: currentHash === 'discover' },
    { label: 'Catalog', href: '#catalog', active: currentHash === 'catalog' },
    { label: 'Runs', href: '#overview', active: runs },
    { label: 'Benchmark', href: '#benchmark', active: currentHash === 'benchmark' },
  ]
  const link = (active: boolean) => `text-sm no-underline transition-colors ${active ? 'text-fg underline decoration-fg-4 decoration-1 underline-offset-[6px]' : 'text-fg-4 hover:text-fg'}`

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-bg font-sans text-fg">
      <header className="shrink-0 border-b border-line-faint bg-bg">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-5 px-5 md:px-8">
          <Wordmark />
          <nav className="hidden items-center gap-8 md:flex" aria-label="Main">
            {links.map((l) => <a key={l.label} href={l.href} className={link(l.active)} aria-current={l.active ? 'page' : undefined}>{l.label}</a>)}
            <a href={REPO} target="_blank" rel="noreferrer" className={link(false)}>GitHub</a>
          </nav>
          <div className="flex items-center gap-4">
            <a href="#discover" className="hidden rounded-full border border-fg px-4 py-1.5 text-[11px] font-medium uppercase tracking-[0.2em] text-fg no-underline transition-colors hover:bg-fg hover:text-bg sm:inline">Try it</a>
            <span className="hidden md:inline"><ThemeSwitch /></span>
            <button onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}
              className="-mr-2 cursor-pointer border-0 bg-transparent p-2 text-fg-3 hover:text-fg-max md:hidden">
              {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
        {menuOpen && (
          <div className="border-t border-line-faint px-5 pb-5 pt-3 md:hidden" onClick={() => setMenuOpen(false)}>
            <nav className="flex flex-col gap-1" aria-label="Main">
              {links.map((l) => <a key={l.label} href={l.href} className={`py-2 ${link(l.active)}`}>{l.label}</a>)}
              <a href={REPO} target="_blank" rel="noreferrer" className={`py-2 ${link(false)}`}>GitHub</a>
            </nav>
            <div className="pt-4" onClick={(e) => e.stopPropagation()}><ThemeSwitch /></div>
          </div>
        )}
      </header>

      <main className="relative min-w-0 flex-1 overflow-y-auto overflow-x-hidden bg-bg">
        {children}
        <footer className="mx-auto mt-10 flex max-w-6xl flex-wrap items-center justify-between gap-4 border-t border-line px-5 py-8 text-sm text-muted md:px-8">
          <span>Nullius in verba: take nobody&rsquo;s word for it.</span>
          <a href="#/" className="inline-flex items-center gap-2 text-fg-4 no-underline hover:text-fg">Start page <ArrowRight className="h-4 w-4" /></a>
        </footer>
      </main>
    </div>
  )
}
