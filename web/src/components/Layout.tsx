import { useState, type ReactNode } from 'react'
import { Home, LayoutDashboard, Layers, CheckCircle, Database, Search, Menu, X, Library } from 'lucide-react'
import { ThemeSwitch } from './ThemeSwitch'

export function Layout({ children, currentHash }: { children: ReactNode, currentHash: string }) {
  const [menuOpen, setMenuOpen] = useState(false)

  const isDiscover = currentHash === 'discover'
  const isCatalog = currentHash === 'catalog'
  const isHome = currentHash === '' || currentHash === '/'
  const isOverview = currentHash === 'overview'
  const item = (active: boolean) => `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${active ? 'bg-raised/80 text-fg' : 'text-fg-4 hover:text-fg hover:bg-raised/50'}`

  const nav = (
    <nav className="space-y-1 px-3" onClick={() => setMenuOpen(false)}>
      <a href="#" className={item(isHome)}>
        <Home className="w-4 h-4" />
        Home
      </a>
      <a href="#overview" className={item(isOverview)}>
        <LayoutDashboard className="w-4 h-4" />
        Overview
      </a>
      <a href="#discover" className={item(isDiscover)}>
        <Search className="w-4 h-4" />
        Discover
      </a>
      <a href="#catalog" className={item(isCatalog)}>
        <Library className="w-4 h-4" />
        Catalog
      </a>
      <a href="#overview" className={item(currentHash.includes('/run/') || currentHash.startsWith('/live/'))}>
        <Layers className="w-4 h-4" />
        Live Runs
      </a>
      <a href="#" className={item(false)}>
        <Database className="w-4 h-4" />
        Datasets
      </a>
    </nav>
  )

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-bg text-fg font-sans md:flex-row">
      {/* Compact top bar below md */}
      <header className="md:hidden shrink-0 border-b border-line-faint bg-surface">
        <div className="flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-3 text-fg">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold text-lg tracking-tight">Nullius</span>
          </div>
          <button onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}
            className="p-2 -mr-2 text-fg-3 hover:text-fg-max cursor-pointer">
            {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
        {menuOpen && <div className="pb-4">{nav}<div className="px-6 pt-4"><ThemeSwitch /></div></div>}
      </header>

      {/* Sidebar from md up */}
      <aside className="hidden md:flex w-64 bg-surface flex-shrink-0 flex-col justify-between py-6 border-r border-line-faint">
        <div>
          <div className="px-6 mb-8 flex items-center gap-3 text-fg">
            <CheckCircle className="w-5 h-5 text-fg" />
            <span className="font-semibold text-lg tracking-tight">Nullius</span>
          </div>
          {nav}
        </div>

        <div className="px-6">
          <ThemeSwitch />
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="min-w-0 flex-1 overflow-y-auto overflow-x-hidden bg-bg relative">
        {children}
      </main>
    </div>
  )
}
