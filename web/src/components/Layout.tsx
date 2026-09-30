import { useState, type ReactNode } from 'react'
import { Home, Layers, CheckCircle, Database, Settings, Search, Menu, X, Library } from 'lucide-react'

export function Layout({ children, currentHash }: { children: ReactNode, currentHash: string }) {
  const [menuOpen, setMenuOpen] = useState(false)

  const isDiscover = currentHash === 'discover'
  const isCatalog = currentHash === 'catalog'
  const isHome = currentHash === '' || currentHash === '#' || currentHash === '#/'
  const item = (active: boolean) => `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${active ? 'bg-zinc-900/80 text-zinc-100' : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/50'}`

  const nav = (
    <nav className="space-y-1 px-3" onClick={() => setMenuOpen(false)}>
      <a href="#" className={item(isHome)}>
        <Home className="w-4 h-4" />
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
      <a href="#" className={item(!isHome && currentHash.includes('/run/'))}>
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
    <div className="flex h-screen w-full flex-col overflow-hidden bg-black text-zinc-100 font-sans md:flex-row">
      {/* Compact top bar below md */}
      <header className="md:hidden shrink-0 border-b border-zinc-900 bg-[#0a0a0a]">
        <div className="flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-3 text-zinc-100">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold text-lg tracking-tight">Nullius</span>
          </div>
          <button onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}
            className="p-2 -mr-2 text-zinc-300 hover:text-white cursor-pointer">
            {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
        {menuOpen && <div className="pb-4">{nav}</div>}
      </header>

      {/* Sidebar from md up */}
      <aside className="hidden md:flex w-64 bg-[#0a0a0a] flex-shrink-0 flex-col justify-between py-6 border-r border-zinc-900">
        <div>
          <div className="px-6 mb-8 flex items-center gap-3 text-zinc-100">
            <CheckCircle className="w-5 h-5 text-zinc-100" />
            <span className="font-semibold text-lg tracking-tight">Nullius</span>
          </div>
          {nav}
        </div>

        <div className="px-6">
          <a href="#" className="flex items-center gap-3 text-sm font-medium text-zinc-400 hover:text-zinc-100 transition-colors">
            <Settings className="w-4 h-4" />
            Settings
          </a>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="min-w-0 flex-1 overflow-y-auto overflow-x-hidden bg-black relative">
        {children}
      </main>
    </div>
  )
}
