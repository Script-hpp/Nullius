import type { ReactNode } from 'react'
import { Home, Layers, CheckCircle, Database, Settings } from 'lucide-react'

export function Layout({ children, currentHash }: { children: ReactNode, currentHash: string }) {
  const isHome = currentHash === '' || currentHash === '#' || currentHash === '#/'

  return (
    <div className="flex h-screen w-full overflow-hidden bg-black text-zinc-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-[#0a0a0a] flex-shrink-0 flex flex-col justify-between py-6 border-r border-zinc-900">
        <div>
          <div className="px-6 mb-8 flex items-center gap-3 text-zinc-100">
            <CheckCircle className="w-5 h-5 text-zinc-100" />
            <span className="font-semibold text-lg tracking-tight">Nullius</span>
          </div>

          <nav className="space-y-1 px-3">
            <a href="#" className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${isHome ? 'bg-zinc-900/80 text-zinc-100' : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/50'}`}>
              <Home className="w-4 h-4" />
              Overview
            </a>
            <a href="#" className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${!isHome && currentHash.includes('/run/') ? 'bg-zinc-900/80 text-zinc-100' : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/50'}`}>
              <Layers className="w-4 h-4" />
              Live Runs
            </a>
            <a href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/50">
              <Database className="w-4 h-4" />
              Datasets
            </a>
          </nav>
        </div>

        <div className="px-6">
          <a href="#" className="flex items-center gap-3 text-sm font-medium text-zinc-400 hover:text-zinc-100 transition-colors">
            <Settings className="w-4 h-4" />
            Settings
          </a>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-auto bg-black relative">
        {children}
      </main>
    </div>
  )
}
