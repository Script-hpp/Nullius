import { useEffect, useRef, type ReactNode } from 'react'
import { RUN_TABS, TAB_LABEL, type RunTab } from '../../runtabs'

// The tab bar of the run page. It sticks to the top of the scrolling area, so the reader always sees where they are.
// Arrow keys, Home and End move between tabs (WAI-ARIA tabs pattern).
export function RunTabs({ tab, onTab, icons, barRef }: { tab: RunTab; onTab: (t: RunTab) => void; icons: Record<RunTab, ReactNode>; barRef: React.Ref<HTMLDivElement> }) {
  const refs = useRef<Partial<Record<RunTab, HTMLButtonElement | null>>>({})
  const listRef = useRef<HTMLDivElement>(null)
  // on a phone the bar scrolls sideways: keep the active tab in view. Only the bar moves: scrollIntoView would also scroll the page
  // down to a tab bar that lies below the fold, so a run page would never open at its top.
  useEffect(() => {
    const list = listRef.current, el = refs.current[tab]
    if (!list || !el) return
    const a = el.getBoundingClientRect(), b = list.getBoundingClientRect()
    list.scrollLeft += a.left + a.width / 2 - (b.left + b.width / 2)
  }, [tab])
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const to = e.key === 'ArrowRight' ? (i + 1) % RUN_TABS.length : e.key === 'ArrowLeft' ? (i + RUN_TABS.length - 1) % RUN_TABS.length : e.key === 'Home' ? 0 : e.key === 'End' ? RUN_TABS.length - 1 : -1
    if (to < 0) return
    e.preventDefault()
    onTab(RUN_TABS[to])
    refs.current[RUN_TABS[to]]?.focus()
  }
  return (
    <div ref={barRef} className="sticky top-0 z-10 -mx-4 border-b border-line bg-bg px-4 md:-mx-8 md:px-8">
      <div ref={listRef} role="tablist" aria-label="Sections of this check" className="flex gap-1 overflow-x-auto overflow-y-hidden">
        {RUN_TABS.map((t, i) => (
          <button key={t} ref={(el) => { refs.current[t] = el }} role="tab" id={`tab-${t}`} aria-selected={tab === t} aria-controls={`panel-${t}`} tabIndex={tab === t ? 0 : -1}
            onClick={() => onTab(t)} onKeyDown={(e) => onKey(e, i)}
            className={`-mb-px flex shrink-0 cursor-pointer items-center gap-2 whitespace-nowrap border-0 border-b-2 bg-transparent px-3 py-3.5 text-[13px] font-medium transition-colors sm:px-4 ${tab === t ? 'border-fg text-fg' : 'border-transparent text-muted hover:text-fg-3'}`}>
            <span className="hidden sm:inline-flex">{icons[t]}</span>{TAB_LABEL[t]}
          </button>
        ))}
      </div>
    </div>
  )
}
