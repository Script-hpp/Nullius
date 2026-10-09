import { Monitor, Moon, Sun } from 'lucide-react'
import { useTheme, type ThemePref } from '../theme'

const OPTIONS: { id: ThemePref; label: string; icon: typeof Sun }[] = [
  { id: 'system', label: 'System', icon: Monitor },
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'dark', label: 'Dark', icon: Moon },
]

export function ThemeSwitch() {
  const [pref, setPref] = useTheme()
  return (
    <div role="group" aria-label="Color theme" className="inline-flex rounded-lg border border-line p-0.5">
      {OPTIONS.map(({ id, label, icon: Icon }) => (
        <button key={id} type="button" onClick={() => setPref(id)} aria-pressed={pref === id} title={label} aria-label={label}
          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium cursor-pointer ${pref === id ? 'bg-raised text-fg' : 'text-muted hover:text-fg-3'}`}>
          <Icon className="w-3.5 h-3.5" />
          <span className="hidden md:inline">{label}</span>
        </button>
      ))}
    </div>
  )
}

// One quiet icon button for the app bars: each click moves on to the next setting (system, light, dark).
export function ThemeToggle() {
  const [pref, setPref] = useTheme()
  const i = OPTIONS.findIndex((o) => o.id === pref)
  const cur = OPTIONS[i]
  const next = OPTIONS[(i + 1) % OPTIONS.length]
  return (
    <button type="button" onClick={() => setPref(next.id)} title={`Theme: ${cur.label}. Click for ${next.label}.`} aria-label={`Color theme: ${cur.label}. Switch to ${next.label}.`}
      className="inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-full border-0 bg-transparent text-fg-4 transition-colors hover:text-fg">
      <cur.icon className="h-4 w-4" aria-hidden />
    </button>
  )
}
