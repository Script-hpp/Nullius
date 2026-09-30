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
