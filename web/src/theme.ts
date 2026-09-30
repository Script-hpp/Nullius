import { useEffect, useState } from 'react'

export type ThemePref = 'system' | 'light' | 'dark'
export const THEME_KEY = 'nullius-theme'   // same key as the inline script in index.html

export function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(THEME_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch { /* storage blocked: follow the system */ }
  return 'system'
}

function systemTheme(): 'light' | 'dark' {
  try { return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark' } catch { return 'dark' }
}

// data-theme on <html> is always the resolved theme; CSS only knows light and dark.
export function applyTheme(pref: ThemePref) {
  document.documentElement.dataset.theme = pref === 'system' ? systemTheme() : pref
}

export function useTheme(): [ThemePref, (p: ThemePref) => void] {
  const [pref, setPref] = useState<ThemePref>(readPref)
  useEffect(() => {
    applyTheme(pref)
    if (pref !== 'system') return
    const mq = matchMedia('(prefers-color-scheme: light)')
    const on = () => applyTheme('system')
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [pref])
  const set = (p: ThemePref) => {
    try { localStorage.setItem(THEME_KEY, p) } catch { /* not persisted, still applied */ }
    setPref(p)
  }
  return [pref, set]
}
