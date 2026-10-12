import { useEffect, useState } from 'react'

// True only after `active` has stayed true for `ms`. Fast loads never show a placeholder, so nothing flickers.
export function useDelayed(active: boolean, ms = 400): boolean {
  const [late, setLate] = useState(false)
  useEffect(() => {
    if (!active) { setLate(false); return }
    const t = setTimeout(() => setLate(true), ms)
    return () => clearTimeout(t)
  }, [active, ms])
  return active && late
}
