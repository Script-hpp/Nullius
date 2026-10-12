import { useEffect } from 'react'
import { takeCollectionPosition } from './runNavigation'

export function useCollectionPosition(ready: boolean) {
  useEffect(() => {
    if (!ready) return
    // Wait until the cards exist and App has reset the new route's scroll position.
    const frame = requestAnimationFrame(() => {
      const position = takeCollectionPosition(location.hash)
      if (position !== null) document.getElementById('app-scroll')?.scrollTo(0, position)
    })
    return () => cancelAnimationFrame(frame)
  }, [ready])
}
