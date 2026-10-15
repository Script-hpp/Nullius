// Landing media helpers (pure, tested): film duration label and loop playback decisions.
export const FILM_SECONDS = 157.8

export function durationLabel(seconds: number): string {
  const total = Math.round(seconds)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

// The proof loop plays only when it is on screen and the viewer has not asked for reduced motion.
export function loopShouldPlay(visible: boolean, reducedMotion: boolean): boolean {
  return visible && !reducedMotion
}

// Tab order inside the modal: wraps at both ends. Returns the index to focus next, or null when nothing is focusable.
export function nextFocusIndex(count: number, current: number, backwards: boolean): number | null {
  if (count <= 0) return null
  if (current < 0) return backwards ? count - 1 : 0
  return (current + (backwards ? count - 1 : 1)) % count
}
