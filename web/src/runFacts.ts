import { minutes } from './eta.ts'

// "3 seeds · 14 min · $0.31": how the check was made, from the run itself. Parts that are unknown are left out, nothing is guessed.
export function runFacts(f: { seeds: number; seconds: number | null; costUsd: number | null | undefined }): string {
  const parts: string[] = []
  if (f.seeds > 0) parts.push(`${f.seeds} ${f.seeds === 1 ? 'seed' : 'seeds'}`)
  if (f.seconds != null && f.seconds >= 0) parts.push(minutes(f.seconds))
  if (f.costUsd != null && Number.isFinite(f.costUsd)) parts.push(`$${f.costUsd.toFixed(2)}`)
  return parts.join(' · ')
}
