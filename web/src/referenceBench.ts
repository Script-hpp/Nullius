// Benchmark figures of a reference catalog entry (written by the export from benchmark.json, never typed by hand).
export type ReferenceBenchmark = {
  version: string
  arm: string
  reached: number
  runs: number
  units_matched: number
  units: number
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many)

// "In benchmark v4 (with Tavily), 3 of 3 fresh runs reached the target. Across all 8 paper-claim units, 2 matched the reference verdict."
export function referenceSentence(b: ReferenceBenchmark | null | undefined): string {
  if (!b || !(b.runs > 0) || !(b.units > 0)) return 'Reference paper from our benchmark.'
  return `In benchmark ${b.version} (${b.arm}), ${b.reached} of ${b.runs} fresh ${plural(b.runs, 'run', 'runs')} reached the target. `
    + `Across all ${b.units} paper-claim ${plural(b.units, 'unit', 'units')}, ${b.units_matched} matched the reference verdict.`
}

// The same facts in one short line for the catalog card. The benchmark ran in the paper-only mode; the card's own result
// (shown above or below it) comes from a newer run in the product mode, so the line names the benchmark mode.
export function referenceShort(b: ReferenceBenchmark | null | undefined): string {
  if (!b || !(b.runs > 0) || !(b.units > 0)) return 'Reference paper.'
  return `Benchmark ${b.version}, rebuild from paper only (${b.arm}): ${b.reached} of ${b.runs} runs reached the target; ${b.units_matched} of ${b.units} units matched the reference verdict.`
}
