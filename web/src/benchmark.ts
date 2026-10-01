import { useEffect, useState } from 'react'
import { getText } from './data'

// Benchmark results come from web/public/data/benchmark.json (written by the export). Without the file the pages say so honestly.
export type BenchArm = { majority: string; verdicts: string[] }
export type BenchUnit = {
  paper: string; title: string; claim: string; metric: string; target: number; reference_verdict: string
  arms: { A: BenchArm; B: BenchArm }; placebo: { A: string[]; B: string[] }
}
export type BenchVersion = {
  id: string; date: string; code_commit: string; n_runs: number; units: BenchUnit[]
  summary: { agree_A: number; agree_B: number; placebo_passed: number; n_units: number }
}
export type Benchmark = { versions: BenchVersion[] }

export function parseBenchmark(text: string | null): Benchmark | null {
  if (!text) return null
  try {
    const b = JSON.parse(text) as Benchmark
    return Array.isArray(b.versions) && b.versions.length ? b : null
  } catch {
    return null
  }
}

// undefined = loading, null = no benchmark file
export function useBenchmark(): Benchmark | null | undefined {
  const [b, setB] = useState<Benchmark | null | undefined>(undefined)
  useEffect(() => { getText('benchmark.json').then((t) => setB(parseBenchmark(t))) }, [])
  return b
}
