import { useCallback, useEffect, useRef, useState } from 'react'
import type { Event, RunRef } from './data'
import type { CatalogEntry } from './catalog'
import { StartRefused, startErrorText, type LiveStatusInfo } from './livestatus'

// Live reproduction: the browser only talks to /api (keys stay on the server). Static mirrors have no API: available = false.
export type JobInfo = { busy: boolean; kind?: 'discovery' | 'reproduction'; job_id?: string; catalog_id?: string | null; started_at?: number }

export const DATA_CHANGED = 'nullius:data-changed'   // index.json / catalog.json changed on the server

export async function fetchJobInfo(): Promise<JobInfo | null> {
  try {
    const r = await fetch('/api/jobs/current', { cache: 'no-store' })
    if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) return null   // SPA fallback of a static host is HTML
    return (await r.json()) as JobInfo
  } catch {
    return null
  }
}

// null = API not reachable (hide the live controls), otherwise the current job state; polled.
export function useJobInfo(everyMs = 10000): { available: boolean; ready: boolean; info: JobInfo | null; refresh: () => void } {
  const [info, setInfo] = useState<JobInfo | null>(null)
  const [ready, setReady] = useState(false)
  const refresh = useCallback(() => { fetchJobInfo().then((i) => { setInfo(i); setReady(true) }) }, [])
  useEffect(() => {
    refresh()
    const t = window.setInterval(refresh, everyMs)
    return () => window.clearInterval(t)
  }, [refresh, everyMs])
  return { available: ready && info !== null, ready, info, refresh }
}

// null = no answer (static mirror or API down): the UI then shows no notice and the start call reports what happens.
export async function fetchLiveStatus(): Promise<LiveStatusInfo | null> {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' })
    if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) return null
    return (await r.json()) as LiveStatusInfo
  } catch {
    return null
  }
}

export function useLiveStatus(everyMs = 30000): { status: LiveStatusInfo | null; refresh: () => void } {
  const [status, setStatus] = useState<LiveStatusInfo | null>(null)
  const refresh = useCallback(() => { fetchLiveStatus().then(setStatus) }, [])
  useEffect(() => {
    refresh()
    const t = window.setInterval(refresh, everyMs)
    return () => window.clearInterval(t)
  }, [refresh, everyMs])
  return { status, refresh }
}

export async function startReproduction(catalogId: string, quick = true): Promise<string> {
  let r: Response
  try {
    r = await fetch('/api/reproduce', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ catalog_id: catalogId, quick }),
    })
  } catch {
    throw new Error('The live backend is not reachable.')
  }
  if (!r.ok) {
    let body: unknown = null
    try { body = await r.json() } catch { /* no body */ }
    const { text, reason } = startErrorText(r.status, body)
    throw new StartRefused(text, reason)
  }
  return (await r.json()).job_id as string
}

export type LiveStatus = 'connecting' | 'running' | 'finished' | 'error' | 'lost' | 'unknown'
export type LiveState = { events: Event[]; status: LiveStatus; runId: string | null; error: string | null; verdict: Record<string, unknown> | null; lastReceived: number }

// Events arrive over SSE. On a dropped connection it reconnects from the number of events already received (server keeps them all).
export function useLiveRun(jobId: string): LiveState {
  const [events, setEvents] = useState<Event[]>([])
  const [status, setStatus] = useState<LiveStatus>('connecting')
  const [runId, setRunId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<Record<string, unknown> | null>(null)
  const [lastReceived, setLastReceived] = useState(0)
  const count = useRef(0)

  useEffect(() => {
    let es: EventSource | null = null
    let timer: number | undefined
    let closed = false
    let tries = 0
    count.current = 0

    const connect = () => {
      if (closed) return
      es = new EventSource(`/api/reproduce/${encodeURIComponent(jobId)}/stream?from=${count.current}`)
      es.onmessage = (m) => {
        tries = 0
        let e: Event
        try { e = JSON.parse(m.data) as Event } catch { return }
        count.current++
        setLastReceived(Date.now())
        setEvents((x) => [...x, e])
        if (e.type === 'finished') {
          setStatus('finished'); setRunId(String(e.run_id)); setVerdict((e.verdict as Record<string, unknown>) ?? null)
          closed = true; es?.close(); window.dispatchEvent(new Event(DATA_CHANGED))
        } else if (e.type === 'error') {
          setStatus('error'); setError(String(e.message ?? 'The run failed.'))
          closed = true; es?.close()
        } else setStatus('running')
      }
      es.onerror = () => {
        es?.close()
        if (closed) return
        if (++tries > 6) { setStatus(count.current === 0 ? 'unknown' : 'lost'); return }
        timer = window.setTimeout(connect, 1500)
      }
    }
    connect()
    return () => { closed = true; es?.close(); window.clearTimeout(timer) }
  }, [jobId])

  return { events, status, runId, error, verdict, lastReceived }
}

export const PHASES = ['Target', 'Setup details', 'Plan', 'Code', 'Seed 0', 'Ablation', 'Seeds', 'Verdict'] as const
export type Phase = (typeof PHASES)[number]

export type Progress = { phase: Phase | null; done: Phase[]; seedsDone: number; seedsTotal: number | null; costNet: number; repairs: number }

// Derives where the loop is from the events seen so far.
export function deriveProgress(events: Event[]): Progress {
  const seen = new Set(events.map((e) => e.type))
  const seedsTotal = (events.find((e) => e.type === 'run_start')?.seeds as number | undefined) ?? null
  const okSeeds = new Set<number>()
  for (const e of events) if (e.type === 'attempt' && e.measured != null) okSeeds.add(Number(e.seed))
  const costNet = Number([...events].reverse().find((e) => e.type === 'llm_call')?.total_net_usd ?? 0)
  const repairs = events.filter((e) => e.type === 'graph_repair').length
  const done: Phase[] = []
  if (seen.has('target_resolved') || seen.has('target_check') || seen.has('setup_details') || seen.has('plan')) done.push('Target')
  if (seen.has('setup_details') || seen.has('plan')) done.push('Setup details')
  if (seen.has('plan')) done.push('Plan')
  if (seen.has('code_written')) done.push('Code')
  if (okSeeds.has(0)) done.push('Seed 0')
  if (seen.has('ablation')) done.push('Ablation')
  if (seen.has('verdict')) { done.push('Seeds'); done.push('Verdict') }
  else if (seedsTotal !== null && okSeeds.size >= seedsTotal) done.push('Seeds')
  const phase = PHASES.find((p) => !done.includes(p)) ?? null
  return { phase, done, seedsDone: okSeeds.size, seedsTotal, costNet, repairs }
}

// Newest finished run of a candidate: the curated run from catalog.json or a live run from index.json.
export function lastFinishedRun(entry: CatalogEntry | undefined, runs: RunRef[]): string | null {
  const live = runs.filter((r) => r.catalog_id && r.catalog_id === entry?.id).map((r) => r.id).sort().pop()
  return live ?? entry?.run_id ?? null
}

