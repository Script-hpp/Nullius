import { useEffect, useRef, useState } from 'react'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { RunTimeline } from './RunTimeline'
import type { Event } from '../../data'

const SPEEDS = [1, 2, 4, 8]
const MAX_GAP_S = 2.5   // long waits in the real run (minutes per seed) are shortened

// Plays a finished run back with the same renderer as the live view. Starts at the end state; Replay starts over.
export function ReplayPlayer({ events, codeLoader, planFull, embedded = true }: {
  events: Event[]; codeLoader?: (hash: string) => Promise<string | null>; planFull?: string | null; embedded?: boolean
}) {
  const total = events.length
  const [count, setCount] = useState(total)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const speedRef = useRef(speed)
  useEffect(() => { speedRef.current = speed }, [speed])
  useEffect(() => { setCount(total); setPlaying(false) }, [total])

  useEffect(() => {
    if (!playing) return
    if (count >= total) { setPlaying(false); return }
    const gap = count === 0 ? 0.4 : Math.min(Math.max(events[count].ts - events[count - 1].ts, 0.15), MAX_GAP_S)
    const t = window.setTimeout(() => setCount((c) => c + 1), (gap * 1000) / speedRef.current)
    return () => window.clearTimeout(t)
  }, [playing, count, total, events])

  const btn = 'inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium cursor-pointer'
  return (
    <div>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <button onClick={() => { if (count >= total) setCount(1); setPlaying((p) => !p) }} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
          {playing ? <><Pause className="w-3.5 h-3.5" />Pause</> : <><Play className="w-3.5 h-3.5" />{count >= total ? 'Replay' : 'Resume'}</>}
        </button>
        <button onClick={() => { setCount(1); setPlaying(true) }} className={`${btn} border-line text-fg-3 hover:bg-raised`}><RotateCcw className="w-3.5 h-3.5" />Restart</button>
        <div role="group" aria-label="Speed" className="inline-flex gap-1">
          {SPEEDS.map((s) => (
            <button key={s} onClick={() => setSpeed(s)} aria-pressed={speed === s}
              className={`rounded-md border px-2.5 py-1.5 font-mono text-xs cursor-pointer ${speed === s ? 'border-line-stronger text-fg' : 'border-line text-muted hover:text-fg-3'}`}>{s}x</button>
          ))}
        </div>
        <span className="ml-auto font-mono text-[11px] text-muted">{count} of {total} events</span>
      </div>
      <RunTimeline events={events.slice(0, count)} mode="replay" running={count < total} embedded={embedded} codeLoader={codeLoader} planFull={planFull} />
    </div>
  )
}
