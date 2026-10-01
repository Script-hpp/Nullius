import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { Scene as SceneView } from './Scenes'
import { ACTOR, FILM_MS, SCENE_LABEL, durationText, eventsForClaim, presentScenes, repairScene, sceneAt, scenePlan } from '../../film'
import { TONE_VAR, type Tone } from '../../labels'
import type { Event, Json } from '../../data'

type Loader = (hash: string) => Promise<string | null>

// "Play this check in 30 seconds" as a film: a 16:9 stage with six scenes made only of the data of this run. One clock drives everything;
// every scene is a pure function of its progress, so pause, jump and replay are exact. With reduced motion the scenes are shown finished.
export function Film({ events, dossier, claim, title, tone, resultLabel, sentence, fidelityUnchecked, overall, fmt, fmtSpread, plan, finalCode, codeLoader, autoplay }: {
  events: Event[]; dossier: Json; claim: Json; title: string; tone: Tone; resultLabel: string; sentence: string; fidelityUnchecked: boolean; overall?: string | null
  fmt: (x: number) => string; fmtSpread: (x: number) => string; plan: string | null; finalCode: string | null; codeLoader: Loader; autoplay: boolean
}) {
  const reduce = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
  const claimEvents = useMemo(() => eventsForClaim(events, claim.id), [events, claim.id])
  const repair = useMemo(() => repairScene(claimEvents, claim.id), [claimEvents, claim.id])
  const firstHash = useMemo(() => String((claimEvents.find((e) => e.type === 'code_written') ?? claimEvents.find((e) => e.type === 'attempt'))?.code_hash ?? '') || null, [claimEvents])
  const [loadedCode, setLoadedCode] = useState<string | null>(null)
  useEffect(() => { let live = true; if (firstHash) codeLoader(firstHash).then((c) => { if (live) setLoadedCode(c) }); return () => { live = false } }, [firstHash, codeLoader])
  const firstCode = loadedCode ?? finalCode   // the first code version, or the final one in older exports

  const present = useMemo(() => presentScenes({ events: claimEvents, claim, hasPlan: !!plan, hasCode: !!(firstCode ?? finalCode), hasVerdict: (dossier.verdicts ?? []).length > 0, repair: !!repair }), [claimEvents, claim, plan, firstCode, finalCode, dossier, repair])
  const scenes = useMemo(() => scenePlan(present), [present])
  const [ms, setMs] = useState(0)
  const [playing, setPlaying] = useState(autoplay)
  const raf = useRef(0)
  const last = useRef(0)

  useEffect(() => {
    if (!playing) return
    last.current = performance.now()
    const tick = (now: number) => {
      const dt = now - last.current
      last.current = now
      let done = false
      setMs((m) => { const n = m + dt; if (n >= FILM_MS) { done = true; return FILM_MS } return n })
      if (done) setPlaying(false)
      else raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing])

  const { index, p } = sceneAt(scenes, ms)
  const scene = scenes[index]
  const ended = ms >= FILM_MS
  const jump = useCallback((i: number) => { setMs(scenes[i].start); setPlaying(true) }, [scenes])
  const toggle = () => { if (ended) { setMs(0); setPlaying(true) } else setPlaying((v) => !v) }
  const took = durationText(events)
  if (!scene) return null
  const btn = 'inline-flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium'

  return (
    <section id="sec-film" aria-label="This check as a film" className="@container min-w-0 pt-6">
      {/* 16:9 on wide screens, but never taller than the space left under the header and above the controls: the stage gets narrower, its content scales with it */}
      <div className="relative mx-auto aspect-[3/4] w-full overflow-hidden rounded-2xl border border-line bg-surface [container-type:inline-size] @xl:aspect-video @xl:w-[min(100%,calc((100svh-12.5rem)*16/9))]">
        <div className="absolute inset-x-0 top-0 flex items-center justify-between gap-3 px-[4cqw] pt-[3cqw] @xl:px-[3cqw] @xl:pt-[2cqw]">
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={scene.id} initial={reduce ? false : { opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={reduce ? undefined : { opacity: 0 }} transition={{ duration: 0.25 }}
              className="inline-flex items-center gap-2 rounded-full border border-line-strong bg-raised px-3 py-1 text-[clamp(10px,1.5cqw,13px)] font-semibold uppercase tracking-[0.14em] text-fg-3">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: scene.id === 'verdict' ? TONE_VAR[tone] : 'var(--c-fg-4)' }} />{ACTOR[scene.id]}
            </motion.span>
          </AnimatePresence>
          <span className="font-mono text-[clamp(10px,1.4cqw,12px)] text-muted">{index + 1} / {scenes.length}</span>
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={scene.id} className="absolute inset-0 pt-[9cqw] @xl:pt-[5.5cqw]"
            initial={reduce ? false : { opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={reduce ? undefined : { opacity: 0, y: -10 }} transition={{ duration: 0.3 }}>
            <SceneView id={scene.id} p={reduce ? 1 : p} data={{ events: claimEvents, dossier, claim, title, tone, resultLabel, sentence, fidelityUnchecked, overall, fmt, fmtSpread, plan, firstCode, firstHash, repair, codeLoader }} />
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button type="button" onClick={toggle} className={`${btn} border-transparent bg-inverse text-on-inverse hover:bg-inverse-hover`}>
          {playing ? <><Pause className="h-3.5 w-3.5" />Pause</> : ended ? <><RotateCcw className="h-3.5 w-3.5" />Play again</> : <><Play className="h-3.5 w-3.5" />{ms === 0 ? 'Play' : 'Resume'}</>}
        </button>
        <div role="group" aria-label="Scenes" className="flex items-center gap-1.5">
          {scenes.map((s, i) => (
            <button key={s.id} type="button" onClick={() => jump(i)} aria-label={`Scene ${i + 1}: ${SCENE_LABEL[s.id]}`} aria-current={i === index ? 'step' : undefined} title={SCENE_LABEL[s.id]}
              className="group flex cursor-pointer flex-col items-center gap-1 border-0 bg-transparent p-1">
              <span className={`h-2 rounded-full transition-all ${i === index ? 'w-7 bg-fg-3' : i < index || ended ? 'w-2 bg-fg-4' : 'w-2 bg-line-strong'}`} />
              <span className={`hidden text-[10px] uppercase tracking-wider @xl:block ${i === index ? 'text-fg-3' : 'text-muted group-hover:text-fg-3'}`}>{SCENE_LABEL[s.id]}</span>
            </button>
          ))}
        </div>
        {took && <span className="ml-auto text-[12px] text-muted">The real run took {took}. This film shows it in 30 seconds.</span>}
      </div>
    </section>
  )
}
