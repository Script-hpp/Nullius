import { useEffect, useMemo, useState } from 'react'
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { BookOpen, Box, CheckCircle2, CircleDashed, FileCode2, FileText, ListChecks, Loader2, Quote, Search, ShieldCheck, XCircle, Gavel, Lightbulb, AlertTriangle } from 'lucide-react'
import { TONE_TEXT_VAR, TONE_VAR } from '../../labels'
import type { Event } from '../../data'
import { buildRunModel } from './phases'
import { ExpandView } from './RunTimeline'
import { buildStepGraph, type StepKind, type StepNode } from './stepGraph'

const ICON: Record<StepKind, typeof Search> = {
  paper: BookOpen, src_paper: Search, src_readme: FileText, src_repo: Search, src_error: AlertTriangle,
  claim: Quote, plan: ListChecks, code: FileCode2, seed: Box, harness: ShieldCheck, verdict: Gavel,
}
const NODE_W = 208
const SEED_W = 92
const NODE_H = 54      // fixed sizes: the view can be fitted before the browser has measured anything
const SEED_H = 44
const GAP = 24
const ROW_H = 112
const SEED_ROW_H = 62

type Data = { step: StepNode; selected: boolean }

function StepNodeView({ data }: NodeProps<Node<Data>>) {
  const { step, selected } = data
  const Icon = ICON[step.kind]
  const color = TONE_VAR[step.tone]
  const small = step.kind === 'seed'
  const border = step.state === 'failed' ? TONE_VAR.red : step.state === 'done' ? (step.tone === 'gray' ? 'var(--c-line-stronger)' : color) : step.state === 'active' ? 'var(--c-fg-4)' : 'var(--c-line)'
  const handle = { opacity: 0, pointerEvents: 'none' as const }
  return (
    <div role="button" tabIndex={0} aria-label={`${step.title}: ${step.sub ?? step.state}`} aria-pressed={selected}
      style={{ width: small ? SEED_W : NODE_W, height: small ? SEED_H : NODE_H, borderColor: selected ? 'var(--c-fg)' : border, borderStyle: step.state === 'pending' ? 'dashed' : 'solid', borderWidth: selected ? 2 : 1 }}
      className={`rounded-lg bg-surface-2 text-left cursor-pointer ${small ? 'px-2 py-1.5' : 'px-3 py-2.5'} ${step.state === 'pending' ? 'opacity-70' : ''} ${step.state === 'active' ? 'animate-pulse' : ''}`}>
      <Handle id="t" type="target" position={Position.Top} style={handle} />
      <Handle id="s" type="source" position={Position.Bottom} style={handle} />
      <Handle id="ls" type="source" position={Position.Left} style={handle} />
      <Handle id="lt" type="target" position={Position.Left} style={handle} />
      <Handle id="rs" type="source" position={Position.Right} style={handle} />
      <Handle id="rt" type="target" position={Position.Right} style={handle} />
      <div className="flex items-center gap-1.5">
        <Icon className={`shrink-0 ${small ? 'w-3 h-3' : 'w-3.5 h-3.5'}`} style={{ color: TONE_TEXT_VAR[step.tone] }} />
        <span className={`truncate font-semibold text-fg ${small ? 'text-[11px]' : 'text-[12px]'}`}>{step.title}</span>
        <span className="ml-auto shrink-0">
          {step.state === 'done' ? <CheckCircle2 className="w-3 h-3 text-tone-green" /> : step.state === 'failed' ? <XCircle className="w-3 h-3 text-tone-red" /> : step.state === 'active' ? <Loader2 className="w-3 h-3 animate-spin text-fg-3" /> : <CircleDashed className="w-3 h-3 text-subtle" />}
        </span>
      </div>
      {step.sub && <div className={`mt-0.5 truncate text-muted ${small ? 'font-mono text-[10px]' : 'text-[11px]'}`}>{step.sub}</div>}
    </div>
  )
}

const nodeTypes = { step: StepNodeView }

function layout(steps: StepNode[]): Record<string, { x: number; y: number }> {
  const seedRows = steps.filter((s) => s.kind === 'seed').reduce((m, s) => Math.max(m, s.row - 4), 0)
  const yOf = (row: number) => (row <= 4 ? row * ROW_H : row < 5 + seedRows ? 5 * ROW_H + (row - 5) * SEED_ROW_H : 5 * ROW_H + seedRows * SEED_ROW_H + 24 + (row - 5 - seedRows) * ROW_H)
  const out: Record<string, { x: number; y: number }> = {}
  for (const s of steps) {
    const w = s.kind === 'seed' ? SEED_W : NODE_W
    out[s.id] = { x: (s.col - (s.rowSize - 1) / 2) * (w + GAP) - w / 2, y: yOf(s.row) }
  }
  return out
}

const STATE_TEXT = { pending: 'Not started', active: 'Happening now', done: 'Done', failed: 'Failed' } as const

function Detail({ step, loader, planFull }: { step: StepNode; loader?: (hash: string) => Promise<string | null>; planFull?: string | null }) {
  const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'
  return (
    <div className="rounded-xl border border-line bg-panel/50 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="m-0 text-base font-semibold text-fg">{step.title}</h3>
        <span className="text-[11px] uppercase tracking-wider text-muted">{STATE_TEXT[step.state]}</span>
      </div>
      <div className="mt-4 grid gap-5 md:grid-cols-2">
        <div>
          <p className={label}>What happens here</p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-fg-3">{step.what}</p>
        </div>
        <div>
          <p className={label}>Why this matters</p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-fg-3">{step.why}</p>
        </div>
      </div>
      <div className="mt-5 border-t border-line/60 pt-4">
        <p className={label}>From this run</p>
        {step.facts.length || step.expand ? (
          <>
            {step.facts.length > 0 && (
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12px] m-0">
                {step.facts.map((f, i) => (
                  <div key={i} className="contents">
                    <dt className="text-muted whitespace-pre">{f.label}</dt>
                    <dd className="m-0 min-w-0 break-words text-fg-2">{f.value}</dd>
                  </div>
                ))}
              </dl>
            )}
            {step.expand && <ExpandView ex={step.expand} loader={loader} planFull={planFull} />}
          </>
        ) : (
          <p className="mt-2 text-[12px] text-muted">Nothing recorded for this step yet.</p>
        )}
      </div>
    </div>
  )
}

// Refits the view when the set of nodes changes (repo search and error search appear during a run), not on every event.
function Refit({ signature }: { signature: string }) {
  const { fitView } = useReactFlow()
  useEffect(() => {
    const t = window.setTimeout(() => fitView({ padding: 0.12, duration: 250 }), 120)
    return () => window.clearTimeout(t)
  }, [signature, fitView])
  return null
}

// The run as a flow graph. Fills step by step in a live run; for a finished run it is the entry point of the evidence page.
export function StepGraph(props: Parameters<typeof StepGraphInner>[0]) {
  return <ReactFlowProvider><StepGraphInner {...props} /></ReactFlowProvider>
}

function StepGraphInner({ events, running, codeLoader, planFull, metricHint }: {
  events: Event[]; running: boolean; codeLoader?: (hash: string) => Promise<string | null>; planFull?: string | null; metricHint?: string | null
}) {
  const model = useMemo(() => buildRunModel(events, { running, metricHint }), [events, running, metricHint])
  const graph = useMemo(() => buildStepGraph(events, model, running), [events, model, running])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = graph.nodes.find((n) => n.id === selectedId) ?? null
  const pos = useMemo(() => layout(graph.nodes), [graph.nodes])
  const signature = graph.nodes.map((n) => n.id).join('|')   // new optional nodes (repo search, error search) refit the view

  const nodes: Node<Data>[] = graph.nodes.map((s) => ({ id: s.id, type: 'step', position: pos[s.id], data: { step: s, selected: s.id === selectedId }, draggable: false, selectable: false,
    width: s.kind === 'seed' ? SEED_W : NODE_W, height: s.kind === 'seed' ? SEED_H : NODE_H, measured: { width: s.kind === 'seed' ? SEED_W : NODE_W, height: s.kind === 'seed' ? SEED_H : NODE_H } }))
  const state = new Map(graph.nodes.map((n) => [n.id, n.state]))
  const edges: Edge[] = graph.edges.map((ed) => {
    const live = state.get(ed.target) === 'active' || state.get(ed.source) === 'active'
    const colour = ed.loop ? 'var(--c-tone-violet-solid)' : state.get(ed.source) === 'done' ? 'var(--c-line-stronger)' : 'var(--c-line)'
    return {
      id: ed.id, source: ed.source, target: ed.target,
      sourceHandle: ed.loop === 'left' ? 'ls' : ed.loop === 'right' ? 'rs' : 's', targetHandle: ed.loop === 'left' ? 'lt' : ed.loop === 'right' ? 'rt' : 't',
      type: 'smoothstep', animated: live || !!ed.loop, label: ed.label,
      labelStyle: { fill: 'var(--c-fg-4)', fontSize: 10 }, labelBgStyle: { fill: 'var(--c-surface-2)' }, labelBgPadding: [4, 2] as [number, number],
      style: { stroke: colour, strokeDasharray: ed.dashed ? '5 4' : undefined, strokeOpacity: ed.source.startsWith('seed_') || ed.target.startsWith('seed_') ? 0.55 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: colour },
    }
  })

  return (
    <div>
      <div className="mb-3 flex items-start gap-2 text-[12px] leading-relaxed text-muted">
        <Lightbulb className="mt-0.5 w-3.5 h-3.5 shrink-0" />
        <span>Each box is one step of the run. Click a box to see what happens there, why it matters and the real data of this run. Dashed purple lines are repair loops.</span>
      </div>
      <div className="h-[540px] overflow-hidden rounded-xl border border-line bg-surface-2" aria-label="Flow graph of the run">
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.15} maxZoom={1.6}
          proOptions={{ hideAttribution: true }} nodesConnectable={false} nodesDraggable={false} elementsSelectable={false}
          onNodeClick={(_, n) => setSelectedId((cur) => (cur === n.id ? null : n.id))} onPaneClick={() => setSelectedId(null)}>
          <Refit signature={signature} />
          <Background color="var(--c-line-strong)" gap={22} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div className="mt-4">
        {selected ? <Detail step={selected} loader={codeLoader} planFull={planFull} />
          : <p className="rounded-xl border border-dashed border-line px-5 py-4 text-[13px] text-muted">Select a step in the graph.</p>}
      </div>
    </div>
  )
}
