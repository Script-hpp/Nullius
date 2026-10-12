import { useEffect, useMemo, useState } from 'react'
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { BookOpen, Box, Check, CheckCircle2, CircleDashed, FileCode2, FileText, ListChecks, Loader2, Quote, Search, ShieldCheck, XCircle, Gavel, Lightbulb, AlertTriangle } from 'lucide-react'
import { TONE_TEXT_VAR, TONE_VAR } from '../../labels'
import type { Event } from '../../data'
import { buildRunModel } from './phases'
import { ExpandView } from './RunTimeline'
import { buildStepGraph, type StepKind, type StepNode } from './stepGraph'

const ICON: Record<StepKind, typeof Search> = {
  paper: BookOpen, src_paper: Search, src_readme: FileText, src_repo: Search, src_error: AlertTriangle,
  claim: Quote, plan: ListChecks, code: FileCode2, seed: Box, harness: ShieldCheck, verdict: Gavel,
}
const NODE_W = 340     // sizes are fixed so the view can be fitted before the browser has measured anything
const NODE_H = 150     // title, one fact of the run and the full sentence on what happens (no tooltip, no clamp)
const SEED_W = 300     // single seed nodes (up to three seeds); more seeds become one node
const GAP = 28
const ROW_H = 190
const NARROW = 760     // below this viewport width the graph is one column

type Data = { step: StepNode; selected: boolean }

function StepNodeView({ data }: NodeProps<Node<Data>>) {
  const { step, selected } = data
  const Icon = ICON[step.kind]
  const color = TONE_VAR[step.tone]
  const border = step.state === 'failed' ? TONE_VAR.red : step.state === 'done' ? (step.tone === 'gray' ? 'var(--c-line-stronger)' : color) : step.state === 'active' ? 'var(--c-fg-4)' : 'var(--c-line)'
  const handle = { opacity: 0, pointerEvents: 'none' as const }
  return (
    <div role="button" tabIndex={0} aria-label={`${step.title}: ${step.sub ?? step.state}`} aria-pressed={selected}
      style={{ width: nodeW(step), height: NODE_H, borderColor: selected ? 'var(--c-fg)' : border, borderStyle: step.state === 'pending' ? 'dashed' : 'solid', borderWidth: selected ? 2 : 1 }}
      className={`rounded-lg bg-surface-2 text-left cursor-pointer overflow-hidden px-4 py-3.5 ${step.state === 'pending' ? 'opacity-70' : ''} ${step.state === 'active' ? 'animate-pulse' : ''}`}>
      <Handle id="t" type="target" position={Position.Top} style={handle} />
      <Handle id="s" type="source" position={Position.Bottom} style={handle} />
      <Handle id="ls" type="source" position={Position.Left} style={handle} />
      <Handle id="lt" type="target" position={Position.Left} style={handle} />
      <Handle id="rs" type="source" position={Position.Right} style={handle} />
      <Handle id="rt" type="target" position={Position.Right} style={handle} />
      <div className="flex items-center gap-1.5">
        <Icon className="w-4 h-4 shrink-0" style={{ color: TONE_TEXT_VAR[step.tone] }} />
        <span className="truncate text-[14px] font-semibold text-fg">{step.title}</span>
        <span className="ml-auto shrink-0">
          {step.state === 'done' ? <CheckCircle2 className="w-4 h-4 text-tone-green" /> : step.state === 'failed' ? <XCircle className="w-4 h-4 text-tone-red" /> : step.state === 'active' ? <Loader2 className="w-4 h-4 animate-spin text-fg-3" /> : <CircleDashed className="w-4 h-4 text-subtle" />}
        </span>
      </div>
      {step.sub && <div className="mt-1 truncate font-mono text-[12px] leading-snug text-fg-3">{step.sub}</div>}
      <div className="mt-1.5 text-[12.5px] leading-snug text-fg-4">{step.what}</div>
    </div>
  )
}

const nodeTypes = { step: StepNodeView }

const nodeW = (s: StepNode) => (s.kind === 'seed' && s.id !== 'seeds' ? SEED_W : NODE_W)

// Wide screens: rows as in the flow. Narrow screens: one column in reading order, so no zoom is needed.
function layout(steps: StepNode[], narrow: boolean): { pos: Record<string, { x: number; y: number }>; width: number; height: number } {
  const pos: Record<string, { x: number; y: number }> = {}
  if (narrow) {
    [...steps].sort((a, b) => a.row - b.row || a.col - b.col).forEach((s, i) => { pos[s.id] = { x: -nodeW(s) / 2, y: i * ROW_H } })
    return { pos, width: NODE_W, height: (steps.length - 1) * ROW_H + NODE_H }
  }
  let width = NODE_W
  for (const s of steps) {
    pos[s.id] = { x: (s.col - (s.rowSize - 1) / 2) * (nodeW(s) + GAP) - nodeW(s) / 2, y: s.row * ROW_H }
    width = Math.max(width, s.rowSize * (nodeW(s) + GAP))
  }
  return { pos, width, height: Math.max(...steps.map((s) => s.row)) * ROW_H + NODE_H }
}

function useNarrow() {
  const [narrow, setNarrow] = useState(() => window.innerWidth < NARROW)
  useEffect(() => {
    const f = () => setNarrow(window.innerWidth < NARROW)
    addEventListener('resize', f)
    return () => removeEventListener('resize', f)
  }, [])
  return narrow
}

const STATE_TEXT = { pending: 'Not started', active: 'Happening now', done: 'Done', failed: 'Failed' } as const

function Detail({ step, loader, planFull }: { step: StepNode; loader?: (hash: string) => Promise<string | null>; planFull?: string | null }) {
  const label = 'text-[10px] font-semibold uppercase tracking-widest text-muted'
  return (
    <div className="nx-card rounded-xl border border-line bg-panel/50 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="m-0 text-base font-semibold text-fg">{step.title}</h3>
        <span className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted">
          {step.state === 'done' && <Check className="h-3.5 w-3.5 shrink-0 text-tone-green" aria-hidden />}{STATE_TEXT[step.state]}
        </span>
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
    const t = window.setTimeout(() => fitView({ padding: 0.04, maxZoom: 1, duration: 250 }), 120)
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
  const narrow = useNarrow()
  const { pos, height } = useMemo(() => layout(graph.nodes, narrow), [graph.nodes, narrow])
  const signature = graph.nodes.map((n) => n.id).join('|') + (narrow ? '|narrow' : '')   // new optional nodes (repo search, error search) refit the view

  const nodes: Node<Data>[] = graph.nodes.map((s) => ({ id: s.id, type: 'step', position: pos[s.id], data: { step: s, selected: s.id === selectedId }, draggable: false, selectable: false,
    width: nodeW(s), height: NODE_H, measured: { width: nodeW(s), height: NODE_H } }))
  const state = new Map(graph.nodes.map((n) => [n.id, n.state]))
  // loops run outside the widest row (a row of single seeds is wider than the code and harness boxes)
  const widest = Math.max(NODE_W, ...graph.nodes.filter((n) => n.kind === 'seed').map((n) => n.rowSize * (nodeW(n) + GAP) - GAP))
  const loopOffset = Math.max(40, (widest - NODE_W) / 2 + 28)
  const edges: Edge[] = graph.edges.map((ed) => {
    const live = state.get(ed.target) === 'active' || state.get(ed.source) === 'active'
    const colour = ed.loop ? 'var(--c-tone-violet-solid)' : state.get(ed.source) === 'done' ? 'var(--c-line-stronger)' : 'var(--c-line)'
    return {
      id: ed.id, source: ed.source, target: ed.target,
      sourceHandle: ed.loop === 'left' ? 'ls' : ed.loop === 'right' ? 'rs' : 's', targetHandle: ed.loop === 'left' ? 'lt' : ed.loop === 'right' ? 'rt' : 't',
      type: 'smoothstep', animated: live || !!ed.loop, label: narrow && ed.loop ? 'repair loop' : ed.label,
      labelStyle: { fill: 'var(--c-fg-4)', fontSize: 12 }, labelBgStyle: { fill: 'var(--c-surface-2)' }, labelBgPadding: [4, 2] as [number, number],
      pathOptions: ed.loop ? { offset: narrow || ed.loop === 'left' ? 28 : loopOffset } : undefined,
      style: { stroke: colour, strokeDasharray: ed.dashed ? '5 4' : undefined, strokeOpacity: ed.source.startsWith('seed_') || ed.target.startsWith('seed_') ? 0.55 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: colour },
    }
  })

  return (
    <div>
      <div className="mb-4 flex items-start gap-2 leading-relaxed text-muted">
        <Lightbulb className="mt-0.5 w-3.5 h-3.5 shrink-0" />
        <span className="text-[14px] text-fg-3">Each box is one step of the run, with a sentence on what happens in it. Click a box to see why it matters and the real data of this run. Dashed purple lines are repair loops.</span>
      </div>
      <div style={{ height: Math.max(420, height + 48) }} className="overflow-hidden rounded-xl border border-line bg-surface-2" aria-label="Flow graph of the run">
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.04, maxZoom: 1 }} minZoom={0.3} maxZoom={1.6}
          zoomOnScroll={false} preventScrolling={false} zoomOnDoubleClick={false} panOnDrag={!narrow} zoomOnPinch={!narrow}
          proOptions={{ hideAttribution: true }} nodesConnectable={false} nodesDraggable={false} elementsSelectable={false}
          onNodeClick={(_, n) => setSelectedId((cur) => (cur === n.id ? null : n.id))} onPaneClick={() => setSelectedId(null)}>
          <Refit signature={signature} />
          <Background color="var(--c-line-strong)" gap={22} />
          {!narrow && <Controls showInteractive={false} />}
        </ReactFlow>
      </div>
      <div className="mt-4">
        {selected ? <Detail step={selected} loader={codeLoader} planFull={planFull} />
          : <p className="rounded-xl border border-dashed border-line px-5 py-4 text-[13px] text-muted">Select a step in the graph.</p>}
      </div>
    </div>
  )
}
