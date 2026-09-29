import { Background, Controls, MarkerType, ReactFlow, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useMemo } from 'react'
import { ATTEMPT, TONE_VAR, de2en } from '../labels'
import type { Json } from '../data'

// Basis-Snapshot -> je Code-Version ein Knoten -> je Lauf (Sandbox-Branch) ein Blatt. Reparaturen verbinden Code-Versionen.
export function BranchTree({ dossier, claimId, events, onSelect, selected }: {
  dossier: Json; claimId: string; events: Json[]; onSelect: (a: Json) => void; selected: string | null
}) {
  const { nodes, edges } = useMemo(() => {
    const attempts: Json[] = (dossier.attempts ?? []).filter((a: Json) => a.claim_id === claimId)
    const versions: string[] = []
    for (const a of attempts) if (!versions.includes(a.code_hash)) versions.push(a.code_hash)
    const nodes: Node[] = []
    const edges: Edge[] = []
    const snap = attempts[0]?.parent_snapshot ?? 'snapshot'
    nodes.push({
      id: 'snap', position: { x: 0, y: 0 }, draggable: false, selectable: false,
      data: { label: (<div><div className="text-xs muted">base snapshot</div><div className="mono text-xs">{String(snap).replace('repro-snapshot-', '').slice(0, 30)}</div></div>) },
      style: { background: 'var(--panel-2)', color: 'var(--text)', border: '1px solid var(--line)', borderRadius: 10, width: 200, padding: 8 },
    })
    const perVersion: Record<string, number> = {}
    const versionCols: Record<string, number> = {}
    const versionX: Record<string, number> = {}
    let x = 260
    for (const v of versions) {
      const n = attempts.filter((a) => a.code_hash === v).length
      versionCols[v] = Math.ceil(n / 5)
      versionX[v] = x
      x += versionCols[v] * 172 + 60
    }
    versions.forEach((v, vi) => {
      const list = attempts.filter((a) => a.code_hash === v)
      nodes.push({
        id: `v-${v}`, position: { x: versionX[v], y: 0 }, draggable: false, selectable: false,
        data: { label: (<div><div className="text-xs muted">code version {vi + 1}</div><div className="mono text-xs">{v}</div><div className="text-xs muted">{list.length} run{list.length > 1 ? 's' : ''}</div></div>) },
        style: { background: 'color-mix(in srgb, var(--violet) 14%, var(--panel))', color: 'var(--text)', border: '1px solid color-mix(in srgb, var(--violet) 40%, transparent)', borderRadius: 10, width: 180, padding: 8 },
      })
      const first = attempts.find((a) => a.code_hash === v)
      edges.push({ id: `e-snap-${v}`, source: vi === 0 ? 'snap' : `v-${versions[vi - 1]}`, target: `v-${v}`, animated: vi > 0,
        label: vi > 0 ? `repair ⟵ ${String(events.find((e) => e.type === 'repair' && e.from_hash === versions[vi - 1])?.error ?? '').split('\n').filter(Boolean).slice(-1)[0]?.slice(0, 40) ?? ''}` : undefined,
        labelStyle: { fill: 'var(--muted)', fontSize: 10 }, labelBgStyle: { fill: 'var(--panel)' },
        style: { stroke: vi > 0 ? 'var(--violet)' : 'var(--line)' }, markerEnd: { type: MarkerType.ArrowClosed, color: vi > 0 ? '#b394ff' : '#243044' } })
      void first
    })
    attempts.forEach((a) => {
      const k = (perVersion[a.code_hash] = (perVersion[a.code_hash] ?? 0) + 1) - 1
      const perCol = 5   // hoechstens 5 Laeufe uebereinander, dann naechste Spalte (30 Laeufe bleiben lesbar)
      const row = (k % perCol) + 1, colIdx = Math.floor(k / perCol)
      const st = de2en(ATTEMPT, a.status)
      const col = TONE_VAR[st.tone]
      const id = `a-${a.branch_id}`
      nodes.push({
        id, position: { x: versionX[a.code_hash] + colIdx * 172, y: 90 + (row - 1) * 66 }, draggable: false,
        data: { label: (<div className="text-left"><div className="text-xs"><b>seed {a.seed}</b> · <span style={{ color: col }}>{st.label}</span></div><div className="mono text-xs muted">{a.measured != null ? a.measured.toFixed(4) : (a.error_summary ?? '').split('\n').filter(Boolean).slice(-1)[0]?.slice(0, 32) ?? '–'}</div></div>) },
        style: { background: `color-mix(in srgb, ${col} 10%, var(--panel))`, color: 'var(--text)', border: `${selected === a.branch_id ? 2 : 1}px solid ${col}`, borderRadius: 8, width: 160, padding: 6 },
      })
      edges.push({ id: `e-${id}`, source: `v-${a.code_hash}`, target: id, type: 'smoothstep', style: { stroke: col, strokeOpacity: 0.45 } })
    })
    return { nodes, edges }
  }, [dossier, claimId, events, selected])

  return (
    <div style={{ height: 460 }} className="panel-2 overflow-hidden">
      <ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.15 }} minZoom={0.2} proOptions={{ hideAttribution: true }}
        nodesConnectable={false} onNodeClick={(_, n) => {
          const a = (dossier.attempts ?? []).find((x: Json) => `a-${x.branch_id}` === n.id)
          if (a) onSelect(a)
        }}>
        <Background color="#243044" gap={22} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
