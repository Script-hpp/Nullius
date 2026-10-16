import { getText, loadIndex, loadRun, type Event, type Json } from '../data'
import { parseBenchmark, type BenchVersion } from '../benchmark'
import { loadSummaries, type PaperSummary } from '../summaries'

// Everything the scenes show comes from exported files in web/public/data: the film run (dossier, events), the run index, the catalog and the
// benchmark export. Nothing is typed in by hand. Trimmed from the film's stage data to what the three landing scenes read.

// The one take, the placebo and the receipt follow this run: Kipf GCN on Cora, ten seeds, a real crash, a planted claim in the same run
// (Token Factory, 2 Oct 2026). The start page features a newer run (web/src/featured.ts); the film keeps this one.
export const FILM_RUN = 'kipf_gcn_2017--20261002-110746'

export type Seed = { seed: number; value: number; seconds: number; ts: number; branch: string; stdout: string | null }
export type StageData = {
  title: string; byline: string
  claim: Json; placebo: Json | null
  verdict: Json; placeboVerdict: Json | null
  seeds: Seed[]                       // final code version, one value per seed, in the order the seeds finished
  wall: { text: string; byline: string }[]     // real claims (benchmark and catalog), the featured run's claim first
  bench: BenchVersion | null
  retrievedAt: string
}

// "Kipf & Welling, ICLR 2017" or "Veličković et al., ICLR 2018", from the paper heads in discovery/summaries.json
function bylineOf(sums: Record<string, PaperSummary>, paperId: string): string {
  const s = Object.values(sums).find((x) => x.paper_ids?.includes(paperId))
  if (!s?.authors?.length) return ''
  const last = s.authors.map((a) => a.split(' ').slice(-1)[0])
  const who = last.length === 1 ? last[0] : last.length === 2 ? `${last[0]} & ${last[1]}` : `${last[0]} et al.`
  const when = s.venue?.name ?? (s.year ? String(s.year) : '')
  return when ? `${who}, ${when}` : who
}

// "shchur_gat_cora_acc" -> GAT on "Cora, random splits"; "dlinear_etth1_96_mse" -> DLinear on ETTh1
export function unitParts(claim: string) {
  const parts = claim.split('_')
  const random = parts[0] === 'shchur'
  const model = (random ? parts[1] : parts[0]).replace(/^([dn])linear$/, (_m, c: string) => c.toUpperCase() + 'Linear').replace(/^(gcn|gat|sgc|dgi)$/, (m) => m.toUpperCase())
  return { model, data: parts.includes('etth1') ? 'ETTh1' : random ? 'Cora, random splits' : 'Cora' }
}

export async function loadStageData(): Promise<StageData> {
  const index = await loadIndex()
  const sums = await loadSummaries()
  const ref = index.find((r) => r.id === FILM_RUN)
  if (!ref) throw new Error(`film run ${FILM_RUN} is not in index.json`)
  const run = await loadRun(ref)
  const d = run.dossier
  const ev: Event[] = run.events
  const claim = d.claims.find((c: Json) => c.origin === 'paper')
  const placebo = d.claims.find((c: Json) => c.origin === 'placebo') ?? null
  const verdict = d.verdicts.find((v: Json) => v.claim_id === claim.id)
  const placeboVerdict = placebo ? d.verdicts.find((v: Json) => v.claim_id === placebo.id) ?? null : null

  const att = ev.filter((e) => e.type === 'attempt' && e.claim_id === claim.id)
  const finalHash = [...att].reverse().find((a) => a.measured != null)?.code_hash
  const bySeed = new Map<number, Seed>()
  for (const a of att) if (a.code_hash === finalHash && a.measured != null) bySeed.set(Number(a.seed), { seed: Number(a.seed), value: Number(a.measured), seconds: Number(a.seconds), ts: a.ts, branch: String(a.branch_id ?? ''), stdout: a.stdout_tail ? String(a.stdout_tail).trim() : null })
  const seeds = [...bySeed.values()].sort((a, b) => a.ts - b.ts)
  const mean = seeds.reduce((s, x) => s + x.value, 0) / seeds.length
  if (Math.abs(mean - verdict.measured_mean) > 1e-6) console.warn('stage: seed values do not give the verdict mean', mean, verdict.measured_mean)

  const paperSrc = (d.sources ?? []).find((s: Json) => s.kind === 'paper')

  const bench = parseBenchmark(await getText('benchmark.json'))?.versions[0] ?? null

  // the wall of claims for the opening: every benchmark claim and every catalog claim, the featured one first
  const wall: StageData['wall'] = []
  const add = (text: string, paperId: string) => { if (!wall.some((w) => w.text === text)) wall.push({ text, byline: bylineOf(sums, paperId) }) }
  const featured = bench?.units.find((u) => u.paper === ref.paper_id)
  for (const u of [...(featured ? [featured] : []), ...(bench?.units ?? [])]) {
    const { model, data } = unitParts(u.claim)
    add(u.metric === 'accuracy' ? `${model} achieves ${(u.target * 100).toFixed(1)}% accuracy on ${data}` : `${model} achieves ${u.target} MSE on ${data}`, u.paper)
  }
  const catalog = JSON.parse((await getText('catalog.json')) ?? '[]') as Json[]
  for (const e of catalog) {
    const c = e.claim ?? {}
    if (!c.method || c.target == null) continue
    if (e.origin === 'reference') continue   // benchmark papers listed in the catalog: their claims are on the wall already
    // accuracy comes as a fraction (reference entries) or in percent (discovered entries), as the paper prints it
    const acc = c.target <= 1 ? (c.target * 100).toFixed(1) : String(c.target)
    add(c.metric === 'accuracy' ? `${c.method} achieves ${acc}% accuracy on ${e.dataset}` : `${c.method} achieves ${c.target} MSE on ${e.dataset}`, e.id)
  }

  return {
    title: ref.title, byline: bylineOf(sums, ref.paper_id),
    claim, placebo, verdict, placeboVerdict, seeds,
    retrievedAt: paperSrc?.retrieved_at ?? '',
    wall, bench,
  }
}
