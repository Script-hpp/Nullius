import { CheckCircle2, MinusCircle } from 'lucide-react'
import { Badge } from './ui'
import { Glossed } from './Glossed'
import { NextStep, PageIntro } from './page'
import { useBenchmark, type BenchArm, type BenchUnit, type BenchVersion } from '../benchmark'
import { RESULT, de2en } from '../labels'

const pct = (x: number, metric: string) => (metric === 'accuracy' ? `${(x * 100).toFixed(1)}%` : x.toFixed(3))
const verdictOf = (v: string) => de2en(RESULT, v)

// One arm of one unit: the majority verdict and one small mark per generation.
function ArmCell({ arm, reference }: { arm: BenchArm; reference: string }) {
  const agrees = arm.majority === reference
  const r = RESULT[arm.majority]
  return (
    <div>
      <div className="flex items-center gap-2">
        {agrees ? <CheckCircle2 className="h-4 w-4 shrink-0 text-tone-green" aria-label="matches the reference" /> : <MinusCircle className="h-4 w-4 shrink-0 text-subtle" aria-label="does not match the reference" />}
        <span className="text-[13px] font-medium text-fg">{r ? verdictOf(arm.majority).label : arm.majority || 'no majority'}</span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1" aria-label="verdict of each generation">
        {arm.verdicts.map((v, i) => <span key={i} title={RESULT[v] ? verdictOf(v).label : v} className={`inline-block h-2 w-2 rounded-full ${RESULT[v]?.tone === 'green' ? 'bg-tone-green-solid' : RESULT[v]?.tone === 'amber' ? 'bg-tone-amber-solid' : RESULT[v]?.tone === 'red' ? 'bg-tone-red-solid' : 'bg-line-stronger'}`} />)}
      </div>
    </div>
  )
}

function UnitRow({ u }: { u: BenchUnit }) {
  return (
    <tr className="border-t border-line align-top">
      <td className="py-5 pr-6">
        <div className="font-grotesk text-[16px] font-semibold leading-snug tracking-[-0.02em] text-fg">{u.title}</div>
        <div className="mt-1 max-w-xs text-[12px] leading-relaxed text-muted">{u.claim}</div>
      </td>
      <td className="py-5 pr-6 font-mono text-[13px] text-fg-3">{pct(u.target, u.metric)}</td>
      <td className="py-5 pr-6"><Badge tone={RESULT[u.reference_verdict] ? verdictOf(u.reference_verdict).tone : 'gray'}>{RESULT[u.reference_verdict] ? verdictOf(u.reference_verdict).label : u.reference_verdict}</Badge></td>
      <td className="py-5 pr-6"><ArmCell arm={u.arms.A} reference={u.reference_verdict} /></td>
      <td className="py-5 pr-6"><ArmCell arm={u.arms.B} reference={u.reference_verdict} /></td>
      <td className="py-5 text-[12px] leading-relaxed text-fg-4">
        <div>Without Tavily: {u.placebo.A.filter((v) => v === 'reproduziert').length} of {u.placebo.A.length} accepted</div>
        <div>With Tavily: {u.placebo.B.filter((v) => v === 'reproduziert').length} of {u.placebo.B.length} accepted</div>
      </td>
    </tr>
  )
}

function VersionBlock({ v }: { v: BenchVersion }) {
  const s = v.summary
  return (
    <div>
      <p className="font-grotesk m-0 max-w-4xl text-2xl font-semibold leading-snug tracking-[-0.03em] text-fg sm:text-3xl">
        With Tavily, {s.agree_B} of {s.n_units} paper-claim units matched the reference verdict; without Tavily, {s.agree_A}. Method fidelity was not checked.
      </p>
      <dl className="m-0 mt-10 grid gap-0 border-y border-line md:grid-cols-4">
        {[
          { k: 'With Tavily', val: `${s.agree_B} of ${s.n_units}`, sub: 'units matched the reference' },
          { k: 'Without Tavily', val: `${s.agree_A} of ${s.n_units}`, sub: 'units matched the reference' },
          { k: 'Placebo claims accepted', val: String(s.placebo_passed), sub: 'wrong targets that passed as reproduced' },
          { k: 'Runs', val: String(v.n_runs), sub: `code ${v.code_commit.slice(0, 7)}, ${v.date}` },
        ].map((x, i) => (
          <div key={x.k} className={`py-6 ${i > 0 ? 'border-t border-line md:border-l md:border-t-0 md:pl-8' : ''}`}>
            <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">{x.k}</dt>
            <dd className="m-0 mt-2 font-mono text-4xl font-light tracking-tight text-fg">{x.val}</dd>
            <p className="mb-0 mt-2 text-[13px] leading-relaxed text-fg-4">{x.sub}</p>
          </div>
        ))}
      </dl>

      <ul className="m-0 mt-10 list-none space-y-2 p-0 text-[14px] leading-relaxed text-fg-3">
        <li className="flex items-center gap-3"><CheckCircle2 className="h-4 w-4 shrink-0 text-tone-green" />The verdict that most generations agree on matches the reference verdict.</li>
        <li className="flex items-center gap-3"><span className="inline-flex shrink-0 gap-1"><i className="inline-block h-2 w-2 rounded-full bg-tone-green-solid" /><i className="inline-block h-2 w-2 rounded-full bg-tone-amber-solid" /><i className="inline-block h-2 w-2 rounded-full bg-tone-red-solid" /></span>One dot is one generation: the whole check started from scratch.</li>
      </ul>

      <div className="mt-8 overflow-x-auto">
        <table className="w-full min-w-[860px] border-collapse text-left">
          <thead>
            <tr className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">
              <th className="pb-3 pr-6 font-semibold">Paper and claim</th><th className="pb-3 pr-6 font-semibold">Claimed</th><th className="pb-3 pr-6 font-semibold">Reference verdict</th>
              <th className="pb-3 pr-6 font-semibold">Without Tavily</th><th className="pb-3 pr-6 font-semibold">With Tavily</th><th className="pb-3 font-semibold">Placebo control</th>
            </tr>
          </thead>
          <tbody>{v.units.map((u) => <UnitRow key={u.paper + u.claim} u={u} />)}</tbody>
        </table>
      </div>
    </div>
  )
}

// Methodology in words, independent of any numbers.
function Method() {
  const items = [
    ['Papers with known results', 'We picked papers whose result has been checked by our own reference solution, written by hand and run in the same harness.'],
    ['With and without Tavily', 'Every claim runs twice: once with Tavily reading the paper and its code page, once without. The comparison shows what Tavily adds.'],
    ['Fresh starts, majority verdict', 'Every claim is generated several times from scratch. The verdict most generations agree on counts for the claim.'],
    ['Placebo claims', 'Next to every real claim we run a placebo: the same experiment with a deliberately wrong target. A trustworthy checker must reject it.'],
  ]
  return (
    <section aria-label="How the benchmark works" className="border-t border-line pt-10">
      <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">How the benchmark works</h2>
      <ol className="m-0 mt-6 grid list-none gap-0 p-0 md:grid-cols-2">
        {items.map(([t, d], i) => (
          <li key={t} className={`py-5 ${i % 2 ? 'md:border-l md:border-line md:pl-10' : 'md:pr-10'} ${i > 1 ? 'border-t border-line' : ''}`}>
            <div className="font-mono text-2xl font-light text-fg-4">{String(i + 1).padStart(2, '0')}</div>
            <h3 className="font-grotesk m-0 mt-3 text-xl font-semibold tracking-[-0.03em] text-fg">{t}</h3>
            <p className="mb-0 mt-2 text-[15px] leading-relaxed text-fg-4"><Glossed text={d} /></p>
          </li>
        ))}
      </ol>
      <p className="mb-0 mt-6 max-w-3xl text-[13px] leading-relaxed text-muted">A match means the target value was reached according to our harness. Whether the code follows the paper&rsquo;s method is not checked. The reference verdicts come from our own reference solutions, not from an independent study.</p>
    </section>
  )
}

export function Benchmark() {
  const bench = useBenchmark()
  const [latest, ...older] = bench?.versions ?? []
  return (
    <main className="mx-auto max-w-6xl px-4 pb-16 md:px-8">
      <PageIntro kicker="Benchmark" title="Does it work?"
        what="We ran the whole procedure on papers whose results are known and counted how often Nullius agrees with the known result, with and without Tavily."
        look={latest ? 'Read the sentence at the top, then the table: a check mark means that arm matched the reference verdict.' : 'Results are published here once the evaluation has run. Until then, read how the benchmark works.'} />
      {bench === undefined ? <p className="text-muted">Loading benchmark results…</p>
        : latest ? (
          <>
            <section aria-label={`Benchmark ${latest.id}`}>
              <div className="mb-8 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Latest evaluation · {latest.id}</div>
              <VersionBlock v={latest} />
            </section>
            {older.length > 0 && (
              <section className="mt-16 border-t border-line pt-10" aria-label="Earlier evaluations">
                <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Earlier evaluations</h2>
                <div className="mt-4 divide-y divide-line border-y border-line">
                  {older.map((v) => (
                    <details key={v.id} className="group py-5">
                      <summary className="flex cursor-pointer list-none flex-wrap items-baseline justify-between gap-3 text-fg">
                        <span className="font-grotesk text-xl font-semibold tracking-[-0.03em]">{v.id} <span className="font-sans text-[13px] font-normal text-muted">{v.date}</span></span>
                        <span className="text-[13px] text-fg-4">With Tavily {v.summary.agree_B} of {v.summary.n_units} · without {v.summary.agree_A} · open for details</span>
                      </summary>
                      <div className="mt-8"><VersionBlock v={v} /></div>
                    </details>
                  ))}
                </div>
              </section>
            )}
          </>
        ) : (
          <section className="rounded-xl border border-line bg-surface p-8" aria-label="No results yet">
            <p className="font-grotesk m-0 text-2xl font-semibold tracking-[-0.03em] text-fg">Results follow after the evaluation</p>
            <p className="mb-0 mt-3 max-w-2xl text-[15px] leading-relaxed text-fg-4">No benchmark results are published on this site yet. Nothing here is estimated: the numbers appear when the evaluation has finished and was exported.</p>
          </section>
        )}
      <div className="mt-16"><Method /></div>
      <NextStep title="Check a paper yourself" text="Discover finds papers that report a result on a benchmark we can rebuild; the catalog lists the ones that were already checked." label="Discover papers" href="#discover" />
    </main>
  )
}
