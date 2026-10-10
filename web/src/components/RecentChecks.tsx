import { ArrowRight } from 'lucide-react'
import { evidenceHref, useCatalog } from '../catalog'

// Curated examples outside the eight benchmark units. Measurements stay in the exported catalog.
const examples = ['disc_2303_06053', 'disc_2405_00946']

export function RecentChecks() {
  const catalog = useCatalog()
  const entries = examples.flatMap((id) => {
    const entry = catalog?.find((e) => e.id === id)
    return entry?.run_id && entry.verdict?.measured_mean != null ? [entry] : []
  })
  if (entries.length !== examples.length) return null
  return (
    <section id="recent-checks" aria-label="Beyond the eight benchmark units" className="mx-auto max-w-4xl scroll-mt-8 px-6 py-16">
      <p className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Beyond the eight benchmark units</p>
      <h2 className="font-grotesk mb-0 mt-4 text-3xl font-semibold leading-snug tracking-[-0.03em] text-fg">Two more papers. Two targets reached.</h2>
      <p className="mb-0 mt-4 max-w-2xl text-[16px] leading-relaxed text-fg-4">Both papers forecast future sensor readings. Nullius wrote its own program, ran it, and checked how close its predictions came to the real readings.</p>
      <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4">How to read these results: MSE measures prediction error; lower is better. Both measurements are within 0.01 of the paper&rsquo;s target. Open a card to follow the evidence.</p>
      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        {entries.map((e) => (
          <article key={e.id} className="rounded-xl border border-line bg-surface p-6">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-grotesk m-0 text-xl font-semibold text-fg">{e.claim.method}</h3>
              <span className="text-[11px] font-medium text-tone-green">Target reached</span>
            </div>
            <p className="mb-0 mt-2 text-[12px] text-muted">{e.task} · {e.claim.horizon}-hour forecast · MSE</p>
            <dl className="mb-0 mt-6 grid grid-cols-2 gap-4">
              <div><dt className="text-[11px] uppercase tracking-wider text-muted">Claimed</dt><dd className="m-0 mt-2 font-mono text-2xl text-fg">{e.claim.target?.toFixed(3)}</dd></div>
              <div><dt className="text-[11px] uppercase tracking-wider text-muted">Measured</dt><dd className="m-0 mt-2 font-mono text-2xl text-fg">{e.verdict!.measured_mean!.toFixed(4)}</dd></div>
            </dl>
            <p className="mb-0 mt-3 text-[12px] text-fg-4">± {e.verdict!.measured_std?.toFixed(4) ?? '–'} over {e.verdict!.seeds} repeated runs · tolerance ±{e.verdict!.tolerance}</p>
            {e.cost_gross_usd != null && <p className="mb-0 mt-2 text-[12px] text-muted">Recorded inference cost: ${e.cost_gross_usd.toFixed(3)} incl. VAT</p>}
            <a href={evidenceHref(e)!} className="mt-6 inline-flex items-center gap-2 text-sm text-fg underline underline-offset-4 hover:text-fg-max">Inspect {e.claim.method} evidence <ArrowRight className="h-4 w-4" aria-hidden /></a>
          </article>
        ))}
      </div>
      <p className="mb-0 mt-5 text-[13px] leading-relaxed text-fg-4">Two additional examples, selected after development. Preliminary results; matching the number still needs a review of the method.</p>
      <details className="mt-3 text-[13px] leading-relaxed text-fg-4">
        <summary className="cursor-pointer text-fg underline underline-offset-4">How these examples were selected</summary>
        <p className="mb-0 mt-3">Candidates were partly selected by the team. These checks are post hoc, outside the eight benchmark units, and do not establish a general success rate. Method fidelity remains unchecked.</p>
      </details>
      <p className="mb-0 mt-3 text-[13px] leading-relaxed text-fg-4">The catalog also records SL-GCN missing its target and HaKAN failing in setup. <a href="#catalog" className="text-fg underline underline-offset-4">Explore all outcomes</a>.</p>
    </section>
  )
}
