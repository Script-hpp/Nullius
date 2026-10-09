import { Eye, FlaskConical, ShieldCheck } from 'lucide-react'
import { Glossed } from './Glossed'
import { FIGURES } from '../data/benchmark'

// "Why you can trust this number": three plain reasons, then the limits. The placebo number is the fixed v4 figure in data/benchmark.ts.
export function Trust({ fidelityUnchecked }: { fidelityUnchecked: boolean }) {
  const [acceptedWrong, ofWrong] = FIGURES.placebo.v4   // benchmark v4, results/benchmark_v4.md
  const points = [
    { icon: Eye, title: 'The agent only predicts', text: 'The agent writes a program that outputs predictions. It never computes or reports a score itself.' },
    { icon: ShieldCheck, title: 'The true answers stay with the harness', text: 'The true labels are never in the agent’s data. The harness loads them itself after the agent has finished, and the agent’s code runs without network access.' },
    {
      icon: FlaskConical, title: 'Wrong claims are rejected',
      text: `We test the checker with placebo claims: targets that are deliberately wrong. A trustworthy checker must reject them. In benchmark v4, ${acceptedWrong} of ${ofWrong} placebo claims were accepted as reproduced.`,
    },
  ]
  return (
    <section aria-label="Why you can trust this number" className="min-w-0">
      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Why you can trust this number</div>
      <ul className="m-0 mt-6 grid list-none gap-0 border-t border-line p-0 md:grid-cols-3">
        {points.map((p, i) => (
          <li key={p.title} className={`py-8 ${i > 0 ? 'border-t border-line md:border-l md:border-t-0 md:pl-8' : ''} ${i < 2 ? 'md:pr-8' : ''}`}>
            <p.icon className="h-5 w-5 text-fg-4" />
            <h3 className="font-grotesk m-0 mt-4 text-xl font-semibold tracking-[-0.03em] text-fg">{p.title}</h3>
            <p className="mb-0 mt-3 text-[14px] leading-relaxed text-fg-4"><Glossed text={p.text} /></p>
          </li>
        ))}
      </ul>
      <div className="border-t border-line pt-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Limits</div>
        <p className="mb-0 mt-3 max-w-3xl text-[14px] leading-relaxed text-fg-4">
          {fidelityUnchecked && <><b className="text-fg-3">Method fidelity not checked.</b> The number was measured by an independent program; whether our code follows the paper&rsquo;s method is not verified. </>}
          A result is a measurement with our setup, not a finding about the paper.
        </p>
      </div>
    </section>
  )
}
