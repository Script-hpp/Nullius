import { ArrowRight, Check, FileText, GitBranch, Quote, Ruler, ShieldCheck, X } from 'lucide-react'

// Start page: one sentence, "words vs. measurement" as a picture, three steps, one button.
// The example numbers come from a real, earlier run (GCN on Cora, web/public/data/kipf_gcn_2017) and are labelled as such.

function WordsVsMeasurement() {
  return (
    <div className="relative grid gap-4 md:grid-cols-[1fr_auto_1fr] md:gap-0 items-stretch">
      <div className="rounded-xl border border-line bg-panel/50 p-6 flex flex-col gap-4">
        <div className="flex items-center gap-2 text-muted text-[10px] font-semibold uppercase tracking-widest">
          <Quote className="w-3.5 h-3.5" /> The paper claims
        </div>
        <p className="text-fg-3 text-sm leading-relaxed">
          Kipf &amp; Welling (2017) report <span className="text-fg font-medium">81.5%</span> accuracy for their GCN model on the Cora benchmark (Table 2).
        </p>
        <div className="mt-auto text-[11px] text-muted">A claim in words, paraphrased. Anyone can write it down.</div>
      </div>

      {/* Divider between the two cards: a line with a ruler mark (horizontal on phones, vertical from md) */}
      <div className="flex md:flex-col items-center justify-center gap-0 py-1 md:py-0 md:px-5" aria-hidden="true">
        <div className="h-px w-10 md:h-10 md:w-px bg-line-strong" />
        <div className="w-9 h-9 rounded-full border border-line-strong bg-surface flex items-center justify-center text-fg-4">
          <Ruler className="w-4 h-4" />
        </div>
        <div className="h-px w-10 md:h-10 md:w-px bg-line-strong" />
      </div>

      <div className="rounded-xl border border-line bg-panel/50 p-6 flex flex-col gap-4">
        <div className="flex items-center justify-between text-muted text-[10px] font-semibold uppercase tracking-widest">
          <span className="flex items-center gap-2"><Ruler className="w-3.5 h-3.5" /> We measured</span>
          <span className="normal-case tracking-normal font-normal">Example, earlier run</span>
        </div>
        <div className="flex items-end gap-3">
          <span className="font-mono text-4xl font-semibold text-tone-green leading-none">82.1%</span>
          <span className="font-mono text-xs text-fg-4 pb-1">± 0.7 over 10 runs</span>
        </div>
        <div className="flex items-center gap-2 text-tone-green text-xs font-medium uppercase tracking-wider">
          <Check className="w-4 h-4" /> Reproduced with our setup
        </div>
        <div className="mt-auto text-[11px] text-muted">A measurement. Taken by a program the agent cannot touch.</div>
      </div>
    </div>
  )
}

const STEPS = [
  { icon: FileText, title: 'Read the claim', text: 'Tavily reads the paper and its code page. We keep only numbers that appear word for word in the text.' },
  { icon: GitBranch, title: 'Rebuild it', text: 'An AI agent writes its own version of the experiment and runs it in isolated sandboxes, one run per random seed.' },
  { icon: ShieldCheck, title: 'Measure it', text: 'A separate checker, which the agent cannot change, scores the results and compares them with the claim.' },
]

function Steps() {
  return (
    <ol className="grid gap-4 md:grid-cols-3 list-none p-0 m-0">
      {STEPS.map((s, i) => (
        <li key={s.title} className="rounded-xl border border-line bg-panel/50 p-6 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <s.icon className="w-5 h-5 text-fg-3" />
            <span className="font-mono text-[11px] text-muted">0{i + 1}</span>
          </div>
          <h3 className="text-base font-semibold text-fg">{s.title}</h3>
          <p className="text-[13px] text-fg-4 leading-relaxed m-0">{s.text}</p>
        </li>
      ))}
    </ol>
  )
}

export function Landing() {
  return (
    <div className="mx-auto max-w-5xl px-4 md:px-8 pb-20">
      <section className="pt-8 md:pt-16 pb-10 md:pb-14 max-w-3xl">
        <div className="text-muted text-[11px] font-semibold uppercase tracking-widest mb-4">Nullius in verba: take nobody&rsquo;s word for it</div>
        <h1 className="text-3xl md:text-5xl font-semibold tracking-tight text-fg leading-tight m-0">
          We check machine-learning papers by rebuilding their experiments.
        </h1>
        <p className="text-fg-4 text-base md:text-lg leading-relaxed mt-5 mb-8">
          A paper reports a number. Nullius builds the experiment again, measures it independently and tells you whether the number can be reproduced.
        </p>
        <div className="flex flex-wrap items-center gap-4">
          <a href="#discover" className="px-5 py-2.5 bg-inverse text-on-inverse text-sm font-medium rounded-lg hover:bg-inverse-hover transition-colors shadow-sm flex items-center gap-2 no-underline">
            Try it <ArrowRight className="w-4 h-4" />
          </a>
          <a href="#catalog" className="text-sm text-fg-4 hover:text-fg transition-colors">Browse finished checks</a>
        </div>
      </section>

      <section className="pb-12" aria-label="Claim and measurement">
        <WordsVsMeasurement />
      </section>

      <section className="pb-12" aria-label="How it works">
        <h2 className="text-[11px] font-semibold tracking-widest uppercase text-muted mb-4">How it works</h2>
        <Steps />
      </section>

      <section className="pb-12 grid gap-4 md:grid-cols-2" aria-label="Results and building blocks">
        <div className="rounded-xl border border-line bg-panel/50 p-6 flex flex-col gap-3">
          <h2 className="text-[11px] font-semibold tracking-widest uppercase text-muted m-0">Does it work?</h2>
          <p className="text-fg-3 text-sm leading-relaxed m-0">
            We tested it on papers with known results: <span className="font-mono text-fg">TBD of 8</span> claims matched the reference verdict with Tavily, <span className="font-mono text-fg">TBD of 8</span> without it.
          </p>
          <p className="text-[12px] text-muted leading-relaxed m-0">
            <X className="inline w-3.5 h-3.5 mr-1 align-text-bottom" />
            A match means the target value was reached. Whether the method matches the paper is not checked yet.
          </p>
        </div>
        <div className="rounded-xl border border-line bg-panel/50 p-6 flex flex-col gap-3">
          <h2 className="text-[11px] font-semibold tracking-widest uppercase text-muted m-0">Built with</h2>
          <ul className="list-none p-0 m-0 flex flex-col gap-2 text-sm text-fg-3">
            <li><span className="text-fg font-medium">NVIDIA Nemotron</span> on Nebius Token Factory plans the experiment and writes the code.</li>
            <li><span className="text-fg font-medium">Nebius Sandboxes</span> run every experiment in isolation.</li>
            <li><span className="text-fg font-medium">Tavily</span> finds and reads the papers, live.</li>
          </ul>
        </div>
      </section>
    </div>
  )
}
