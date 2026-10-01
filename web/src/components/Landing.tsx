import { useState } from 'react'
import { ArrowRight, ChevronDown } from 'lucide-react'
import { ThemeSwitch } from './ThemeSwitch'
import { Wordmark } from './Wordmark'
import { WrittenText } from './hero/WrittenText'
import { prefersReducedMotion } from './hero/config'

// Start page: a full-screen hero with a handwritten motto that is drawn stroke by stroke, then one sentence and two buttons.
// Further down: claim vs. measurement, three steps, limits. The example numbers come from a real, earlier run
// (GCN on Cora, web/public/data/kipf_gcn_2017) and are labelled as such. Benchmark numbers stay TBD until the evaluation exists.

const REPO = 'https://github.com/Script-hpp/Nullius'
const EXAMPLE_RUN = '#/run/kipf_gcn_2017--20260929-130133'

function TopBar() {
  const link = 'hidden text-sm text-fg-4 transition-colors hover:text-fg no-underline sm:inline'
  return (
    <header className="absolute inset-x-0 top-0 z-10 flex h-16 items-center justify-between gap-5 px-5 sm:px-10">
      <Wordmark />
      <div className="flex items-center gap-5 sm:gap-7">
        <a href="#discover" className={link}>Discover</a>
        <a href="#catalog" className={link}>Catalog</a>
        <a href="#overview" className={link}>Runs</a>
        <a href={REPO} target="_blank" rel="noreferrer" className={link}>GitHub</a>
        <a href="#discover" className="rounded-full border border-fg px-4 py-1.5 text-[11px] font-medium uppercase tracking-[0.2em] text-fg no-underline transition-colors hover:bg-fg hover:text-bg">Try it</a>
        <ThemeSwitch />
      </div>
    </header>
  )
}

function Hero() {
  const [shown, setShown] = useState(prefersReducedMotion())   // false until the lettering is written
  const fade = (delay = 0) => ({
    style: { opacity: shown ? 1 : 0, transform: shown ? 'none' : 'translateY(10px)', transition: `opacity 900ms ease ${delay}ms, transform 900ms ease ${delay}ms` },
    'data-hero-fade': true,
  })
  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center px-6 pb-24 pt-24 text-center">
      <WrittenText onDone={() => setShown(true)} className="w-[min(880px,88vw)] text-fg" />
      {/* collapsed while the lettering is written (it sits in the middle of the screen), then opens and moves it up */}
      <div className="grid w-full" data-hero-fade style={{ gridTemplateRows: shown ? '1fr' : '0fr', transition: 'grid-template-rows 1000ms ease' }}>
        <div className="min-h-0 overflow-hidden">
          <div className="mx-auto mt-12 max-w-4xl sm:mt-16" {...fade()}>
            <h1 className="font-grotesk m-0 text-[2.6rem] font-semibold leading-[0.98] tracking-[-0.05em] sm:text-6xl lg:text-7xl">
              <span className="block text-muted">papers make claims.</span>
              <span className="block text-fg">we measure them.</span>
            </h1>
            <p className="mx-auto mb-0 mt-7 max-w-xl text-base leading-relaxed text-fg-4 sm:text-lg">
              We check machine-learning papers by rebuilding their experiments. A paper reports a number; Nullius builds the experiment again, measures it independently and tells you whether it can be reproduced.
            </p>
          </div>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4" {...fade(150)}>
            <a href="#discover" className="inline-flex items-center gap-2 rounded-full bg-inverse px-6 py-3 text-sm font-medium text-on-inverse no-underline transition-colors hover:bg-inverse-hover">
              Try it <ArrowRight className="h-4 w-4" />
            </a>
            <a href={EXAMPLE_RUN} className="inline-flex items-center gap-2 rounded-full border border-line-strong px-6 py-3 text-sm font-medium text-fg no-underline transition-colors hover:bg-raised/60">
              See a finished check
            </a>
          </div>
        </div>
      </div>
      <div className="absolute inset-x-0 bottom-6 flex flex-col items-center gap-3 px-6" {...fade(400)}>
        <button type="button" onClick={() => document.getElementById('claim')?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' })}
          aria-label="Scroll down" className="cursor-pointer border-0 bg-transparent p-1 text-muted hover:text-fg-3">
          <ChevronDown className="h-5 w-5" />
        </button>
        <p className="m-0 text-center text-[11px] uppercase tracking-[0.18em] text-muted">Nemotron · Nebius Token Factory · Nebius Sandboxes · Tavily</p>
      </div>
    </section>
  )
}

function ClaimVsMeasurement() {
  return (
    <section id="claim" className="mx-auto max-w-5xl px-6 py-24 sm:py-32">
      <div className="grid gap-14 md:grid-cols-2 md:gap-0">
        <div className="md:pr-14">
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">The paper reports</div>
          <div className="mt-6 font-mono text-6xl font-light tracking-tight text-fg-3 sm:text-7xl">81.5%</div>
          <p className="mt-6 max-w-sm text-[15px] leading-relaxed text-fg-4">
            Kipf &amp; Welling (2017) report 81.5% accuracy for their GCN model on the Cora benchmark (Table 2). A claim in words, paraphrased. Anyone can write it down.
          </p>
        </div>
        <div className="border-t border-line pt-14 md:border-l md:border-t-0 md:pl-14 md:pt-0">
          <div className="flex items-baseline justify-between gap-4 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">
            <span>We measured</span>
            <span className="font-normal normal-case tracking-normal">Example, earlier run</span>
          </div>
          <div className="mt-6 font-mono text-6xl font-light tracking-tight text-tone-green sm:text-7xl">82.1%</div>
          <p className="mt-6 max-w-sm text-[15px] leading-relaxed text-fg-4">
            <span className="text-fg">± 0.7</span> over 10 runs, scored by a program the agent cannot touch. Reproduced with our setup.
          </p>
        </div>
      </div>
    </section>
  )
}

const STEPS = [
  { n: '01', title: 'Read the claim', text: 'Tavily reads the paper and its code page. We keep only numbers that appear word for word in the text.' },
  { n: '02', title: 'Rebuild it', text: 'An AI agent writes its own version of the experiment and runs it in isolated sandboxes, one run per random seed.' },
  { n: '03', title: 'Measure it', text: 'A separate checker, which the agent cannot change, scores the results and compares them with the claim.' },
]

function Steps() {
  return (
    <section className="mx-auto max-w-5xl px-6 pb-24 sm:pb-32">
      <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">How it works</h2>
      <ol className="m-0 mt-8 grid list-none gap-0 border-t border-line p-0 md:grid-cols-3">
        {STEPS.map((s, i) => (
          <li key={s.n} className={`py-10 md:py-12 ${i > 0 ? 'border-t border-line md:border-l md:border-t-0 md:pl-10' : ''} ${i < 2 ? 'md:pr-10' : ''}`}>
            <div className="font-mono text-4xl font-light text-fg-4">{s.n}</div>
            <h3 className="font-grotesk m-0 mt-5 text-2xl font-semibold tracking-[-0.04em] text-fg">{s.title}</h3>
            <p className="mb-0 mt-3 text-[15px] leading-relaxed text-fg-4">{s.text}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

function Footer() {
  return (
    <section className="mx-auto max-w-5xl px-6 pb-28">
      <div className="grid gap-14 border-t border-line pt-14 md:grid-cols-2 md:gap-0">
        <div className="md:pr-14">
          <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Does it work?</h2>
          <p className="mb-0 mt-5 text-lg leading-relaxed text-fg-3">
            We tested it on papers with known results: <span className="font-mono text-fg">TBD of 8</span> claims matched the reference verdict with Tavily, <span className="font-mono text-fg">TBD of 8</span> without it.
          </p>
          <p className="mb-0 mt-4 max-w-md text-[13px] leading-relaxed text-muted">
            A match means the target value was reached. Whether the method matches the paper is not checked yet.
          </p>
        </div>
        <div className="md:border-l md:border-line md:pl-14">
          <h2 className="m-0 text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Built with</h2>
          <ul className="m-0 mt-5 flex list-none flex-col gap-3 p-0 text-[15px] leading-relaxed text-fg-4">
            <li><span className="font-medium text-fg">NVIDIA Nemotron</span> on Nebius Token Factory plans the experiment and writes the code.</li>
            <li><span className="font-medium text-fg">Nebius Sandboxes</span> run every experiment in isolation.</li>
            <li><span className="font-medium text-fg">Tavily</span> finds and reads the papers, live.</li>
          </ul>
        </div>
      </div>
      <div className="mt-20 flex flex-wrap items-center justify-between gap-4 border-t border-line pt-8 text-sm text-muted">
        <span>Nullius in verba: take nobody&rsquo;s word for it.</span>
        <a href="#discover" className="inline-flex items-center gap-2 text-fg no-underline hover:text-fg-max">Try it <ArrowRight className="h-4 w-4" /></a>
      </div>
    </section>
  )
}

export function Landing() {
  return (
    <div className="landing relative min-h-screen font-sans">
      <TopBar />
      <Hero />
      <ClaimVsMeasurement />
      <Steps />
      <Footer />
    </div>
  )
}
