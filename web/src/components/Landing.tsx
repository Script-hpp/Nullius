import { useEffect } from 'react'
import { ChevronDown, Play } from 'lucide-react'
import { ThemeToggle } from './ThemeSwitch'
import { Wordmark } from './Wordmark'
import { SiteFooter } from './SiteFooter'
import { REPO } from '../data/benchmark'
import { TIMELAPSE_RUN, Problem, Steps, CertificatePreview, Proof, Lesson, TavilySection, Limits, TryIt, useFeatured } from './LandingSections'
import { WrittenText } from './hero/WrittenText'
import { prefersReducedMotion } from './hero/config'

// Start page: a full-screen hero with a handwritten motto that is drawn stroke by stroke (under a second), above one sentence, one button and one quiet link.
// Further down it reads as one story while scrolling (see LandingSections). The certificate numbers come from the exported example run.

function TopBar() {
  const link = 'hidden text-sm text-fg-4 transition-colors hover:text-fg no-underline sm:inline'
  return (
    <header className="absolute inset-x-0 top-0 z-10 flex h-16 items-center justify-between gap-3 px-4 sm:px-10">
      <Wordmark />
      <div className="flex items-center gap-3 sm:gap-7">
        <a href="#catalog" className={link}>Catalog</a>
        <a href="#benchmark" className={link}>Benchmark</a>
        <a href={REPO} target="_blank" rel="noreferrer" className={link}>GitHub</a>
        <a href="#catalog" className="shrink-0 whitespace-nowrap rounded-full border border-fg px-3 py-1.5 sm:px-4 text-[11px] font-medium uppercase tracking-[0.2em] text-fg no-underline transition-colors hover:bg-fg hover:text-bg">Try it</a>
        <ThemeToggle />
      </div>
    </header>
  )
}

function Hero() {
  // Text and buttons are on screen from the first paint; only the lettering above them is animated.
  return (
    <section className="relative flex min-h-screen flex-col items-center justify-center px-6 pb-16 pt-24 text-center">
      <WrittenText className="w-[min(880px,88vw)] text-fg" />
      <div className="mx-auto mt-12 max-w-4xl sm:mt-16">
        <h1 className="font-grotesk m-0 text-[2.2rem] font-semibold leading-[0.98] tracking-[-0.05em] sm:text-6xl lg:text-7xl">
          <span className="block text-muted">papers make claims.</span>
          <span className="block text-fg">we measure them.</span>
        </h1>
        <p className="mx-auto mb-0 mt-7 max-w-xl text-balance text-base leading-relaxed text-fg-4 sm:text-lg">
          Nullius rebuilds a paper&rsquo;s experiment and measures it independently. Tavily researches live.
        </p>
      </div>
      <div className="mt-10 flex flex-col items-center gap-3">
        <a href={TIMELAPSE_RUN} className="inline-flex items-center gap-2 rounded-full bg-inverse px-7 py-3 text-sm font-medium text-on-inverse no-underline transition-colors hover:bg-inverse-hover">
          <Play className="h-4 w-4" aria-hidden />See it in action
        </a>
      </div>
      <div className="mt-12 flex flex-col items-center gap-3 sm:mt-14">
        <button type="button" onClick={() => document.getElementById('claim')?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' })}
          aria-label="Scroll down" className="cursor-pointer border-0 bg-transparent p-1 text-muted hover:text-fg-3">
          <ChevronDown className="h-5 w-5" />
        </button>
        <BuiltWith />
      </div>
    </section>
  )
}

// "Built with" strip, kept small. Logos come only from the companies' own brand kits, unmodified, light and dark variant as the kit ships them:
// - Nebius: https://nebius.com/media-kit (NEBIUS-logos 07.25.zip, SVGs/NEBIUS-outline-black.svg and -white.svg). The page states no usage terms; used
//   unaltered and with clear space, to name the services we use.
// - Tavily: https://tavily.com/brand (/logos/tavily-full.svg and tavily-full-offwhite.svg). Terms: unaltered proportions, clear space at least the icon
//   height, black on light and off-white on dark, no effects; referential identification of Tavily products is allowed without suggesting endorsement.
// - NVIDIA: no downloadable kit, and https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/ requires written approval for any use
//   outside its listed cases, so NVIDIA stays text until approved.
const BUILT_WITH: { name: string; product: string; logo?: { light: string; dark: string; h: string } }[] = [
  { name: 'NVIDIA', product: 'Nemotron' },
  { name: 'Nebius', product: 'Token Factory, Sandboxes', logo: { light: 'logos/nebius-black.svg', dark: 'logos/nebius-white.svg', h: 'h-5 sm:h-6' } },
  { name: 'Tavily', product: 'Live research', logo: { light: 'logos/tavily-full.svg', dark: 'logos/tavily-full-offwhite.svg', h: 'h-6 sm:h-7' } },
]

function BuiltWith() {
  return (
    <div className="flex flex-col items-center gap-3" aria-label="Built with">
      <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-muted">Built with</div>
      <ul className="m-0 flex list-none flex-wrap items-start justify-center gap-x-7 gap-y-3 p-0 sm:gap-x-10">
        {BUILT_WITH.map((b) => (
          <li key={b.name} className="flex flex-col items-center gap-1.5">
            <div className="flex h-7 items-center">
              {b.logo ? (
                <>
                  <img src={b.logo.light} alt={b.name} className={`logo-light w-auto ${b.logo.h}`} />
                  <img src={b.logo.dark} alt={b.name} className={`logo-dark w-auto ${b.logo.h}`} />
                </>
              ) : (
                <span className="font-grotesk text-lg font-semibold tracking-[0.04em] text-fg-3">{b.name}</span>
              )}
            </div>
            <span className="text-[10.5px] tracking-wide text-muted">{b.product}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// scrollTo: id of a section to open on (route #/lesson), so a link can point into the start page.
export function Landing({ scrollTo }: { scrollTo?: string }) {
  const run = useFeatured()
  useEffect(() => {
    if (!scrollTo) return
    const t = setTimeout(() => document.getElementById(scrollTo)?.scrollIntoView({ behavior: 'auto' }), 50)
    return () => clearTimeout(t)
  }, [scrollTo])
  return (
    <div className="landing relative min-h-screen font-sans">
      <TopBar />
      <Hero />
      <Problem run={run} />
      <Steps />
      <CertificatePreview run={run} />
      <TavilySection />
      <Proof />
      <Lesson />
      <Limits />
      <TryIt />
      <SiteFooter className="pb-16" />
    </div>
  )
}
