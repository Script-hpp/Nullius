import { useEffect } from 'react'
import { Play } from 'lucide-react'
import { SiteHeader } from './Layout'
import { GoArrow } from './ui'
import { SiteFooter } from './SiteFooter'
import { TIMELAPSE_RUN, Problem, Steps, HeroCard, DoesItWork, Findings, Limits, TryIt, useFeatured } from './LandingSections'
import { WrittenText } from './hero/WrittenText'
import { FilmButton } from './FilmModal'
import { ScrollScene } from '../scroll/ScrollScene'

// Start page: hero with the real result card, then one story while scrolling: the problem, how it works, does it work, findings, limits, try it.
// The card numbers come from the exported example run (see LandingSections and heroCard.ts).

function Hero({ run }: { run: ReturnType<typeof useFeatured> }) {
  // Text and buttons are on screen from the first paint; only the small lettering above the headline is animated.
  return (
    <section className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-6xl flex-col justify-between gap-10 px-6 pb-8 pt-10 md:px-8 lg:pt-14">
      <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-16">
        <div>
          <WrittenText className="w-[132px] text-fg-4 sm:w-[160px]" />
          <h1 className="font-grotesk m-0 mt-5 text-[2.5rem] font-semibold leading-[1.02] tracking-[-0.045em] sm:text-6xl lg:text-[4.1rem]">
            <span className="text-muted">Papers make claims.</span>{' '}
            <span className="text-fg">We measure them.</span>
          </h1>
          <p className="mb-0 mt-6 max-w-lg text-balance text-base leading-relaxed text-fg-4 sm:text-lg">
            Before you build on a paper, Nullius rebuilds the experiment and measures the claim independently.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-7 gap-y-4">
            <a href="#catalog" className="nx-go inline-flex items-center gap-2 rounded-full bg-inverse px-7 py-3 text-sm font-medium text-on-inverse no-underline transition-opacity hover:opacity-85">
              Check a paper <GoArrow className="h-4 w-4 shrink-0" />
            </a>
            <a href={TIMELAPSE_RUN} className="inline-flex items-center gap-2 text-sm text-fg-3 no-underline hover:text-fg">
              <Play className="h-4 w-4" aria-hidden />Watch a check in 30 seconds
            </a>
          </div>
          <div className="mt-4"><FilmButton /></div>
        </div>
        <HeroCard run={run} />
      </div>
      <BuiltWith />
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

// scrollTo: id of a section to open on (routes #/lesson, #/tavily), so a link can point into the start page.
export function Landing({ scrollTo }: { scrollTo?: string }) {
  const run = useFeatured()
  useEffect(() => {
    if (!scrollTo) return
    const t = setTimeout(() => document.getElementById(scrollTo)?.scrollIntoView({ behavior: 'auto' }), 50)
    return () => clearTimeout(t)
  }, [scrollTo])
  return (
    <div className="landing relative min-h-screen font-sans">
      <SiteHeader currentHash="" className="sticky top-0 z-20" />
      <Hero run={run} />
      <Problem />
      <ScrollScene id="opening" />
      <Steps />
      <DoesItWork />
      <Findings />
      <Limits />
      <TryIt />
      <SiteFooter className="pb-16" />
    </div>
  )
}
