// Scroll behaviour on route changes. Needs `npm run build && npx vite preview --port 4173` and playwright-core
// (`npm i --no-save playwright-core`); CHROME=<path to chrome> picks the browser. Not part of `npm test`.
import assert from 'node:assert/strict'
import { chromium } from 'playwright-core'

const BASE = process.env.BASE ?? 'http://localhost:4173/'
const RUN = 'disc_1609_02907--20261009-115012'
const b = await chromium.launch(process.env.CHROME ? { executablePath: process.env.CHROME } : {})
const p = await b.newPage({ viewport: { width: 1280, height: 800 } })
const inner = () => p.evaluate(() => document.getElementById('app-scroll')?.scrollTop ?? 0)
const win = () => p.evaluate(() => scrollY)

// The scroll container is <main id="app-scroll"> (Layout); the start page itself scrolls the window.
// Start page scrolled down, "See it in action": the player wrapper lies fully inside the container, below its top edge.
const inView = (sel) => p.evaluate((sel) => {
  const m = document.getElementById('app-scroll'), el = document.querySelector(sel)
  if (!m || !el) return null
  const mr = m.getBoundingClientRect(), r = el.getBoundingClientRect()
  return { top: r.top - mr.top, bottom: r.bottom - mr.top, h: mr.height, scrollTop: m.scrollTop }
}, sel)
await p.goto(`${BASE}#/`); await p.waitForTimeout(1500)
await p.evaluate(() => window.scrollTo(0, 1500))
assert.equal(await win(), 1500)
await p.click('a[href*="play=1"]'); await p.waitForTimeout(2500)
assert.equal(await win(), 0)
const film = await inView('[data-testid="film"]')
assert.ok(film, 'player not rendered')
assert.ok(film.top >= 0 && film.bottom <= film.h, `player not fully visible: ${JSON.stringify(film)}`)
assert.ok(film.scrollTop < 100, `container scrolled past the page top: ${film.scrollTop}`)

// the run is already open and scrolled down, the ?play=1 link is followed: player in view again
await p.evaluate((r) => { location.hash = `#/run/${r}?tab=code` }, RUN); await p.waitForTimeout(500)
await p.evaluate(() => document.getElementById('app-scroll').scrollTo(0, 900))
await p.evaluate((r) => { location.hash = `#/run/${r}?play=1` }, RUN); await p.waitForTimeout(1500)
const again = await inView('[data-testid="film"]')
assert.ok(again && again.top >= 0 && again.bottom <= again.h, `player not in view after the link on an open run: ${JSON.stringify(again)}`)

// same page, other tab: position stays
await p.goto(`${BASE}#/run/${RUN}`); await p.waitForTimeout(1500)
await p.evaluate(() => document.getElementById('app-scroll').scrollTo(0, 700))
await p.evaluate((r) => { location.hash = `#/run/${r}?tab=code` }, RUN); await p.waitForTimeout(600)
assert.equal(await inner(), 700)

// catalog to certificate: top
await p.goto(`${BASE}#/catalog`); await p.waitForTimeout(1200)
await p.evaluate(() => document.getElementById('app-scroll').scrollTo(0, 500))
await p.evaluate((r) => { location.hash = `#/run/${r}` }, RUN); await p.waitForTimeout(1200)
assert.equal(await inner(), 0)

// short window (tab bar below the fold) and a phone: a new page opens at the very top
for (const vp of [{ width: 1280, height: 560 }, { width: 390, height: 700 }]) {
  await p.setViewportSize(vp)
  await p.goto(`${BASE}#/catalog`); await p.waitForTimeout(1200)
  await p.evaluate(() => document.getElementById('app-scroll').scrollTo(0, 300))
  await p.evaluate((r) => { location.hash = `#/run/${r}` }, RUN); await p.waitForTimeout(1200)
  assert.equal(await inner(), 0, `run page not at top in ${JSON.stringify(vp)}`)
}

// the Tavily chip on the verdict card opens the Sources & Tavily tab
await p.setViewportSize({ width: 1280, height: 800 })
await p.goto(`${BASE}#/run/${RUN}`); await p.waitForTimeout(1500)
await p.click('[data-testid="tavily-chip"]'); await p.waitForTimeout(600)
assert.equal(await p.getAttribute('#tab-sources', 'aria-selected'), 'true')
console.log('scroll e2e ok')
await b.close()
