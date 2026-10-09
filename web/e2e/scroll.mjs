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

// start page scrolled down, "See it in action": top of the page, player fully in view
await p.goto(`${BASE}#/`); await p.waitForTimeout(1500)
await p.evaluate(() => window.scrollTo(0, 1500))
assert.equal(await win(), 1500)
await p.click('a[href*="play=1"]'); await p.waitForTimeout(2500)
assert.equal(await win(), 0)
const film = await p.evaluate(() => { const m = document.getElementById('app-scroll').getBoundingClientRect(); const r = document.querySelector('#app-scroll main > div').getBoundingClientRect(); return { top: r.top - m.top, bottom: r.bottom - m.top, h: m.height } })
assert.ok(film.top >= 0 && film.bottom <= film.h, `player not fully visible: ${JSON.stringify(film)}`)

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
console.log('scroll e2e ok')
await b.close()
