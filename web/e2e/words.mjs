// Visible word count per page (details closed, hidden elements skipped). Needs a preview on :4173 and playwright-core; CHROME=<path> optional.
import { chromium } from 'playwright-core'
const BASE = process.env.BASE ?? 'http://localhost:4173/'
const RUN = 'kipf_gcn_2017'
const pages = { landing: '#/', benchmark: '#benchmark', catalog: '#catalog' }
const b = await chromium.launch(process.env.CHROME ? { executablePath: process.env.CHROME } : {})
const p = await b.newPage({ viewport: { width: 1280, height: 800 } })
const count = () => p.evaluate(() => {
  const root = document.body
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let n = 0, t
  const out = []
  while ((t = w.nextNode())) {
    const el = t.parentElement
    if (!el || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(el.tagName)) continue
    if (el.closest('details:not([open])') && !el.closest('summary')) continue
    const cs = getComputedStyle(el)
    if (el.closest('.sr-only')) continue
    if (cs.display === 'none' || cs.visibility === 'hidden') continue
    if (!el.getClientRects().length) continue
    const k = t.textContent.trim().split(/\s+/).filter(Boolean).length
    n += k
  }
  return n
})
const res = {}
for (const [k, h] of Object.entries(pages)) {
  await p.goto(BASE + h); await p.waitForTimeout(2000); res[k] = await count()
}
const idx = await (await fetch(BASE + 'data/index.json')).json()
const ids = idx.map((r) => r.id)
const pick = process.env.RUN ?? ids.find((i) => i.startsWith('kipf')) ?? ids[0]
await p.goto(`${BASE}#/run/${pick}`); await p.waitForTimeout(2500); res['run:' + pick] = await count()
console.log(JSON.stringify(res, null, 1))
await b.close()
