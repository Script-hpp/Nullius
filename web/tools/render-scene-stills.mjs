// Renders the stills of the landing scenes from the real stage code (tools/stills.html -> SceneStage), so the placeholder is exactly the first
// frame of the live stage. Needs a running vite dev server and playwright-core with chromium; ffmpeg turns the PNGs into WebP.
//   npx vite --port 5391 &   PLAYWRIGHT=/path/to/playwright-core/index.mjs node tools/render-scene-stills.mjs http://localhost:5391
// Start frames come from SCENES[id].start, poster frames from SCENES[id].poster (src/scroll/scenes.ts, kept in sync below).
import { execFileSync } from 'node:child_process'
import { mkdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const base = process.argv[2] ?? 'http://localhost:5391'
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright-core')
const SHOTS = { opening: { start: 2500, poster: 19500 }, placebo: { start: 0, poster: 8700 } }
const out = new URL('../public/img/scenes/', import.meta.url).pathname
mkdirSync(out, { recursive: true })
const tmp = join(tmpdir(), 'scene-stills'); mkdirSync(tmp, { recursive: true })

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1200, height: 675 } })
for (const [id, at] of Object.entries(SHOTS)) for (const kind of ['start', 'poster']) for (const pal of ['paper', 'night']) {
  await page.goto(`${base}/tools/stills.html?id=${id}&t=${at[kind]}&p=${pal}`)
  await page.waitForSelector('#r > div')
  await page.evaluate(() => document.fonts.ready)
  await page.waitForTimeout(600)
  const png = join(tmp, `${id}-${kind}-${pal}.png`)
  await page.locator('#r').screenshot({ path: png })
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', png, '-c:v', 'libwebp', '-quality', '82', '-compression_level', '6', join(out, `${id}-${kind}-${pal}.webp`)])
}
await browser.close()
rmSync(tmp, { recursive: true, force: true })
