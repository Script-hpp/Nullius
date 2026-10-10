process.env.PW_TEST_SCREENSHOT_NO_FONTS_READY='1'
import { chromium } from 'playwright-core'
import { mkdirSync, writeFileSync } from 'node:fs'
const base=process.argv[2] || 'http://127.0.0.1:5186'
const out=new URL('./out/',import.meta.url).pathname
mkdirSync(out,{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1})
const errors=[]; page.on('pageerror',e=>errors.push(e.message))
// Capture is strictly local/static. Block API traffic rather than accidentally starting a paid job.
await page.route('**/api/**',r=>r.fulfill({status:503,contentType:'application/json',body:'{"live":false,"error":"Recorded evidence capture"}'}))
await page.addInitScript(()=>localStorage.setItem('nullius-theme','light'))
const run='disc_2501_01087--20261010-161916'
async function open(hash){await page.goto(`${base}/${hash}`); await page.waitForTimeout(1000);await page.evaluate(()=>document.fonts.ready)}
async function snap(name,loc){if(loc){await loc.evaluate(el=>el.scrollIntoView({block:'start'}));await page.waitForTimeout(300)} await page.screenshot({path:`${out}${name}.png`});console.log(name)}
await open('#/');await snap('homepage')
await open(`#/run/${run}`);await page.getByRole('button',{name:'Export certificate (PDF)'}).waitFor();await snap('result')
await page.getByRole('button',{name:'Inspect sources & setup'}).click();await page.waitForTimeout(300)
const target=page.locator('#f-target > summary');if(await target.count() && !await target.locator('..').evaluate(e=>e.open))await target.click()
await snap('sources',target)
await open(`#/run/${run}?tab=code`)
const plan=page.locator('#f-plancode > summary');await plan.click();await snap('code',plan)
await open(`#/run/${run}`)
const popupPromise=page.waitForEvent('popup');await page.getByRole('button',{name:'Export certificate (PDF)'}).click();const pdf=await popupPromise;console.log('popup opened')
await pdf.waitForTimeout(500);console.log('popup ready');await pdf.pdf({path:`${out}glinear-certificate.pdf`,format:'A4',printBackground:true,timeout:15000});console.log('pdf saved')
await pdf.screenshot({path:`${out}pdf.png`});await pdf.close()
// Record the real What-if interaction, including its measured values remaining fixed.
await open(`#/run/${run}`)
await page.locator('#sec-whatif > summary').click();const slider=page.locator('#sec-whatif input[type=range]').first();await slider.scrollIntoViewIfNeeded();await snap('whatif-before')
await slider.fill(await slider.getAttribute('max'));await snap('whatif-after')
writeFileSync(`${out}capture-check.json`,JSON.stringify({base,run,errors,api:'blocked',pdf:'glinear-certificate.pdf'},null,2))
if(errors.length)throw Error(errors.join('\n'))
await browser.close()
