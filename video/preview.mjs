import {chromium} from 'playwright-core'
const out=new URL('./out/',import.meta.url).pathname
const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}})
await p.goto(`file://${out}film.html`);await p.evaluate(()=>document.fonts.ready)
for(const t of [8,20,32,46,55,61,69,75,81,92,102,107,116,130,144]){await p.evaluate(t=>window.seek(t),t);await p.screenshot({path:`${out}frame-${t}.png`})}
await p.goto(`file://${process.cwd()}/docs/img/nullius-logo.svg`);await p.screenshot({path:`${out}logo-preview.png`});await b.close()
