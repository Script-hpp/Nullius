// Export the existing product wordmark to outlined SVG via Chromium PDF + Cairo.
import {chromium} from 'playwright-core'
import {writeFileSync} from 'node:fs'
import {execFileSync} from 'node:child_process'
const out=new URL('./out/',import.meta.url).pathname
writeFileSync(`${out}logo.html`,`<style>@font-face{font-family:Inter;src:url(inter-tight-latin-wght-normal.woff2)}@font-face{font-family:Serif;src:url(instrument-serif-latin-400-italic.woff2);font-style:italic}body{margin:0;color:#252520}.logo{display:flex;align-items:center;justify-content:center;gap:24px;width:700px;height:180px;font:600 72px Inter;letter-spacing:-2.88px}.logo i{font:italic 400 82px Serif;letter-spacing:0}</style><div class="logo"><svg width="76" height="76" viewBox="0 0 24 24" fill="none" stroke="#252520" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5.5H6v13h3M15 5.5h3v13h-3"/><circle cx="12" cy="12" r="2.3" fill="#252520" stroke="none"/></svg><span>nullius <i>in verba</i></span></div>`)
const b=await chromium.launch();const p=await b.newPage();await p.goto(`file://${out}logo.html`);await p.evaluate(()=>document.fonts.ready);await p.pdf({path:`${out}logo.pdf`,width:'700px',height:'180px',margin:{top:0,bottom:0,left:0,right:0}});await b.close()
execFileSync('pdftocairo',['-svg',`${out}logo.pdf`,`${out}logo.svg`])
