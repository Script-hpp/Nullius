// Frame-exact local render. Run compose.py and capture.mjs first; no provider requests.
import { chromium } from 'playwright-core'
import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
const out=new URL('./out/',import.meta.url).pathname
const rows=JSON.parse(readFileSync(`${out}timeline.json`,'utf8'))
const fps=24,total=rows.reduce((a,s)=>a+s.duration,0)
const start=Number(process.argv[2]||0),end=Number(process.argv[3]||total),filename=start?'patch.mp4':'picture.mp4'
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1920,height:1080}})
const errors=[];page.on('pageerror',e=>errors.push(e.message))
await page.goto(`file://${out}film.html`);await page.evaluate(()=>document.fonts.ready)
const ff=spawn('ffmpeg',['-y','-loglevel','error','-f','image2pipe','-framerate',String(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p',`${out}${filename}`],{stdio:['pipe','inherit','inherit']})
const done=new Promise((resolve,reject)=>{ff.on('error',reject);ff.on('close',c=>c===0?resolve():reject(Error(`ffmpeg exit ${c}`)))})
for(let f=0;f<(end-start)*fps;f++){
 await page.evaluate(sec=>{window.seek(sec)},start+f/fps)
 const jpg=await page.screenshot({type:'jpeg',quality:92})
 if(!ff.stdin.write(jpg))await new Promise(r=>ff.stdin.once('drain',r))
 if(f%(fps*10)===0)console.log(`${f/fps}/${total}s`)
}
ff.stdin.end();await done;await browser.close()
if(errors.length)throw Error(errors.join('\n'))
console.log(`Picture rendered: 1920×1080, 24 fps, ${end-start} seconds`)
