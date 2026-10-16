import { useEffect, useState } from 'react'
import { GROTESK, H, PAL, W } from './kit'
import { loadStageData, type StageData } from './data'
import { SHOTS } from './shots'
import type { SceneId } from './scenes'
import type { Theme } from './palette'

// The lazy half of a scene (its own chunk): loads the film run's exported data and draws the shot at time t on a 1920x1080 canvas, scaled to the frame.

let data: Promise<StageData> | null = null
function loadOnce(): Promise<StageData> {
  data ??= (async () => {
    // fonts the shots use, so the first frame is not drawn in a fallback face; never wait long for them
    const faces = ["400 20px 'Inter Tight Variable'", "600 20px 'Inter Tight Variable'", "italic 400 20px 'Instrument Serif'"]
    const fonts = Promise.all(faces.map((f) => document.fonts.load(f))).catch(() => null)
    const [d] = await Promise.all([loadStageData(), Promise.race([fonts, new Promise((r) => setTimeout(r, 1500))])])
    return d
  })()
  data.catch(() => { data = null })   // a failed load may be tried again on the next mount
  return data
}

export default function SceneStage({ id, t, palette, scale }: { id: SceneId; t: number; palette: Theme; scale: number }) {
  const [d, setD] = useState<StageData | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => { loadOnce().then(setD).catch(() => setFailed(true)) }, [])
  const shot = SHOTS[id]
  const pal = PAL[palette]
  if (failed) return <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', fontSize: 15, color: '#8a857b', padding: 24, textAlign: 'center' }}>This animation could not load its data.</div>
  if (!d) return null
  return (
    <div style={{ position: 'absolute', left: 0, top: 0, width: W, height: H, transform: `scale(${scale})`, transformOrigin: '0 0', overflow: 'hidden', background: pal.bg, color: pal.ink, fontFamily: GROTESK }}>
      {shot.render(t, d, pal)}
    </div>
  )
}
