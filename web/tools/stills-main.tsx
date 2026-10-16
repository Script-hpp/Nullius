import { createRoot } from 'react-dom/client'
import '@fontsource-variable/inter-tight'
import '@fontsource/instrument-serif/latin-400-italic.css'
import SceneStage from '../src/scroll/SceneStage'
const q = new URLSearchParams(location.search)
createRoot(document.getElementById('r')!).render(<SceneStage id={q.get('id') as 'opening'} t={Number(q.get('t'))} palette={q.get('p') as 'paper'} scale={1200 / 1920} />)
