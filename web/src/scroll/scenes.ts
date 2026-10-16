import { PAL, type Theme } from './palette.ts'

// What the landing scenes are, without any of the stage code, so the page can size and label a scene before its chunk has loaded.
// duration/poster are the single source: shots.tsx takes them from here.
export const SCENE_IDS = ['opening', 'placebo'] as const
export type SceneId = (typeof SCENE_IDS)[number]

export type SceneMeta = {
  theme: Theme        // the shot's own palette; the page forces night in dark mode
  duration: number    // ms of film
  poster: number      // ms of the still shown under reduced motion and on phones
  label: string       // aria-label: what the scene shows, one plain sentence
  caption: string     // visible caption under the scene
}

export const SCENES: Record<SceneId, SceneMeta> = {
  opening: {
    theme: 'paper', duration: 21000, poster: 19500,
    label: 'Animation, driven by scrolling: one claim from a paper among many other claims, then Nullius runs the experiment ten times and measures the result.',
    caption: 'One claim among many, then measured: the experiment is run ten times and the result is shown.',
  },
  placebo: {
    theme: 'paper', duration: 9000, poster: 8700,
    label: 'Animation, driven by scrolling: the same ten measurements are checked against the real claim, which is confirmed, and against a false claim planted on purpose, which is rejected.',
    caption: 'The same ten measurements, checked against the real claim and against a false claim we planted. The false one is rejected.',
  },
}

// "paper" carries words, "night" the machinery. In dark mode every scene uses night so no bright sheet sits on a dark page.
export const paletteFor = (theme: Theme, dark: boolean): Theme => (dark ? 'night' : theme)
// the frame's colour before the stage code has loaded
export const FRAME_BG: Record<Theme, string> = { paper: PAL.paper.bg, night: PAL.night.bg }
