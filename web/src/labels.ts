// Deutsche Werte aus dem Dossier -> englische Anzeige, plus Farbe. Nur Anzeige, keine Bewertung.
export type Tone = 'green' | 'amber' | 'red' | 'gray' | 'violet'

export const RESULT: Record<string, { label: string; tone: Tone }> = {
  'reproduziert': { label: 'reproduced', tone: 'green' },
  'teilweise': { label: 'partially', tone: 'amber' },
  'nicht reproduziert': { label: 'not reproduced', tone: 'red' },
  'nicht reproduzierbar (Setup)': { label: 'not reproducible (setup)', tone: 'gray' },
  'Agenten-Fehler vermutet': { label: 'agent fault suspected', tone: 'violet' },
}
export const METHOD: Record<string, { label: string; tone: Tone }> = {
  'ja': { label: 'yes', tone: 'green' },
  'teilweise': { label: 'partially', tone: 'amber' },
  'nein': { label: 'no', tone: 'red' },
  'nicht geprueft': { label: 'not checked', tone: 'gray' },
}
export const OVERALL: Record<string, string> = {
  'vorlaeufig': 'provisional',
  'Paper reproduziert': 'paper reproduced',
  'Zielwert erreicht mit anderem Modell': 'target reached with a different model',
  'nicht reproduziert': 'not reproduced',
  'teilweise reproduziert': 'partially reproduced',
  'nicht bewertbar (Methode nicht umgesetzt)': 'not assessable (method not implemented)',
  'Agenten-Fehler vermutet': 'agent fault suspected',
  'nicht reproduzierbar (Setup)': 'not reproducible (setup)',
}
export const ATTEMPT: Record<string, { label: string; tone: Tone }> = {
  'stabil': { label: 'within tolerance', tone: 'green' },
  'abweichung': { label: 'deviates', tone: 'amber' },
  'fehler': { label: 'error', tone: 'red' },
  'ungueltig': { label: 'invalid', tone: 'gray' },
  'timeout': { label: 'timeout', tone: 'gray' },
}
export const ABLATION: Record<string, string> = {
  'Graph genutzt': 'graph used',
  'Graph nicht genutzt': 'graph not used',
  'unklar': 'unclear',
  'nicht messbar': 'not measurable',
}
export const TONE_VAR: Record<Tone, string> = {
  green: 'var(--green)', amber: 'var(--amber)', red: 'var(--red)', gray: 'var(--muted)', violet: 'var(--violet)',
}
export const de2en = (map: Record<string, { label: string; tone: Tone }>, k: string) =>
  map[k] ?? { label: k, tone: 'gray' as Tone }
