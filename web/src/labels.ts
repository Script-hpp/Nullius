export type Tone = 'green' | 'amber' | 'red' | 'gray' | 'violet'

export const RESULT: Record<string, { label: string; tone: Tone }> = {
  'reproduziert': { label: 'VERIFIED', tone: 'green' },
  'teilweise': { label: 'PARTIAL', tone: 'amber' },
  'nicht reproduziert': { label: 'FAILED', tone: 'red' },
  'nicht reproduzierbar (Setup)': { label: 'INVALID SETUP', tone: 'gray' },
  'Agenten-Fehler vermutet': { label: 'AGENT ERROR', tone: 'violet' },
}
export const METHOD: Record<string, { label: string; tone: Tone }> = {
  'ja': { label: 'YES', tone: 'green' },
  'teilweise': { label: 'PARTIAL', tone: 'amber' },
  'nein': { label: 'NO', tone: 'red' },
  'nicht geprueft': { label: 'UNTESTED', tone: 'gray' },
}
export const OVERALL: Record<string, string> = {
  'vorlaeufig': 'Provisional',
  'Paper reproduziert': 'Fully Verified',
  'Zielwert erreicht mit anderem Modell': 'Target reached with different model',
  'nicht reproduziert': 'Failed Verification',
  'teilweise reproduziert': 'Partially Verified',
  'nicht bewertbar (Methode nicht umgesetzt)': 'Not Assessable',
  'Agenten-Fehler vermutet': 'Agent Fault Suspected',
  'nicht reproduzierbar (Setup)': 'Setup Not Reproducible',
}
export const ATTEMPT: Record<string, { label: string; tone: Tone }> = {
  'stabil': { label: 'PASSED', tone: 'green' },
  'abweichung': { label: 'OUT OF RANGE', tone: 'amber' },
  'fehler': { label: 'FAILED', tone: 'red' },
  'ungueltig': { label: 'INVALID', tone: 'gray' },
  'timeout': { label: 'TIMEOUT', tone: 'gray' },
}
export const ABLATION: Record<string, string> = {
  'Graph genutzt': 'Graph utilized',
  'Graph nicht genutzt': 'Graph ignored',
  'unklar': 'Unclear',
  'nicht messbar': 'Not measurable',
}
export const de2en = (map: Record<string, { label: string; tone: Tone }>, k: string) =>
  map[k] ?? { label: k.toUpperCase(), tone: 'gray' as Tone }

export const TONE_VAR: Record<Tone, string> = {
  green: 'var(--green, #34d399)', amber: 'var(--amber, #fbbf24)', red: 'var(--red, #f87171)', gray: 'var(--muted, #9ca3af)', violet: 'var(--violet, #a78bfa)',
}
