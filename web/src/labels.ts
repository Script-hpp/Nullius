export type Tone = 'green' | 'amber' | 'red' | 'gray' | 'violet'

export const RESULT: Record<string, { label: string; tone: Tone }> = {
  'reproduziert': { label: 'VERIFIED', tone: 'green' },
  'teilweise': { label: 'PARTIAL', tone: 'amber' },
  'nicht reproduziert': { label: 'FAILED', tone: 'red' },
  'nicht reproduzierbar (Setup)': { label: 'INVALID SETUP', tone: 'gray' },
  'Agenten-Fehler vermutet': { label: 'AGENT ERROR', tone: 'violet' },
}
// Discovery candidates that ran once through the loop: never a verdict on the paper, always "with our setup".
export const RESULT_CATALOG: Record<string, { label: string; tone: Tone }> = {
  'reproduziert': { label: 'REPRODUCED', tone: 'green' },
  'teilweise': { label: 'PARTIALLY REPRODUCED', tone: 'amber' },
  'nicht reproduziert': { label: 'NOT REPRODUCED', tone: 'red' },
  'nicht reproduzierbar (Setup)': { label: 'SETUP NOT REPRODUCIBLE', tone: 'gray' },
  'Agenten-Fehler vermutet': { label: 'AGENT ERROR SUSPECTED', tone: 'violet' },
}
export const PRELIMINARY = 'Preliminary: method fidelity not checked'
export const isPreliminary = (gesamt: string | undefined) => gesamt === 'vorlaeufig'
export const NOT_ASSESSABLE = 'nicht bewertbar (Methode nicht umgesetzt)'

// Label for a catalog result: a run whose code did not implement the method is "not assessable", whatever the number says.
export function catalogLabel(urteil: string, gesamt: string | undefined): { label: string; tone: Tone } {
  if (gesamt === NOT_ASSESSABLE) return { label: 'NOT ASSESSABLE', tone: 'gray' }
  return de2en(RESULT_CATALOG, urteil)
}

// Warnings in the dossier are written in German by the loop; the UI shows English.
const WARNING_PATTERNS: [RegExp, (m: RegExpMatchArray) => string][] = [
  [/^Info: Das Paper nennt Streuung \+-([\d.]+)/, (m) => `The paper reports a spread of ±${m[1]} across splits or runs. That is not the tolerance of the mean; the tolerance stays fixed.`],
  [/^Protokoll im Paper nicht belegt/, () => 'Protocol not stated in the paper: the task was assumed, not confirmed. The overall statement stays preliminary.'],
  [/^Tabellenspalte nicht gepr/, () => 'Table column not verified: the number could belong to a comparison method.'],
  [/^Eingabel[aä]nge im Paper nicht belegt/, () => 'Input length not stated in the paper: the agent chose it freely (up to 720).'],
  [/^Zielwert weicht ab: Rolle 2 liest ([\d.]+).*die Discovery ([\d.]+)/, (m) => `Target differs: re-reading the paper gave ${m[1]}, discovery gave ${m[2]}. The discovery value is used.`],
  [/^Rolle 2 konnte den Zielwert/, () => 'The target could not be cross-checked against the paper text. The discovery value is used.'],
]
export function translateWarning(text: string): string {
  for (const [re, fn] of WARNING_PATTERNS) {
    const m = text.match(re)
    if (m) return fn(m)
  }
  return text
}

// Text color per tone, shared by Badge (ui.tsx) and the catalog components: one place to change for a light theme.
export const TONE_TEXT: Record<Tone, string> = {
  green: 'text-tone-green', amber: 'text-tone-amber', red: 'text-tone-red', gray: 'text-muted', violet: 'text-tone-violet',
}

export const METHOD: Record<string, { label: string; tone: Tone }> = {
  'ja': { label: 'YES', tone: 'green' },
  'teilweise': { label: 'PARTIAL', tone: 'amber' },
  'nein': { label: 'NO', tone: 'red' },
  'nicht geprueft': { label: 'UNTESTED', tone: 'gray' },
}
export const OVERALL: Record<string, string> = {
  'vorlaeufig': 'Preliminary',
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

// Fills, borders and strokes (solid tone colors); TONE_TEXT_VAR for text drawn in inline styles (readable in both themes).
export const TONE_VAR: Record<Tone, string> = {
  green: 'var(--c-tone-green-solid)', amber: 'var(--c-tone-amber-solid)', red: 'var(--c-tone-red-solid)', gray: 'var(--c-line-strong)', violet: 'var(--c-tone-violet-solid)',
}
export const TONE_TEXT_VAR: Record<Tone, string> = {
  green: 'var(--c-tone-green)', amber: 'var(--c-tone-amber)', red: 'var(--c-tone-red)', gray: 'var(--c-muted)', violet: 'var(--c-tone-violet)',
}
