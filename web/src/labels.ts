export type Tone = 'green' | 'amber' | 'red' | 'gray' | 'violet'

// Discovery candidates that ran once through the loop: never a verdict on the paper, always "with our setup".
export const RESULT_CATALOG: Record<string, { label: string; tone: Tone }> = {
  'reproduziert': { label: 'REPRODUCED', tone: 'green' },
  'teilweise': { label: 'PARTIALLY REPRODUCED', tone: 'amber' },
  'nicht reproduziert': { label: 'NOT REPRODUCED', tone: 'red' },
  'nicht reproduzierbar (Setup)': { label: 'SETUP NOT REPRODUCIBLE', tone: 'gray' },
  'Agenten-Fehler vermutet': { label: 'AGENT ERROR SUSPECTED', tone: 'violet' },
}
// Same wording for every run: "reproduced" means the target value was reached with our setup; method fidelity is not checked.
export const RESULT = RESULT_CATALOG
// Tooltip for the green badge: the badge says "target reached", never "method reproduced".
export const TARGET_HINT = 'The independent harness measured the target value. Method fidelity not checked.'
export const resultHint = (urteil: string | undefined) => (urteil === 'reproduziert' ? TARGET_HINT : undefined)
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
  [/^Repo nicht gelesen/, () => 'The code repository could not be cloned in the sandbox. The check went on with the paper and the README only.'],
  [/^Sandbox-Setup fehlgeschlagen/, () => 'The sandbox setup failed (infrastructure, for example a data download). This is not a verdict about the paper; the run was stopped before any check.'],
  [/^Kurzbeschreibung nicht erstellt/, () => 'The plain-language summary of the paper could not be written. The check itself is not affected.'],
  [/^Autoren und Jahr nicht gelesen/, () => 'Authors and year could not be read from the arXiv page. The summary was written without them.'],
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
  'Paper reproduziert': 'Paper reproduced (target value and method)',
  'Zielwert erreicht mit anderem Modell': 'Target reached with different model',
  'nicht reproduziert': 'Not reproduced with our setup',
  'teilweise reproduziert': 'Partially reproduced',
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

// Jury reasons and failure reasons are written in German by scripts/curate_catalog.py (catalog.py); the UI shows English.
export function translateJuryReason(reason: string): string {
  const g = reason.match(/^Gesamtaussage: (.+)$/)
  if (g) return OVERALL[g[1]] ?? g[1]
  const s = reason.match(/^Status (\w+)$/)
  if (s) return ({ ungeprueft: 'not run yet', lauf_fehlgeschlagen: 'run failed', nicht_laufbar: 'not runnable', geprueft: 'checked' } as Record<string, string>)[s[1]] ?? s[1]
  if (reason === 'kein Urteil') return 'no verdict'
  return reason
}

export function translateReason(reason: string): string {
  let m: RegExpMatchArray | null
  if (/^Cora ohne matched_task/.test(reason)) return 'the paper does not say which Cora protocol it uses (Planetoid or random splits)'
  if ((m = reason.match(/^ETTh1: Horizont (\d+), Task misst nur (\d+)/))) return `horizon ${m[1]}, the task only measures ${m[2]}`
  if (/^ETTh1: Horizont fehlt/.test(reason)) return 'the horizon is missing or unreadable'
  if ((m = reason.match(/^Metrik (.+) auf Task (.+) nicht messbar/))) return `metric ${m[1]} cannot be measured on ${m[2]}`
  if (/^Zielwert fehlt/.test(reason)) return 'the target value is missing or unreadable'
  if ((m = reason.match(/^Datensatz (.+) ohne Task/))) return `no task for dataset ${m[1]}`
  return reason
}

// Claim texts from the loop are German ("erreicht ... Accuracy auf ..."); the UI shows English.
export function translateClaim(text: string) {
  if (!text) return text
  let t = text
  t = t.replace(/erreicht/g, 'achieves')
  t = t.replace(/Accuracy auf/g, 'accuracy on')
  t = t.replace(/\bauf\b/g, 'on')
  t = t.replace(/groesste Komponente/g, 'largest component')
  t = t.replace(/zufaellige/g, 'random')
  t = t.replace(/20\/30\/Rest-Splits/g, '20/30/rest splits')
  t = t.replace(/Standard-Split/g, 'standard split')
  t = t.replace(/Horizont (\d+)/g, 'horizon $1')
  t = t.replace(/,(\d) %/g, '.$1%')
  return t
}

export const VERDICT_SENTENCE: Record<string, string> = {
  'reproduziert': 'Reproduced with our setup',
  'teilweise': 'Partially reproduced with our setup',
  'nicht reproduziert': 'Not reproduced with our setup',
  'nicht reproduzierbar (Setup)': 'No valid run with our setup',
  'Agenten-Fehler vermutet': 'Agent error suspected',
}
