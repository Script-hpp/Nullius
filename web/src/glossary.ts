// Plain-language explanations of technical terms, shown as tooltips (components/Glossed.tsx).
// Static on purpose: no model call per page view.
export const GLOSSARY: Record<string, string> = {
  accuracy: 'The share of test cases a model gets right. 81.5% accuracy means about 82 of 100 articles were sorted into the correct topic.',
  MSE: 'Mean squared error: how far a forecast is from the real values, on average. Lower is better, 0 would be perfect.',
  Cora: 'A standard benchmark: about 2,700 scientific articles that cite each other, each belonging to one of seven topics.',
  CiteSeer: 'A standard benchmark: about 3,300 scientific articles that cite each other, each belonging to one of six topics.',
  PubMed: 'A standard benchmark: about 19,700 medical research articles on diabetes that cite each other, each belonging to one of three topics.',
  ETTh1: 'A standard benchmark for forecasting: hourly temperature readings from an electricity transformer.',
  ETTh2: 'A standard benchmark for forecasting: hourly temperature readings from a second electricity transformer.',
  ETTm1: 'A standard benchmark for forecasting: temperature readings from an electricity transformer, every 15 minutes.',
  ETTm2: 'A standard benchmark for forecasting: temperature readings from a second electricity transformer, every 15 minutes.',
  seed: 'A starting number for the random choices in a training run. Different seeds give slightly different results, so we run several and use the average.',
  tolerance: 'How far our measurement may differ from the claimed value and still count as a match. We fix it before the run.',
  reproduced: 'Our rebuild reached the claimed number within the tolerance. It does not prove the method matches the paper; that is not checked yet.',
  placebo: 'A control claim with a deliberately wrong target value. A trustworthy checker must reject it.',
  sandbox: 'An isolated computer in the cloud where the experiment code runs. It has no network access during training.',
  harness: 'Our checking program. The agent only writes predictions; the harness loads the true answers itself, computes the score and compares it with the claim. The agent cannot touch it.',
  'sandbox branch': 'A copy of the sandbox. Each seed runs in its own branch, and a failed attempt can be thrown away without harm.',
  branch: 'A copy of the sandbox. Each seed runs in its own branch, and a failed attempt can be thrown away without harm.',
}

// Longest first so that "accuracy" never hides inside a longer term.
const KEYS = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length)
const PATTERN = new RegExp(`\\b(${KEYS.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi')

export type Piece = { text: string; term?: string }

// Split text into plain pieces and glossary terms (matched case-insensitively, whole words).
export function splitTerms(text: string): Piece[] {
  const out: Piece[] = []
  let last = 0
  for (const m of text.matchAll(PATTERN)) {
    const i = m.index ?? 0
    if (i > last) out.push({ text: text.slice(last, i) })
    const key = KEYS.find((k) => k.toLowerCase() === m[0].toLowerCase())
    out.push({ text: m[0], term: key })
    last = i + m[0].length
  }
  if (last < text.length) out.push({ text: text.slice(last) })
  return out
}
