// Paper titles come from arXiv metadata and can carry inline LaTeX ("FITS: Modeling Time Series with $10k$ Parameters").
// Titles are shown as plain text: the math marks go, the content stays. Pure, tested in title.test.ts.

const WRAPPERS = /\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|mathcal|emph|operatorname)\s*\{([^{}]*)\}/g
const SYMBOLS: Record<string, string> = { '\\times': '×', '\\cdot': '·', '\\%': '%', '\\&': '&', '\\_': '_', '\\#': '#', '\\$': '$', '\\to': ' to ', '\\rightarrow': ' to ', '\\ell': 'ℓ', '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\lambda': 'λ', '\\mu': 'μ', '\\sigma': 'σ', '\\pi': 'π', '\\ldots': '…', '\\dots': '…' }

function mathToText(inner: string): string {
  let s = inner
  for (let i = 0; i < 3; i++) s = s.replace(WRAPPERS, '$1')
  s = s.replace(/\\[A-Za-z]+|\\[%&_#$]/g, (m) => SYMBOLS[m] ?? '')
  s = s.replace(/_\{([^{}]*)\}/g, '$1').replace(/\^\{([^{}]*)\}/g, '^$1').replace(/[{}]/g, '')
  return s.replace(/\s+/g, ' ').trim()
}

export function cleanTitle(title: string): string {
  if (!/[$\\]/.test(title)) return title
  // $...$ with no space inside next to the marks, so "$0.42 each ... $5" prose is not eaten
  let out = title.replace(/\$(?=\S)([^$\n]*?\S)\$/g, (_, inner: string) => mathToText(inner))
  out = out.replace(WRAPPERS, '$1').replace(/\\[%&_#]/g, (m) => SYMBOLS[m])
  return out.replace(/\s+/g, ' ').trim()
}
