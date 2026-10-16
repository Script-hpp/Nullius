// Quotes from a paper's LaTeX source are table rows: `\hline \parbox[t]{2mm}{\multirow{4}{*}{\rotatebox[origin=c]{90}{ETTh1}}} & 96 & \textbf{0.381} & ...`.
// The page shows the cells; the exact quote stays reachable (it is the evidence). Pure, tested in latexRow.test.ts.

// layout wrappers whose last brace group is the content: the leading arguments go, the content stays as "{content}"
const OPENERS: RegExp[] = [
  /\\multirow\s*\{[^{}]*\}\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\}\s*\{/g,
  /\\multicolumn\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{/g,
  /\\(?:parbox|rotatebox|makebox|raisebox|resizebox)\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\}\s*(?:\[[^\]]*\]\s*)?\{/g,
  /\\(?:textbf|textit|textsc|textrm|texttt|text|underline|ul|uline|emph|mathbf|mathrm|mathit|boldsymbol|bm)\s*\{/g,
]
const NOISE = /\\(?:hline|midrule|toprule|bottomrule|cline\s*\{[^{}]*\}|cmidrule\s*(?:\([^)]*\))?\s*\{[^{}]*\}|noalign\s*\{[^{}]*\}|centering|small|footnotesize|scriptsize|rowcolor\s*\{[^{}]*\}|cellcolor\s*\{[^{}]*\}|bf|it|rm|hfill)\b/g
const AND = '\u0001'   // an escaped \& is text, not a cell border

export function looksLikeLatexRow(quote: string | undefined): boolean {
  return !!quote && /&/.test(quote.replace(/\\&/g, '')) && /\\(?:hline|midrule|toprule|bottomrule|textbf|underline|ul|multirow|multicolumn|parbox|rotatebox|\\)|[0-9]\s*&/.test(quote)
}

function cleanCell(cell: string): string {
  let s = cell
  for (let i = 0; i < 6; i++) { const before = s; for (const re of OPENERS) s = s.replace(re, '{'); if (s === before) break }
  return s
    .replace(NOISE, ' ')
    .replace(/\$/g, '')
    .replace(/\\pm\b/g, '±').replace(/\\times\b/g, '×').replace(/\\%/g, '%').replace(/\\_/g, '_').replace(/\\#/g, '#').replace(/~/g, ' ')
    .replace(/\\[A-Za-z]+\*?/g, ' ')      // any other command word goes, its text stays
    .replace(/[{}]/g, '')
    .replace(AND, '&')
    .replace(/\s+/g, ' ').trim()
}

// Rows of cells (empty cells, left by spanned layout cells, are dropped). Empty when the quote is not a LaTeX table row.
export function latexRows(quote: string | undefined): string[][] {
  if (!quote || !looksLikeLatexRow(quote)) return []
  const text = quote.replace(/\\&/g, AND)
  return text.split(/\\\\(?:\s*\[[^\]]*\])?/)
    .map((row) => row.split('&').map(cleanCell).filter(Boolean))
    .filter((cells) => cells.length > 0)
}

export function matchesTarget(cell: string, target: number | undefined): boolean {
  if (target === undefined) return false
  const m = cell.match(/-?\d+(?:\.\d+)?/)
  if (!m) return false
  const n = Number(m[0])
  return Math.abs(n - target) < 1e-9 || Math.abs(n / 100 - target) < 1e-9 || Math.abs(n * 100 - target) < 1e-9
}
