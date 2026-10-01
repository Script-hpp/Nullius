// Formula spans in plan text and their LaTeX, converted mechanically (no interpretation). Rendering is done by KaTeX, loaded on demand
// (components/film/Scenes.tsx); if KaTeX cannot parse a span, the plain text stays (see mathWords in film.ts).
export type Seg = { kind: 'text'; text: string } | { kind: 'math'; text: string; latex: string }

const COMBINING: Record<string, string> = { '\u0303': 'tilde', '\u0302': 'hat', '\u0304': 'bar', '\u0307': 'dot' }
const ACCENT = /[\u0302\u0303\u0304\u0307]/
const hasAccent = (s: string) => ACCENT.test(s.normalize('NFD'))

// Mechanical conversion of one formula span. Accents on a letter become \tilde{X}, \hat{X}, ...; ^(...) becomes ^{...}; ^x becomes ^{x};
// arrows become \to; characters TeX would choke on are escaped. Everything else stays as written.
export function toLatex(span: string): string {
  const s = span.normalize('NFD')
  let out = ''
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]
    const next = s[i + 1]
    if (next && COMBINING[next] && /[\p{L}]/u.test(ch)) { out += `\\${COMBINING[next]}{${ch}}`; i++; continue }
    if (ch === '^') {
      const rest = s.slice(i + 1)
      const paren = rest.match(/^\(([^)]*)\)/)
      if (paren) { out += `^{${paren[1]}}`; i += paren[0].length; continue }
      const plain = rest.match(/^-?[A-Za-z0-9.]+(?:\/[A-Za-z0-9.]+)?/)
      if (plain) { out += `^{${plain[0]}}`; i += plain[0].length; continue }
    }
    if (ch === '→' || ch === '⟶') out += '\\to '
    else if ('%#&$'.includes(ch)) out += '\\' + ch
    else if (ch === '_') out += '\\_'
    else out += ch
  }
  return out.replace(/\s+/g, ' ').trim()
}

// A word is "mathy" if it is an operator, a number, a single letter or has a formula mark (^, accent). Trailing punctuation does not count.
const core = (w: string) => w.replace(/[;,.:]+$/, '')
const strong = (w: string) => core(w).includes('^') || hasAccent(core(w))
const mathy = (w: string) => {
  const c = core(w)
  return c !== '' && (strong(w) || /^[=+\-*/<>≈≤≥×·→⟶]$/.test(c) || /^[A-Za-z]$/.test(c) || /^\d+(\.\d+)?$/.test(c))
}

// Split text into plain text and formula spans. $...$ is taken as written; otherwise a span is a run of mathy words around a word with
// a ^ or an accent mark (so "lr=0.01" and "Train=140" stay text), ending before trailing punctuation.
export function splitFormulas(text: string): Seg[] {
  const segs: Seg[] = []
  const dollar = /\$([^$]+)\$/g
  let last = 0
  const plain = (t: string) => { if (t) segs.push(...splitPlain(t)) }
  for (const m of text.matchAll(dollar)) {
    plain(text.slice(last, m.index))
    segs.push({ kind: 'math', text: m[0], latex: m[1].trim() })
    last = m.index! + m[0].length
  }
  plain(text.slice(last))
  return segs
}

function splitPlain(text: string): Seg[] {
  const parts = text.split(/(\s+)/)   // words and the white space between them, so the text can be put back together exactly
  const idx = parts.map((_, i) => i).filter((i) => i % 2 === 0)   // positions of words
  const used = new Set<number>()
  const spans: [number, number][] = []   // [first word, last word] as positions in `parts`
  for (const wi of idx) {
    if (used.has(wi) || parts[wi] === '' || !strong(parts[wi])) continue
    let a = wi, b = wi
    for (let k = wi - 2; k >= 0 && mathy(parts[k]) && !/[;,.:]$/.test(parts[k]); k -= 2) a = k
    for (let k = wi + 2; k < parts.length && mathy(parts[k]); k += 2) { b = k; if (/[;,.:]$/.test(parts[k])) break }
    if (/[;,.:]$/.test(parts[wi]) && b === wi) b = wi
    for (let k = a; k <= b; k += 2) used.add(k)
    spans.push([a, b])
  }
  const segs: Seg[] = []
  let at = 0
  const flush = (from: number, to: number) => { const t = parts.slice(from, to).join(''); if (t) segs.push({ kind: 'text', text: t }) }
  for (const [a, b] of spans.sort((x, y) => x[0] - y[0])) {
    if (a < at) continue
    let end = b
    let tail = ''
    const last = parts[end]
    const m = last.match(/[;,.:]+$/)
    if (m) { tail = m[0] }
    const spanText = parts.slice(a, end + 1).join('').slice(0, parts.slice(a, end + 1).join('').length - tail.length)
    flush(at, a)
    segs.push({ kind: 'math', text: spanText, latex: toLatex(spanText) })
    if (tail) segs.push({ kind: 'text', text: tail })
    at = end + 1
  }
  flush(at, parts.length)
  return segs
}
