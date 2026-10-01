import { useMemo, useState, type ReactNode } from 'react'

// Small Python highlighter: comments, strings, numbers, keywords, def/class names. No dependency; colors come from the tone tokens.
const KEYWORDS = new Set(['import', 'from', 'as', 'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False', 'with', 'try', 'except', 'finally', 'raise', 'lambda', 'pass', 'break', 'continue', 'yield', 'global', 'assert', 'del'])
const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[fbrFBR]{0,2}"(?:\\.|[^"\\\n])*"|[fbrFBR]{0,2}'(?:\\.|[^'\\\n])*')|(\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\b[A-Za-z_][A-Za-z_0-9]*\b)/g

export function highlight(code: string): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let prevWord = ''
  let k = 0
  for (const m of code.matchAll(TOKEN)) {
    const i = m.index ?? 0
    if (i > last) out.push(code.slice(last, i))
    const [text, comment, str, number, word] = m
    if (comment) out.push(<span key={k++} className="text-muted italic">{text}</span>)
    else if (str) out.push(<span key={k++} className="text-tone-green">{text}</span>)
    else if (number) out.push(<span key={k++} className="text-tone-amber">{text}</span>)
    else if (word && KEYWORDS.has(word)) out.push(<span key={k++} className="text-tone-violet">{text}</span>)
    else if (word && (prevWord === 'def' || prevWord === 'class')) out.push(<span key={k++} className="text-tone-blue">{text}</span>)
    else out.push(text)
    if (word) prevWord = word
    last = i + text.length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}

// Code with line count, collapsed to the first lines until "Show all".
export function PythonCode({ code, collapsedLines = 14 }: { code: string; collapsedLines?: number }) {
  const [all, setAll] = useState(false)
  const lines = useMemo(() => code.replace(/\n$/, '').split('\n'), [code])
  const shown = all ? code.replace(/\n$/, '') : lines.slice(0, collapsedLines).join('\n')
  const nodes = useMemo(() => highlight(shown), [shown])
  return (
    <div className="rounded-lg border border-line bg-surface">
      <pre className="max-h-[28rem] overflow-auto p-3 font-mono text-[11.5px] leading-relaxed text-fg-3"><code>{nodes}</code></pre>
      {lines.length > collapsedLines && (
        <button onClick={() => setAll((v) => !v)} className="w-full border-t border-line px-3 py-1.5 text-left text-[11px] text-muted hover:text-fg-3 cursor-pointer">
          {all ? 'Show less' : `Show all ${lines.length} lines`}
        </button>
      )}
    </div>
  )
}
