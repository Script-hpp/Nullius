import type { ReactNode } from 'react'

// Plans use headings, lists, bold labels and inline code. Render those as React
// elements; model output and HTML remain text, never executable markup.
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*.+?\*\*|__.+?__)/g).map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i} className="rounded bg-raised px-1 py-0.5 font-mono text-[11px] text-fg-2 [overflow-wrap:anywhere]">{part.slice(1, -1)}</code>
    if ((part.startsWith('**') && part.endsWith('**')) || (part.startsWith('__') && part.endsWith('__'))) return <strong key={i} className="font-semibold text-fg-2">{inline(part.slice(2, -2))}</strong>
    return part
  })
}

export function PlanText({ lines }: { lines: string[] }) {
  const blocks: ReactNode[] = []
  for (let i = 0; i < lines.length;) {
    const line = lines[i]
    const heading = /^#{1,6}\s+(.+)$/.exec(line)
    const bullet = /^[-*+]\s+/.test(line)
    const ordered = /^(\d+)[.)]\s+/.exec(line)
    if (bullet || ordered) {
      const items: ReactNode[] = []
      const pattern = bullet ? /^[-*+]\s+(.+)$/ : /^\d+[.)]\s+(.+)$/
      const start = i
      for (; i < lines.length; i++) {
        const match = pattern.exec(lines[i])
        if (!match) break
        items.push(<li key={i} className="pl-0.5 [overflow-wrap:anywhere]">{inline(match[1])}</li>)
      }
      blocks.push(bullet
        ? <ul key={start} className="m-0 list-disc space-y-1 pl-4 marker:text-muted">{items}</ul>
        : <ol key={start} start={Number(ordered![1])} className="m-0 list-decimal space-y-1 pl-4 marker:text-muted">{items}</ol>)
    } else {
      blocks.push(heading
        ? <h4 key={i} className="m-0 pt-1 text-[13px] font-semibold text-fg-2">{inline(heading[1])}</h4>
        : <p key={i} className="m-0 [overflow-wrap:anywhere]">{inline(line)}</p>)
      i++
    }
  }
  return <div className="space-y-2 text-[12px] leading-relaxed text-fg-3">{blocks}</div>
}
