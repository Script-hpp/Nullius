// pure part of Arrows.tsx (kept apart so node's test runner can import it without JSX)
export const arrowParts = (text: string): string[] => text.split(/\s*[\u2192\u27F6]\s*/u)
