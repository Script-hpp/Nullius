// How a measured value compares with the claim, in words. One rule for every page: inside half of the last shown digit it "matches the claim",
// otherwise the signed difference ("+0.003 vs. claim", "−1.2 pp vs. claim"). Pure, tested in delta.test.ts.
export type Diff = { text: string; sign: -1 | 0 | 1; match: boolean }

// diff is in the shown unit (percentage points for shares, raw value otherwise); share = shown with one decimal as percent points.
export function describeDiff(diff: number, share: boolean): Diff {
  const d = share ? 1 : 3
  if (Math.abs(diff) < 0.5 * 10 ** -d) return { text: 'matches the claim', sign: 0, match: true }
  return { text: `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(d)}${share ? ' pp' : ''} vs. claim`, sign: diff > 0 ? 1 : -1, match: false }
}
