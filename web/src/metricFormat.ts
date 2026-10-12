// Accuracy is a proportion; error metrics are decimals. Keep small spreads visible.
export function formatMetric(metric: string | undefined, value: unknown): string {
  if (value == null || !Number.isFinite(Number(value))) return '–'
  const n = Number(value)
  if (!metric || metric === 'accuracy') return `${(n > 0 && n <= 2 ? n * 100 : n).toFixed(1)}%`
  return n.toFixed(4)
}

export function formatSpread(metric: string | undefined, value: unknown): string {
  if (value == null || !Number.isFinite(Number(value))) return '–'
  const n = Number(value)
  const scaled = metric === 'accuracy' ? n * 100 : n
  const decimals = metric === 'accuracy' ? 1 : 4
  const rounded = scaled.toFixed(decimals)
  return scaled !== 0 && Number(rounded) === 0 ? scaled.toPrecision(2) : rounded
}
