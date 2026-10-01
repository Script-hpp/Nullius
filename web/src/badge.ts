// "Show this result in your README": the Markdown for the badge of one run. The badge file itself is written at export time
// (src/repro_agent/badge.py) next to dossier.json; the link leads to the certificate.
export function badgeUrls(base: string, paperId: string, timestamp: string) {
  const root = base.replace(/\/+$/, '')
  const runId = `${paperId}--${timestamp}`
  return { image: `${root}/data/${paperId}/${timestamp}/badge.svg`, link: `${root}/#/run/${runId}`, runId }
}

export function badgeMarkdown(base: string, paperId: string, timestamp: string): string {
  const u = badgeUrls(base, paperId, timestamp)
  return `[![Nullius](${u.image})](${u.link})`
}
