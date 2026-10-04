// "Add this badge to your README": the Markdown for the badge of one run. The badge file itself is written at export time
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

// Text above the snippet, by verdict (German dossier keys). Same words as the badge file (src/repro_agent/badge.py WORDS).
// A verdict other than "reproduced" is never advertised: the note says it is not a finding about the paper.
export function badgeNote(urteil: string | undefined, gesamt?: string, catalog = false): { text: string; caution: boolean } {
  if (catalog && gesamt === 'nicht bewertbar (Methode nicht umgesetzt)') {
    return { text: 'The generated code did not implement the method, so this run is not assessable. The badge says so. It is not a finding about the paper.', caution: true }
  }
  switch (urteil) {
    case 'reproduziert':
      return { text: 'Reproduced with our setup (target value reached; method fidelity not checked). The badge shows the measured and the claimed value.', caution: false }
    case 'teilweise':
      return { text: 'Partially reproduced with our setup (within three times the tolerance; method fidelity not checked). The badge says "partially reproduced", not "reproduced".', caution: false }
    case 'nicht reproduziert':
      return { text: 'Not reproduced with our setup (target value not reached; method fidelity not checked). This is not a finding about the paper: the cause may be our generated code. The badge says so in the same words.', caution: true }
    case 'nicht reproduzierbar (Setup)':
      return { text: 'Not reproducible with our setup: the experiment could not be run. This is not a finding about the paper.', caution: true }
    case 'Agenten-Fehler vermutet':
      return { text: 'Agent error suspected: the result is below a trivial baseline. This is not a finding about the paper.', caution: true }
    default:
      return { text: 'No verdict, so no badge.', caution: true }
  }
}
