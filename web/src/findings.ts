import { REPO_BLOB } from './data/benchmark.ts'

// The four findings shown in "What we found" on the start page. Observations from our own checks, not statistics.
// Every number is traced to a file in the repo (see the README block "What we found" and the commit message).

export type Finding = {
  id: string
  paper: string
  dataset: string
  title: string
  text: string
  // Two short columns: what the paper (or script) says, and what we found.
  says: { label: string; value: string }
  found: { label: string; value: string }
  link: { href: string; label: string }
  // Set when the link is an external file; the test checks that it exists in the repo.
  repoPath?: string
}

const gh = (path: string) => `${REPO_BLOB}/${path}`

export const FINDINGS: Finding[] = [
  {
    id: 'fits',
    paper: 'FITS',
    dataset: 'ETTh1',
    title: 'The decisive settings sit in the launch script',
    text: 'The results table lists four input lengths (90, 180, 360, 720). The authors’ launch script for ETTh1 uses 360, and its batch size of 64 is doubled to an effective 128 by the code.',
    says: { label: 'Target (paper)', value: '0.372' },
    found: { label: 'With the script’s settings', value: '0.37203 ± 0.00005' },
    link: { href: '#/run/disc_2307_03756--20261014-080542', label: 'Open the FITS check' },
  },
  {
    id: 'sparsetsf',
    paper: 'SparseTSF',
    dataset: 'ETTh1',
    title: 'The launch script, run as is, trains a different variant',
    text: 'The authors’ launch script for the linear variant does not set the model type, and the code default is the MLP variant. The recorded run used the linear variant and reached the target. A new check would have to guess the variant, so it is locked.',
    says: { label: 'Target (paper)', value: '0.359' },
    found: { label: 'Script, run as is', value: 'MLP variant' },
    link: { href: '#/run/disc_2405_00946--20261009-220822', label: 'Open the recorded check' },
  },
  {
    id: 'rlinear',
    paper: 'RLinear',
    dataset: 'ETTh1',
    title: 'One key setting appears only in a table caption',
    text: 'The input length 336 is stated only in the caption of the claim table. Our check used the code default of 96 and measured 0.397 against 0.366. We now read captions; whether the target is reached with that fix is untested.',
    says: { label: 'Caption', value: 'length 336' },
    found: { label: 'Code default won', value: 'length 96' },
    link: { href: gh('docs/audits/rlinear_live_ba9fa94f_20261014.json'), label: 'Run record (GitHub)' },
    repoPath: 'docs/audits/rlinear_live_ba9fa94f_20261014.json',
  },
  {
    id: 'tsmixer',
    paper: 'TSMixer',
    dataset: 'ETTh1',
    title: 'Reaching the number is not reproducing the paper',
    text: 'The target was reached with a smaller model: 1 mixer block and hidden size 128 instead of 6 and 512. We show it as target reached with a reduced model, not as reproduced.',
    says: { label: 'Paper model', value: '6 blocks, 512' },
    found: { label: 'Model that reached it', value: '1 block, 128' },
    link: { href: '#/run/disc_2303_06053--20261009-201230', label: 'Open the recorded check' },
  },
]
