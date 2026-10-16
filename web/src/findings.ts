import { FIGURES, REPO_BLOB } from './data/benchmark.ts'

// The findings page (#/findings) and its teaser on the start page. Observations from our own checks, not statistics or a success rate.
// Visible text is short and plain on purpose (the jury is not assumed to know ML); everything else sits in `details`, closed by default.
// Every number is traced to a file in the repo; the sources are listed next to each finding and in the commit message.

export type Tone = 'amber' | 'red' | 'green' | 'gray'

export type Side = { label: string; value: string; tone: Tone }

export type Finding = {
  id: string
  part: 'A' | 'B'
  // Small tag above the headline, e.g. "FITS · ETTh1" (a paper and the data it was checked on).
  tag: string
  // Everyday words, about ten words at most.
  headline: string
  // At most two short sentences.
  text: string
  // The one visual: two big values side by side. Part B entries may leave it out.
  left?: Side
  right?: Side
  // Closed by default. Label and text pairs; the commit ids are printed, not linked (the repository may be private).
  details: { label: string; text: string }[]
  commits: string[]
  links: { href: string; label: string; repoPath?: string }[]
}

const gh = (path: string) => `${REPO_BLOB}/${path}`
const doc = (path: string, label: string) => ({ href: gh(path), label, repoPath: path })

export const INTRO = {
  title: 'What checking real papers taught us',
  summary: 'The number in a paper is often not enough to rebuild it. These are observations from our own checks, not a success rate.',
  analogy: 'Like a recipe that lists the oven temperature only in the photo caption.',
}

export const PART_A = { title: 'What papers leave out', sub: 'Each paper is shown on its own data. Lower error is better.' }
export const PART_B = { title: 'What checking taught us about checking', sub: 'Mistakes a checker can make, and what Nullius does now.' }

const placebo = FIGURES.placebo.v4

export const FINDINGS: Finding[] = [
  {
    id: 'fits',
    part: 'A',
    tag: 'FITS · ETTh1',
    headline: 'The key setting hides in the authors’ start file',
    text: 'The paper’s table lists four choices for how much past data the model sees. The authors’ start file picks one. With it, our error matched the paper.',
    left: { label: 'Paper table', value: '4 choices', tone: 'amber' },
    right: { label: 'Start file', value: '360', tone: 'green' },
    details: [
      { label: 'What the paper says', text: 'The results table lists four input lengths (90, 180, 360, 720). Target error (MSE): 0.372.' },
      { label: 'What actually ran', text: 'The authors’ start file for this data uses 360. Its batch size of 64 is doubled to an effective 128 by the code.' },
      { label: 'What we measured', text: 'With these settings, a live run without human edits measured 0.37203 ± 0.00005 over 3 seeds. An earlier live run with other settings measured about 0.70.' },
      { label: 'Why it matters', text: 'Anyone who rebuilds from the table alone has to guess the settings. In our earlier run, other settings nearly doubled the error.' },
      { label: 'What Nullius does now', text: 'It reads the start file and the paper’s source text, and checks the settings against the paper before any paid step.' },
    ],
    commits: ['be1a643'],
    links: [
      { href: '#/run/disc_2307_03756--20261014-080542', label: 'Open the FITS check' },
      doc('docs/audits/fits_live_a514edeb_20261014.json', 'Run record (GitHub)'),
      doc('docs/fits_setup_binding_fixes_20261013.md', 'What we fixed (GitHub)'),
    ],
  },
  {
    id: 'sparsetsf',
    part: 'A',
    tag: 'SparseTSF · ETTh1',
    headline: 'Run as is, the start file builds another model',
    text: 'The authors’ start file for the simple version of the model does not say which version to build. The code picks a different one, so we locked this check instead of guessing.',
    left: { label: 'Paper’s model', value: 'linear version', tone: 'green' },
    right: { label: 'Start file, run as is', value: 'MLP version', tone: 'amber' },
    details: [
      { label: 'What the paper says', text: 'SparseTSF has a linear version and an MLP version. Target error (MSE): 0.359.' },
      { label: 'What actually ran', text: 'The authors’ start file for the linear version does not set the model type. The code’s default is the MLP version.' },
      { label: 'What we measured', text: 'The recorded run of 9 October used the linear version and reached the target: 0.3653 ± 0.0034 over 3 seeds. In the re-check of 14 October the pipeline stopped before running anything, because the start file does not match.' },
      { label: 'Why it matters', text: 'Anyone who runs the start file as is trains a different model than the paper describes.' },
      { label: 'What Nullius does now', text: 'It compares the model named in the start file with the model of the claim and stops with a reason when they differ. This entry stays locked in the demo.' },
    ],
    commits: [],
    links: [
      { href: '#/run/disc_2405_00946--20261009-220822', label: 'Open the recorded check' },
      doc('docs/fits_code_generation_fixes_20261014.md', 'Re-check notes (GitHub)'),
    ],
  },
  {
    id: 'rlinear',
    part: 'A',
    tag: 'RLinear · ETTh1',
    headline: 'One key setting sits in a table’s small print',
    text: 'The paper names how much past data the model sees only in the small text under a table. Our check used the code’s default instead and landed further from the paper.',
    left: { label: 'Small print', value: '336', tone: 'green' },
    right: { label: 'What ran', value: '96', tone: 'amber' },
    details: [
      { label: 'What the paper says', text: 'The input length 336 is stated only in the caption of the claim table. Target error (MSE): 0.366.' },
      { label: 'What actually ran', text: 'The generated code used the code default of 96. A second cause: the default early-stopping patience in the function signature beat the value we passed in.' },
      { label: 'What we measured', text: '0.397 ± 0.013, not reproduced. Cost 0.19 $ gross.' },
      { label: 'Why it matters', text: 'A reader who skips captions rebuilds a different experiment and gets a different number.' },
      { label: 'What Nullius does now', text: 'It reads table captions and keeps the value that was passed in. Whether the target is reached with these fixes is untested.' },
    ],
    commits: ['8abc8da', 'bb5f890'],
    links: [doc('docs/audits/rlinear_live_ba9fa94f_20261014.json', 'Run record (GitHub)')],
  },
  {
    id: 'tsmixer',
    part: 'A',
    tag: 'TSMixer · ETTh1',
    headline: 'Reaching the number is not the same as rebuilding',
    text: 'We reached the paper’s number with a shrunk copy of the model. We call that “target reached with a reduced model”, not “reproduced”.',
    left: { label: 'Paper’s model', value: '6 blocks', tone: 'green' },
    right: { label: 'Model that ran', value: '1 block', tone: 'amber' },
    details: [
      { label: 'What the paper says', text: 'Six building blocks, size 512, dropout 0.9. Target error (MSE): 0.361.' },
      { label: 'What actually ran', text: 'The generated code shrank the model for speed: 1 block instead of 6, size 128 instead of 512.' },
      { label: 'What we measured', text: '0.3624 ± 0.0003 over 3 seeds on 9 October. In the re-check of 14 October, a first try read the wrong defaults (dropout 0.05, size 64) because the paper’s settings table in the source was not read. With the right settings, each try took longer than 10 minutes on a CPU, so the check could not be finished within the budget.' },
      { label: 'Why it matters', text: 'A matching number can come from a different, smaller model. The label has to say so.' },
      { label: 'What Nullius does now', text: 'It reads settings tables in the paper’s source. The catalog labels this run “target reached with a reduced model”.' },
    ],
    commits: [],
    links: [
      { href: '#/run/disc_2303_06053--20261009-201230', label: 'Open the recorded check' },
      doc('docs/fits_code_generation_fixes_20261014.md', 'Re-check notes (GitHub)'),
    ],
  },
  {
    id: 'saraf',
    part: 'A',
    tag: 'SARAF · ETTh1',
    headline: 'The method needs calendar data we did not pass in',
    text: 'SARAF looks for similar past moments and also uses hour and weekday. Our input had only plain numbers. Rebuilt by hand with the authors’ own model core and calendar data, the error was within range.',
    left: { label: 'Paper target', value: '0.371', tone: 'gray' },
    right: { label: 'By hand, with calendar', value: '0.3696', tone: 'green' },
    details: [
      { label: 'What the paper says', text: 'Target error (MSE): 0.371. The method weighs calendar features (year, month, day, weekday, hour) at 0.5 when it picks similar moments.' },
      { label: 'What actually ran', text: 'Until then, the input to the generated code held only the numbers. A test on real data shows that leaving the calendar out changes the similarity scores by up to 0.63.' },
      { label: 'What we measured', text: 'A person wrote an adapter around the unchanged authors’ model core and gave it the calendar data: 0.3696 ± 0.0014 over 3 seeds. This is not an autonomous Nullius run. Earlier live runs gave 0.4917 ± 0.0031 (the authors’ model code was cut off in the prompt) and one stop before any measurement.' },
      { label: 'Why it matters', text: 'It shows the target can be reached with the authors’ method when the input is complete. It does not show that Nullius rebuilds the method by itself.' },
      { label: 'What Nullius does now', text: 'The input now carries the calendar fields next to the numbers. A new autonomous run is still open.' },
    ],
    commits: ['db5e571'],
    links: [
      doc('docs/saraf_local_contract_verification_20261013.md', 'Calendar check (GitHub)'),
      doc('docs/saraf_local_audit_20261013.md', 'Earlier live runs (GitHub)'),
    ],
  },
  {
    id: 'cutoff',
    part: 'B',
    tag: 'Sandbox output',
    headline: 'A long file list was cut off, a file looked missing',
    text: 'The sandbox keeps only the first 65,535 bytes of output. Our list of 1,410 files was longer. Nullius now moves such lists as a file and checks its size and fingerprint.',
    details: [
      { label: 'What happened', text: 'The FITS file list had 83,540 bytes. The cut fell before the model file, so the check ran with an incomplete setup (0 of 3 seeds measured, no verdict).' },
      { label: 'What Nullius does now', text: 'It copies the list through its own file with byte count and hash, and stops if they do not match. We also reported this in our Nebius feedback.' },
    ],
    commits: ['30c67b0'],
    links: [
      doc('docs/fits_deployed_verification_20261013.md', 'Check record (GitHub)'),
      doc('docs/nebius_feedback.md', 'Our Nebius feedback (GitHub)'),
    ],
  },
  {
    id: 'fewshot',
    part: 'B',
    tag: 'Finding papers',
    headline: 'A “few examples” table looked like a normal result',
    text: 'One discovered claim came from a table where the model saw only 1 or 5 labeled examples per class. Discovery now skips such tables.',
    details: [
      { label: 'What happened', text: 'The GILT claim (73.22 on Cora) sat in a table whose caption names 1-shot and 5-shot accuracy. We caught it at the paper’s source before any paid run.' },
      { label: 'What Nullius does now', text: 'Discovery blocks tables that mention shots. On 12 GILT tables it flagged all 12 and none of 37 tables of other papers.' },
    ],
    commits: ['6618c29'],
    links: [doc('docs/fits_code_generation_fixes_20261014.md', 'Notes (GitHub)')],
  },
  {
    id: 'quote',
    part: 'B',
    tag: 'Reading targets',
    headline: 'A quote must really exist in the paper',
    text: 'An AI once wrote a simplified quote for a target number. Nullius now checks the number in the freshly downloaded table before any paid step.',
    details: [
      { label: 'What happened', text: 'For the SARAF target 0.371 a second AI invented a simplified, non-literal LaTeX snippet, and the check stopped.' },
      { label: 'What Nullius does now', text: 'A shared table reader checks method, data, horizon and column in the source file first, without a model call. A wrong old target stays stopped.' },
    ],
    commits: [],
    links: [doc('docs/discovery_target_handoff_20261013.md', 'Notes (GitHub)')],
  },
  {
    id: 'placebo',
    part: 'B',
    tag: 'Benchmark v4',
    headline: `None of ${placebo[1]} planted false claims was accepted`,
    text: `We planted ${placebo[1]} claims with a wrong target on purpose. ${placebo[0]} of ${placebo[1]} were reported as reproduced. That is a result of this test, not a guarantee for new papers.`,
    details: [
      { label: 'What it is', text: 'Each run in the benchmark also gets one false claim. A checker that says “reproduced” for these is too easy to fool.' },
      { label: 'Earlier versions', text: `Benchmark v2 and v3 gave ${FIGURES.placebo.v2[0]} of ${FIGURES.placebo.v2[1]} each.` },
    ],
    commits: [],
    links: [{ href: '#benchmark', label: 'Open the Benchmark' }],
  },
]

export const partA = FINDINGS.filter((f) => f.part === 'A')
export const partB = FINDINGS.filter((f) => f.part === 'B')

// Every word a visitor sees with all details closed (used by the test that keeps the page short).
export function visibleText(): string {
  const words = [INTRO.title, INTRO.summary, INTRO.analogy, PART_A.title, PART_A.sub, PART_B.title, PART_B.sub]
  for (const f of FINDINGS) {
    words.push(f.tag, f.headline, f.text)
    if (f.left && f.right) words.push(f.left.label, f.left.value, f.right.label, f.right.value)
  }
  return words.join(' ')
}
