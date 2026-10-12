import { Mail } from 'lucide-react'
import { REPO } from '../data/benchmark'
import { CONTACT_NAME, CONTACT_EMAIL } from '../contact'
import { ExtLink, INLINE } from './ui'

// Contact information only, no form.
export function Contact() {
  return (
    <main className="mx-auto max-w-3xl px-4 pb-16 md:px-8">
      <header className="pb-10 pt-10 md:pt-14">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Contact</div>
        <h1 className="font-grotesk m-0 mt-3 text-4xl font-semibold leading-[1.02] text-fg sm:text-5xl">Get in touch</h1>
        <p className="mb-0 mt-6 max-w-xl text-[16px] leading-relaxed text-fg-4">Nullius is a hackathon project demo. Questions, corrections and ideas are welcome.</p>
      </header>
      <dl className="m-0 grid gap-0 border-t border-line">
        <div className="grid gap-1 border-b border-line py-5 sm:grid-cols-[10rem_1fr]">
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Name</dt>
          <dd className="m-0 text-[15px] text-fg">{CONTACT_NAME}</dd>
        </div>
        <div className="grid gap-1 border-b border-line py-5 sm:grid-cols-[10rem_1fr]">
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">E-mail</dt>
          <dd className="m-0 text-[15px]"><a href={`mailto:${CONTACT_EMAIL}`} className={`${INLINE} text-fg underline underline-offset-4 hover:text-fg-max`}><Mail className="h-3.5 w-3.5 shrink-0" aria-hidden />{CONTACT_EMAIL}</a></dd>
        </div>
        <div className="grid gap-1 border-b border-line py-5 sm:grid-cols-[10rem_1fr]">
          <dt className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted">Code and issues</dt>
          <dd className="m-0 text-[15px]"><ExtLink href={REPO}>GitHub repository</ExtLink></dd>
        </div>
      </dl>
    </main>
  )
}
