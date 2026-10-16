import { useState } from 'react'
import { Download } from 'lucide-react'
import type { RunData } from '../data'
import { certificateHtml } from '../certificate'

export function CertificateActions({ run }: { run: RunData }) {
  const [notice, setNotice] = useState('')
  const exportPdf = () => {
    const preview = window.open('', '_blank')
    if (!preview) { setNotice('Allow a new tab to open the certificate print view.'); return }
    preview.opener = null
    preview.document.open()
    preview.document.write(certificateHtml(run, location.href))
    preview.document.close()
    preview.document.getElementById('certificate-print')?.addEventListener('click', () => preview.print())
    preview.focus()
    setNotice('Certificate opened. Choose “Save as PDF / Print” in the new tab.')
  }
  return <span className="inline-flex flex-col items-end gap-1">
    <button type="button" onClick={exportPdf} className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-line-strong bg-transparent px-3.5 py-1.5 text-[12px] font-medium text-fg hover:bg-raised/60"><Download className="h-3.5 w-3.5" />Export PDF</button>
    {notice && <span role="status" className="max-w-56 text-right text-xs text-muted">{notice}</span>}
  </span>
}
