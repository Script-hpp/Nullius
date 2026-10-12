import test from 'node:test'
import assert from 'node:assert/strict'
import { scrollToSection } from './sectionScroll.ts'

const fake = (present: boolean) => {
  const calls: unknown[] = []
  return { calls, doc: { getElementById: () => (present ? { scrollIntoView: (o: unknown) => calls.push(o) } : null) } }
}

test('scrolls smoothly and reports handled; works on every call', () => {
  const f = fake(true)
  assert.equal(scrollToSection('tavily', f.doc, false), true)
  assert.equal(scrollToSection('tavily', f.doc, false), true)
  assert.deepEqual(f.calls, [{ behavior: 'smooth', block: 'start' }, { behavior: 'smooth', block: 'start' }])
})

test('reduced motion jumps instead of animating', () => {
  const f = fake(true)
  scrollToSection('tavily', f.doc, true)
  assert.deepEqual(f.calls, [{ behavior: 'auto', block: 'start' }])
})

test('missing section: not handled, the link navigates', () => {
  assert.equal(scrollToSection('tavily', fake(false).doc, false), false)
})
