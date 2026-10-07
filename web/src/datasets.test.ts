import { describe, expect, it } from 'vitest'
import { benchmarkText, type DPaper } from './discovery'
import { translateReason } from './labels'

const paper = (matched_task: string, horizon?: number) =>
  ({ dataset: matched_task, protocol: 'passt', claim: { matched_task, horizon } }) as unknown as DPaper

describe('supported datasets', () => {
  it('labels released Planetoid and ETT tasks', () => {
    expect(benchmarkText(paper('CiteSeer'))).toBe('CiteSeer · Planetoid split')
    expect(benchmarkText(paper('ETTh1'))).toBe('ETTh1 · horizon 96')
  })
  it('translates horizon reasons for ETT tasks', () => {
    expect(translateReason('ETTm2: Horizont 192, Task misst nur 96')).toBe('horizon 192, the task only measures 96')
    expect(translateReason('ETTh2: Horizont fehlt oder nicht lesbar (None), Task misst nur 96')).toBe('the horizon is missing or unreadable')
  })
})
