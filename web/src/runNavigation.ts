export type CollectionPosition = { hash: string; scrollTop: number }
const origins = new Map<string, CollectionPosition>()
const positions = new Map<string, number>()

export function collectionHash(hash: string): string | null {
  if (/^#catalog(?:\?view=(?:featured|all|failed|unchecked))?$/.test(hash)) return hash
  return hash === '#overview' ? hash : null
}

export function rememberRunOrigin(href: string, hash: string, scrollTop: number) {
  const collection = collectionHash(hash)
  const run = href.match(/^#\/run\/([^?]+)(?:\?.*)?$/)?.[1]
  if (!collection || !run || !Number.isFinite(scrollTop)) return
  const position = Math.max(0, scrollTop)
  origins.set(run, { hash: collection, scrollTop: position })
  positions.set(collection, position)
}

export function runOrigin(id: string): CollectionPosition {
  return origins.get(encodeURIComponent(id)) ?? { hash: '#overview', scrollTop: 0 }
}

export function takeCollectionPosition(hash: string): number | null {
  const position = positions.get(hash) ?? null
  positions.delete(hash)
  return position
}
