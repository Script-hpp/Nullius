// Nullius mark: a tolerance bracket around a point, [ • ] (the claim lies within the tolerance). Geometry is fixed; only the stroke weight
// follows the size (large 1.8, up to 32 px 2.2, up to 16 px 2.6). Colour is currentColor, so it follows the theme.
export const markStroke = (size: number) => (size <= 16 ? 2.6 : size <= 32 ? 2.2 : 1.8)

export function Mark({ size = 16, className = '' }: { size?: number; className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={markStroke(size)} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" className={className}>
      <path d="M9 5.5H6v13h3M15 5.5h3v13h-3" />
      <circle cx="12" cy="12" r="2.3" fill="currentColor" stroke="none" />
    </svg>
  )
}
