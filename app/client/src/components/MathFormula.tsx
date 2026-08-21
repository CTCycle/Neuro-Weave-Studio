import katex from 'katex'
import { useMemo } from 'react'
import 'katex/dist/katex.min.css'

export function MathFormula({ formula }: { formula: string }) {
  const markup = useMemo(() => katex.renderToString(formula, { displayMode: false, throwOnError: false }), [formula])
  return <span className="math-formula" aria-label="Formula" dangerouslySetInnerHTML={{ __html: markup }} />
}
