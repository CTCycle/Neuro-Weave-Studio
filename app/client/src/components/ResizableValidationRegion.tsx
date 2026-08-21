import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'

const MIN_HEIGHT = 76
const MIN_CANVAS_HEIGHT = 150
const DEFAULT_HEIGHT = 150

interface Props {
  children: ReactNode
}
export function ResizableValidationRegion({ children }: Props) {
  const regionRef = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(DEFAULT_HEIGHT)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    if (!dragging) return
    const handleMove = (event: PointerEvent) => {
      const workspace = regionRef.current?.parentElement
      if (!workspace) return
      const bounds = workspace.getBoundingClientRect()
      const maximum = Math.max(MIN_HEIGHT, bounds.height - MIN_CANVAS_HEIGHT - 24)
      setHeight(Math.min(maximum, Math.max(MIN_HEIGHT, bounds.bottom - event.clientY)))
    }
    const stopDragging = () => setDragging(false)
    window.addEventListener('pointermove', handleMove)
    window.addEventListener('pointerup', stopDragging, { once: true })
    window.addEventListener('pointercancel', stopDragging, { once: true })
    document.body.style.userSelect = 'none'
    return () => {
      window.removeEventListener('pointermove', handleMove)
      window.removeEventListener('pointerup', stopDragging)
      window.removeEventListener('pointercancel', stopDragging)
      document.body.style.userSelect = ''
    }
  }, [dragging])

  const startDragging = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    setDragging(true)
  }

  return (
    <section ref={regionRef} className={`resizable-validation-region ${dragging ? 'is-dragging' : ''}`} style={{ height }}>
      <div className="validation-splitter" role="separator" aria-label="Resize validation panel" aria-orientation="horizontal" aria-valuemin={MIN_HEIGHT} aria-valuemax={Math.max(MIN_HEIGHT, height)} aria-valuenow={height} tabIndex={0} onPointerDown={startDragging}>
        <span aria-hidden="true" />
      </div>
      <div className="validation-region-content">{children}</div>
    </section>
  )
}
