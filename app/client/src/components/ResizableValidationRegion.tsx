import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'

const MIN_HEIGHT = 76
const MIN_CANVAS_HEIGHT = 150
export const DEFAULT_VALIDATION_HEIGHT = 160
export const COLLAPSED_VALIDATION_HEIGHT = 42

interface Props {
  children: ReactNode
  expanded: boolean
  height: number
  onHeightChange: (height: number) => void
}
export function ResizableValidationRegion({ children, expanded, height, onHeightChange }: Props) {
  const regionRef = useRef<HTMLDivElement>(null)
  const [maximumHeight, setMaximumHeight] = useState(DEFAULT_VALIDATION_HEIGHT)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    const workspace = regionRef.current?.parentElement
    if (!workspace || typeof ResizeObserver === 'undefined') return
    const updateMaximum = () => setMaximumHeight(Math.max(MIN_HEIGHT, workspace.clientHeight - MIN_CANVAS_HEIGHT - 24))
    const observer = new ResizeObserver(updateMaximum)
    observer.observe(workspace)
    updateMaximum()
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!dragging || !expanded) return
    const handleMove = (event: PointerEvent) => {
      const workspace = regionRef.current?.parentElement
      if (!workspace) return
      const bounds = workspace.getBoundingClientRect()
      const maximum = Math.max(MIN_HEIGHT, bounds.height - MIN_CANVAS_HEIGHT - 24)
      onHeightChange(Math.min(maximum, Math.max(MIN_HEIGHT, bounds.bottom - event.clientY)))
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
  }, [dragging, expanded, onHeightChange])

  const startDragging = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    setDragging(true)
  }

  return (
    <section ref={regionRef} className={`resizable-validation-region ${expanded ? 'is-expanded' : 'is-collapsed'} ${dragging ? 'is-dragging' : ''}`} style={{ height: expanded ? Math.min(Math.max(MIN_HEIGHT, height), maximumHeight) : COLLAPSED_VALIDATION_HEIGHT }}>
      {expanded && <div className="validation-splitter" role="separator" aria-label="Resize validation panel" aria-orientation="horizontal" aria-valuemin={MIN_HEIGHT} aria-valuemax={maximumHeight} aria-valuenow={Math.min(Math.max(MIN_HEIGHT, height), maximumHeight)} tabIndex={0} onPointerDown={startDragging}>
        <span aria-hidden="true" />
      </div>}
      <div className="validation-region-content">{children}</div>
    </section>
  )
}
