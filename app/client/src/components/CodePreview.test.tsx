import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CodePreview } from './CodePreview'

describe('CodePreview', () => {
  afterEach(cleanup)

  it('reports successful source copying', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })

    render(<CodePreview code="generated source" embedded />)
    await userEvent.click(screen.getByRole('button', { name: 'Copy code' }))

    expect(writeText).toHaveBeenCalledWith('generated source')
    expect(screen.getByRole('status')).toHaveTextContent('Copied')
  })
})
