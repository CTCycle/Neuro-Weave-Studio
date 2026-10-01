// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { DOCUMENT } from '@angular/common'
import { Directive, ElementRef, Injectable, booleanAttribute, effect, inject, input } from '@angular/core'

type ModalSnapshot = {
  root: HTMLElement | null
  rootOverflow: string
  rootOverscroll: string
  rootScrollTop: number
  bodyOverflow: string
  focused: Element | null
}

@Injectable({ providedIn: 'root' })
export class ModalCoordinatorService {
  private readonly document = inject(DOCUMENT)
  private readonly stack: HTMLElement[] = []
  private snapshot: ModalSnapshot | null = null
  private readonly onKeydown = (event: KeyboardEvent): void => {
    const top = this.top()
    if (!top) return
    if (!(event.target instanceof Node) || !top.contains(event.target)) {
      event.preventDefault()
      event.stopImmediatePropagation()
      return
    }
    if (event.key === 'Tab') this.trapFocus(event, top)
  }
  private readonly onFocusin = (event: FocusEvent): void => {
    const top = this.top()
    if (!top || (event.target instanceof Node && top.contains(event.target))) return
    event.preventDefault()
    this.focusFirst(top)
  }
  private readonly onPointer = (event: Event): void => {
    const top = this.top()
    if (!top || (event.target instanceof Node && top.contains(event.target))) return
    event.preventDefault()
    event.stopImmediatePropagation()
  }

  open(element: HTMLElement): () => void {
    if (this.stack.includes(element)) return () => this.close(element)
    if (!this.stack.length) this.lockPage()
    this.stack.push(element)
    queueMicrotask(() => {
      if (this.top() === element) this.focusFirst(element)
    })
    return () => this.close(element)
  }

  private close(element: HTMLElement): void {
    const index = this.stack.indexOf(element)
    if (index < 0) return
    this.stack.splice(index, 1)
    if (this.stack.length) {
      queueMicrotask(() => this.focusFirst(this.top()!))
      return
    }
    this.unlockPage()
  }

  private top(): HTMLElement | null { return this.stack[this.stack.length - 1] ?? null }

  private lockPage(): void {
    const root = this.document.querySelector<HTMLElement>('.app-content')
    this.snapshot = {
      root,
      rootOverflow: root?.style.overflow ?? '',
      rootOverscroll: root?.style.overscrollBehavior ?? '',
      rootScrollTop: root?.scrollTop ?? 0,
      bodyOverflow: this.document.body.style.overflow,
      focused: this.document.activeElement,
    }
    if (root) {
      root.style.overflow = 'hidden'
      root.style.overscrollBehavior = 'none'
    }
    this.document.documentElement.classList.add('modal-open')
    this.document.body.classList.add('modal-open')
    this.document.addEventListener('keydown', this.onKeydown, true)
    this.document.addEventListener('focusin', this.onFocusin, true)
    this.document.addEventListener('pointerdown', this.onPointer, true)
    this.document.addEventListener('click', this.onPointer, true)
    this.document.addEventListener('wheel', this.onPointer, { capture: true, passive: false })
    this.document.addEventListener('touchmove', this.onPointer, { capture: true, passive: false })
  }

  private unlockPage(): void {
    const snapshot = this.snapshot
    if (!snapshot) return
    const { root } = snapshot
    if (root) {
      root.style.overflow = snapshot.rootOverflow
      root.style.overscrollBehavior = snapshot.rootOverscroll
      root.scrollTop = snapshot.rootScrollTop
    }
    this.document.body.style.overflow = snapshot.bodyOverflow
    this.document.documentElement.classList.remove('modal-open')
    this.document.body.classList.remove('modal-open')
    this.document.removeEventListener('keydown', this.onKeydown, true)
    this.document.removeEventListener('focusin', this.onFocusin, true)
    this.document.removeEventListener('pointerdown', this.onPointer, true)
    this.document.removeEventListener('click', this.onPointer, true)
    this.document.removeEventListener('wheel', this.onPointer, true)
    this.document.removeEventListener('touchmove', this.onPointer, true)
    const focused = snapshot.focused
    this.snapshot = null
    queueMicrotask(() => {
      if (focused instanceof HTMLElement && focused.isConnected) focused.focus()
    })
  }

  private focusFirst(element: HTMLElement): void {
    const focusable = this.focusables(element)
    const target = focusable[0] ?? element
    if (target === element && !element.hasAttribute('tabindex')) element.setAttribute('tabindex', '-1')
    target.focus()
  }

  private trapFocus(event: KeyboardEvent, element: HTMLElement): void {
    const focusable = this.focusables(element)
    if (!focusable.length) {
      event.preventDefault()
      this.focusFirst(element)
      return
    }
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && this.document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && this.document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  private focusables(element: HTMLElement): HTMLElement[] {
    return Array.from(element.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((item) => item.getAttribute('aria-hidden') !== 'true')
  }
}

@Directive({
  selector: '[nwModalGuard]',
  standalone: true,
})
export class ModalGuardDirective {
  readonly nwModalGuard = input(true, { transform: booleanAttribute })
  private release?: () => void
  private readonly element = inject(ElementRef<HTMLElement>)
  private readonly coordinator = inject(ModalCoordinatorService)

  constructor() {
    effect(() => {
      const active = this.nwModalGuard()
      if (active && !this.release) this.release = this.coordinator.open(this.element.nativeElement)
      if (!active && this.release) {
        this.release()
        this.release = undefined
      }
    })
  }

  ngOnDestroy(): void { this.release?.(); this.release = undefined }
}
