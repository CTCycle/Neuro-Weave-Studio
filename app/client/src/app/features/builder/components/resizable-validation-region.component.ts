// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, effect, inject, input, output, signal, type AfterViewInit, type OnDestroy } from '@angular/core'

const MIN_HEIGHT = 76
const MIN_CANVAS_HEIGHT = 150
export const DEFAULT_VALIDATION_HEIGHT = 160
export const COLLAPSED_VALIDATION_HEIGHT = 42

@Component({
  selector: 'nw-resizable-validation-region',
  standalone: true,
  template: `<section class="resizable-validation-region" [class.is-expanded]="expanded()" [class.is-collapsed]="!expanded()" [class.is-dragging]="dragging()" [style.height.px]="expanded() ? boundedHeight() : COLLAPSED_VALIDATION_HEIGHT">@if (expanded()) { <div class="validation-splitter" role="separator" aria-label="Resize validation panel" aria-orientation="horizontal" [attr.aria-valuemin]="MIN_HEIGHT" [attr.aria-valuemax]="maximumHeight()" [attr.aria-valuenow]="boundedHeight()" tabindex="0" (pointerdown)="startDragging($event)"><span aria-hidden="true"></span></div> }<div class="validation-region-content"><ng-content /></div></section>`,
})
export class ResizableValidationRegionComponent implements AfterViewInit, OnDestroy {
  readonly expanded = input(false)
  readonly height = input(DEFAULT_VALIDATION_HEIGHT)
  readonly heightChange = output<number>()
  readonly dragging = signal(false)
  readonly maximumHeight = signal(DEFAULT_VALIDATION_HEIGHT)
  readonly MIN_HEIGHT = MIN_HEIGHT
  readonly COLLAPSED_VALIDATION_HEIGHT = COLLAPSED_VALIDATION_HEIGHT
  private readonly elementRef = inject(ElementRef<HTMLElement>)
  private observer?: ResizeObserver
  private readonly handleMove = (event: PointerEvent): void => {
    const workspace = this.elementRef.nativeElement.parentElement
    if (!workspace) return
    const bounds = workspace.getBoundingClientRect()
    const maximum = Math.max(MIN_HEIGHT, bounds.height - MIN_CANVAS_HEIGHT - 24)
    this.heightChange.emit(Math.min(maximum, Math.max(MIN_HEIGHT, bounds.bottom - event.clientY)))
  }
  private readonly stopDragging = (): void => {
    if (!this.dragging()) return
    this.dragging.set(false)
    window.removeEventListener('pointermove', this.handleMove)
    window.removeEventListener('pointerup', this.stopDragging)
    window.removeEventListener('pointercancel', this.stopDragging)
    document.body.style.userSelect = ''
  }

  constructor() {
    effect(() => { if (!this.expanded() && this.dragging()) this.stopDragging() })
  }

  ngAfterViewInit(): void {
    const workspace = this.elementRef.nativeElement.parentElement
    if (!workspace || typeof ResizeObserver === 'undefined') return
    const updateMaximum = (): void => this.maximumHeight.set(Math.max(MIN_HEIGHT, workspace.clientHeight - MIN_CANVAS_HEIGHT - 24))
    this.observer = new ResizeObserver(updateMaximum)
    this.observer.observe(workspace)
    updateMaximum()
  }

  ngOnDestroy(): void { this.stopDragging(); this.observer?.disconnect() }

  boundedHeight(): number { return Math.min(Math.max(MIN_HEIGHT, this.height()), this.maximumHeight()) }

  startDragging(event: PointerEvent): void {
    if (event.button !== 0) return
    event.preventDefault()
    this.dragging.set(true)
    window.addEventListener('pointermove', this.handleMove)
    window.addEventListener('pointerup', this.stopDragging, { once: true })
    window.addEventListener('pointercancel', this.stopDragging, { once: true })
    document.body.style.userSelect = 'none'
  }
}
