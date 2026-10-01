// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, ElementRef, HostListener, computed, inject, input, output, signal, viewChild } from '@angular/core'
import { LucideCheck, LucideChevronDown } from '@lucide/angular'

export interface ThemedSelectOption {
  value: string
  label: string
  disabled?: boolean
}

let nextSelectId = 0

@Component({
  selector: 'nw-themed-select',
  standalone: true,
  imports: [LucideCheck, LucideChevronDown],
  template: `
    <div class="themed-select">
      <button #trigger type="button" class="themed-select-trigger" role="combobox" [attr.aria-label]="ariaLabel()" [attr.data-value]="value()" [attr.aria-controls]="listboxId" [attr.aria-expanded]="isOpen()" [attr.aria-activedescendant]="isOpen() ? optionId(activeIndex()) : null" (click)="toggle()" (keydown)="handleKeydown($event)">
        <span class="themed-select-value">{{ selectedOption()?.label || 'Select…' }}</span><svg lucideChevronDown [size]="14" aria-hidden="true"></svg>
      </button>
      @if (isOpen()) {
        <div #listbox class="themed-select-menu" [id]="listboxId" role="listbox" [attr.aria-label]="ariaLabel()">
          @for (option of options(); track option.value; let index = $index) {
            <button type="button" class="themed-select-option" [class.is-active]="activeIndex() === index" [class.is-selected]="option.value === value()" [disabled]="option.disabled" role="option" [id]="optionId(index)" [attr.aria-selected]="option.value === value()" (mouseenter)="activeIndex.set(index)" (click)="choose(option)"><span>{{ option.label }}</span>@if (option.value === value()) { <svg lucideCheck [size]="13" aria-hidden="true"></svg> }</button>
          }
        </div>
      }
    </div>
  `,
})
export class ThemedSelectComponent {
  readonly options = input<readonly ThemedSelectOption[]>([])
  readonly value = input('')
  readonly ariaLabel = input('Select an option')
  readonly valueChange = output<string>()
  readonly isOpen = signal(false)
  readonly activeIndex = signal(0)
  private readonly host = injectElementRef()
  private readonly trigger = viewChild<ElementRef<HTMLButtonElement>>('trigger')
  readonly selectId = `nw-themed-select-${nextSelectId++}`
  readonly listboxId = `${this.selectId}-listbox`
  readonly selectedOption = computed(() => this.options().find((option) => option.value === this.value()))

  toggle(): void {
    if (this.isOpen()) {
      this.close()
      return
    }
    this.syncActiveIndex()
    this.isOpen.set(true)
  }

  close(restoreFocus = false): void {
    this.isOpen.set(false)
    if (restoreFocus) queueMicrotask(() => this.trigger()?.nativeElement.focus())
  }

  choose(option: ThemedSelectOption): void {
    if (option.disabled) return
    this.valueChange.emit(option.value)
    this.close(true)
  }

  handleKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && this.isOpen()) {
      event.preventDefault()
      this.close(true)
      return
    }
    if ((event.key === 'Enter' || event.key === ' ') && !this.isOpen()) {
      event.preventDefault()
      this.toggle()
      return
    }
    if (!this.isOpen()) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        this.toggle()
      }
      return
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      this.moveActive(event.key === 'ArrowDown' ? 1 : -1)
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault()
      this.activeIndex.set(event.key === 'Home' ? this.firstEnabledIndex() : this.lastEnabledIndex())
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      const option = this.options()[this.activeIndex()]
      if (option) this.choose(option)
    }
  }

  optionId(index: number): string { return `${this.selectId}-option-${index}` }

  @HostListener('document:pointerdown', ['$event'])
  closeOnOutsidePointer(event: PointerEvent): void {
    if (!this.isOpen()) return
    const target = event.target instanceof Node ? event.target : null
    if (!target || !this.host.nativeElement.contains(target)) this.close()
  }

  private syncActiveIndex(): void {
    const selectedIndex = this.options().findIndex((option) => option.value === this.value() && !option.disabled)
    this.activeIndex.set(selectedIndex >= 0 ? selectedIndex : this.firstEnabledIndex())
  }

  private moveActive(delta: number): void {
    const options = this.options()
    if (!options.length) return
    let index = this.activeIndex()
    for (let attempt = 0; attempt < options.length; attempt += 1) {
      index = (index + delta + options.length) % options.length
      if (!options[index].disabled) {
        this.activeIndex.set(index)
        return
      }
    }
  }

  private firstEnabledIndex(): number { return Math.max(0, this.options().findIndex((option) => !option.disabled)) }
  private lastEnabledIndex(): number {
    const options = this.options()
    for (let index = options.length - 1; index >= 0; index -= 1) if (!options[index].disabled) return index
    return 0
  }
}

function injectElementRef(): ElementRef<HTMLElement> {
  return inject(ElementRef)
}
