// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, forwardRef, input, output, signal } from '@angular/core'
import type { ParameterCapability, ParameterDefinition, ParameterValueSchema } from '../../../domain/types'
import { parameterValueMatchesDefinition } from '../../../domain/types'
import { parseParameterValue, type ParameterDraftValue } from '../graph/parameter-value'
import { ParameterEditorComponent, type ParameterEditorState } from './parameter-editor.component'

interface StructuredField {
  id: string
  definition: ParameterDefinition
  value: unknown
}

@Component({
  selector: 'nw-structured-parameter-editor',
  standalone: true,
  imports: [forwardRef(() => ParameterEditorComponent), forwardRef(() => StructuredParameterEditorComponent)],
  template: `
    <div class="structured-editor" [class.is-disabled]="disabled() || state() !== 'editable'">
      <div class="structured-toolbar"><span>{{ expertJson() ? 'Expert JSON' : 'Structured values' }}</span><button type="button" [disabled]="disabled()" (click)="expertJson.update((value) => !value)">{{ expertJson() ? 'Use fields' : 'Edit JSON' }}</button></div>
      @if (expertJson()) {
        <textarea rows="4" [value]="jsonValue()" [disabled]="disabled()" [attr.aria-invalid]="invalid() ? 'true' : null" (input)="updateJson(($any($event.target)).value)"></textarea>
      } @else if (fields().length) {
        <div class="structured-fields">
          @for (field of fields(); track field.id) {
            <div class="structured-field"><label>{{ field.definition.display_name }}@if (field.definition.description) { <small>{{ field.definition.description }}</small> }</label>
              @if (isNested(field.definition)) {
                <nw-structured-parameter-editor [definition]="field.definition" [value]="field.value" [disabled]="disabled()" [state]="state()" (valueChange)="updateObjectField(field.id, $event)" (validChange)="validChange.emit($event)" />
              } @else {
                <nw-parameter-editor [definition]="field.definition" [value]="field.value" [disabled]="disabled()" [state]="state()" (valueChange)="updateObjectField(field.id, $event)" (validChange)="validChange.emit($event)" />
              }
            </div>
          }
        </div>
      } @else if (arrayItems().length) {
        <div class="structured-fields">
          @for (item of arrayItems(); track item.id) {
            <div class="structured-field"><label>{{ item.definition.display_name }}</label>
              @if (isNested(item.definition)) {
                <nw-structured-parameter-editor [definition]="item.definition" [value]="item.value" [disabled]="disabled()" [state]="state()" (valueChange)="updateArrayItem(item.index, $event)" (validChange)="validChange.emit($event)" />
              } @else {
                <nw-parameter-editor [definition]="item.definition" [value]="item.value" [disabled]="disabled()" [state]="state()" (valueChange)="updateArrayItem(item.index, $event)" (validChange)="validChange.emit($event)" />
              }
            </div>
          }
        </div>
      } @else {
        <textarea rows="3" [value]="jsonValue()" [disabled]="disabled()" [attr.aria-invalid]="invalid() ? 'true' : null" (input)="updateJson(($any($event.target)).value)"></textarea>
      }
      @if (invalid()) { <small class="structured-error">Enter a valid structured value.</small> }
    </div>
  `,
  styles: [`
    :host { display:block; min-width:0; }
    .structured-editor { display:grid; gap:7px; min-width:0; }
    .structured-toolbar { display:flex; align-items:center; justify-content:space-between; gap:8px; color:var(--muted); font:500 8px/1.2 'IBM Plex Mono', Consolas, monospace; text-transform:uppercase; letter-spacing:.05em; }
    .structured-toolbar button { border:0; background:transparent; color:var(--cyan); cursor:pointer; font:inherit; text-transform:inherit; }
    .structured-fields { display:grid; gap:8px; padding:8px; border:1px solid var(--border); border-radius:5px; background:color-mix(in srgb, var(--surface-2) 75%, transparent); }
    .structured-field { display:grid; gap:5px; min-width:0; }
    .structured-field > label { display:grid; gap:2px; color:var(--muted); font-size:9px; }
    .structured-field > label small { line-height:1.3; }
    .structured-editor textarea { width:100%; min-width:0; resize:vertical; }
    .structured-error { color:var(--error); line-height:1.35; }
    .is-disabled { opacity:.75; }
  `],
})
export class StructuredParameterEditorComponent {
  readonly definition = input.required<ParameterDefinition>()
  readonly value = input<unknown>()
  readonly capability = input<ParameterCapability | null>(null)
  readonly disabled = input(false)
  readonly state = input<ParameterEditorState>('editable')
  readonly valueChange = output<unknown>()
  readonly validChange = output<boolean>()
  readonly invalid = signal(false)
  readonly expertJson = signal(false)
  readonly currentValue = computed(() => this.value() === undefined ? this.definition().default : this.value())
  readonly jsonValue = computed(() => {
    const value = this.currentValue()
    return typeof value === 'string' ? value : JSON.stringify(value ?? null, null, 2)
  })
  readonly fields = computed<StructuredField[]>(() => {
    const properties = this.definition().value_schema?.properties
    if (!properties || !isRecord(this.currentValue())) return []
    const value = this.currentValue() as Record<string, unknown>
    return Object.entries(properties).map(([id, schema]) => ({ id, definition: schemaDefinition(id, schema, value[id], this.definition().value_schema?.required_properties), value: value[id] }))
  })
  readonly arrayItems = computed<Array<StructuredField & { index: number }>>(() => {
    const itemSchema = this.definition().value_schema?.item
    const value = this.currentValue()
    if (!itemSchema || !Array.isArray(value)) return []
    return value.map((item, index) => ({ id: `${index}`, index, definition: schemaDefinition(`item_${index}`, itemSchema, item), value: item }))
  })

  isNested(definition: ParameterDefinition): boolean {
    return definition.type === 'object' || definition.type === 'shape' || definition.type === 'list' || definition.type === 'tuple'
      || ((definition.type === 'initializer' || definition.type === 'regularizer' || definition.type === 'constraint') && typeof definition.default === 'object' && definition.default !== null)
  }

  updateObjectField(id: string, value: unknown): void {
    const current = isRecord(this.currentValue()) ? this.currentValue() as Record<string, unknown> : {}
    this.commit({ ...current, [id]: value })
  }

  updateArrayItem(index: number, nextValue: unknown): void {
    const value = this.currentValue()
    const current = Array.isArray(value) ? [...value] : []
    current[index] = nextValue
    this.commit(current)
  }

  updateJson(text: string): void {
    try {
      const value: unknown = JSON.parse(text)
      this.commit(value)
    } catch {
      this.invalid.set(true)
      this.validChange.emit(false)
    }
  }

  private commit(value: unknown): void {
    const definition = this.definition()
    const parsed = Array.isArray(value) || isRecord(value)
      ? { valid: parameterValueMatchesDefinition(definition, value, this.capability() ?? undefined), value }
      : parseParameterValue(definition, value as ParameterDraftValue, this.capability() ?? undefined)
    this.invalid.set(!parsed.valid)
    this.validChange.emit(parsed.valid)
    if (parsed.valid) this.valueChange.emit(parsed.value)
  }
}

function schemaDefinition(id: string, schema: ParameterValueSchema, value: unknown, requiredProperties?: string[]): ParameterDefinition {
  return {
    id,
    display_name: id.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
    type: schema.type,
    default: value,
    required: requiredProperties?.includes(id) ?? false,
    constraints: {},
    options: [],
    description: '',
    nullable: Boolean(schema.nullable),
    value_schema: schema,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
