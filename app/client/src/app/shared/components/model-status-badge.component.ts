import { Component, Input } from '@angular/core'
import { LucideCheckCircle2, LucideCircleAlert, LucideCircleHelp, LucideClock3 } from '@lucide/angular'

@Component({
  selector: 'nw-validation-status-badge',
  standalone: true,
  imports: [LucideCheckCircle2, LucideCircleAlert, LucideCircleHelp],
  template: `
    <span class="status-indicator" [class.valid]="status === 'valid'" [class.invalid]="status === 'invalid'">
      @if (status === 'valid') { <svg lucideCheckCircle2 [size]="14"></svg> }
      @else if (status === 'invalid') { <svg lucideCircleAlert [size]="14"></svg> }
      @else { <svg lucideCircleHelp [size]="14"></svg> }
      {{ status === 'valid' ? 'Ready' : status === 'invalid' ? 'Needs attention' : status || 'Not checked' }}
    </span>
  `,
})
export class ValidationStatusBadgeComponent {
  @Input() status = ''
}

@Component({
  selector: 'nw-compilation-status-badge',
  standalone: true,
  imports: [LucideCheckCircle2, LucideClock3],
  template: `
    <span class="status-indicator" [class.valid]="compiled">
      @if (compiled) { <svg lucideCheckCircle2 [size]="14"></svg>Compiled{{ target ? ' · ' + target : '' }} }
      @else { <svg lucideClock3 [size]="14"></svg>Not compiled }
    </span>
  `,
})
export class CompilationStatusBadgeComponent {
  @Input() compiled = false
  @Input() target = ''
}
