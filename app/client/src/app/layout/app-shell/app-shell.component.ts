import { afterNextRender, Component, inject } from '@angular/core'
import { toSignal } from '@angular/core/rxjs-interop'
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router'
import { LucideActivity, LucideBox, LucideBrainCircuit, LucideCpu, LucideFolderOpen, LucideHammer, LucideSettings, LucideX } from '@lucide/angular'
import { filter, map, startWith } from 'rxjs'
import { AppResourcesService } from '../../core/state/app-resources.service'
import { TensorInspectorModalComponent } from '../../features/tensor-inspection/tensor-inspector-modal.component'

@Component({
  selector: 'nw-app-shell',
  standalone: true,
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    LucideActivity,
    LucideBox,
    LucideBrainCircuit,
    LucideCpu,
    LucideFolderOpen,
    LucideHammer,
    LucideSettings,
    LucideX,
    TensorInspectorModalComponent,
  ],
  template: `
    <div class="app-shell" [class.builder-shell]="isBuilder()">
      <header class="app-header">
        <a class="brand-mark" routerLink="/models" aria-label="NeuroWeave models">
          <span class="brand-icon"><svg lucideBrainCircuit [size]="18"></svg></span>
          <span><strong>NeuroWeave</strong><small>MODEL STUDIO</small></span>
        </a>
        <nav class="primary-nav" aria-label="Primary navigation">
          <a routerLink="/models" routerLinkActive="active" ariaCurrentWhenActive="page" [routerLinkActiveOptions]="{ exact: false }"><svg lucideBox [size]="16"></svg>Models</a>
          <a routerLink="/builder" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideHammer [size]="16"></svg>Model Builder</a>
          <a routerLink="/nodes" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideBox [size]="16"></svg>Nodes</a>
          <a routerLink="/datasets" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideFolderOpen [size]="16"></svg>Datasets</a>
          <a routerLink="/training" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideBrainCircuit [size]="16"></svg>Training</a>
          <a routerLink="/metrics" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideActivity [size]="16"></svg>Metrics</a>
          <a routerLink="/environment" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideCpu [size]="16"></svg>Environment</a>
          <a routerLink="/settings" routerLinkActive="active" ariaCurrentWhenActive="page"><svg lucideSettings [size]="16"></svg>Settings</a>
        </nav>
      </header>
      <main class="app-content" [class.builder-content]="isBuilder()" [class.nodes-content]="isNodes()"><router-outlet /></main>
      @if (resources.error(); as error) {
        <button class="notice app-notice" (click)="resources.clearError()" aria-label="Dismiss application notice">
          <span>{{ error }}</span><svg lucideX [size]="14"></svg>
        </button>
      }
      <nw-tensor-inspector-modal />
    </div>
  `,
  styles: [`
    .app-header { min-width: 0; }
    .primary-nav { min-width: 0; overflow-x: auto; overflow-y: hidden; scrollbar-width: none; }
    .primary-nav::-webkit-scrollbar { display: none; }
    .primary-nav a { flex: 0 0 auto; }
  `],
})
export class AppShellComponent {
  readonly resources = inject(AppResourcesService)
  private readonly router = inject(Router)
  readonly isBuilder = toSignal(this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), map((event) => event.urlAfterRedirects.startsWith('/builder')), startWith(this.router.url.startsWith('/builder'))), { initialValue: this.router.url.startsWith('/builder') })
  readonly isNodes = toSignal(this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), map((event) => event.urlAfterRedirects.startsWith('/nodes')), startWith(this.router.url.startsWith('/nodes'))), { initialValue: this.router.url.startsWith('/nodes') })

  constructor() {
    afterNextRender(() => performance.mark('nw:shell:rendered'))
  }
}
