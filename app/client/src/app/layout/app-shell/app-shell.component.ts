// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

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
          <a #modelsNav routerLink="/models" routerLinkActive="active" ariaCurrentWhenActive="page" [routerLinkActiveOptions]="{ exact: false }" (isActiveChange)="revealActiveLink($event, modelsNav)"><svg lucideBox [size]="16"></svg>Models</a>
          <a #builderNav routerLink="/builder" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, builderNav)"><svg lucideHammer [size]="16"></svg>Model Builder</a>
          <a #nodesNav routerLink="/nodes" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, nodesNav)"><svg lucideBox [size]="16"></svg>Nodes</a>
          <a #datasetsNav routerLink="/datasets" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, datasetsNav)"><svg lucideFolderOpen [size]="16"></svg>Datasets</a>
          <a #trainingNav routerLink="/training" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, trainingNav)"><svg lucideBrainCircuit [size]="16"></svg>Training</a>
          <a #metricsNav routerLink="/metrics" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, metricsNav)"><svg lucideActivity [size]="16"></svg>Metrics</a>
          <a #environmentNav routerLink="/environment" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, environmentNav)"><svg lucideCpu [size]="16"></svg>Environment</a>
          <a #settingsNav routerLink="/settings" routerLinkActive="active" ariaCurrentWhenActive="page" (isActiveChange)="revealActiveLink($event, settingsNav)"><svg lucideSettings [size]="16"></svg>Settings</a>
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
    .primary-nav { min-width: 0; overflow-x: auto; overflow-y: hidden; overscroll-behavior-x: contain; scrollbar-width: thin; scrollbar-color: var(--scrollbar-thumb) transparent; }
    .primary-nav::-webkit-scrollbar { height: 4px; }
    .primary-nav::-webkit-scrollbar-track { background: transparent; }
    .primary-nav a { flex: 0 0 auto; }
  `],
})
export class AppShellComponent {
  readonly resources = inject(AppResourcesService)
  private readonly router = inject(Router)
  readonly isBuilder = toSignal(this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), map((event) => event.urlAfterRedirects.startsWith('/builder')), startWith(this.router.url.startsWith('/builder'))), { initialValue: this.router.url.startsWith('/builder') })
  readonly isNodes = toSignal(this.router.events.pipe(filter((event): event is NavigationEnd => event instanceof NavigationEnd), map((event) => event.urlAfterRedirects.startsWith('/nodes')), startWith(this.router.url.startsWith('/nodes'))), { initialValue: this.router.url.startsWith('/nodes') })

  revealActiveLink(isActive: boolean, link: HTMLElement): void {
    if (isActive) requestAnimationFrame(() => link.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'auto' }))
  }

  constructor() {
    afterNextRender(() => performance.mark('nw:shell:rendered'))
  }
}
