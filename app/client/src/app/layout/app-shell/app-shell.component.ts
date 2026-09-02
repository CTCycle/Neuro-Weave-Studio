import { afterNextRender, Component, inject } from '@angular/core'
import { toSignal } from '@angular/core/rxjs-interop'
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router'
import { LucideActivity, LucideBox, LucideBrainCircuit, LucideCpu, LucideFolderOpen, LucideHammer, LucideX } from '@lucide/angular'
import { filter, map, startWith } from 'rxjs'
import { AppResourcesService } from '../../core/state/app-resources.service'

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
    LucideX,
  ],
  template: `
    <div class="app-shell" [class.builder-shell]="isBuilder()">
      <header class="app-header">
        <a class="brand-mark" routerLink="/models" aria-label="NeuroWeave models">
          <span class="brand-icon"><svg lucideBrainCircuit [size]="18"></svg></span>
          <span><strong>NeuroWeave</strong><small>MODEL STUDIO</small></span>
        </a>
        <nav class="primary-nav" aria-label="Primary navigation">
          <a routerLink="/models" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: false }"><svg lucideBox [size]="16"></svg>Models</a>
          <a routerLink="/builder" routerLinkActive="active"><svg lucideHammer [size]="16"></svg>Model Builder</a>
          <a routerLink="/datasets" routerLinkActive="active"><svg lucideFolderOpen [size]="16"></svg>Datasets</a>
          <a routerLink="/training" routerLinkActive="active"><svg lucideBrainCircuit [size]="16"></svg>Training</a>
          <a routerLink="/environment" routerLinkActive="active"><svg lucideCpu [size]="16"></svg>Environment</a>
          <a routerLink="/metrics" routerLinkActive="active"><svg lucideActivity [size]="16"></svg>Metrics</a>
        </nav>
      </header>
      <main class="app-content" [class.builder-content]="isBuilder()"><router-outlet /></main>
      @if (resources.error(); as error) {
        <button class="notice app-notice" (click)="resources.clearError()" aria-label="Dismiss application notice">
          <span>{{ error }}</span><svg lucideX [size]="14"></svg>
        </button>
      }
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

  constructor() {
    afterNextRender(() => performance.mark('nw:shell:rendered'))
  }
}
