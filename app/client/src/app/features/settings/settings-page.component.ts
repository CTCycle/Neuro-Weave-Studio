import { Component } from '@angular/core'
import { RouterLink } from '@angular/router'
import { LucideBox, LucideCpu } from '@lucide/angular'

@Component({
  selector: 'nw-settings-page',
  standalone: true,
  imports: [RouterLink, LucideBox, LucideCpu],
  template: `
    <div class="settings-page">
      <header class="catalog-page-header">
        <div><span class="eyebrow page-eyebrow">APPLICATION</span><h1>Settings</h1><p>Manage installation-wide NeuroWeave configuration and shared authoring resources.</p></div>
      </header>

      <section class="settings-list panel">
        <a class="settings-row" routerLink="/nodes">
          <span class="settings-icon"><svg lucideBox [size]="17"></svg></span>
          <span class="settings-copy"><strong>Node library</strong><span>Inspect built-in node contracts and manage reusable custom node definitions.</span></span>
          <span class="settings-meta">Authoring</span>
        </a>
        <a class="settings-row" routerLink="/environment">
          <span class="settings-icon"><svg lucideCpu [size]="17"></svg></span>
          <span class="settings-copy"><strong>Runtime environment</strong><span>Configure and inspect installation-wide execution backends, devices, and runtime capabilities.</span></span>
          <span class="settings-meta">Runtime</span>
        </a>
      </section>

      <section class="settings-note">
        <span class="eyebrow">SCOPE</span>
        <p>Global settings apply to this local NeuroWeave installation. Model-specific compilation and graph settings remain inside each model workspace.</p>
      </section>
    </div>
  `,
  styles: [`
    .settings-page{display:grid;gap:18px;max-width:980px}.settings-list{overflow:hidden}.settings-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:14px;padding:16px 18px;border-bottom:1px solid var(--border);color:var(--text);text-decoration:none}.settings-row:last-child{border-bottom:0}.settings-row:hover{background:var(--surface-2)}.settings-icon{width:34px;height:34px;display:grid;place-items:center;border:1px solid var(--border);border-radius:6px;background:var(--surface-2);color:var(--cyan)}.settings-copy{display:grid;gap:4px}.settings-copy strong{font-size:11px}.settings-copy span{color:var(--muted);font-size:9px;line-height:1.45}.settings-meta{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}.settings-note{display:grid;gap:6px;padding:2px 2px 0}.settings-note p{margin:0;color:var(--muted);font-size:10px;line-height:1.55}@media(max-width:650px){.settings-row{grid-template-columns:auto minmax(0,1fr)}.settings-meta{grid-column:2}}
  `],
})
export class SettingsPageComponent {}
