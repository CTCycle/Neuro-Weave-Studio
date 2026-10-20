// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { Component, inject, signal, type OnInit } from '@angular/core'
import { FormsModule } from '@angular/forms'
import { RouterLink } from '@angular/router'
import { LucideBox, LucideCpu } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import type { ApplicationSettings, DatabaseSettings, DatabaseStatus } from '../../domain/settings.types'

const DEFAULT_SETTINGS: ApplicationSettings = {
  app_name: 'NeuroWeave Studio',
  app_version: '1.0.0',
  cors_origins: 'http://127.0.0.1:5173,http://localhost:5173',
  log_level: 'INFO',
  log_file: 'data/runtime/logs/backend.log',
  export_root: 'data/runtime/exports',
  ml_job_root: 'data/runtime/ml-jobs',
  tensor_inspection_root: 'data/runtime/tensor-inspections',
  tensor_inspection_ttl_seconds: 600,
  ml_worker_timeout_seconds: 300,
  runtime_discovery_timeout_seconds: 60,
  ml_cancel_grace_seconds: 5,
  ml_terminate_grace_seconds: 5,
  ml_job_history_limit: 100,
  runtime_discovery_path: 'runtimes/cache/environment-discovery.json',
}

@Component({
  selector: 'nw-settings-page',
  standalone: true,
  imports: [FormsModule, RouterLink, LucideBox, LucideCpu],
  template: `
    <div class="settings-page">
      <header class="settings-header"><h1>Settings</h1></header>
      <div class="settings-layout">
        <nav class="settings-nav" aria-label="Settings sections">
          <a href="#application-general" [class.active]="activeSection() === 'application-general'" [attr.aria-current]="activeSection() === 'application-general' ? 'location' : null" (click)="activeSection.set('application-general')">Application<span>General</span></a>
          <a href="#database" [class.active]="activeSection() === 'database'" [attr.aria-current]="activeSection() === 'database' ? 'location' : null" (click)="activeSection.set('database')">Database<span>Connection</span></a>
          <a href="#related-management" [class.active]="activeSection() === 'related-management'" [attr.aria-current]="activeSection() === 'related-management' ? 'location' : null" (click)="activeSection.set('related-management')">Related management<span>Installation resources</span></a>
        </nav>

        <div class="settings-content">
          <section id="application-general" class="settings-section">
            <div class="section-heading"><h2>General</h2><p>Application preferences are stored in the currently selected database.</p></div>
            @if (settingsError()) {
              <div class="notice error" role="alert">{{ settingsError() }}</div>
            } @else if (settingsNotice()) {
              <div class="notice success" role="status">{{ settingsNotice() }}</div>
            }
            <form class="settings-form" (ngSubmit)="saveSettings()">
              <fieldset [disabled]="!settingsAvailable() || settingsBusy()">
                <div class="field-grid">
                  <label class="setting-field wide">Allowed browser origins
                    <input name="cors_origins" type="text" [ngModel]="settings().cors_origins" (ngModelChange)="updateSetting('cors_origins', $event)" required>
                    <small>Comma-separated origins allowed to call the API.</small>
                  </label>
                  <label class="setting-field">Log level
                    <select name="log_level" [ngModel]="settings().log_level" (ngModelChange)="updateSetting('log_level', $event)">
                      <option>DEBUG</option><option>INFO</option><option>WARNING</option><option>ERROR</option><option>CRITICAL</option>
                    </select>
                  </label>
                  <label class="setting-field">Worker timeout (seconds)
                    <input name="ml_worker_timeout_seconds" type="number" min="1" max="86400" [ngModel]="settings().ml_worker_timeout_seconds" (ngModelChange)="updateSetting('ml_worker_timeout_seconds', $event)">
                  </label>
                  <label class="setting-field">Runtime discovery timeout (seconds)
                    <input name="runtime_discovery_timeout_seconds" type="number" min="1" max="600" [ngModel]="settings().runtime_discovery_timeout_seconds" (ngModelChange)="updateSetting('runtime_discovery_timeout_seconds', $event)">
                  </label>
                  <label class="setting-field">Tensor inspection retention (seconds)
                    <input name="tensor_inspection_ttl_seconds" type="number" min="60" max="86400" [ngModel]="settings().tensor_inspection_ttl_seconds" (ngModelChange)="updateSetting('tensor_inspection_ttl_seconds', $event)">
                  </label>
                  <label class="setting-field">Cancel grace period (seconds)
                    <input name="ml_cancel_grace_seconds" type="number" min="1" max="300" [ngModel]="settings().ml_cancel_grace_seconds" (ngModelChange)="updateSetting('ml_cancel_grace_seconds', $event)">
                  </label>
                  <label class="setting-field">Terminate grace period (seconds)
                    <input name="ml_terminate_grace_seconds" type="number" min="1" max="300" [ngModel]="settings().ml_terminate_grace_seconds" (ngModelChange)="updateSetting('ml_terminate_grace_seconds', $event)">
                  </label>
                  <label class="setting-field">Job history limit
                    <input name="ml_job_history_limit" type="number" min="10" max="1000" [ngModel]="settings().ml_job_history_limit" (ngModelChange)="updateSetting('ml_job_history_limit', $event)">
                  </label>
                </div>
                <details class="advanced-settings">
                  <summary>Storage and log locations</summary>
                  <div class="field-grid path-grid">
                    <label class="setting-field wide">Log file<input name="log_file" type="text" [ngModel]="settings().log_file" (ngModelChange)="updateSetting('log_file', $event)"></label>
                    <label class="setting-field">Export folder<input name="export_root" type="text" [ngModel]="settings().export_root" (ngModelChange)="updateSetting('export_root', $event)"></label>
                    <label class="setting-field">ML job folder<input name="ml_job_root" type="text" [ngModel]="settings().ml_job_root" (ngModelChange)="updateSetting('ml_job_root', $event)"></label>
                    <label class="setting-field">Tensor inspection folder<input name="tensor_inspection_root" type="text" [ngModel]="settings().tensor_inspection_root" (ngModelChange)="updateSetting('tensor_inspection_root', $event)"></label>
                    <label class="setting-field">Runtime discovery cache<input name="runtime_discovery_path" type="text" [ngModel]="settings().runtime_discovery_path" (ngModelChange)="updateSetting('runtime_discovery_path', $event)"></label>
                  </div>
                </details>
              </fieldset>
              <div class="form-actions"><button class="primary" type="submit" [disabled]="!settingsAvailable() || settingsBusy()">{{ settingsBusy() ? 'Saving…' : 'Save application settings' }}</button></div>
              @if (!settingsAvailable() && !settingsError()) { <p class="field-help">Application settings will be available after connecting to a database.</p> }
            </form>
          </section>

          <section id="database" class="settings-section">
            <div class="section-heading"><div class="section-title-row"><div><span class="eyebrow">CONNECTION</span><h2>Database</h2></div>
              @if (databaseStatus(); as status) { <span class="status-chip" [class.is-online]="status.available">{{ status.available ? status.backend.toUpperCase() + ' · Connected' : 'Unavailable' }}</span> }
            </div><p>Choose where this application stores projects, datasets, and its settings.</p></div>
            @if (databaseStatus()?.sqlite_path) { <p class="active-database-path"><strong>Active file</strong> <code>{{ databaseStatus()?.sqlite_path }}</code></p> }
            @if (databaseError()) { <div class="notice error" role="alert">{{ databaseError() }}</div> }
            @if (databaseNotice()) { <div class="notice success" role="status">{{ databaseNotice() }}</div> }
            @if (databaseStatus()?.message && !databaseStatus()?.available) { <p class="database-message" role="status">{{ databaseStatus()?.message }}</p> }
            <form class="settings-form" (ngSubmit)="switchDatabase()">
              <div class="field-grid database-grid">
                <label class="setting-field">Backend
                  <select name="database_backend" [ngModel]="databaseSettings().backend" disabled><option value="sqlite">SQLite</option></select>
                  <small>SQLite is the database backend currently supported by this installation.</small>
                </label>
                <label class="setting-field">Connection timeout (seconds)
                  <input name="database_timeout" type="number" min="1" max="300" [ngModel]="databaseSettings().connect_timeout_seconds" (ngModelChange)="updateDatabaseSetting('connect_timeout_seconds', $event)" required>
                </label>
                <label class="setting-field wide">Database file path
                  <input name="sqlite_path" type="text" [ngModel]="databaseSettings().sqlite_path" (ngModelChange)="updateDatabaseSetting('sqlite_path', $event)" required>
                  <small>Each database is independent. Switching does not copy projects or settings.</small>
                </label>
              </div>
              <div class="form-actions database-actions">
                <button class="secondary" type="button" [disabled]="databaseBusy()" (click)="testDatabase()">{{ databaseBusy() ? 'Checking…' : 'Test connection' }}</button>
                <button class="primary" type="submit" [disabled]="databaseBusy() || !databaseSettings().sqlite_path.trim()">{{ switchingDatabase() ? 'Switching…' : 'Save and switch' }}</button>
              </div>
              <p class="field-help">Switching waits for active API requests to finish and is blocked while runtime jobs are active. The selected database’s own settings load as soon as it becomes active.</p>
            </form>
          </section>

          <section id="related-management" class="settings-section related-section">
            <div class="section-heading"><span class="eyebrow">RELATED MANAGEMENT</span><h2>Installation resources</h2><p>These separate management surfaces describe installation-wide capabilities.</p></div>
            <div class="management-links">
              <a class="management-row" routerLink="/nodes"><span class="management-icon"><svg lucideBox [size]="17"></svg></span><span><strong>Node Library</strong><small>Inspect built-in node contracts and manage reusable custom node definitions.</small></span><span class="management-meta">Authoring</span></a>
              <a class="management-row" routerLink="/environment"><span class="management-icon"><svg lucideCpu [size]="17"></svg></span><span><strong>Runtime Environment</strong><small>Inspect installation-wide execution backends, devices, and runtime capabilities.</small></span><span class="management-meta">Runtime</span></a>
            </div>
          </section>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host{display:block}.settings-page{display:grid;gap:24px;max-width:1180px}.settings-header h1{margin:0}.settings-layout{display:grid;grid-template-columns:190px minmax(0,760px);gap:48px;align-items:start}.settings-nav{position:sticky;top:18px;display:grid;gap:2px;border-left:1px solid var(--border)}.settings-nav a{display:grid;gap:4px;padding:10px 12px;border-left:2px solid transparent;color:var(--muted);text-decoration:none;font-size:10px}.settings-nav a:hover,.settings-nav a.active{border-left-color:var(--cyan);color:var(--text);background:rgba(101,201,212,.06)}.settings-nav a span{color:var(--muted);font-size:8px}.settings-content{display:grid;gap:34px;min-width:0}.settings-section{display:grid;gap:16px;scroll-margin-top:20px}.section-heading{display:grid;gap:5px;padding-bottom:14px;border-bottom:1px solid var(--border)}.section-heading h2{margin:0;font-size:16px}.section-heading p{margin:0;color:var(--muted);font-size:10px;line-height:1.45}.section-title-row{display:flex;align-items:center;justify-content:space-between;gap:12px}.section-title-row>div{display:grid;gap:4px}.eyebrow{color:var(--cyan);font:500 8px 'IBM Plex Mono',Consolas,monospace;letter-spacing:.08em}.status-chip{padding:4px 8px;border:1px solid var(--border);border-radius:999px;color:var(--error);font-size:9px}.status-chip.is-online{border-color:rgba(101,201,212,.35);color:var(--cyan)}.active-database-path{display:grid;gap:5px;margin:0;color:var(--muted);font-size:9px}.active-database-path code{overflow-wrap:anywhere;color:var(--text);font-size:9px}.active-database-path strong{color:var(--muted);font-size:8px;text-transform:uppercase}.settings-form{display:grid;gap:16px}.settings-form fieldset{min-width:0;margin:0;padding:0;border:0}.field-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 16px}.setting-field{display:grid;gap:6px;min-width:0;color:var(--muted);font-size:9px}.setting-field.wide{grid-column:1/-1}.setting-field input,.setting-field select{width:100%;min-width:0;height:34px;padding:0 9px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text);font:400 11px 'IBM Plex Mono',Consolas,monospace}.setting-field select{font-family:inherit}.setting-field input:focus,.setting-field select:focus{outline:1px solid var(--cyan);border-color:var(--cyan)}.setting-field small,.field-help{color:var(--muted);font-size:9px;line-height:1.5}.form-actions{display:flex;justify-content:flex-end;gap:8px}.form-actions button{font-size:10px}.advanced-settings{padding-top:12px;border-top:1px solid var(--border)}.advanced-settings summary{color:var(--muted);font-size:10px;cursor:pointer}.path-grid{padding-top:14px}.database-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.database-actions{justify-content:space-between}.field-help{margin:0}.database-message{margin:0;padding:9px 11px;border:1px solid rgba(239,106,114,.25);border-radius:5px;color:var(--muted);font-size:10px;line-height:1.5}.notice{padding:9px 11px;border:1px solid var(--border);border-radius:5px;font-size:10px;line-height:1.5}.notice.error{border-color:rgba(239,106,114,.4);color:var(--error)}.notice.success{border-color:rgba(101,201,212,.35);color:var(--cyan)}.related-section{padding-top:2px}.management-links{display:grid}.management-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:12px;padding:13px 0;border-block-end:1px solid var(--border);color:var(--text);text-decoration:none}.management-row:first-child{border-top:1px solid var(--border)}.management-row:hover .management-icon{border-color:var(--cyan);color:var(--cyan)}.management-row>span:nth-child(2){display:grid;gap:4px}.management-row strong{font-size:11px}.management-row small{color:var(--muted);font-size:9px;line-height:1.45}.management-icon{display:grid;place-items:center;width:34px;height:34px;border:1px solid var(--border);border-radius:5px;color:var(--muted)}.management-meta{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace;text-transform:uppercase}
    @media(max-width:800px){.settings-layout{grid-template-columns:1fr;gap:22px}.settings-nav{position:static;display:flex;overflow:auto;border-left:0;border-bottom:1px solid var(--border)}.settings-nav a{min-width:max-content;border-left:0;border-bottom:2px solid transparent}.settings-nav a:hover,.settings-nav a.active{border-left:0;border-bottom-color:var(--cyan)}}@media(max-width:520px){.settings-page{gap:18px}.settings-content{gap:25px}.field-grid{grid-template-columns:1fr}.setting-field.wide{grid-column:auto}.database-actions{flex-wrap:wrap;justify-content:flex-end}.management-row{grid-template-columns:auto minmax(0,1fr)}.management-meta{grid-column:2}}
  `],
})
export class SettingsPageComponent implements OnInit {
  private readonly api = inject(ApiService)

  readonly activeSection = signal('application-general')
  readonly settings = signal<ApplicationSettings>(structuredClone(DEFAULT_SETTINGS))
  readonly databaseSettings = signal<DatabaseSettings>({
    backend: 'sqlite',
    schema_version: 1,
    sqlite_path: 'data/database.db',
    connect_timeout_seconds: 30,
  })
  readonly databaseStatus = signal<DatabaseStatus | null>(null)
  readonly settingsAvailable = signal(false)
  readonly settingsBusy = signal(false)
  readonly databaseBusy = signal(false)
  readonly switchingDatabase = signal(false)
  readonly settingsNotice = signal('')
  readonly settingsError = signal('')
  readonly databaseNotice = signal('')
  readonly databaseError = signal('')

  ngOnInit(): void {
    void this.loadSettings()
    void this.loadDatabaseStatus()
  }

  updateSetting<K extends keyof ApplicationSettings>(field: K, value: ApplicationSettings[K] | string | number): void {
    this.settings.update((current) => ({ ...current, [field]: this.coerceSettingValue(field, value) }))
    this.settingsNotice.set('')
  }

  updateDatabaseSetting(field: 'sqlite_path' | 'connect_timeout_seconds', value: string | number): void {
    this.databaseSettings.update((current) => ({
      ...current,
      [field]: field === 'connect_timeout_seconds' ? Number(value) : String(value),
    }))
    this.databaseNotice.set('')
  }

  async saveSettings(): Promise<void> {
    this.settingsBusy.set(true)
    this.settingsError.set('')
    this.settingsNotice.set('')
    try {
      const saved = await this.api.saveApplicationSettings(this.settings())
      this.settings.set(saved)
      this.settingsNotice.set('Application settings saved to the active database.')
    } catch (error) {
      this.settingsError.set(this.message(error))
    } finally {
      this.settingsBusy.set(false)
    }
  }

  async testDatabase(): Promise<void> {
    this.databaseBusy.set(true)
    this.databaseError.set('')
    this.databaseNotice.set('')
    try {
      const result = await this.api.testDatabase(this.databaseSettings())
      this.databaseNotice.set(result.message)
    } catch (error) {
      this.databaseError.set(this.message(error))
    } finally {
      this.databaseBusy.set(false)
    }
  }

  async switchDatabase(): Promise<void> {
    this.databaseBusy.set(true)
    this.switchingDatabase.set(true)
    this.databaseError.set('')
    this.databaseNotice.set('')
    try {
      const result = await this.api.switchDatabase(this.databaseSettings())
      this.settings.set(result.settings)
      this.settingsAvailable.set(true)
      this.databaseStatus.set(result.status)
      this.databaseSettings.update((current) => ({
        ...current,
        sqlite_path: result.status.sqlite_path ?? current.sqlite_path,
        connect_timeout_seconds: result.status.connect_timeout_seconds ?? current.connect_timeout_seconds,
      }))
      this.databaseNotice.set('Database switched. Application settings now reflect the selected database.')
      this.settingsNotice.set('Settings loaded from the selected database.')
    } catch (error) {
      this.databaseError.set(this.message(error))
    } finally {
      this.switchingDatabase.set(false)
      this.databaseBusy.set(false)
      void this.loadDatabaseStatus()
    }
  }

  private async loadSettings(): Promise<void> {
    try {
      this.settings.set(await this.api.getApplicationSettings())
      this.settingsAvailable.set(true)
    } catch (error) {
      this.settingsAvailable.set(false)
      this.settingsError.set(this.message(error))
    }
  }

  private async loadDatabaseStatus(): Promise<void> {
    try {
      const status = await this.api.getDatabaseStatus()
      this.databaseStatus.set(status)
      if (status.sqlite_path) this.databaseSettings.update((current) => ({
        ...current,
        sqlite_path: status.sqlite_path ?? current.sqlite_path,
        connect_timeout_seconds: status.connect_timeout_seconds ?? current.connect_timeout_seconds,
      }))
    } catch (error) {
      this.databaseError.set(this.message(error))
    }
  }

  private coerceSettingValue<K extends keyof ApplicationSettings>(field: K, value: ApplicationSettings[K] | string | number): ApplicationSettings[K] {
    const current = this.settings()[field]
    return (typeof current === 'number' ? Number(value) : String(value)) as ApplicationSettings[K]
  }

  private message(error: unknown): string {
    return error instanceof Error ? error.message : String(error)
  }
}
