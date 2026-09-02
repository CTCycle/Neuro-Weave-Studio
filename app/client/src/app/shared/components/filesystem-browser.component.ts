import { Component, EventEmitter, Input, OnChanges, Output, inject, signal } from '@angular/core'
import { FormControl, ReactiveFormsModule } from '@angular/forms'
import { LucideChevronLeft, LucideFile, LucideFolder, LucideHardDrive, LucideRefreshCw, LucideX } from '@lucide/angular'
import { ApiService } from '../../core/api/api.service'
import type { FilesystemEntry } from '../../domain/filesystem.types'

@Component({
  selector: 'nw-filesystem-browser',
  standalone: true,
  imports: [ReactiveFormsModule, LucideChevronLeft, LucideFile, LucideFolder, LucideHardDrive, LucideRefreshCw, LucideX],
  template: `
    @if (open) {
      <div class="browser-backdrop" (click)="close()">
        <section class="browser-dialog panel" role="dialog" aria-modal="true" aria-labelledby="filesystem-browser-title" (click)="$event.stopPropagation()">
          <header>
            <div><span class="eyebrow">LOCAL FILESYSTEM</span><h2 id="filesystem-browser-title">{{ title }}</h2></div>
            <button class="icon-button" type="button" aria-label="Close filesystem explorer" (click)="close()"><svg lucideX [size]="16"></svg></button>
          </header>
          <div class="browser-path-row">
            <button class="secondary compact" type="button" [disabled]="busy()" (click)="goToParent()" aria-label="Go to parent folder"><svg lucideChevronLeft [size]="15"></svg></button>
            <input [formControl]="pathControl" aria-label="Filesystem path" placeholder="Select a drive or enter a folder path" (keydown.enter)="browse(pathControl.value)" />
            <button class="secondary compact" type="button" [disabled]="busy()" (click)="browse(pathControl.value)"><svg lucideRefreshCw [size]="15"></svg>Browse</button>
          </div>
          @if (error()) { <p class="browser-error" role="alert">{{ error() }}</p> }
          <div class="browser-body" [attr.aria-busy]="busy()">
            @if (busy()) {
              <div class="browser-empty">Reading folder…</div>
            } @else if (!items().length) {
              <div class="browser-empty">No matching {{ selectionMode === 'directory' ? 'folders' : 'files or folders' }} in this location.</div>
            } @else {
              @for (item of items(); track item.path) {
                <button class="browser-item" type="button" [class.selected]="selectedFile()?.path === item.path" (click)="activate(item)">
                  <span class="browser-item-icon">@if (item.is_dir) { <svg lucideFolder [size]="17"></svg> } @else { <svg lucideFile [size]="17"></svg> }</span>
                  <span class="browser-item-copy"><strong>{{ item.name }}</strong><small>{{ item.is_dir ? 'Folder' : fileMeta(item) }}</small></span>
                  @if (item.is_dir) { <span class="browser-item-open">Open</span> }
                </button>
              }
            }
          </div>
          @if (truncated()) { <p class="browser-warning">Only the first 500 matching entries are shown.</p> }
          <footer>
            <div class="browser-location"><svg lucideHardDrive [size]="14"></svg><span>{{ currentPath() || 'Computer' }}</span></div>
            <div class="browser-actions">
              <button class="secondary" type="button" (click)="close()">Cancel</button>
              <button class="primary" type="button" [disabled]="!canSelect()" (click)="confirmSelection()">{{ selectionMode === 'directory' ? 'Use this folder' : 'Use selected file' }}</button>
            </div>
          </footer>
        </section>
      </div>
    }
  `,
  styles: [`
    .browser-backdrop{position:fixed;inset:0;z-index:80;display:grid;place-items:center;padding:22px;background:rgba(3,8,12,.76);backdrop-filter:blur(4px)}
    .browser-dialog{width:min(900px,calc(100vw - 44px));height:min(680px,calc(100vh - 44px));display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto auto;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}
    header,footer{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 18px} header{border-bottom:1px solid var(--border)} footer{border-top:1px solid var(--border)} h2{margin:4px 0 0}
    .browser-path-row{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;padding:12px 18px;border-bottom:1px solid var(--border);background:#0d151c}.browser-path-row input{min-width:0;height:34px;border:1px solid var(--border);border-radius:6px;background:#091117;color:var(--text);padding:0 10px;font:500 11px 'IBM Plex Mono',Consolas,monospace}.compact{min-height:34px!important;height:34px;display:inline-flex;align-items:center;gap:6px}
    .browser-error,.browser-warning{margin:0;padding:9px 18px;font-size:10px}.browser-error{color:var(--error);background:rgba(239,106,114,.07)}.browser-warning{color:var(--warning);border-top:1px solid var(--border)}
    .browser-body{min-height:0;overflow:auto;padding:8px 10px;scrollbar-gutter:stable}.browser-item{width:100%;min-height:46px;display:grid;grid-template-columns:28px minmax(0,1fr) auto;align-items:center;gap:8px;padding:7px 9px;border:1px solid transparent;border-radius:5px;background:transparent;color:var(--text);text-align:left;cursor:pointer}.browser-item:hover,.browser-item.selected{border-color:var(--border-strong);background:var(--surface-2)}.browser-item.selected{box-shadow:inset 2px 0 0 var(--cyan)}.browser-item-icon{color:var(--cyan)}.browser-item-copy{min-width:0;display:grid;gap:2px}.browser-item-copy strong{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px}.browser-item-copy small,.browser-item-open{color:var(--muted);font:500 8px 'IBM Plex Mono',Consolas,monospace}.browser-item-open{color:var(--cyan)}.browser-empty{display:grid;place-items:center;min-height:180px;color:var(--muted);font-size:11px}
    .browser-location{min-width:0;display:flex;align-items:center;gap:7px;color:var(--muted);font:500 9px 'IBM Plex Mono',Consolas,monospace}.browser-location span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.browser-actions{display:flex;gap:8px;flex:0 0 auto}
    @media(max-width:700px){.browser-backdrop{padding:8px}.browser-dialog{width:100%;height:calc(100vh - 16px)}.browser-path-row{grid-template-columns:auto minmax(0,1fr)}.browser-path-row button:last-child{grid-column:1/-1}.browser-location{display:none}}
  `],
})
export class FilesystemBrowserComponent implements OnChanges {
  @Input() open = false
  @Input() title = 'Select path'
  @Input() selectionMode: 'file' | 'directory' = 'directory'
  @Input() extensions: string[] = []
  @Output() readonly closed = new EventEmitter<void>()
  @Output() readonly selected = new EventEmitter<string>()

  private readonly api = inject(ApiService)
  readonly pathControl = new FormControl('', { nonNullable: true })
  readonly items = signal<FilesystemEntry[]>([])
  readonly currentPath = signal('')
  readonly parentPath = signal<string | null>(null)
  readonly truncated = signal(false)
  readonly busy = signal(false)
  readonly error = signal<string | null>(null)
  readonly selectedFile = signal<FilesystemEntry | null>(null)

  ngOnChanges(): void {
    if (this.open) void this.browse(this.currentPath())
  }

  async browse(path: string): Promise<void> {
    this.busy.set(true)
    this.error.set(null)
    this.selectedFile.set(null)
    try {
      const response = await this.api.browseFilesystem(path, this.selectionMode === 'file', this.extensions)
      this.items.set(response.items)
      this.currentPath.set(response.current_path)
      this.parentPath.set(response.parent_path)
      this.pathControl.setValue(response.current_path, { emitEvent: false })
      this.truncated.set(response.truncated)
    } catch (reason) {
      this.error.set(reason instanceof Error ? reason.message : String(reason))
    } finally {
      this.busy.set(false)
    }
  }

  activate(item: FilesystemEntry): void {
    if (item.is_dir) {
      void this.browse(item.path)
      return
    }
    if (this.selectionMode === 'file') this.selectedFile.set(item)
  }

  goToParent(): void {
    void this.browse(this.parentPath() ?? '')
  }

  canSelect(): boolean {
    return this.selectionMode === 'directory' ? !!this.currentPath() : !!this.selectedFile()
  }

  confirmSelection(): void {
    const path = this.selectionMode === 'directory' ? this.currentPath() : this.selectedFile()?.path
    if (path) this.selected.emit(path)
  }

  close(): void { this.closed.emit() }

  fileMeta(item: FilesystemEntry): string {
    const size = item.size_bytes == null ? '' : ` · ${this.formatBytes(item.size_bytes)}`
    return `${item.suffix || 'file'}${size}`
  }

  private formatBytes(value: number): string {
    if (value < 1024) return `${value} B`
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
    if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`
    return `${(value / (1024 * 1024 * 1024)).toFixed(1)} GB`
  }
}
