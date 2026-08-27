import { CheckCircle2, CircleAlert, CircleHelp, Clock3 } from 'lucide-react'

export function ValidationStatusBadge({ status }: { status: string }) {
  const valid = status === 'valid'
  const invalid = status === 'invalid'
  return <span className={`status-indicator ${valid ? 'valid' : invalid ? 'invalid' : ''}`}>{valid ? <CheckCircle2 size={14} /> : invalid ? <CircleAlert size={14} /> : <CircleHelp size={14} />}{valid ? 'Ready' : invalid ? 'Needs attention' : status || 'Not checked'}</span>
}

export function CompilationStatusBadge({ compiled, target }: { compiled: boolean; target?: string }) {
  return <span className={`status-indicator ${compiled ? 'valid' : ''}`}>{compiled ? <CheckCircle2 size={14} /> : <Clock3 size={14} />}{compiled ? `Compiled${target ? ` · ${target}` : ''}` : 'Not compiled'}</span>
}
