import { BrainCircuit, Box, Cpu, Hammer, X } from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useAppResources } from '../state/useAppResources'

export function AppShell() {
  const { error, clearError } = useAppResources()
  const location = useLocation()
  const isBuilder = location.pathname.startsWith('/builder')
  return (
    <div className="app-shell">
      <header className="app-header">
        <NavLink className="brand-mark" to="/models" aria-label="NeuroWeave models">
          <span className="brand-icon"><BrainCircuit size={18} /></span>
          <span><strong>NeuroWeave</strong><small>MODEL STUDIO</small></span>
        </NavLink>
        <nav className="primary-nav" aria-label="Primary navigation">
          <NavLink to="/models" className={({ isActive }) => isActive ? 'active' : ''}><Box size={16} />Models</NavLink>
          <NavLink to="/builder" className={({ isActive }) => isActive ? 'active' : ''}><Hammer size={16} />Model Builder</NavLink>
          <NavLink to="/environment" className={({ isActive }) => isActive ? 'active' : ''}><Cpu size={16} />Environment</NavLink>
        </nav>
      </header>
      <main className={`app-content ${isBuilder ? 'builder-content' : ''}`}><Outlet /></main>
      {error && <button className="notice app-notice" onClick={clearError} aria-label="Dismiss application notice"><span>{error}</span><X size={14} /></button>}
    </div>
  )
}
