# TypeScript Coding Standards

Last updated: 2026-08-31

Use strict TypeScript, standalone Angular components, signal-backed state, typed API boundaries, and editor-domain modules. Avoid `any` in domain contracts and keep backend field names stable at API boundaries. Canvas behavior belongs in store actions and the Foblex adapter, not ad hoc template closures. Use reactive forms for editable parameter surfaces, preserve keyboard accessibility and reduced-motion behavior, and keep components lazy-loadable where routes permit. ESLint, Vitest, the Angular production build, and Playwright E2E are required frontend gates.
