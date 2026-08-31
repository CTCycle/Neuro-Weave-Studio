# ADR 001: Frontend Framework

Last updated: 2026-08-31

Status: accepted; this decision supersedes the original React selection. Angular 22 with standalone components, strict TypeScript templates, zoneless change detection, signals for local/application state, and RxJS for asynchronous streams is the canonical frontend framework. The migration keeps the backend-neutral graph contract and existing drafting-board visual language while replacing React-specific ownership and routing with Angular-native boundaries. Angular’s dependency injection, route/component lifecycle, reactive forms, and signal model provide explicit ownership for the editor, runtime workflows, and navigation guards.
