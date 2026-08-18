# Python Coding Standards

Last updated: 2026-08-05

Use Python 3.13.14, Pydantic v2, SQLAlchemy 2 typed mappings, built-in generics, explicit return types, and dependency-injected services. Keep route handlers thin. CPU-heavy model construction or training must not run directly in async handlers. Do not use arbitrary code evaluation. Ruff, strict Pyright, and pytest are required gates over server, scripts, and backend tests. Imports are explicit and top-level; nested functions, conditional/local imports, mutable process-global state, and files over 1000 lines are structural violations. Business rules belong in domain, validation, compilation, service, or repository modules.
