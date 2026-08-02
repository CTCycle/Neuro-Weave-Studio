# Python Coding Standards

Last updated: 2026-07-30

Use Python 3.12 or newer, Pydantic v2, SQLAlchemy 2 typed mappings, built-in generics, explicit return types, and dependency-injected services. Keep route handlers thin. CPU-heavy model construction or training must not run directly in async handlers. Do not use arbitrary code evaluation. Ruff and pytest are required gates. Business rules belong in domain, validation, compilation, service, or repository modules.
