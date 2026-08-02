# Installation

Last updated: 2026-08-02

Install Python 3.12 or newer, Node.js 20 or newer, and `uv`. Run `uv sync --extra test` in `app/server` and `npm install` in `frontend`. Add `--extra keras` only when local TensorFlow-backed binary export is required because it is a large optional dependency. Copy `.env.example` to `.env` only when overriding defaults.
