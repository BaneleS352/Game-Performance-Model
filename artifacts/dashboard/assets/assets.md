# `artifacts/dashboard/assets/` Folder Guide

## Folder purpose

Generated Vite JavaScript and CSS assets for the static dashboard.

## File inventory

| Pattern | Purpose | Producer |
|---|---|---|
| `*.js` | Compiled React bundle | Vite |
| `*.css` | Compiled styles | Vite |
| `assets.md` | Folder guide | Markdown |

Assets are disposable build output. `dashboard/scripts/clean-assets.cjs` removes
only this directory before a rebuild.
