# `artifacts/dashboard/` Folder Guide

## Folder purpose

Published dashboard data and static build output consumed by the browser or
opened locally after a successful run.

## File inventory

| Pattern | Purpose | Producer |
|---|---|---|
| `dashboard_data.json` | JSON dashboard contract | `game_performance.dashboard.export_dashboard` |
| `dashboard_data.js` | Browser global fallback | `game_performance.dashboard.export_dashboard` |
| `index.html` | Static dashboard shell | Vite build |
| `assets/` | Compiled CSS/JS | Vite build |
| `dashboard.md` | Folder guide | Markdown |

## Key processes

The upload pipeline stages dashboard data per run and atomically publishes the
two data files here. Vite writes the static shell/assets here during `pnpm
build`. Do not edit generated files by hand.
