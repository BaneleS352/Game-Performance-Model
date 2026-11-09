# `dashboard/` Folder Guide

## Folder purpose

The dashboard is the browser-facing React/Vite application and its build
configuration. It renders published forecast data, provides workbook-sheet
previews, and drives the upload/run/status workflow through the local Python
API.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `index.html` | Browser entry HTML | Vite/React build | Mounts the React root. |
| `package.json` | Package metadata/scripts | Node, pnpm | Declares `dev` and `build` commands. |
| `package-lock.json` | npm lockfile | npm | Generated dependency lock information. |
| `pnpm-lock.yaml` | pnpm lockfile | pnpm | Preferred dependency lock information. |
| `pnpm-workspace.yaml` | pnpm workspace config | pnpm | Workspace configuration. |
| `.npmrc` | npm/pnpm settings | pnpm/npm | Local package-manager behavior. |
| `vite.config.js` | Dev/build orchestration | Vite, React plugin, Node child process | Configures proxy, dashboard data middleware, Python API child process, and output directory. |
| `src/main.jsx` | UI and interaction | React, browser Fetch API | Dashboard views, upload, polling, filters, sorting, and report links. |
| `src/styles.css` | UI styling | CSS | Responsive arcade-style layout, tables, states, and accessibility focus styles. |
| `scripts/clean-assets.cjs` | Build cleanup | Node `fs`, `path` | Removes only the dashboard compiled-assets subtree. |
| `scripts/fix-local-html.cjs` | Build post-processing | Node `fs`, `path` | Makes static HTML usable from local files. |

## Key processes

`pnpm dev` starts Vite. `vite.config.js` also starts
`python -m game_performance.server` and proxies API/report paths. The React app
first uses `window.DASHBOARD_DATA` when available, then fetches
`/api/dashboard-data`; the upload panel posts raw file bytes, starts a run, and
polls job status.

## Related documentation

- [User guide](../docs/user-guide.md)
- [API reference](../docs/api-reference.md)
- [Technology stack](../docs/technology-stack.md)
- [Scripts folder guide](scripts/scripts.md)
- [Frontend source guide](src/src.md)
