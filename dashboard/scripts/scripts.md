# `dashboard/scripts/` Folder Guide

## Folder purpose

Build-time Node scripts used by `pnpm build`.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `clean-assets.cjs` | Remove stale compiled assets | Node `fs`, `path` | Validates the target is inside `artifacts/dashboard` before recursive removal. |
| `fix-local-html.cjs` | Make static build locally openable | Node `fs`, `path` | Rewrites the generated module script and CSS cross-origin attribute. |
| `scripts.md` | Folder guide | Markdown | Describes the scripts and build order. |

## Execution order

`pnpm build` runs `clean-assets.cjs`, then `vite build`, then
`fix-local-html.cjs`. The clean script is intentionally scoped to the generated
assets directory; the post-processor assumes Vite has already emitted
`artifacts/dashboard/index.html`.

## Related documentation

- [Clean assets](clean-assets.md)
- [Fix local HTML](fix-local-html.md)
- [Dashboard folder](../dashboard.md)
