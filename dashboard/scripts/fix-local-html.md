# `fix-local-html.cjs`

## Purpose and location

Build post-processor at `dashboard/scripts/fix-local-html.cjs`. It adjusts the
Vite-generated `artifacts/dashboard/index.html` so the compiled dashboard can
be opened as a local static file.

## Inputs

No command-line arguments or environment variables. Requires Vite to have
already created `artifacts/dashboard/index.html`.

## Outputs

The same HTML file is rewritten with the generated JavaScript script converted
from a module/cross-origin form to a deferred local script and with the linked
CSS `crossorigin` attribute removed.

## Dependencies

Node built-ins `node:fs` and `node:path`.

## Error handling

Missing or unreadable HTML causes `readFileSync` to throw. If the expected Vite
pattern is absent, the file is still written unchanged.

## Example

```powershell
cd dashboard
pnpm build
Start-Process ..\artifacts\dashboard\index.html
```

## Related scripts

Runs after Vite and follows [`clean-assets.cjs`](clean-assets.md).
