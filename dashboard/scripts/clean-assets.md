# `clean-assets.cjs`

## Purpose and location

Build helper at `dashboard/scripts/clean-assets.cjs`. It removes stale compiled
frontend assets before Vite writes a new build.

## Inputs

No command-line arguments or environment variables. It resolves the repository
root relative to `__dirname` and targets `artifacts/dashboard/assets`.

## Outputs

The target assets directory is removed recursively if present. Vite recreates
it during the subsequent build.

## Dependencies

Node built-ins `node:fs` and `node:path`.

## Execution and error handling

The script verifies the resolved target begins with the dashboard artifact
directory plus a path separator. If the safety check fails it throws rather
than deleting. It then calls `fs.rmSync(..., { recursive: true, force: true })`.

## Example

```powershell
cd dashboard
pnpm build
```

## Related scripts

Runs immediately before Vite and [`fix-local-html.cjs`](fix-local-html.md).
