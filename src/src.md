# `src/` Folder Guide

## Folder purpose

Python source root for the installable `game_performance` package and its
generated setuptools metadata.

## File inventory

| Path | Purpose | Dependencies | Description |
|---|---|---|---|
| `game_performance/` | Reusable application package | Python dependencies in `pyproject.toml` | Data, feature, modeling, scoring, reporting, API, and persistence modules. |
| `game_performance_ml.egg-info/` | Packaging metadata | setuptools | Generated metadata from editable/package installation. |
| `src.md` | Folder guide | Markdown | Source tree ownership and links. |

## Key processes

Setuptools discovers packages under `src/`. The package is imported by notebooks,
the Vite-spawned server, tests, and command-line module execution.

## Related documentation

- [Package folder](game_performance/game_performance.md)
- [Technology stack](../docs/technology-stack.md)
