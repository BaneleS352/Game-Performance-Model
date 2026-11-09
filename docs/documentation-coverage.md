# Documentation Coverage and Templates

## Maintained hierarchy

```text
README.md
├── docs/
│   ├── README.md
│   ├── docs.md
│   ├── platform-overview.md
│   ├── user-guide.md
│   ├── technology-stack.md
│   ├── data-and-reporting.md
│   ├── data-flow.md
│   ├── api-reference.md
│   ├── operational-runbook.md
│   ├── maintenance.md
│   ├── module-reference.md
│   └── documentation-coverage.md
├── dashboard/
│   ├── dashboard.md
│   ├── scripts/scripts.md
│   │   ├── clean-assets.md
│   │   └── fix-local-html.md
│   └── src/src.md
├── data/
│   ├── data.md
│   └── uploads/uploads.md
├── notebooks/
│   ├── notebooks.md
│   ├── 01_data_exploration.md
│   ├── 02_feature_engineering.md
│   ├── 03_model_experiments.md
│   └── 04_training_and_forecasting.md
├── src/
│   ├── src.md
│   ├── game_performance/game_performance.md
│   └── game_performance/*.md (one guide per Python module)
├── tests/tests.md
├── artifacts/
│   ├── artifacts.md
│   ├── dashboard/dashboard.md
│   ├── dashboard/assets/assets.md
│   ├── features/features.md
│   ├── models/models.md
│   ├── processed/processed.md
│   ├── reports/reports.md
│   └── uploads/uploads.md
└── tools/tools.md
```

Runtime-created upload-run subdirectories under `artifacts/uploads/` are
documented by the pattern and artifact table in [Data and reporting](data-and-reporting.md)
rather than by committing a Markdown file into every timestamped run.
Virtual environments, package caches, `.git`, and build caches are excluded
from maintained repository documentation.

## Coverage rules

- Every logical repository folder has a `.md` file named after the folder.
- Every Python module has a matching `.md` file in the same folder.
- Every executable Node script has a matching `.md` guide.
- Every orchestration notebook has a matching `.md` guide.
- Cross-folder links point back to this suite and module/folder guides.
- Generated artifacts are documented by stable folder and file-pattern guides,
  not by copying volatile binary contents into Markdown.

## Folder documentation template

```markdown
# `<folder>/` Folder Guide

## Folder purpose
Describe the business and technical responsibility.

## File inventory
| File | Purpose | Dependencies | Description |
|---|---|---|---|

## Key processes
Explain execution order and collaboration between files.

## Data dependencies
List inputs, outputs, schemas, and storage locations.

## Ownership and related services
Name the responsible team/module and linked docs.
```

## Script documentation template

```markdown
# `<script>`

## Purpose
## Location
## Inputs and required environment variables
## Outputs
## Dependencies
## Execution process
## Error handling
## Example usage
## Data processing logic
## Related scripts and services
```

## Synchronization evidence

The current implementation was inspected across `pyproject.toml`, the Python
package, Vite configuration, React entry point, notebooks, tests, and the
runtime artifact layout. The documentation intentionally describes local
runtime storage and the absence of authentication/database/external-service
integrations rather than implying capabilities that are not implemented.
