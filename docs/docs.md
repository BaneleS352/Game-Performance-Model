# `docs/` Folder Guide

## Folder purpose

This folder contains the maintained documentation suite for the Game
Performance ML repository. It is the stable knowledge-transfer layer between
the implementation and its users, support staff, developers, and maintainers.

## File inventory

| File | Purpose | Dependencies | Description |
|---|---|---|---|
| `README.md` | Documentation index | All documents | Audience-based entry point and cross-reference map. |
| `docs.md` | Folder guide | `README.md` | Describes this documentation folder. |
| `platform-overview.md` | Business and architecture guide | Source package, dashboard, notebooks | Explains the platform’s purpose, capabilities, and main workflows. |
| `user-guide.md` | End-user guide | Dashboard and server behavior | Explains upload, worksheet selection, scoring, navigation, and troubleshooting. |
| `technology-stack.md` | Stack reference | `pyproject.toml`, dashboard package files | Records versions, locations, purpose, and configuration requirements. |
| `data-and-reporting.md` | Dataset/output reference | `data.py`, `features.py`, `reporting.py`, `dashboard.py` | Defines source fields, calculated fields, workbook sheets, and dashboard semantics. |
| `data-flow.md` | Pipeline and integration guide | `upload_pipeline.py`, `server.py`, package modules | Maps validation, transformations, storage, publication, and failure paths. |
| `api-reference.md` | Local HTTP API reference | `server.py`, `main.jsx`, Vite proxy | Documents upload, run, status, dashboard-data, and report endpoints. |
| `operational-runbook.md` | Run and support guide | `README.md`, package and dashboard scripts | Provides setup, commands, verification, and incident procedures. |
| `maintenance.md` | Maintainer guide | All implementation modules | Explains extension points, compatibility constraints, and synchronization rules. |
| `module-reference.md` | Python module index | `src/game_performance/*.py` | Links to dedicated module documentation. |
| `documentation-coverage.md` | Coverage and templates | Repository folders and files | Shows the documentation hierarchy, folder/script coverage, and suggested templates. |

## Key processes

Documentation changes should be made alongside implementation changes. When a
module changes its inputs, outputs, endpoints, workbook sheets, configuration,
or user-facing behavior, update the corresponding document and run the
coverage checks described in [Maintenance](maintenance.md).

## Ownership

The repository maintainer owns technical accuracy. The product or operations
owner owns business terminology, user responsibilities, retention rules, and
support escalation instructions.
