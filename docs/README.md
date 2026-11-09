# Game Performance ML Documentation

This is the documentation entry point for the Game Performance ML platform.
The suite is written for business users, support engineers, data scientists,
developers, and maintainers. It describes the implementation currently in this
repository as of 2026-11-09.

## Start by audience

| Audience | Recommended reading |
|---|---|
| End user / business analyst | [User guide](user-guide.md), [Data and reporting](data-and-reporting.md) |
| Support engineer | [Operational runbook](operational-runbook.md), [API reference](api-reference.md), [Data flow](data-flow.md) |
| Developer / data scientist | [Platform overview](platform-overview.md), [Technology stack](technology-stack.md), [Module reference](module-reference.md) |
| Future maintainer | [Maintenance guide](maintenance.md), [Documentation coverage](documentation-coverage.md), folder and module guides linked below |

## Documentation map

1. [Platform overview](platform-overview.md) — business purpose, capabilities, architecture, and workflows.
2. [User guide](user-guide.md) — access, upload, scoring, navigation, responsibilities, and troubleshooting.
3. [Technology stack](technology-stack.md) — versions, locations, purpose, and configuration.
4. [Data and reporting](data-and-reporting.md) — input schema, output workbooks, dashboard datasets, fields, filters, sorting, and interpretation.
5. [Data flow](data-flow.md) — source-to-output movement, validation, storage, integrations, and failure paths.
6. [API reference](api-reference.md) — local server endpoints and request/response behavior.
7. [Operational runbook](operational-runbook.md) — setup, development, production-style local runs, checks, and incident response.
8. [Maintenance guide](maintenance.md) — extension points, model changes, artifact lifecycle, security, and synchronization rules.
9. [Module reference](module-reference.md) — package-level index and links to the dedicated module documents.
10. [Documentation coverage](documentation-coverage.md) — repository hierarchy, folder coverage, script coverage, and templates.

## Repository-local references

- [Root repository guide](../README.md)
- [Python package folder guide](../src/game_performance/game_performance.md)
- [Dashboard folder guide](../dashboard/dashboard.md)
- [Notebook guide](../notebooks/notebooks.md)
- [Tests folder guide](../tests/tests.md)
- [Artifacts folder guide](../artifacts/artifacts.md)

## Scope and terminology

The platform forecasts the next seven calendar days for games after an
observation window, using daily game-performance data. “Game” means the stable
`Game_ID` created from `SystemName::TournamentName`; “checkpoint” means the
number of observed lifecycle days used to create a scoring snapshot; and
“holdout” means the final chronological evaluation partition that is not used
to fit production models.

Generated workbooks and uploaded source files are local runtime data. They may
contain commercially sensitive information and should be handled according to
the operating team’s data-retention policy.
