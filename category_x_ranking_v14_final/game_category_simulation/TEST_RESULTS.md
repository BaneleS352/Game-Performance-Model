# Persistence movement test results - v14

## Purpose

This version keeps the original daily ranking architecture. It uses daily Current Performance, Momentum, Acceleration and Stability calculations. Persistence is used only in the movement layer.

Persistence now acts as a multiplier on movement pressure. It is not a ranking-score component and it is not a fixed waiting period.

## Ranking configuration

- Current Performance: 65%
- Momentum: 15%
- Acceleration: 5%
- Stability: 15%
- Forecast: excluded
- Anomaly risk: diagnostic only

## Real-data final test

The final movement-policy test used the existing 3,326 games and 30 evaluation days. The signal calculations were held constant while movement policies were compared.

| Measure | v13 baseline | v14 targeted multiplier |
|---|---:|---:|
| Spearman correlation | 0.6791 | 0.6816 |
| Top-10 overlap | 52.33% | 51.67% |
| Top-20 overlap | 55.00% | 54.00% |
| Top-100 overlap | 76.33% | 76.23% |
| Mean rank error | 380.1 | 377.2 |
| Average movement | 6.08 | 7.02 |
| Median movement | 4 | 5 |
| Maximum movement | 32 | 36 |
| Moves >20 positions | 2.33% | 5.72% |
| Direction reversals | 24.89% | 24.72% |

The targeted policy increases movement responsiveness while keeping broad ranking quality close to the baseline.

High-evidence games with at least four days of persistence moved about 12.8 positions on average. This demonstrates that persistent strong growth is not restricted to one-position-per-day movement.

## Persistence effect

For the targeted movement policy, average upward movement increases with longer persistence:

| Persistence | Mean upward move |
|---|---:|
| 1-2 days | 7.35 |
| 3-4 days | 8.23 |
| 5+ days | 8.66 |
| 6+ days | 9.64 |

## Pipeline validation

The v14 pipeline was rerun successfully on the merged dataset:

- 408,998 source rows loaded
- 3,326 unique games
- 179 source dates detected
- 60-day working window selected
- 30 calibration days
- 30 unseen evaluation days
- Runtime: about 18.8 seconds

The generated website positions are unique per day. Missing-data games hold position and do not move because of artificial zeros.

## Server validation

The server started successfully and responded to `/api/status` with HTTP 200.

The Dashboard and Website views continue to use the same daily `website_position` data.

## Decision

The v14 targeted persistence multiplier is integrated into the main pipeline. It was selected because it increases movement for persistent evidence while keeping ranking quality close to the baseline. It should continue to be monitored using real production data before being treated as final.
