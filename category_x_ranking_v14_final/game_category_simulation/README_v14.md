# Category X Ranking Lab - v14

## Overview

Category X Ranking Lab v14 is a daily game-placement simulation for testing how game performance can be converted into a controlled website order.

The project uses the merged slot-game dataset, selects the latest 60 dated days, uses the first 30 days as a calibration/validation period, and evaluates the ranking engine on the following 30 unseen days using a walk-forward process. Ranking signals for an evaluation day use only information available before that day.

The v14 architecture separates two decisions:

1. **Final Score / Ideal Rank** - where the game's performance signals say it ideally belongs.
2. **Adaptive Movement / Website Position** - how quickly the game is allowed to move toward that ideal position.

This separation is intentional. The goal is to surface genuinely strong and persistently improving games without allowing short-term noise to make the website order change excessively.

## v14 ranking architecture

```text
Daily game data
    -> Current Performance
    -> Momentum
    -> Acceleration
    -> Stability
    -> Final Score
    -> Ideal Rank
    -> Evidence + Direction Persistence
    -> Adaptive Movement
    -> Website Position
```

The Dashboard and Website views use the same daily `website_position` output.

## Ranking signals

### Current Performance - 65%

Current Performance measures how strong a game is now. It uses the most recent seven days available before the ranking date, with newer observations receiving more influence.

The pipeline combines four performance metrics. In the v14 code their internal metric weights are:

- Clients: 25%
- Bet Count: 20%
- Stakes: 30%
- GGR: 25%

Each metric is normalized relative to the catalogue before being combined.

### Momentum - 15%

Momentum measures whether a game is improving or declining and the strength of that direction.

The engine compares recent activity with preceding activity over three windows:

- 3-day: 35%
- 7-day: 45%
- 14-day: 20%

Positive Momentum means improvement. Negative Momentum means decline.

### Acceleration - 5%

Acceleration measures how Momentum itself is changing. It helps distinguish a trend that is strengthening from one that is cooling.

Acceleration is fixed at 5% in v14 after sensitivity testing showed that larger acceleration weights could increase movement while reducing near-top ranking quality.

### Stability - 15%

In v14, Stability measures the consistency of the recent performance pattern. The implementation uses rolling 30-day variability for Clients, Bet Count, Stakes and GGR and converts that variability into a 0-100 consistency score.

A high Stability value means the recent performance pattern is relatively consistent. It does **not** mean that the website position must remain unchanged.

## Final Score and Ideal Rank

The v14 Final Score is:

```text
Final Score =
    65% Current Performance
  + 15% Momentum Score
  +  5% Acceleration Score
  + 15% Stability
```

Games are sorted by Final Score to produce the **Ideal Rank**.

Ideal Rank is the destination suggested by the ranking signals. It is not automatically the Website Position because v14 applies controlled adaptive movement.

## Direction Persistence

Direction Persistence counts consecutive ranking days in which Momentum continues to support the same direction.

- Momentum >= +5: improving direction
- Momentum <= -5: declining direction
- Momentum between -5 and +5: no active directional streak

Persistence is **not** a fixed waiting period and it is **not** another Final Score weight.

In v14 it acts as a movement multiplier. Strong evidence can create movement immediately, while sustained evidence increases the amount of movement pressure that is trusted. The movement calculation uses up to five days of persistence maturity.

## Strong-performance streak

v14 also tracks how long a game has remained strongly performing. A strong-performance day is identified when Current Performance is at least 70 and Momentum is above -10.

This streak is used mainly to protect established strong performers from abrupt downward replacement. Protection weakens when the game's performance evidence deteriorates.

## Movement Confidence

Movement Confidence is an explanatory movement-layer value. It combines:

- evidence strength,
- Direction Persistence maturity, and
- whether the game has an established strong-performance streak.

It indicates how much confidence the engine has that changing the game's website position is justified. It is not a separate ranking score.

## Adaptive Movement

v14 does not use one fixed movement limit for every game.

Movement pressure is based on:

- distance between current Website Position and Ideal Rank,
- evidence strength,
- Momentum strength,
- Acceleration strength,
- Direction Persistence, and
- established-game protection.

Persistence multiplies already-supported movement pressure. This means weak or noisy signals remain controlled, while strong and persistent signals can produce larger progressive movement.

The current implementation allows a maximum adaptive upward cap of 35 positions and a maximum downward cap of 23 positions before catalogue-wide position collision resolution. Established strong games can receive additional downward protection.

Because every game must occupy a unique website position, the final movement can also be affected by other games competing for positions.

## Missing data

Missing data is never converted into artificial zero performance.

If a game has no actual record for an evaluation date:

- its state is shown as `No Data`,
- raw Clients, Bet Count, Stakes and GGR are not shown as zero,
- its previous Website Position is held, and
- it is not deliberately moved because of the missing record.

## Dataset and evaluation design

The included v14 run was built from:

- 408,998 source rows after loading,
- 3,326 unique games,
- 179 source dates,
- latest 60 dated days selected for the ranking test,
- first 30 selected days used for calibration/validation,
- next 30 days used for unseen walk-forward evaluation.

The pipeline writes one compact JSON file per evaluation day under `web/data/days/` together with metadata, evaluation summaries, calibration information and per-game history.

## v14 validation summary

The v14 targeted persistence-multiplier test produced approximately:

| Measure | v14 result |
|---|---:|
| Spearman correlation | 0.682 |
| Top-10 overlap | 51.7% |
| Top-20 overlap | 54.0% |
| Top-100 overlap | 76.2% |
| Mean rank error | 377.2 |
| Average absolute daily movement | 7.0 positions |
| Median movement | 5 positions |
| Maximum movement | 36 positions |
| Moves above 20 positions | 5.7% |
| Direction reversals | 24.7% |

High-evidence games with at least four days of persistence moved about 12.8 positions on average in the real-data movement test. These are prototype/back-test results and should continue to be validated before production use.

## Project structure

```text
game_category_simulation/
|-- build_ranking_pipeline.py
|-- merged_slot_games.xlsx
|-- requirements.txt
|-- serve.py
|-- run_pipeline.bat
|-- run_ui.bat
|-- test_persistence_movement.py
|-- test_persistence_multiplier_realdata.py
|-- TEST_RESULTS.md
|-- FINAL_PERSISTENCE_DECISION.md
|-- Category_X_Ranking_Algorithm_v14_Explanation.pdf
|-- web/
|   |-- index.html
|   |-- app.js
|   |-- styles.css
|   |-- assets/
|   `-- data/
|       |-- metadata.json
|       |-- evaluation.json
|       |-- game_series.json
|       |-- pipeline_status.json
|       `-- days/
`-- README.md
```

## First-time setup on Windows

Open a terminal in the `game_category_simulation` folder.

Create a virtual environment:

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

Install the required packages:

```cmd
python -m pip install -r requirements.txt
```

If a test that calculates Spearman correlation reports that SciPy is missing, install it with:

```cmd
python -m pip install scipy
```

## Run the application WITHOUT rerunning the ranking pipeline

The v14 project archive already contains generated results under `web\data`. If those files are present, **you do not need to run `build_ranking_pipeline.py` to open the application**.

After completing the first-time setup, run:

```cmd
.venv\Scripts\activate
python serve.py
```

Then open:

```text
http://127.0.0.1:8000
```

Keep the terminal running while using the application. Stop the server with `Ctrl + C`.

For later sessions, the normal startup is only:

```cmd
.venv\Scripts\activate
python serve.py
```

You can also use:

```cmd
run_ui.bat
```

provided the `.venv` has already been created and the dependencies are installed.

### When is a pipeline rerun required?

Do **not** rerun the pipeline simply because you reopened or re-extracted the project.

Rerun it only when you intentionally want to regenerate ranking outputs, for example after:

- changing `merged_slot_games.xlsx`,
- changing ranking-signal calculations,
- changing score weights,
- changing persistence/adaptive-movement logic, or
- deleting/replacing the generated `web\data` results.

## Regenerate the ranking results

If you intentionally need to rebuild the results:

```cmd
.venv\Scripts\activate
python build_ranking_pipeline.py "merged_slot_games.xlsx"
```

Wait for the pipeline to report successful completion. Then start the UI:

```cmd
python serve.py
```

The same pipeline can also be launched with:

```cmd
run_pipeline.bat
```

## Persistence movement tests

The project contains two movement-related test scripts.

To run the persistence movement test:

```cmd
.venv\Scripts\activate
python test_persistence_movement.py
```

To run the real-data targeted multiplier test:

```cmd
python test_persistence_multiplier_realdata.py
```

These tests are for evaluation and do not need to be run simply to view the existing v14 UI.

Some test comments and reports refer to a **v13 baseline**. Those references are intentional historical comparisons used to measure the improvement introduced in v14; they do not mean that this project is v13.

## Server and UI

`serve.py` starts a lightweight local Python HTTP server on:

```text
http://127.0.0.1:8000
```

The server provides API endpoints used by the browser, including bootstrap metadata, daily rankings, the Top-300 Website view, individual game history, evaluation results and pipeline status.

The Website page loads only the first 300 games for the selected date. The Dashboard can inspect the full catalogue and uses the same daily Website Position data as the Website view.

## Important notes

- v14 is the current project baseline documented by this README.
- Forecast is excluded from the current ranking test.
- Anomaly risk does not influence placement.
- Acceleration remains fixed at 5%.
- Persistence affects movement, not the Final Score.
- Missing data is held rather than treated as zero performance.
- Website Position is the authoritative simulated placement shown by the UI.
- The project is a prototype/back-testing environment and should be validated with production/shadow data before live deployment.
