"""Build the game-movement data file from the checked-in position scrapes."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
SCRAPES = BASE / "Top Games Position Scrapes"


def clean_name(value: object) -> str:
    value = str(value or "").lower().replace("\u2019", "'").replace("&", " and ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"^\s*[a-z0-9]{2,10}\s*[-:]\s*", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def _load_positions(scrapes_dir: Path = SCRAPES) -> tuple[pd.DataFrame, list[str], dict[str, list[dict[str, str | int]]]]:
    files = sorted(scrapes_dir.glob("hollywoodbets_games_*.csv"))
    if not files:
        raise FileNotFoundError(f"No position scrapes found in {scrapes_dir}")

    history = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    history["position"] = pd.to_numeric(history["position"], errors="raise").astype(int)
    history = history.sort_values(["date", "scraped_at", "position"]).drop_duplicates(["date", "title"], keep="last")

    latest_date = history["date"].max()
    current = history.loc[history["date"].eq(latest_date)].sort_values("position").drop_duplicates("title", keep="first").copy()
    position_dates = [value.strftime("%Y-%m-%d") for value in sorted(history["date"].unique())]
    position_history = {
        title: [
            {"date": row.date.strftime("%Y-%m-%d"), "position": int(row.position)}
            for row in rows.itertuples()
        ]
        for title, rows in history.groupby("title", sort=False)
    }
    return current, position_dates, position_history


def _best_match(key: str, choices: list[str]) -> tuple[str, float] | None:
    sorted_key = " ".join(sorted(key.split()))
    scored = sorted(
        ((candidate, SequenceMatcher(None, sorted_key, " ".join(sorted(candidate.split()))).ratio() * 100) for candidate in choices),
        key=lambda item: item[1],
        reverse=True,
    )[:2]
    if not scored:
        return None
    best = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0
    return best if best[1] >= 94 or (best[1] >= 88 and best[1] - second_score >= 5) else None


def build(base: Path = BASE, scrapes_dir: Path = SCRAPES) -> dict[str, object]:
    positions, position_dates, position_history = _load_positions(scrapes_dir)
    performance_files = sorted(scrapes_dir.glob("FB_DAY_TOURNAMENT*.csv"))
    if not performance_files:
        raise FileNotFoundError(f"No performance CSV found in {scrapes_dir}")
    performance = pd.concat((pd.read_csv(path) for path in performance_files), ignore_index=True)
    performance = performance.rename(columns={"Clients": "players", "BetCount": "bets", "Stakes": "stakes", "GGR": "ggr"})

    positions["match_key"] = positions["title"].map(clean_name)
    performance["match_key"] = performance["TournamentName"].map(clean_name)
    daily = performance.groupby(["EventDateID", "match_key"], as_index=False)[["players", "bets", "stakes", "ggr"]].sum()
    performance_dates = sorted(int(value) for value in daily["EventDateID"].unique())
    choices = daily["match_key"].drop_duplicates().tolist()
    choice_set = set(choices)

    match_map: dict[str, str] = {}
    score_map: dict[str, float] = {}
    for row in positions.itertuples():
        if row.match_key in choice_set:
            match_map[row.title] = row.match_key
            score_map[row.title] = 100
            continue
        match = _best_match(row.match_key, choices)
        if match:
            match_map[row.title], score_map[row.title] = match[0], round(match[1], 1)

    groups = {key: rows.set_index("EventDateID") for key, rows in daily.groupby("match_key", sort=False)}
    records = []
    for row in positions.itertuples():
        selected = match_map.get(row.title)
        group = groups.get(selected) if selected else None
        series = []
        for day in performance_dates:
            values = group.loc[day] if group is not None and day in group.index else None
            series.append(
                {
                    "d": str(day),
                    "players": int(values["players"]) if values is not None else None,
                    "bets": int(values["bets"]) if values is not None else None,
                    "stakes": float(values["stakes"]) if values is not None else None,
                    "ggr": float(values["ggr"]) if values is not None else None,
                }
            )
        records.append(
            {
                "name": row.title,
                "position": int(row.position),
                "image": row.src,
                "matchScore": score_map.get(row.title),
                "positionHistory": position_history[row.title],
                "series": series,
            }
        )

    meta = {
        "dates": performance_dates,
        "positionDates": position_dates,
        "positionSnapshot": position_dates[-1],
        "positionsCount": len(positions),
        "matchedCount": len(match_map),
        "performanceGames": int(performance["TournamentName"].nunique()),
        "rows": len(performance),
    }
    (base / "data.js").write_text(
        "window.GAME_DATA=" + json.dumps({"meta": meta, "games": records}, separators=(",", ":")) + ";",
        encoding="utf-8",
    )
    return meta


if __name__ == "__main__":
    print(build())
