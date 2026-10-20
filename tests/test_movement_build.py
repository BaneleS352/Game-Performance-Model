import pandas as pd

from game_movement_platform.build_app import _load_positions


def test_position_builder_uses_latest_scrape_and_keeps_history(tmp_path):
    columns = ["date", "scraped_at", "position", "title", "src"]
    pd.DataFrame([
        ["2026-08-01", "2026-08-01 10:00:00", 2, "Game A", "a.webp"],
        ["2026-08-01", "2026-08-01 10:00:00", 1, "Game B", "b.webp"],
    ], columns=columns).to_csv(tmp_path / "hollywoodbets_games_2026-08-01.csv", index=False)
    pd.DataFrame([
        ["2026-08-02", "2026-08-02 10:00:00", 1, "Game A", "a.webp"],
        ["2026-08-02", "2026-08-02 10:00:00", 2, "Game B", "b.webp"],
    ], columns=columns).to_csv(tmp_path / "hollywoodbets_games_2026-08-02.csv", index=False)

    current, dates, history = _load_positions(tmp_path)

    assert current[["title", "position"]].to_dict("records") == [
        {"title": "Game A", "position": 1},
        {"title": "Game B", "position": 2},
    ]
    assert dates == ["2026-08-01", "2026-08-02"]
    assert history["Game A"][-1] == {"date": "2026-08-02", "position": 1}
