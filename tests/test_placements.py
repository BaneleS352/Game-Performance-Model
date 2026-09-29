import json
import threading
from http.client import RemoteDisconnected
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from game_performance import server as server_module
from game_performance.placements import (
    RankedGame,
    build_placement_recommendations,
    get_placement_recommendations,
    normalize_game_name,
    parse_game_detail,
    parse_ranked_links,
    score_forecasts,
)
from game_performance.server import DashboardRequestHandler, UploadState


def _forecast(game_id, name, stakes, clients, probability, confidence):
    return {
        "Game_ID": f"SYS::{game_id}",
        "TournamentName": name,
        "SystemName": "SYS",
        "predicted_stakes": stakes,
        "predicted_clients": clients,
        "probability_high_performance": probability,
        "confidence_clients": confidence,
        "confidence_stakes": confidence,
    }


def test_ranked_link_parser_uses_visible_ranks_and_deduplicates():
    source = """
    <a href="/gaming/games/first" aria-label="#1">First</a>
    <a href="/gaming/games/second">#2 Second</a>
    <a href="/gaming/games/first">First again</a>
    <a href="/gaming/games/third">#3 Third</a>
    """

    games = parse_ranked_links(source)

    assert [(game.current_rank, game.url, game.lobby_title) for game in games] == [
        (1, "https://www.hollywoodbets.net/gaming/games/first", "First"),
        (2, "https://www.hollywoodbets.net/gaming/games/second", "Second"),
        (3, "https://www.hollywoodbets.net/gaming/games/third", "Third"),
    ]


def test_ranked_link_parser_uses_embedded_trending_data_when_present():
    source = '<script id="__NEXT_DATA__">{"props":{"pageProps":{"trendingGames":{"items":[{"name":"Hot Game","slug":"hot-game"},{"name":"Next Game","slug":"next-game"}]}}}}</script>'

    games = parse_ranked_links(source)

    assert [(game.current_rank, game.lobby_title, game.url) for game in games] == [
        (1, "Hot Game", "https://www.hollywoodbets.net/gaming/games/hot-game"),
        (2, "Next Game", "https://www.hollywoodbets.net/gaming/games/next-game"),
    ]


def test_detail_parser_extracts_title_and_provider():
    detail = parse_game_detail("<title>Ignored | Hollywoodbets</title><h1>Jet X</h1><span>Provider</span><span>Smartsoft</span>")

    assert detail == {"title": "Jet X", "provider": "Smartsoft"}


def test_normalize_game_name_handles_slugs_and_accents():
    assert normalize_game_name("Book-of-Bloodlines: Café") == "book of bloodlines cafe"


def test_model_matching_accepts_system_prefix_in_tournament_name():
    forecasts = [_forecast("A", "SYS-A Game", 100, 10, 0.5, 0.5)]
    visible = [RankedGame(1, "https://www.hollywoodbets.net/gaming/games/a-game", "A Game")]

    payload = build_placement_recommendations(forecasts, visible)

    assert payload["summary"]["matched_games"] == 1
    assert payload["recommendations"][0]["placement_action"] == "Hold"


def test_score_forecasts_applies_balanced_score_and_rank():
    scored = score_forecasts([
        _forecast("A", "A", 1000, 100, 0.9, 0.8),
        _forecast("B", "B", 100, 10, 0.1, 0.4),
    ])

    assert [row["Game_ID"] for row in scored] == ["SYS::A", "SYS::B"]
    assert scored[0]["placement_score"] > scored[1]["placement_score"]
    assert scored[0]["confidence"] == 0.8


def test_score_forecasts_keeps_explicit_zero_confidence_in_average():
    row = _forecast("A", "A", 100, 10, 0.5, 0.0)
    row["confidence_stakes"] = 0.8

    assert score_forecasts([row])[0]["confidence"] == 0.4


def test_build_recommendations_classifies_moves_promotions_and_unmatched():
    forecasts = [
        _forecast("A", "A Game", 1000, 100, 0.9, 0.9),
        _forecast("C", "C Game", 800, 80, 0.8, 0.8),
        _forecast("E", "E Game", 700, 70, 0.7, 0.7),
        _forecast("B", "B Game", 100, 10, 0.1, 0.4),
    ]
    visible = [
        RankedGame(1, "https://www.hollywoodbets.net/gaming/games/b-game", "B Game"),
        RankedGame(2, "https://www.hollywoodbets.net/gaming/games/c-game", "C Game"),
        RankedGame(3, "https://www.hollywoodbets.net/gaming/games/unknown", "Unknown Game"),
        RankedGame(4, "https://www.hollywoodbets.net/gaming/games/a-game", "A Game"),
    ]

    payload = build_placement_recommendations(forecasts, visible, scraped_at="2026-09-29T00:00:00+00:00")
    actions = {row["lobby_title"]: row["placement_action"] for row in payload["recommendations"]}

    assert actions["A Game"] == "Move higher"
    assert actions["B Game"] == "Move lower"
    assert actions["C Game"] == "Hold"
    assert actions["Unknown Game"] == "No model match"
    assert actions["E Game"] == "Promote into visible ranks"
    assert payload["summary"] == {
        "scraped_games": 4,
        "matched_games": 3,
        "move_higher": 1,
        "move_lower": 1,
        "hold": 1,
        "promote": 1,
        "unmatched": 1,
    }


def test_placement_cache_reuses_payload_until_refresh(tmp_path, monkeypatch):
    dashboard_dir = tmp_path / "artifacts" / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "dashboard_data.json").write_text(json.dumps({"forecasts": [_forecast("A", "A", 100, 10, 0.5, 0.5)]}), encoding="utf-8")
    calls = []
    monkeypatch.setattr("game_performance.placements.scrape_ranked_games", lambda: calls.append("scrape") or [])
    cache = {}

    get_placement_recommendations(tmp_path, cache=cache)
    get_placement_recommendations(tmp_path, cache=cache)
    get_placement_recommendations(tmp_path, refresh=True, cache=cache)

    assert calls == ["scrape", "scrape"]


def test_placements_endpoint_returns_scraper_payload(tmp_path, monkeypatch):
    expected = {"source_url": "https://example.test", "summary": {"scraped_games": 0}, "recommendations": []}
    calls = []

    def fake_placements(project_root, *, refresh, cache, cache_lock):
        calls.append(refresh)
        return expected

    monkeypatch.setattr(server_module, "get_placement_recommendations", fake_placements)
    state = UploadState(tmp_path)

    class Handler(DashboardRequestHandler):
        pass

    Handler.state = state
    http_server = server_module.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        lambda *args, **kwargs: Handler(*args, directory=str(tmp_path), **kwargs),
    )
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    try:
        port = http_server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/placements?refresh=1") as response:
            assert json.loads(response.read()) == expected
        assert calls == [True]
    finally:
        http_server.shutdown()
        thread.join(timeout=2)
        http_server.server_close()
