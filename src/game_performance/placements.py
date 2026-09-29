"""Scrape public lobby ranks and turn model forecasts into placement advice."""

from __future__ import annotations

import html
import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.hollywoodbets.net/gaming"
MAX_VISIBLE_RANK = 24
CACHE_TTL = timedelta(minutes=10)
FETCH_TIMEOUT_SECONDS = 15


def _text(value: Any) -> str:
    return html.unescape(str(value or "")).strip()


def normalize_game_name(value: Any) -> str:
    """Normalize titles and identifiers enough for lobby/model matching."""
    text = unicodedata.normalize("NFKD", _text(value)).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _rank_from_text(value: str) -> int | None:
    if not value:
        return None
    medals = {"🥇": 1, "🥈": 2, "🥉": 3}
    for medal, rank in medals.items():
        if medal in value:
            return rank
    match = re.search(r"(?:#|rank\s*)(\d{1,3})\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


@dataclass
class RankedGame:
    current_rank: int
    url: str
    lobby_title: str
    provider: str | None = None


class _LobbyLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, Any]] = []
        self._anchor: dict[str, Any] | None = None
        self._next_data: list[str] = []
        self._in_next_data = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self._in_next_data = True
            return
        if tag.lower() != "a":
            return
        attributes = {key.lower(): value or "" for key, value in attrs}
        href = attributes.get("href", "")
        if "/gaming/games/" not in href:
            return
        self._anchor = {
            "href": href,
            "text": [],
            "rank": _rank_from_text(" ".join(attributes.get(key, "") for key in ("aria-label", "title", "data-rank"))),
        }

    def handle_data(self, data: str) -> None:
        if self._in_next_data:
            self._next_data.append(data)
        if self._anchor is not None:
            self._anchor["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_next_data:
            self._in_next_data = False
            return
        if tag.lower() != "a" or self._anchor is None:
            return
        entry = self._anchor
        self._anchor = None
        text = _text(" ".join(entry["text"]))
        entry["rank"] = entry["rank"] or _rank_from_text(text)
        entry["text"] = re.sub(r"(?:🥇|🥈|🥉|#\s*\d+)\s*", "", text, count=1).strip()
        self.links.append(entry)


class _DetailParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.headings: list[str] = []
        self.text_parts: list[str] = []
        self._capture_title = False
        self._capture_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._capture_title = True
        if lowered in {"h1", "h2"}:
            self._capture_heading = True

    def handle_data(self, data: str) -> None:
        value = _text(data)
        if not value:
            return
        self.text_parts.append(value)
        if self._capture_title:
            self.title_parts.append(value)
        if self._capture_heading:
            self.headings.append(value)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._capture_title = False
        if lowered in {"h1", "h2"}:
            self._capture_heading = False


def parse_ranked_links(source_html: str, *, source_url: str = SOURCE_URL, limit: int = MAX_VISIBLE_RANK) -> list[RankedGame]:
    parser = _LobbyLinkParser()
    parser.feed(source_html)
    try:
        next_data = json.loads("".join(parser._next_data))
        items = next_data["props"]["pageProps"]["trendingGames"]["items"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        items = []
    if items:
        return [
            RankedGame(
                index,
                urljoin(source_url, f"/gaming/games/{item['slug']}"),
                _text(item.get("name")) or item["slug"],
            )
            for index, item in enumerate(items[:limit], start=1)
            if isinstance(item, dict) and item.get("slug")
        ]
    explicit_links = [entry for entry in parser.links if entry["rank"] and entry["rank"] <= limit]
    source_links = explicit_links or parser.links
    ranked: list[RankedGame] = []
    seen_urls: set[str] = set()
    next_rank = 1
    for entry in source_links:
        url = urljoin(source_url, entry["href"])
        if url in seen_urls:
            continue
        rank = entry["rank"] or next_rank
        if rank < 1 or rank > limit:
            continue
        seen_urls.add(url)
        ranked.append(RankedGame(rank, url, entry["text"] or url.rsplit("/", 1)[-1]))
        next_rank = max(next_rank, rank + 1)
        if len(ranked) >= limit:
            break
    return sorted(ranked, key=lambda item: item.current_rank)


def parse_game_detail(source_html: str, *, fallback_title: str = "") -> dict[str, str | None]:
    parser = _DetailParser()
    parser.feed(source_html)
    title = _text(parser.headings[0] if parser.headings else " ".join(parser.title_parts).split("|")[0]) or fallback_title
    provider: str | None = None
    for index, value in enumerate(parser.text_parts[:-1]):
        if value.strip().lower().rstrip(":") == "provider":
            provider = parser.text_parts[index + 1]
            break
    if provider is None:
        match = re.search(r"provider\s*:?\s*([A-Za-z0-9][A-Za-z0-9 &'().-]{1,80})", " ".join(parser.text_parts), re.IGNORECASE)
        provider = match.group(1).strip(" .,") if match else None
    return {"title": title, "provider": provider}


def _fetch_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "GamePerformanceModel/1.0"})
    with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def scrape_ranked_games(fetch: Callable[[str], str] = _fetch_url, *, enrich_details: bool = False) -> list[RankedGame]:
    ranked = parse_ranked_links(fetch(SOURCE_URL))
    if not enrich_details:
        return ranked
    enriched: list[RankedGame] = []
    for game in ranked:
        try:
            detail = parse_game_detail(fetch(game.url), fallback_title=game.lobby_title)
        except Exception:
            detail = {"title": game.lobby_title, "provider": None}
        enriched.append(RankedGame(game.current_rank, game.url, detail["title"] or game.lobby_title, detail["provider"]))
    return enriched


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _normalized(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _performance_probability(row: dict[str, Any]) -> float:
    direct = row.get("probability_high_performance")
    if direct is not None:
        return max(0.0, min(1.0, _number(direct)))
    present = [
        _number(row.get(key))
        for key in ("probability_high_client_performer", "probability_high_stakes_performer")
        if row.get(key) is not None
    ]
    return sum(present) / len(present) if present else 0.0


def _confidence(row: dict[str, Any]) -> float:
    present = [
        max(0.0, min(1.0, _number(row.get(key))))
        for key in ("confidence_clients", "confidence_stakes")
        if row.get(key) is not None
    ]
    return sum(present) / len(present) if present else 0.0


def score_forecasts(forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stakes = _normalized([_number(row.get("predicted_stakes")) for row in forecasts])
    clients = _normalized([_number(row.get("predicted_clients")) for row in forecasts])
    scored = []
    for row, normalized_stakes, normalized_clients in zip(forecasts, stakes, clients):
        probability = _performance_probability(row)
        confidence = _confidence(row)
        score = 100 * (0.35 * normalized_stakes + 0.25 * normalized_clients + 0.25 * probability + 0.15 * confidence)
        scored.append({**row, "placement_score": round(score, 2), "confidence": round(confidence, 4)})
    return sorted(scored, key=lambda row: (-row["placement_score"], str(row.get("Game_ID", ""))))


def _model_keys(row: dict[str, Any]) -> set[str]:
    game_id = _text(row.get("Game_ID"))
    suffix = game_id.split("::", 1)[-1]
    tournament = _text(row.get("TournamentName"))
    system = _text(row.get("SystemName"))
    keys = {normalize_game_name(tournament), normalize_game_name(suffix)}
    if system and tournament.lower().startswith(f"{system.lower()}-"):
        keys.add(normalize_game_name(tournament[len(system) + 1:]))
    return {key for key in keys if key}


def _lobby_keys(game: RankedGame) -> set[str]:
    slug = urlparse(game.url).path.rstrip("/").rsplit("/", 1)[-1]
    keys = {normalize_game_name(game.lobby_title), normalize_game_name(slug.replace("-", " "))}
    return {key for key in keys if key}


def match_model_game(game: RankedGame, forecasts: list[dict[str, Any]]) -> dict[str, Any] | None:
    keys = _lobby_keys(game)
    matches = [row for row in forecasts if keys & _model_keys(row)]
    return matches[0] if len(matches) == 1 else (matches[0] if matches else None)


def _recommendation(row: dict[str, Any], *, current_rank: int | None, lobby_title: str, provider: str | None) -> dict[str, Any]:
    recommended_rank = int(row["recommended_rank"])
    rank_delta = current_rank - recommended_rank if current_rank is not None else None
    if current_rank is None:
        action = "Promote into visible ranks" if recommended_rank <= MAX_VISIBLE_RANK else "Model-ranked"
    elif rank_delta is not None and rank_delta >= 3:
        action = "Move higher"
    elif rank_delta is not None and rank_delta <= -3:
        action = "Move lower"
    else:
        action = "Hold"
    return {
        "game_id": row.get("Game_ID"),
        "tournament_name": row.get("TournamentName"),
        "system_name": row.get("SystemName"),
        "lobby_title": lobby_title,
        "provider": provider,
        "current_rank": current_rank,
        "recommended_rank": recommended_rank,
        "rank_delta": rank_delta,
        "placement_action": action,
        "placement_score": row["placement_score"],
        "predicted_stakes": _number(row.get("predicted_stakes")),
        "predicted_clients": _number(row.get("predicted_clients")),
        "probability_high_performance": round(_performance_probability(row), 4),
        "confidence": row["confidence"],
    }


def build_placement_recommendations(
    forecasts: list[dict[str, Any]],
    ranked_games: list[RankedGame],
    *,
    source_url: str = SOURCE_URL,
    scraped_at: str | None = None,
) -> dict[str, Any]:
    scored = score_forecasts(forecasts)
    for index, row in enumerate(scored, start=1):
        row["recommended_rank"] = index

    recommendations: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    for game in ranked_games:
        match = match_model_game(game, scored)
        if match is None:
            recommendations.append({
                "game_id": None,
                "tournament_name": None,
                "system_name": None,
                "lobby_title": game.lobby_title,
                "provider": game.provider,
                "current_rank": game.current_rank,
                "recommended_rank": None,
                "rank_delta": None,
                "placement_action": "No model match",
                "placement_score": None,
                "predicted_stakes": None,
                "predicted_clients": None,
                "probability_high_performance": None,
                "confidence": None,
            })
            continue
        matched_ids.add(str(match.get("Game_ID")))
        recommendations.append(_recommendation(match, current_rank=game.current_rank, lobby_title=game.lobby_title, provider=game.provider))

    for row in scored[:MAX_VISIBLE_RANK]:
        if str(row.get("Game_ID")) not in matched_ids:
            recommendations.append(_recommendation(row, current_rank=None, lobby_title=row.get("TournamentName") or row.get("Game_ID") or "", provider=None))

    action_order = {"Move higher": 0, "Promote into visible ranks": 1, "Hold": 2, "Move lower": 3, "No model match": 4}
    recommendations.sort(key=lambda row: (action_order.get(row["placement_action"], 9), row["recommended_rank"] or 999, row["current_rank"] or 999))
    actions = [row["placement_action"] for row in recommendations]
    summary = {
        "scraped_games": len(ranked_games),
        "matched_games": sum(1 for action in actions if action != "No model match" and action != "Promote into visible ranks"),
        "move_higher": actions.count("Move higher"),
        "move_lower": actions.count("Move lower"),
        "hold": actions.count("Hold"),
        "promote": actions.count("Promote into visible ranks"),
        "unmatched": actions.count("No model match"),
    }
    return {
        "source_url": source_url,
        "scraped_at": scraped_at or datetime.now(UTC).isoformat(),
        "summary": summary,
        "recommendations": recommendations,
    }


def _load_forecasts(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "artifacts" / "dashboard" / "dashboard_data.json"
    if not path.is_file():
        raise FileNotFoundError("Dashboard data has not been exported yet. Score a workbook first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("Dashboard data does not contain any forecasted games.")
    return forecasts


def get_placement_recommendations(
    project_root: str | Path,
    *,
    refresh: bool = False,
    cache: dict[str, Any] | None = None,
    cache_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    cache = cache if cache is not None else {}
    cache_lock = cache_lock or threading.Lock()
    cache_key = str(project_root.resolve())
    now = datetime.now(UTC)
    with cache_lock:
        cached = cache.get(cache_key)
        if not refresh and cached and now - cached["created_at"] < CACHE_TTL:
            return cached["payload"]
    payload = build_placement_recommendations(_load_forecasts(project_root), scrape_ranked_games(), scraped_at=now.isoformat())
    with cache_lock:
        cache[cache_key] = {"created_at": now, "payload": payload}
    return payload
