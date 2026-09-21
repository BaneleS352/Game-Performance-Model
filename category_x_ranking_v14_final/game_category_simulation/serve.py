from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = WEB / "data"
DAYS = DATA / "days"
ASSETS = WEB / "assets"
HOST = "127.0.0.1"
PORT = 8000

ROW_KEYS = [
    "game_idx", "website_position", "score", "momentum", "acceleration_raw",
    "stability", "evidence_strength", "state", "previous_position",
    "movement", "movement_direction", "actual_rank", "actual_performance_score",
    "current_performance", "momentum_score", "acceleration_score",
    "Clients", "Betcount", "Stakes", "GGR", "data_status",
    "signal_as_of", "movement_reason", "persistence_days", "strong_performance_days", "movement_confidence", "movement_cap", "ideal_rank"
]

_cache: dict[str, object] = {}
_art_map: dict[str, str] = {}


def read_json(path: Path):
    key = str(path.resolve())
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[key]


def load_art_map() -> dict[str, str]:
    global _art_map
    if _art_map:
        return _art_map
    p = ASSETS / "game-art-map.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                gid, fn = line.split(":", 1)
                _art_map[gid.strip()] = fn.strip()
    return _art_map


def row_to_obj(row: list) -> dict:
    out = {ROW_KEYS[i]: row[i] if i < len(row) else None for i in range(len(ROW_KEYS))}
    return out


def load_day(date: str) -> dict:
    path = DAYS / f"{date}.json"
    data = read_json(path)
    rows = [row_to_obj(r) for r in data.get("rows", [])]
    signal_as_of = data.get("signal_as_of")
    for row in rows:
        if row.get("signal_as_of") is None:
            row["signal_as_of"] = signal_as_of
        # Never expose absent raw metrics as numeric zero.
        if row.get("data_status") != "Available":
            row["data_status"] = "No Data"
            for key in ("Clients", "Betcount", "Stakes", "GGR"):
                row[key] = None
        if not row.get("movement_reason"):
            row["movement_reason"] = "No data - position held" if row.get("data_status") == "No Data" else ("No movement - position held" if row.get("movement", 0) in (0, None) else "Moving toward ideal position")
    return {"date": data.get("date", date), "signal_as_of": signal_as_of, "rows": rows}


def game_obj(meta: dict, idx: int) -> dict:
    g = meta["games"][idx]
    return {
        "id": g.get("id", idx),
        "game_id": g.get("game_id", ""),
        "name": g.get("name", ""),
        "system": g.get("system", ""),
    }


class Handler(BaseHTTPRequestHandler):
    def send_body(self, code: int, content_type: str, body: bytes | str) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code: int, obj) -> None:
        self.send_body(
            code,
            "application/json; charset=utf-8",
            json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
        )

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            meta = read_json(DATA / "metadata.json")

            if path == "/api/bootstrap":
                art = load_art_map()
                games = []
                for g in meta["games"]:
                    x = dict(g)
                    fn = art.get(str(x.get("id")))
                    x["image"] = f"assets/game-icons/{fn}" if fn else None
                    games.append(x)
                return self.send_json(
                    200,
                    {"config": meta["config"], "dates": meta["dates"], "games": games},
                )

            if path == "/api/ranking":
                date = qs.get("date", [None])[0]
                if not date:
                    return self.send_json(400, {"error": "Missing date"})
                day = load_day(date)
                games = meta["games"]
                for row in day["rows"]:
                    idx = int(row["game_idx"])
                    g = games[idx]
                    row["name"] = g.get("name", "")
                    row["system"] = g.get("system", "")
                    fn = load_art_map().get(str(g.get("id")))
                    row["image"] = f"assets/game-icons/{fn}" if fn else None
                return self.send_json(200, day)

            if path == "/api/website":
                date = qs.get("date", [None])[0]
                if not date:
                    return self.send_json(400, {"error": "Missing date"})
                limit = max(1, min(300, int(qs.get("limit", ["300"])[0])))
                day = load_day(date)
                games = meta["games"]
                art = load_art_map()
                rows = sorted(day["rows"], key=lambda r: (int(r["website_position"]), int(r["game_idx"])))[:limit]
                cards = []
                for r in rows:
                    g = games[int(r["game_idx"])]
                    fn = art.get(str(g.get("id")))
                    cards.append({
                        "game_idx": r["game_idx"],
                        "website_position": r["website_position"],
                        "score": r["score"],
                        "movement": r["movement"],
                        "movement_direction": r["movement_direction"],
                        "momentum": r["momentum"],
                        "state": r["state"],
                        "name": g.get("name", ""),
                        "system": g.get("system", ""),
                        "image": f"assets/game-icons/{fn}" if fn else None,
                        "data_status": r.get("data_status", "No Data"),
                        "movement_reason": r.get("movement_reason"),
                    })
                return self.send_json(200, {"date": day["date"], "total_games": len(day["rows"]), "limit": limit, "cards": cards})

            if path == "/api/game":
                raw = qs.get("id", [None])[0]
                if raw is None:
                    return self.send_json(400, {"error": "Missing id"})
                idx = int(raw)
                if idx < 0 or idx >= len(meta["games"]):
                    return self.send_json(404, {"error": "Game not found"})
                series_all = read_json(DATA / "game_series.json")
                series = dict(series_all.get(str(idx)) or {})
                if not series:
                    return self.send_json(404, {"error": "Game history not found"})

                # Build a compact per-game signal history on demand. This is only
                # 30 rows for one inspected game, so it is inexpensive and lets
                # the UI explain whether today's signals actually changed.
                signal_history = []
                for day_name in meta.get("dates", []):
                    try:
                        day = load_day(day_name)
                    except FileNotFoundError:
                        continue
                    found = next((r for r in day["rows"] if int(r["game_idx"]) == idx), None)
                    if found is None:
                        continue
                    signal_history.append({
                        "date": day["date"],
                        "website_position": found.get("website_position"),
                        "score": found.get("score"),
                        "current_performance": found.get("current_performance"),
                        "momentum": found.get("momentum"),
                        "acceleration": found.get("acceleration_raw"),
                        "stability": found.get("stability"),
                        "movement": found.get("movement"),
                        "state": found.get("state"),
                        "data_status": found.get("data_status"),
                        "movement_reason": found.get("movement_reason"),
                    })
                series["signal_history"] = signal_history

                fn = load_art_map().get(str(meta["games"][idx].get("id")))
                g = game_obj(meta, idx)
                g["image"] = f"assets/game-icons/{fn}" if fn else None
                return self.send_json(200, {"game": g, "series": series})

            if path == "/api/evaluation":
                return self.send_json(200, read_json(DATA / "evaluation.json"))

            if path == "/api/status":
                p = DATA / "pipeline_status.json"
                status = read_json(p) if p.exists() else {"status": "unknown", "message": "Pipeline status file not available."}
                return self.send_json(200, status)

            rel = path.lstrip("/") or "index.html"
            target = (WEB / rel).resolve()
            base = WEB.resolve()
            if base not in target.parents and target != base:
                return self.send_json(403, {"error": "Forbidden"})
            if target.is_file():
                return self.send_body(
                    200,
                    mimetypes.guess_type(str(target))[0] or "application/octet-stream",
                    target.read_bytes(),
                )
            return self.send_json(404, {"error": "Not found"})
        except FileNotFoundError as exc:
            return self.send_json(404, {"error": f"Not found: {exc}"})
        except (ValueError, TypeError) as exc:
            return self.send_json(400, {"error": f"Bad request: {exc}"})
        except Exception as exc:  # pragma: no cover
            return self.send_json(500, {"error": f"Server error: {type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main() -> None:
    required = [DATA / "metadata.json", DATA / "evaluation.json", DAYS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Simulation results missing: " + ", ".join(missing))

    meta = read_json(DATA / "metadata.json")
    print(f"Loaded simulation metadata: {len(meta['games']):,} unique games, {len(meta['dates'])} evaluation days", flush=True)
    print(f"Serving Category X simulation at http://{HOST}:{PORT}", flush=True)
    print("Dashboard and website views use the same daily website_position data.", flush=True)
    print("Website view loads only the top 300 cards for the selected date.", flush=True)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
