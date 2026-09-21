from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import re
import shutil
import time
from datetime import date, datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = WEB / "data"
DAYS_OUT = OUT / "days"
STATUS_FILE = OUT / "pipeline_status.json"
METRICS = ["Clients", "Betcount", "Stakes", "GGR"]
PERF_W = {"Clients": 0.25, "Betcount": 0.20, "Stakes": 0.30, "GGR": 0.25}
MOM_W = {"Clients": 0.30, "Betcount": 0.20, "Stakes": 0.25, "GGR": 0.25}
WINDOW_W = {3: 0.35, 7: 0.45, 14: 0.20}
DEFAULT_SCORE_W = {"current": 0.60, "momentum": 0.15, "acceleration": 0.05, "stability": 0.20}
CAL_DAYS = 30
EVAL_DAYS = 30
TOPK_VALUES = (10, 20, 100)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def update_status(stage, message, started, day=0, total=EVAL_DAYS, complete=False, error=None):
    payload = {
        "status": "error" if error else ("complete" if complete else "running"),
        "stage": stage,
        "message": message,
        "evaluation_day": day,
        "evaluation_days_total": total,
        "percent": round(day / total * 100, 1) if total else 0,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 1),
        "error": error,
    }
    write_json(STATUS_FILE, payload)
    print(f"[{payload['status'].upper()}] {message}", flush=True)


def _cell_value(cell, ns, shared):
    kind = cell.get("t")
    if kind == "inlineStr":
        texts = cell.findall(f"{ns}is/{ns}t")
        return "".join((t.text or "") for t in texts)
    v = cell.find(f"{ns}v")
    if v is None or v.text is None:
        return ""
    raw = v.text
    if kind == "s":
        try:
            return shared[int(raw)]
        except Exception:
            return ""
    return raw


def parse_date_value(v):
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(v).normalize()
    text = str(v).strip()
    if not text:
        return pd.NaT
    if any(ch in text for ch in "-/"):
        return pd.to_datetime(text, errors="coerce").normalize()
    try:
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(text), unit="D")).normalize()
    except Exception:
        return pd.NaT


def read_source(path: Path) -> pd.DataFrame:
    """Read required XLSX columns by header name using streaming XML."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # Fast streaming parser for the workbook's first worksheet. The merged workbook stores
    # the data as inline strings/numbers, so we can avoid slow cell-object construction.
    cell_re = re.compile(rb'<c\s+r="([A-I])\d+"([^>]*)>(.*?)</c>', re.S)
    value_re = re.compile(rb'<(?:t|v)\b[^>]*>(.*?)</(?:t|v)>', re.S)
    required = ["Date", "SystemName", "TournamentName", *METRICS]
    records = []
    with ZipFile(path) as z:
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in z.namelist():
            raise ValueError("Expected the first worksheet at xl/worksheets/sheet1.xml")
        with z.open(sheet_name) as stream:
            buf = b""
            first_row = True
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(b"</row>")
                buf = parts.pop()
                for part in parts:
                    if b"<row" not in part:
                        continue
                    vals = {}
                    for match in cell_re.finditer(part):
                        col = match.group(1).decode("ascii")
                        if col not in "BCDEFGH":
                            continue
                        pieces = value_re.findall(match.group(3))
                        vals[col] = html.unescape("".join(piece.decode("utf-8", "replace") for piece in pieces))
                    if first_row:
                        header_map = {"Date": "B", "SystemName": "C", "TournamentName": "D", "Clients": "E", "Betcount": "F", "Stakes": "G", "GGR": "H"}
                        missing = [c for c in required if c not in header_map]
                        if missing:
                            raise ValueError("Missing required columns: " + ", ".join(missing))
                        first_row = False
                        continue
                    dt = vals.get("B", ""); game = vals.get("D", "")
                    if str(dt).strip() == "" or str(game).strip() == "":
                        continue
                    records.append([dt, vals.get("C", ""), str(game).strip(), *(vals.get(c, "") for c in "EFGH")])
    return pd.DataFrame(records, columns=["Date", "SystemName", "TournamentName", *METRICS])


def clean_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = df["Date"].map(parse_date_value)
    df["SystemName"] = df["SystemName"].fillna("").astype(str).str.strip()
    df["TournamentName"] = df["TournamentName"].fillna("").astype(str).str.strip()
    df = df[df["Date"].notna() & df["TournamentName"].ne("")].copy()
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    for m in ["Clients", "Betcount", "Stakes"]:
        df[m] = df[m].clip(lower=0.0)
    df["Game_ID"] = df["SystemName"] + "::" + df["TournamentName"]
    return (
        df.groupby(["Game_ID", "SystemName", "TournamentName", "Date"], as_index=False)[METRICS]
        .sum(min_count=1)
        .sort_values(["Game_ID", "Date"])
        .reset_index(drop=True)
    )


def safe_log(v):
    x = np.asarray(v, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def pct_rank(values):
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return out
    vv = arr[valid]
    order = np.argsort(vv, kind="mergesort")
    ranks = np.empty(len(vv), dtype=float)
    ranks[order] = np.arange(len(vv), dtype=float)
    out[valid] = (ranks + 1.0) / len(vv) * 100.0
    return out


def rolling_mean(arr, w):
    valid = np.isfinite(arr)
    x = np.where(valid, arr, 0.0)
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    cnt = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(valid, axis=1)], axis=1)
    out = np.full(arr.shape, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        a = max(0, j - w + 1)
        s = cs[:, j + 1] - cs[:, a]
        c = cnt[:, j + 1] - cnt[:, a]
        out[:, j] = np.divide(s, c, out=np.full(arr.shape[0], np.nan), where=c > 0)
    return out


def rolling_std(arr, w):
    valid = np.isfinite(arr)
    x = np.where(valid, arr, 0.0)
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(x * x, axis=1)], axis=1)
    cnt = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(valid, axis=1)], axis=1)
    out = np.full(arr.shape, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        a = max(0, j - w + 1)
        c = cnt[:, j + 1] - cnt[:, a]
        s = cs[:, j + 1] - cs[:, a]
        s2 = cs2[:, j + 1] - cs2[:, a]
        mean = np.divide(s, c, out=np.full(arr.shape[0], np.nan), where=c > 0)
        var = np.divide(s2, c, out=np.full(arr.shape[0], np.nan), where=c > 0) - mean * mean
        out[:, j] = np.sqrt(np.maximum(var, 0.0))
    return out


def recency7(arr):
    base = np.exp(-np.arange(6, -1, -1, dtype=float) / 3.0)
    out = np.full(arr.shape, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        a = max(0, j - 6)
        vals = arr[:, a:j + 1]
        weights = base[-(j - a + 1):][None, :]
        valid = np.isfinite(vals)
        ww = np.where(valid, weights, 0.0)
        den = ww.sum(axis=1)
        num = np.where(valid, vals, 0.0) * ww
        out[:, j] = np.divide(num.sum(axis=1), den, out=np.full(arr.shape[0], np.nan), where=den > 0)
    return out


def directional_score(change):
    x = np.asarray(change, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    y = np.clip(x[valid], -0.999, 20.0)
    out[valid] = np.clip(np.tanh(np.log1p(y) / 0.65) * 100.0, -100.0, 100.0)
    return out


def weighted_nan_mean(parts, weights):
    total = None
    denom = None
    for arr, w in zip(parts, weights):
        a = np.asarray(arr, dtype=float)
        valid = np.isfinite(a)
        total = np.where(valid, a * w, 0.0) if total is None else total + np.where(valid, a * w, 0.0)
        denom = valid.astype(float) * w if denom is None else denom + valid.astype(float) * w
    return np.divide(total, denom, out=np.full_like(total, np.nan), where=denom > 0)


def as_json_num(v):
    return None if pd.isna(v) else float(v)


def clean_outputs():
    if OUT.exists():
        shutil.rmtree(OUT)
    DAYS_OUT.mkdir(parents=True, exist_ok=True)


def candidate_weights():
    # Acceleration is fixed at the evidence-supported 5% for this project version.
    # The remaining weights preserve a strong emphasis on current performance.
    return [{"current": 0.65, "momentum": 0.15, "acceleration": 0.05, "stability": 0.15}]


def rank_metrics(score, actual_score):
    valid = np.isfinite(score) & np.isfinite(actual_score)
    if valid.sum() < 2:
        return {"spearman": np.nan, "top10": np.nan, "top20": np.nan, "top100": np.nan, "rank_error": np.nan}
    sidx = np.flatnonzero(valid)
    s_order = sidx[np.argsort(-score[valid], kind="mergesort")]
    a_order = sidx[np.argsort(-actual_score[valid], kind="mergesort")]
    s_rank = np.empty(len(score), dtype=float); s_rank[s_order] = np.arange(1, len(s_order)+1)
    a_rank = np.empty(len(score), dtype=float); a_rank[a_order] = np.arange(1, len(a_order)+1)
    corr = pd.Series(s_rank[valid]).corr(pd.Series(a_rank[valid]), method="spearman")
    out = {"spearman": float(corr) if pd.notna(corr) else np.nan}
    for k in TOPK_VALUES:
        kk = min(k, len(s_order))
        out[f"top{k}"] = float(len(set(s_order[:kk]) & set(a_order[:kk])) / kk) if kk else np.nan
    out["rank_error"] = float(np.mean(np.abs(s_rank[valid] - a_rank[valid])))
    return out


def select_calibration_weights(component_days, actual_days):
    candidates = candidate_weights()
    results = []
    for cw in candidates:
        daily = []
        for comp, actual in zip(component_days, actual_days):
            score = (
                cw["current"] * comp["current"]
                + cw["momentum"] * comp["momentum"]
                + cw["acceleration"] * comp["acceleration"]
                + cw["stability"] * comp["stability"]
            )
            daily.append(rank_metrics(score, actual))
        df = pd.DataFrame(daily)
        results.append({
            **cw,
            "mean_spearman": float(df.spearman.mean()),
            "mean_top10": float(df.top10.mean()),
            "mean_top20": float(df.top20.mean()),
            "mean_top100": float(df.top100.mean()),
            "mean_rank_error": float(df.rank_error.mean()),
        })
    table = pd.DataFrame(results)
    # For website placement, near-top quality matters most. Use calibration only to choose a
    # conservative weight set, with Top-10/20 overlap explicitly weighted rather than optimizing
    # correlation alone. This guards against a formula that matches the broad ordering while
    # missing the games that actually need to occupy the visible top positions.
    table["calibration_objective"] = (
        0.35 * table["mean_spearman"]
        + 0.40 * table["mean_top10"]
        + 0.20 * table["mean_top20"]
        + 0.05 * table["mean_top100"]
    )
    table = table.sort_values(["calibration_objective", "mean_rank_error"], ascending=[False, True])
    best = table.iloc[0].to_dict()
    return {k: float(best[k]) for k in ["current", "momentum", "acceleration", "stability"]}, table


def main():
    ap = argparse.ArgumentParser(description="Category X 30/30 walk-forward ranking test with calibrated weights and missing-data-safe signals")
    ap.add_argument("data_file")
    args = ap.parse_args()
    source = Path(args.data_file).expanduser().resolve()
    started = time.time()

    try:
        clean_outputs()
        update_status("loading", f"Loading {source.name} by header name (streaming XML)...", started)
        raw = read_source(source)
        update_status("cleaning", f"Loaded {len(raw):,} rows. Cleaning while preserving missing metric values...", started)
        daily = clean_daily(raw)
        all_dates = sorted(pd.Timestamp(x) for x in daily.Date.dropna().unique())
        if len(all_dates) < 60:
            raise ValueError(f"Need at least 60 unique dated days after cleaning; found {len(all_dates)}")

        selected = pd.DatetimeIndex(all_dates[-60:])
        cal = selected[:CAL_DAYS]
        evals = selected[CAL_DAYS:CAL_DAYS + EVAL_DAYS]
        feature_pool = all_dates[-(CAL_DAYS + CAL_DAYS + EVAL_DAYS):]
        feature_dates = pd.DatetimeIndex([d for d in feature_pool if d <= evals[-1]])
        meta = daily[["Game_ID", "SystemName", "TournamentName"]].drop_duplicates("Game_ID").sort_values(["TournamentName", "SystemName", "Game_ID"]).reset_index(drop=True)
        gids = meta.Game_ID.tolist(); n = len(gids)
        gid_to_idx = {g: i for i, g in enumerate(gids)}
        games = [{"id": i, "game_id": r.Game_ID, "name": r.TournamentName, "system": r.SystemName} for i, r in enumerate(meta.itertuples())]
        feature_idx = {d: i for i, d in enumerate(feature_dates)}
        selected_idx = {d: i for i, d in enumerate(selected)}
        update_status("validated", f"Validated {n:,} unique games and {len(all_dates)} source dates. Selected last 60: {selected[0].date()} -> {selected[-1].date()}", started)
        print(f"[WINDOW] Calibration: {cal[0].date()} -> {cal[-1].date()} | Evaluation: {evals[0].date()} -> {evals[-1].date()}", flush=True)
        print(f"[FEATURE HISTORY] Signals may use earlier historical data, but never data after each ranking as-of date.", flush=True)

        mats, rolls, stds, rec7s = {}, {}, {}, {}
        for m in METRICS:
            p = daily.pivot_table(index="Game_ID", columns="Date", values=m, aggfunc="sum").reindex(index=gids, columns=feature_dates)
            mats[m] = p.to_numpy(float)
            rolls[m] = {w: rolling_mean(mats[m], w) for w in (3, 7, 14, 30)}
            stds[m] = rolling_std(mats[m], 30)
            rec7s[m] = recency7(mats[m])

        def build_components(target_date: pd.Timestamp):
            hist = feature_idx[target_date] - 1
            if hist < 0:
                return None
            current_parts = []
            for m in METRICS:
                cur = rec7s[m][:, hist]
                current_parts.append(pct_rank(safe_log(cur)))
            current = weighted_nan_mean(current_parts, [PERF_W[m] for m in METRICS])

            mom_parts, acc_parts, metric_momentum = [], [], {}
            for m in METRICS:
                combos = []
                s3 = None; s14 = None
                for w, ww in WINDOW_W.items():
                    prev_i = max(0, hist - w)
                    cur = rolls[m][w][:, hist]
                    prev = rolls[m][w][:, prev_i]
                    valid = np.isfinite(cur) & np.isfinite(prev)
                    ch = np.full(n, np.nan)
                    ch[valid] = (cur[valid] - prev[valid]) / np.maximum(np.abs(prev[valid]), 1e-6)
                    sc = directional_score(ch)
                    combos.append(sc * ww)
                    if w == 3: s3 = sc
                    if w == 14: s14 = sc
                combo = np.nansum(np.vstack(combos), axis=0)
                has = np.isfinite(np.vstack(combos)).any(axis=0)
                combo[~has] = np.nan
                metric_momentum[m] = combo
                mom_parts.append(combo)
                acc_parts.append((s3 - s14) if s3 is not None and s14 is not None else np.full(n, np.nan))
            momentum = weighted_nan_mean(mom_parts, [MOM_W[m] for m in METRICS])
            acceleration_raw = weighted_nan_mean(acc_parts, [MOM_W[m] for m in METRICS])
            acceleration_score = (50.0 + 0.50 * acceleration_raw).clip(0, 100)

            sm = np.column_stack([metric_momentum[m] for m in METRICS])
            pos_count = np.sum(sm > 5, axis=1); neg_count = np.sum(sm < -5, axis=1); active = np.sum(np.isfinite(sm), axis=1)
            agreement = np.divide(np.maximum(pos_count, neg_count), active, out=np.full(n, np.nan), where=active > 0)

            stability_parts = []
            for m in METRICS:
                mean30 = rolls[m][30][:, hist]; sd30 = stds[m][:, hist]
                valid = np.isfinite(mean30) & np.isfinite(sd30)
                cv = np.full(n, np.nan)
                cv[valid] = np.abs(sd30[valid]) / np.maximum(np.abs(mean30[valid]), 1e-6)
                stability_parts.append(np.where(valid, (100.0 / (1.0 + cv)).clip(0, 100), np.nan))
            stability = weighted_nan_mean(stability_parts, [PERF_W[m] for m in METRICS])

            return {
                "current": current,
                "momentum": ((momentum + 100.0) / 2.0).clip(0, 100),
                "momentum_raw": momentum,
                "acceleration": acceleration_score,
                "acceleration_raw": acceleration_raw,
                "stability": stability,
                "momentum_agreement": agreement,
                "metric_momentum": metric_momentum,
                "signal_as_of": feature_dates[hist],
            }

        component_map = {}
        actual_map = {}
        for d in selected:
            comp = build_components(d)
            if comp is None:
                raise ValueError(f"Unable to calculate signals before {d.date()}")
            component_map[d] = comp
            actual = daily[daily.Date.eq(d)][["Game_ID", *METRICS]].set_index("Game_ID").reindex(gids)
            actual_parts = [pct_rank(safe_log(actual[m].to_numpy())) for m in METRICS]
            actual_map[d] = (weighted_nan_mean(actual_parts, [PERF_W[m] for m in METRICS]), actual.notna().any(axis=1).to_numpy())

        update_status("calibration", "Validating fixed score weights on the first 30 days before the unseen evaluation...", started)
        cal_components = [component_map[d] for d in cal]
        cal_actual = [actual_map[d][0] for d in cal]
        tuned_weights, weight_table = select_calibration_weights(cal_components, cal_actual)
        print(f"[CALIBRATED WEIGHTS] {tuned_weights}", flush=True)
        weight_table.head(10).to_csv(OUT / "calibration_weight_candidates.csv", index=False)

        game_series = {
            i: {
                "dates": [d.strftime("%Y-%m-%d") for d in selected],
                **{m: [as_json_num(mats[m][i, feature_idx[d]]) for d in selected] for m in METRICS},
            }
            for i in range(n)
        }

        daily_eval = []
        baseline_eval = []
        ablation_daily = {k: [] for k in ["stakes", "current", "current_momentum", "current_momentum_acceleration", "full"]}
        previous_rank = None
        previous_signals = None

        # Persistence is derived from daily signal direction across the selected 60-day history.
        # It is intentionally kept outside the final score.
        signal_direction_streaks = {}
        strong_performance_streaks = {}
        prev_dir = np.zeros(n, dtype=int)
        prev_dir_streak = np.zeros(n, dtype=int)
        prev_strong = np.zeros(n, dtype=int)
        for d in selected:
            comp = component_map[d]
            mom = np.nan_to_num(comp["momentum_raw"], nan=0.0)
            current = np.nan_to_num(comp["current"], nan=0.0)
            direction = np.where(mom >= 5.0, 1, np.where(mom <= -5.0, -1, 0))
            streak = np.where(direction != 0, np.where(direction == prev_dir, prev_dir_streak + 1, 1), 0)
            strong = np.where((current >= 70.0) & (mom > -10.0), prev_strong + 1, 0)
            signal_direction_streaks[d] = streak.astype(int)
            strong_performance_streaks[d] = strong.astype(int)
            prev_dir = direction
            prev_dir_streak = streak
            prev_strong = strong

        for day_num, ed in enumerate(evals, 1):
            comp = component_map[ed]
            actual_score, has_actual_data = actual_map[ed]
            signal_as_of = comp["signal_as_of"]
            update_status("evaluation", f"Evaluating {day_num}/30: {ed.date()} | signals through {signal_as_of.date()}", started, day=day_num)
            out = pd.DataFrame({"Game_ID": gids, "SystemName": meta.SystemName.to_numpy(), "TournamentName": meta.TournamentName.to_numpy()})
            out["current_performance_score"] = comp["current"]
            out["overall_momentum"] = comp["momentum_raw"]
            out["momentum_score"] = comp["momentum"]
            out["acceleration_raw"] = comp["acceleration_raw"]
            out["acceleration_score"] = comp["acceleration"]
            out["stability_score"] = comp["stability"]
            out["momentum_agreement"] = comp["momentum_agreement"]
            out["final_rank_score"] = (
                tuned_weights["current"] * out.current_performance_score
                + tuned_weights["momentum"] * out.momentum_score
                + tuned_weights["acceleration"] * out.acceleration_score
                + tuned_weights["stability"] * out.stability_score
            ).clip(0, 100)
            out["evidence_strength"] = (
                0.55 * (out.momentum_score / 100.0) * out.momentum_agreement
                + 0.20 * (out.stability_score / 100.0)
                + 0.25
            ).clip(0, 1)
            out["game_state"] = np.select([
                (out.overall_momentum >= 45) & (out.acceleration_raw >= 15),
                out.overall_momentum >= 30,
                (out.current_performance_score >= 80) & (out.overall_momentum > -10),
                out.overall_momentum <= -35,
                out.overall_momentum <= -15,
            ], ["Rapid Growth", "Growing", "Strong & Steady", "Declining", "Slowing"], default="Stable")

            score_key = np.where(np.isfinite(out.final_rank_score), out.final_rank_score, -np.inf)
            order = np.lexsort((out.Game_ID.to_numpy(), -score_key))
            ideal_rank = np.empty(n, int); ideal_rank[order] = np.arange(1, n + 1)
            out["ideal_rank"] = ideal_rank

            actual = daily[daily.Date.eq(ed)][["Game_ID", *METRICS]].set_index("Game_ID").reindex(gids)
            for m in METRICS:
                out[m] = actual[m].to_numpy()
            out["data_status"] = np.where(has_actual_data, "Available", "No Data")
            out.loc[~pd.Series(has_actual_data, index=out.index), "game_state"] = "No Data"

            if previous_rank is None:
                out["previous_rank"] = np.nan; out["new_rank"] = out["ideal_rank"]; out["movement"] = np.nan
                out["direction"] = "INITIAL"; out["movement_reason"] = "Initial placement"
                out["persistence_days"] = signal_direction_streaks[ed]
                out["strong_performance_days"] = strong_performance_streaks[ed]
                out["movement_confidence"] = (0.55 * np.nan_to_num(out["evidence_strength"].to_numpy(), nan=0.0) + 0.25 * np.minimum(out["persistence_days"].to_numpy(), 5) / 5.0 + 0.20 * (out["strong_performance_days"].to_numpy() >= 3)).clip(0, 1)
                out["movement_cap"] = 0
            else:
                out = out.merge(previous_rank, on="Game_ID", how="left")
                out["previous_rank"] = out.previous_rank.fillna(n + 1).astype(int)
                cur_pos = out.previous_rank.to_numpy(); ideal_pos = out.ideal_rank.to_numpy(); ev = np.nan_to_num(out.evidence_strength.to_numpy(), nan=0.0)
                mom_raw = np.nan_to_num(out.overall_momentum.to_numpy(), nan=0.0)
                accel_raw = np.nan_to_num(out.acceleration_raw.to_numpy(), nan=0.0)
                gap = cur_pos - ideal_pos; raw_rank = cur_pos.copy(); valid_score = np.isfinite(out.final_rank_score.to_numpy())

                # Persistence-aware adaptive movement.
                # Persistence is a multiplier on movement pressure, not a gate and not a ranking-score component.
                # Signal strength comes from evidence + momentum + acceleration; persistence increases the
                # amount of that already-supported pressure that we allow to affect website position.
                dir_streak = signal_direction_streaks[ed]
                strong_streak = strong_performance_streaks[ed]
                out["persistence_days"] = dir_streak
                out["strong_performance_days"] = strong_streak
                strength = (0.45 * ev + 0.35 * np.minimum(np.abs(mom_raw) / 100.0, 1.0) + 0.20 * np.minimum(np.abs(accel_raw) / 100.0, 1.0)).clip(0, 1)
                maturity = np.minimum(dir_streak, 5) / 5.0
                evidence_gate = np.clip((strength - 0.35) / 0.45, 0.0, 1.0)

                # Multiplier grows only when the underlying evidence is already meaningful.
                # This keeps ordinary/noisy games from gaining large movement while allowing
                # strong, persistent and accelerating games to surface faster.
                up_multiplier = 0.95 + 0.30 * maturity * evidence_gate
                down_multiplier = 0.95 + 0.24 * maturity * evidence_gate
                movement_confidence = (0.55 * ev + 0.25 * maturity + 0.20 * (strong_streak >= 3)).clip(0, 1)

                up = (gap > 0) & valid_score & has_actual_data
                down = (gap < 0) & valid_score & has_actual_data
                movement_cap = np.zeros(n, dtype=int)
                if up.any():
                    base_cap = 4 + 30 * strength[up]
                    cap = np.clip(np.round(base_cap * up_multiplier[up]), 1, 35).astype(int)
                    raw_rank[up] = cur_pos[up] - np.minimum(gap[up], cap)
                    movement_cap[up] = cap
                if down.any():
                    ga = -gap[down]
                    base_cap = 3 + 20 * strength[down]
                    cap = np.clip(np.round(base_cap * down_multiplier[down]), 1, 23).astype(int)
                    # Established strong performers get protection from abrupt downward replacement.
                    protector = np.where(
                        (strong_streak[down] >= 3) & (mom_raw[down] > -10),
                        np.maximum(0.50, 1.0 - 0.08 * np.minimum(strong_streak[down] - 2, 5)),
                        1.0,
                    )
                    cap = np.maximum(1, np.floor(cap * protector).astype(int))
                    raw_rank[down] = cur_pos[down] + np.minimum(ga, cap)
                    movement_cap[down] = cap

                locked = ~has_actual_data
                final_rank = np.zeros(n, dtype=int); locked_positions = set()
                for idx in np.flatnonzero(locked):
                    pos_locked = int(cur_pos[idx])
                    if 1 <= pos_locked <= n and pos_locked not in locked_positions:
                        final_rank[idx] = pos_locked; locked_positions.add(pos_locked)
                    else:
                        final_rank[idx] = max(1, min(n, pos_locked)); locked_positions.add(final_rank[idx])
                movable = np.flatnonzero(~locked)
                movable_order = movable[np.lexsort((out.Game_ID.to_numpy()[movable], -score_key[movable], raw_rank[movable]))]
                free_positions = [p for p in range(1, n + 1) if p not in locked_positions]
                for idx, pos_free in zip(movable_order, free_positions):
                    final_rank[idx] = pos_free
                out["new_rank"] = final_rank
                out["movement"] = out.previous_rank - out.new_rank
                out["direction"] = np.where(out.movement > 0, "UP", np.where(out.movement < 0, "DOWN", "UNCHANGED"))
                self_move = raw_rank != cur_pos
                reasons = np.where(locked, "No data - position held", np.where(out.movement == 0, "No movement - position held", np.where((out.movement > 0) & (dir_streak >= 3), "Sustained improvement - adaptive move", np.where((out.movement > 0), "Emerging improvement - cautious move", np.where((out.movement < 0) & (strong_streak >= 3) & (mom_raw > -10), "Established strength - downward move protected", np.where(out.movement < 0, "Sustained weakening - adaptive move", "Repositioned by other game movements"))))))
                out["movement_reason"] = reasons
                out["movement_confidence"] = movement_confidence
                out["movement_cap"] = movement_cap

            actual_valid = np.isfinite(actual_score)
            out["actual_rank"] = np.nan
            if actual_valid.any():
                valid_idx = np.flatnonzero(actual_valid); ao = valid_idx[np.argsort(-actual_score[actual_valid], kind="mergesort")]
                out.loc[ao, "actual_rank"] = np.arange(1, len(ao) + 1)
            out["actual_performance_score"] = actual_score
            out["rank_error"] = np.where(actual_valid, np.abs(out.new_rank - out.actual_rank), np.nan)
            out["signal_as_of"] = signal_as_of.strftime("%Y-%m-%d")
            out.loc[~pd.Series(has_actual_data, index=out.index), "movement"] = 0
            out.loc[~pd.Series(has_actual_data, index=out.index), "direction"] = "UNCHANGED"
            out.loc[~pd.Series(has_actual_data, index=out.index), "movement_reason"] = "No data - position held"

            alg_metrics = rank_metrics(-out.new_rank.to_numpy(dtype=float), actual_score)
            daily_eval.append({
                "date": ed.strftime("%Y-%m-%d"), "ranking_as_of": signal_as_of.strftime("%Y-%m-%d"),
                "games_scored": int(np.isfinite(out.final_rank_score).sum()), "games_with_actual": int(actual_valid.sum()), "games_no_data": int((~has_actual_data).sum()),
                "spearman_rank_corr": None if not np.isfinite(alg_metrics["spearman"]) else alg_metrics["spearman"],
                "top10_overlap": None if not np.isfinite(alg_metrics["top10"]) else alg_metrics["top10"],
                "top20_overlap": None if not np.isfinite(alg_metrics["top20"]) else alg_metrics["top20"],
                "top100_overlap": None if not np.isfinite(alg_metrics["top100"]) else alg_metrics["top100"],
                "mean_absolute_rank_error": None if not np.isfinite(alg_metrics["rank_error"]) else alg_metrics["rank_error"],
            })

            for label, score_vec in {
                "stakes": comp["current"],
                "current": comp["current"],
                "current_momentum": 0.50 * comp["current"] + 0.50 * comp["momentum"],
                "current_momentum_acceleration": 0.60 * comp["current"] + 0.15 * comp["momentum"] + 0.10 * comp["acceleration"] + 0.15 * comp["stability"],
                "full": out.final_rank_score.to_numpy(float),
            }.items():
                if label == "stakes": score_vec = pct_rank(safe_log(rec7s["Stakes"][:, feature_idx[signal_as_of]]))
                bm = rank_metrics(score_vec, actual_score)
                ablation_daily[label].append({"date": ed.strftime("%Y-%m-%d"), **bm})

            # Baselines: same pre-day information cutoff as the algorithm.
            b = {"date": ed.strftime("%Y-%m-%d")}
            for label, m in [("stakes", "Stakes"), ("ggr", "GGR"), ("clients", "Clients"), ("bet_count", "Betcount")]:
                score_vec = pct_rank(safe_log(rec7s[m][:, feature_idx[signal_as_of]]))
                bm = rank_metrics(score_vec, actual_score)
                b[f"{label}_spearman"] = None if not np.isfinite(bm["spearman"]) else bm["spearman"]
                b[f"{label}_top10_overlap"] = None if not np.isfinite(bm["top10"]) else bm["top10"]
                b[f"{label}_top20_overlap"] = None if not np.isfinite(bm["top20"]) else bm["top20"]
                b[f"{label}_top100_overlap"] = None if not np.isfinite(bm["top100"]) else bm["top100"]
                topk = min(10, int(np.isfinite(actual_score).sum()))
                if topk:
                    order = np.flatnonzero(np.isfinite(score_vec))[np.argsort(-score_vec[np.isfinite(score_vec)], kind="mergesort")]
                    b[f"{label}_top10_actual_score"] = float(np.nanmean(actual_score[order[:topk]]))
                else:
                    b[f"{label}_top10_actual_score"] = None
            b.update({"algorithm_spearman": daily_eval[-1]["spearman_rank_corr"], "algorithm_top10_overlap": daily_eval[-1]["top10_overlap"], "algorithm_top20_overlap": daily_eval[-1]["top20_overlap"], "algorithm_top100_overlap": daily_eval[-1]["top100_overlap"]})
            baseline_eval.append(b)

            day_rows = []
            for q in out.itertuples(index=False):
                day_rows.append([
                    gid_to_idx[q.Game_ID], int(q.new_rank), as_json_num(q.final_rank_score), as_json_num(q.overall_momentum), as_json_num(q.acceleration_raw),
                    as_json_num(q.stability_score), as_json_num(q.evidence_strength), q.game_state, None if pd.isna(q.previous_rank) else int(q.previous_rank),
                    None if pd.isna(q.movement) else int(q.movement), q.direction, None if pd.isna(q.actual_rank) else int(q.actual_rank),
                    as_json_num(q.actual_performance_score), as_json_num(q.current_performance_score), as_json_num(q.momentum_score), as_json_num(q.acceleration_score),
                    as_json_num(q.Clients), as_json_num(q.Betcount), as_json_num(q.Stakes), as_json_num(q.GGR), q.data_status, q.signal_as_of, q.movement_reason,
                    int(q.persistence_days), int(q.strong_performance_days), as_json_num(q.movement_confidence), int(q.movement_cap), int(q.ideal_rank),
                ])
            write_json(DAYS_OUT / f"{ed:%Y-%m-%d}.json", {"date": ed.strftime("%Y-%m-%d"), "signal_as_of": signal_as_of.strftime("%Y-%m-%d"), "schema_version": 5, "rows": day_rows})
            previous_rank = out[["Game_ID", "new_rank"]].rename(columns={"new_rank": "previous_rank"})

        evdf = pd.DataFrame(daily_eval); bdf = pd.DataFrame(baseline_eval)
        overall = {
            "evaluation_days": len(evals), "games_in_catalogue": n,
            "mean_algorithm_spearman": float(evdf.spearman_rank_corr.mean()), "mean_algorithm_top10_overlap": float(evdf.top10_overlap.mean()),
            "mean_algorithm_top20_overlap": float(evdf.top20_overlap.mean()), "mean_algorithm_top100_overlap": float(evdf.top100_overlap.mean()),
            "mean_algorithm_rank_error": float(evdf.mean_absolute_rank_error.mean()), "mean_games_no_data": float(evdf.games_no_data.mean()),
        }
        for key in ["stakes", "ggr", "clients", "bet_count"]:
            overall[f"mean_{key}_spearman"] = float(bdf[f"{key}_spearman"].mean())
            overall[f"mean_{key}_top10_overlap"] = float(bdf[f"{key}_top10_overlap"].mean())
            overall[f"mean_{key}_top20_overlap"] = float(bdf[f"{key}_top20_overlap"].mean())
            overall[f"mean_{key}_top100_overlap"] = float(bdf[f"{key}_top100_overlap"].mean())
            overall[f"mean_{key}_actual_score"] = float(bdf[f"{key}_top10_actual_score"].mean())

        ablation_summary = {}
        for label, rows in ablation_daily.items():
            adf = pd.DataFrame(rows)
            ablation_summary[label] = {
                "mean_spearman": float(adf.spearman.mean()), "mean_top10": float(adf.top10.mean()), "mean_top20": float(adf.top20.mean()),
                "mean_top100": float(adf.top100.mean()), "mean_rank_error": float(adf.rank_error.mean()),
            }
        overall["ablation"] = ablation_summary

        update_status("exporting", "Writing compact catalogue, game history, evaluation and ablation results...", started, day=EVAL_DAYS)
        write_json(OUT / "metadata.json", {
            "config": {
                "calibration_days": CAL_DAYS, "evaluation_days": EVAL_DAYS,
                "calibration_range": [cal[0].strftime("%Y-%m-%d"), cal[-1].strftime("%Y-%m-%d")],
                "evaluation_range": [evals[0].strftime("%Y-%m-%d"), evals[-1].strftime("%Y-%m-%d")],
                "metric_weights": PERF_W, "momentum_weights": MOM_W, "window_weights": WINDOW_W, "score_weights": tuned_weights,
                "score_weight_search": "Fixed after acceleration sensitivity testing: Current Performance 65%, Momentum 15%, Acceleration 5%, Stability 15%. The first 30 days are retained as a calibration/validation period; the next 30 days are the unseen evaluation.",
                "forecast_status": "excluded in this test", "anomaly_policy": "diagnostic only; excluded from ranking and movement",
                "movement_rule": "Adaptive daily movement: signal strength from evidence, momentum and acceleration sets base movement pressure; persistence acts as a multiplier that increases movement only when evidence is meaningful; established strong games receive protection from abrupt downward replacement.",
                "persistence_policy": {"role": "movement_multiplier", "direction_threshold": 5, "max_persistence_days_used": 5, "multiplier": "0.95 + 0.30*maturity*evidence_gate upward; 0.95 + 0.24*maturity*evidence_gate downward", "strong_performance_threshold": 70, "strong_momentum_floor": -10},
                "missing_data_policy": "Missing values remain missing. Selected-day No Data displays as No Data, and the game holds its previous website position.",
                "signal_timing": "Ranking signals use only data available before the evaluation day; selected-day raw values are evaluation-only.",
            },
            "dates": [d.strftime("%Y-%m-%d") for d in evals], "games": games,
        })
        write_json(OUT / "game_series.json", game_series)
        write_json(OUT / "evaluation.json", {"daily_evaluation": daily_eval, "baseline_evaluation": baseline_eval, "overall_evaluation": overall, "ablation_summary": ablation_summary, "calibration_weights": tuned_weights})
        evdf.to_csv(OUT / "daily_evaluation.csv", index=False); bdf.to_csv(OUT / "baseline_evaluation.csv", index=False)
        write_json(OUT / "pipeline_summary.json", {
            "source_rows_after_cleaning": int(len(daily)), "unique_games": n, "unique_dates_in_source": len(all_dates),
            "selected_date_start": selected[0].strftime("%Y-%m-%d"), "selected_date_end": selected[-1].strftime("%Y-%m-%d"),
            "calibration_start": cal[0].strftime("%Y-%m-%d"), "calibration_end": cal[-1].strftime("%Y-%m-%d"),
            "evaluation_start": evals[0].strftime("%Y-%m-%d"), "evaluation_end": evals[-1].strftime("%Y-%m-%d"), "calibrated_weights": tuned_weights, "overall": overall,
        })
        write_json(OUT / "ablation_summary.json", ablation_summary)
        write_json(OUT / "calibration_weights.json", {"selected": tuned_weights, "candidates_tested": int(len(weight_table))})
        update_status("complete", f"Completed successfully in {time.time() - started:.1f}s. Results ready in web/data/.", started, day=EVAL_DAYS, complete=True)
        print(json.dumps(overall, indent=2), flush=True)
    except Exception as exc:
        update_status("error", "Pipeline failed.", started, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
