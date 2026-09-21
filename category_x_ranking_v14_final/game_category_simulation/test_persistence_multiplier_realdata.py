from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "web" / "data"
DAYS = DATA / "days"
OUT = DATA / "persistence_multiplier_test"
TOPKS = (10, 20, 100)

# Current agreed scoring context. This test does not recalculate these signals.
ACCELERATION_WEIGHT = 0.05

ROW_KEYS = [
    "game_idx", "website_position", "score", "momentum", "acceleration",
    "stability", "evidence_strength", "state", "previous_position",
    "movement", "direction", "actual_rank", "actual_performance_score",
    "current_performance", "momentum_score", "acceleration_score", "Clients",
    "Betcount", "Stakes", "GGR", "data_status", "signal_as_of",
    "movement_reason", "persistence_days", "strong_performance_days",
    "movement_confidence", "movement_cap", "ideal_rank",
]


def load_days():
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    days = []
    for date in meta["dates"]:
        payload = json.loads((DAYS / f"{date}.json").read_text(encoding="utf-8"))
        days.append((date, pd.DataFrame(payload["rows"], columns=ROW_KEYS)))
    return meta, days


def rank_metrics(df: pd.DataFrame, pos_col: str) -> dict:
    valid = np.isfinite(df[pos_col].to_numpy(float)) & np.isfinite(df["actual_rank"].to_numpy(float))
    if valid.sum() < 2:
        return {"spearman": np.nan, "top10": np.nan, "top20": np.nan, "top100": np.nan, "rank_error": np.nan}
    pos = df.loc[valid, pos_col].to_numpy(float)
    actual = df.loc[valid, "actual_rank"].to_numpy(float)
    p_order = np.argsort(pos, kind="mergesort")
    a_order = np.argsort(actual, kind="mergesort")
    p_rank = np.empty(len(pos)); p_rank[p_order] = np.arange(1, len(pos) + 1)
    a_rank = np.empty(len(actual)); a_rank[a_order] = np.arange(1, len(actual) + 1)
    corr = pd.Series(p_rank).corr(pd.Series(a_rank), method="spearman")
    out = {"spearman": float(corr) if pd.notna(corr) else np.nan}
    for k in TOPKS:
        kk = min(k, len(p_order))
        out[f"top{k}"] = float(len(set(p_order[:kk]) & set(a_order[:kk])) / kk) if kk else np.nan
    out["rank_error"] = float(np.mean(np.abs(p_rank - a_rank)))
    return out


def strength_from_rows(df: pd.DataFrame) -> np.ndarray:
    ev = np.nan_to_num(df["evidence_strength"].to_numpy(float), nan=0.0)
    mom = np.nan_to_num(df["momentum"].to_numpy(float), nan=0.0)
    accel = np.nan_to_num(df["acceleration"].to_numpy(float), nan=0.0)
    return np.clip(
        0.45 * ev + 0.35 * np.minimum(np.abs(mom) / 100.0, 1.0)
        + 0.20 * np.minimum(np.abs(accel) / 100.0, 1.0),
        0.0, 1.0,
    )


def replay(days, variant: str) -> List[pd.DataFrame]:
    previous = None
    out_days = []
    for date, source in days:
        df = source.copy()
        n = len(df)
        if previous is None:
            df["test_position"] = df["ideal_rank"].astype(int)
            df["test_move"] = np.nan
            df["test_direction"] = "INITIAL"
            previous = df[["game_idx", "test_position"]].rename(columns={"test_position": "previous"})
            out_days.append(df)
            continue

        df = df.merge(previous, on="game_idx", how="left")
        df["previous"] = df["previous"].fillna(n + 1).astype(int)
        cur = df["previous"].to_numpy(int)
        ideal = df["ideal_rank"].to_numpy(int)
        valid = (df["data_status"].eq("Available").to_numpy() & np.isfinite(df["score"].to_numpy(float)))
        gap = cur - ideal
        raw = cur.copy()
        strength = strength_from_rows(df)
        streak = np.nan_to_num(df["persistence_days"].to_numpy(float), nan=0.0)
        strong = np.nan_to_num(df["strong_performance_days"].to_numpy(float), nan=0.0)
        maturity = np.clip(streak, 0, 5) / 5.0

        up = (gap > 0) & valid
        down = (gap < 0) & valid
        caps = np.zeros(n, dtype=int)

        if variant == "baseline":
            # Reproduce the v13 movement formula from stored signals.
            effective = strength * (0.55 + 0.45 * maturity)
            if up.any():
                cap = np.clip(np.round(4 + 31 * effective[up] * (0.70 + 0.30 * maturity[up])), 1, 35).astype(int)
                raw[up] = cur[up] - np.minimum(gap[up], cap)
                caps[up] = cap
            if down.any():
                cap = np.clip(np.round(3 + 20 * effective[down] * (0.70 + 0.30 * maturity[down])), 1, 23).astype(int)
                protector = np.where(
                    (strong[down] >= 3) & (np.nan_to_num(df.loc[down, "momentum"].to_numpy(float), nan=0.0) > -10),
                    np.maximum(0.45, 1.0 - 0.10 * np.minimum(strong[down] - 2, 5)),
                    1.0,
                )
                cap = np.maximum(1, np.floor(cap * protector).astype(int))
                raw[down] = cur[down] + np.minimum(-gap[down], cap)
                caps[down] = cap

        else:
            # Persistence is a multiplier applied AFTER signal strength is calculated.
            # This is the feature being evaluated.
            if variant == "targeted":
                # Persistence acts as a multiplier only when signal strength is already meaningful.
                # This concentrates larger movement on games with strong evidence instead of all games.
                base_up = 4.0 + 30.0 * strength
                evidence_gate = np.clip((strength - 0.35) / 0.45, 0.0, 1.0)
                up_mult = 0.95 + 0.30 * maturity * evidence_gate
                base_down = 3.0 + 20.0 * strength
                down_mult = 0.95 + 0.24 * maturity * evidence_gate
                max_up, max_down = 35, 23
            elif variant == "balanced":
                base_up = 4.0 + 30.0 * strength
                up_mult = 0.78 + 0.12 * np.clip(streak, 0, 5)
                base_down = 3.0 + 20.0 * strength
                down_mult = 0.82 + 0.10 * np.clip(streak, 0, 5)
                max_up, max_down = 35, 23
            elif variant == "responsive":
                base_up = 4.0 + 31.0 * strength
                up_mult = 0.82 + 0.15 * np.clip(streak, 0, 5)
                base_down = 3.0 + 21.0 * strength
                down_mult = 0.86 + 0.11 * np.clip(streak, 0, 5)
                max_up, max_down = 35, 23
            elif variant == "fast_persistence":
                base_up = 4.0 + 32.0 * strength
                up_mult = 0.85 + 0.18 * np.clip(streak, 0, 5)
                base_down = 3.0 + 22.0 * strength
                down_mult = 0.88 + 0.13 * np.clip(streak, 0, 5)
                max_up, max_down = 35, 23
            else:
                raise ValueError(variant)

            if up.any():
                cap = np.clip(np.round(base_up[up] * up_mult[up]), 1, max_up).astype(int)
                raw[up] = cur[up] - np.minimum(gap[up], cap)
                caps[up] = cap
            if down.any():
                cap = np.clip(np.round(base_down[down] * down_mult[down]), 1, max_down).astype(int)
                # Established strong games retain some protection.
                momentum_down = np.nan_to_num(df.loc[down, "momentum"].to_numpy(float), nan=0.0)
                protector = np.where(
                    (strong[down] >= 3) & (momentum_down > -10),
                    np.maximum(0.50, 1.0 - 0.08 * np.minimum(strong[down] - 2, 5)),
                    1.0,
                )
                cap = np.maximum(1, np.floor(cap * protector).astype(int))
                raw[down] = cur[down] + np.minimum(-gap[down], cap)
                caps[down] = cap

        locked = ~valid
        final = np.zeros(n, dtype=int)
        occupied = set()
        for idx in np.flatnonzero(locked):
            pos = int(cur[idx])
            pos = max(1, min(n, pos))
            if pos in occupied:
                free = next(p for p in range(1, n + 1) if p not in occupied)
                pos = free
            final[idx] = pos
            occupied.add(pos)

        movable = np.flatnonzero(~locked)
        score = np.nan_to_num(df["score"].to_numpy(float), nan=-np.inf)
        order = movable[np.lexsort((df["game_idx"].to_numpy()[movable], -score[movable], raw[movable]))]
        free_positions = [p for p in range(1, n + 1) if p not in occupied]
        for idx, pos in zip(order, free_positions):
            final[idx] = pos

        df["test_position"] = final
        df["test_move"] = df["previous"] - df["test_position"]
        df["test_direction"] = np.where(df["test_move"] > 0, "UP", np.where(df["test_move"] < 0, "DOWN", "UNCHANGED"))
        df["test_cap"] = caps
        previous = df[["game_idx", "test_position"]].rename(columns={"test_position": "previous"})
        out_days.append(df)
    return out_days


def movement_summary(records: List[pd.DataFrame]) -> dict:
    moves = []
    rows = []
    for df in records[1:]:
        mv = df["test_move"].fillna(0).to_numpy(float)
        moves.extend(np.abs(mv).tolist())
        rows.append(df)
    arr = np.asarray(moves, float)
    if arr.size == 0:
        return {}
    all_rows = pd.concat(rows, ignore_index=True)
    nonzero = arr[arr > 0]
    gt10 = (arr > 10).mean()
    gt20 = (arr > 20).mean()
    gt30 = (arr > 30).mean()
    holds = (arr == 0).mean()

    reversals = 0
    comparisons = 0
    prev = {}
    for df in records[1:]:
        for r in df.itertuples(index=False):
            idx = int(r.game_idx)
            d = r.test_direction
            if d in ("UP", "DOWN"):
                if idx in prev:
                    comparisons += 1
                    if prev[idx] != d:
                        reversals += 1
                prev[idx] = d
    return {
        "avg_abs_daily_movement": float(arr.mean()),
        "median_abs_daily_movement": float(np.median(arr)),
        "max_single_day_movement": int(arr.max()),
        "pct_moves_gt_10": float(gt10),
        "pct_moves_gt_20": float(gt20),
        "pct_moves_gt_30": float(gt30),
        "position_hold_rate": float(holds),
        "direction_reversal_rate": float(reversals / max(comparisons, 1)),
        "moves_count": int(arr.size),
    }


def persistence_effect(records: List[pd.DataFrame]) -> pd.DataFrame:
    frames=[]
    for df in records[1:]:
        s=df[(df["test_direction"]=="UP") & (df["test_move"].abs()>0)].copy()
        if not s.empty:
            frames.append(s[["persistence_days","test_move","score","momentum","acceleration","stability"]])
    if not frames:
        return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True)
    x["abs_move"]=x.test_move.abs()
    x["bucket"]=pd.cut(x.persistence_days, bins=[-1,2,4,5,100], labels=["1-2 days","3-4 days","5+ days","6+ days"])
    return x.groupby("bucket", observed=False).agg(games=("abs_move","size"), mean_move=("abs_move","mean"), median_move=("abs_move","median")).reset_index()


def main():
    meta, days = load_days()
    variants=["baseline","targeted","balanced","responsive","fast_persistence"]
    all_summaries=[]
    daily_quality=[]
    streak_tables=[]
    full_records={}

    for variant in variants:
        records=replay(days,variant)
        full_records[variant]=records
        ms=movement_summary(records)
        q=[]
        for (date,_), df in zip(days, records):
            q.append({"date":date, **rank_metrics(df,"test_position"), "method":variant})
        qdf=pd.DataFrame(q)
        means=qdf[["spearman","top10","top20","top100","rank_error"]].mean(numeric_only=True).to_dict()
        s=persistence_effect(records)
        if not s.empty:
            s["method"]=variant
            streak_tables.append(s)
        all_summaries.append({"method":variant,**means,**ms})
        daily_quality.append(qdf)

    summary=pd.DataFrame(all_summaries)
    quality=pd.concat(daily_quality,ignore_index=True)
    streak=pd.concat(streak_tables,ignore_index=True) if streak_tables else pd.DataFrame()
    OUT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(OUT/"persistence_multiplier_summary.csv",index=False)
    quality.to_csv(OUT/"persistence_multiplier_daily_quality.csv",index=False)
    streak.to_csv(OUT/"persistence_multiplier_streak_effect.csv",index=False)

    # Decide whether a multiplier variant is a credible candidate.
    base=summary.iloc[0]
    candidates=summary[summary.method!="baseline"].copy()
    candidates["top10_gain"]=candidates.top10-base.top10
    candidates["top20_gain"]=candidates.top20-base.top20
    candidates["movement_reduction"]=base.avg_abs_daily_movement-candidates.avg_abs_daily_movement
    candidates["large_move_reduction"]=base.pct_moves_gt_20-candidates.pct_moves_gt_20
    candidates["rank_error_change"]=candidates.rank_error-base.rank_error
    candidates["score"]=candidates.top10_gain*5+candidates.top20_gain*3+candidates.large_move_reduction*2-candidates.rank_error_change*0.001
    best=candidates.sort_values("score",ascending=False).iloc[0].to_dict() if len(candidates) else None

    report={
        "test":"Real-data persistence multiplier test",
        "games":int(len(days[0][1])),
        "evaluation_days":len(days),
        "acceleration_weight":ACCELERATION_WEIGHT,
        "purpose":"Compare the existing movement policy with persistence-as-a-multiplier variants without changing signal calculations.",
        "results":all_summaries,
        "candidate_comparison":candidates.to_dict("records"),
        "preferred_candidate_by_test_score":best,
        "notes":"Selection is diagnostic, not a production approval. Final integration should be based on ranking quality, movement stability, and breakout responsiveness together.",
    }
    (OUT/"persistence_multiplier_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("[REAL-DATA PERSISTENCE MULTIPLIER TEST]")
    print("="*96)
    print(f"Games: {report['games']} | Evaluation days: {report['evaluation_days']} | Acceleration: {ACCELERATION_WEIGHT:.0%}")
    print()
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print()
    if not streak.empty:
        print("[UPWARD MOVEMENT BY PERSISTENCE]")
        print(streak.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print()
    print(f"[PREFERRED CANDIDATE FOR REVIEW] {best['method'] if best else 'None'}")
    print(f"[OUTPUT] {OUT}")

if __name__=="__main__":
    main()
