from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'web' / 'data'
DAYS = DATA / 'days'
TOPKS = (10, 20, 100)
ROW_KEYS = [
    'game_idx','website_position','score','momentum','acceleration_raw','stability','evidence_strength','state',
    'previous_position','movement','movement_direction','actual_rank','actual_performance_score','current_performance',
    'momentum_score','acceleration_score','Clients','Betcount','Stakes','GGR','data_status','signal_as_of','movement_reason',
    'persistence_days','strong_performance_days','movement_confidence','movement_cap','ideal_rank'
]

def row_objects(rows):
    return [dict(zip(ROW_KEYS, r)) for r in rows]


def load_days():
    meta = json.loads((DATA / 'metadata.json').read_text(encoding='utf-8'))
    dates = meta['dates']
    out = []
    for d in dates:
        payload = json.loads((DAYS / f'{d}.json').read_text(encoding='utf-8'))
        rows = row_objects(payload['rows'])
        out.append((d, rows))
    return meta, out


def replay_policy(days, policy):
    previous = None
    records = []
    for d, rows in days:
        df = pd.DataFrame(rows)
        n = len(df)
        if previous is None:
            df['prev_policy_pos'] = np.nan
            df['policy_pos'] = df['ideal_rank'].astype(int)
            df['policy_move'] = np.nan
            df['policy_dir'] = 'INITIAL'
            previous = df[['game_idx', 'policy_pos']].copy()
            previous.columns = ['game_idx', 'previous']
        else:
            df = df.merge(previous, on='game_idx', how='left')
            df['previous'] = df['previous'].fillna(n + 1).astype(int)
            cur = df['previous'].to_numpy()
            ideal = df['ideal_rank'].to_numpy()
            ev = np.nan_to_num(df['evidence_strength'].to_numpy(dtype=float), nan=0.0)
            gap = cur - ideal
            raw = cur.copy()
            valid = df['data_status'].eq('Available').to_numpy() & np.isfinite(df['score'].to_numpy(dtype=float))
            up = (gap > 0) & valid
            down = (gap < 0) & valid
            if policy == 'legacy':
                if up.any():
                    cap = np.maximum(1, np.round(5 + 35 * (0.65 * ev[up] + 0.35 * np.minimum(gap[up] / 100.0, 1.0))).astype(int))
                    raw[up] = cur[up] - np.minimum(gap[up], cap)
                if down.any():
                    ga = -gap[down]
                    cap = np.maximum(1, np.round(3 + 22 * (0.65 * ev[down] + 0.35 * np.minimum(ga / 100.0, 1.0))).astype(int))
                    raw[down] = cur[down] + np.minimum(ga, cap)
            elif policy == 'persistence':
                movement_cap = np.nan_to_num(df.get('movement_cap', pd.Series(0, index=df.index)).to_numpy(dtype=float), nan=0).astype(int)
                # The v13 row already contains the adaptive cap used by the new engine.
                raw[up] = cur[up] - np.minimum(gap[up], movement_cap[up])
                raw[down] = cur[down] + np.minimum(-gap[down], movement_cap[down])
            else:
                raise ValueError(policy)
            locked = ~valid
            final = np.zeros(n, dtype=int)
            locked_positions = set()
            for idx in np.flatnonzero(locked):
                pos = int(cur[idx])
                pos = max(1, min(n, pos))
                if pos not in locked_positions:
                    final[idx] = pos
                    locked_positions.add(pos)
                else:
                    free = next((p for p in range(1, n + 1) if p not in locked_positions), pos)
                    final[idx] = free
                    locked_positions.add(free)
            movable = np.flatnonzero(~locked)
            score = np.nan_to_num(df['score'].to_numpy(dtype=float), nan=-np.inf)
            order = movable[np.lexsort((df['game_idx'].to_numpy()[movable], -score[movable], raw[movable]))]
            free_positions = [p for p in range(1, n + 1) if p not in locked_positions]
            for idx, pos in zip(order, free_positions):
                final[idx] = pos
            df['prev_policy_pos'] = df['previous']
            df['policy_pos'] = final
            df['policy_move'] = df['prev_policy_pos'] - df['policy_pos']
            df['policy_dir'] = np.where(df['policy_move'] > 0, 'UP', np.where(df['policy_move'] < 0, 'DOWN', 'UNCHANGED'))
            previous = df[['game_idx', 'policy_pos']].copy()
            previous.columns = ['game_idx', 'previous']
        records.append(df)
    return records


def rank_metrics_from_positions(df, pos_col='policy_pos'):
    valid = np.isfinite(df['actual_rank'].to_numpy(dtype=float)) & np.isfinite(df[pos_col].to_numpy(dtype=float))
    if valid.sum() < 2:
        return {'spearman': np.nan, 'top10': np.nan, 'top20': np.nan, 'top100': np.nan, 'rank_error': np.nan}
    pos = df.loc[valid, pos_col].to_numpy(float)
    actual = df.loc[valid, 'actual_rank'].to_numpy(float)
    p_order = np.argsort(pos, kind='mergesort')
    a_order = np.argsort(actual, kind='mergesort')
    p_rank = np.empty(len(pos), dtype=float); p_rank[p_order] = np.arange(1, len(pos)+1)
    a_rank = np.empty(len(actual), dtype=float); a_rank[a_order] = np.arange(1, len(actual)+1)
    corr = pd.Series(p_rank).corr(pd.Series(a_rank), method='spearman')
    out = {'spearman': float(corr) if pd.notna(corr) else np.nan}
    for k in TOPKS:
        kk=min(k,len(p_order))
        out[f'top{k}']=float(len(set(p_order[:kk]) & set(a_order[:kk]))/kk) if kk else np.nan
    out['rank_error']=float(np.mean(np.abs(p_rank-a_rank)))
    return out


def movement_stats(records, pos_col='policy_pos', move_col='policy_move'):
    abs_moves=[]; reversals=0; comparisons=0; gt10=0; gt20=0; gt30=0; holds=0; max_move=0
    by_streak=[]
    prev_dir_by_game={}
    for df in records:
        mv=df[move_col].fillna(0).to_numpy(float)
        abs_moves.extend(np.abs(mv[1:]) if len(records)>1 else np.abs(mv))
        if len(records) > 1:
            av=np.abs(mv)
        else:
            av=np.abs(mv)
        gt10 += int((av > 10).sum()); gt20 += int((av > 20).sum()); gt30 += int((av > 30).sum()); holds += int((av == 0).sum()); max_move=max(max_move, int(av.max()) if len(av) else 0)
        for r in df.itertuples(index=False):
            idx=int(r.game_idx); direction=r.policy_dir; prev=prev_dir_by_game.get(idx)
            if direction in ('UP','DOWN') and prev in ('UP','DOWN'):
                comparisons += 1
                if direction != prev: reversals += 1
            if direction in ('UP','DOWN'):
                prev_dir_by_game[idx]=direction
            streak = getattr(r, 'persistence_days', np.nan)
            if direction == 'UP' and np.isfinite(streak):
                by_streak.append((int(streak), abs(float(r.policy_move))))
    total = sum(len(df) for df in records[1:]) if len(records)>1 else 0
    return {
        'avg_abs_daily_movement': float(np.mean(abs_moves)) if abs_moves else 0.0,
        'median_abs_daily_movement': float(np.median(abs_moves)) if abs_moves else 0.0,
        'max_single_day_movement': max_move,
        'pct_moves_gt_10': gt10 / max(total,1),
        'pct_moves_gt_20': gt20 / max(total,1),
        'pct_moves_gt_30': gt30 / max(total,1),
        'position_hold_rate': holds / max(total,1),
        'direction_reversal_rate': reversals / max(comparisons,1),
    }, by_streak


def main():
    meta, days = load_days()
    legacy = replay_policy(days, 'legacy')
    new = replay_policy(days, 'persistence')
    rows=[]
    for d, ldf, ndf in zip(meta['dates'], legacy, new):
        lm=rank_metrics_from_positions(ldf)
        nm=rank_metrics_from_positions(ndf)
        for method, m in [('Legacy movement',lm),('Persistence movement',nm)]:
            rows.append({'date':d,'method':method,**m})
    eval_df=pd.DataFrame(rows)
    legacy_stats, _ = movement_stats(legacy)
    new_stats, streak = movement_stats(new)
    # Persistence effectiveness checks.
    ups = pd.concat([df[df.policy_dir=='UP'][['persistence_days','policy_move']] for df in new], ignore_index=True)
    streak_summary=[]
    if not ups.empty:
        ups['abs_move']=ups.policy_move.abs()
        for bucket, sub in [('1-2 days', ups[ups.persistence_days <= 2]), ('3-4 days', ups[ups.persistence_days.between(3,4)]), ('5+ days', ups[ups.persistence_days >= 5])]:
            streak_summary.append({'persistence_bucket':bucket,'games_with_upward_move':int(len(sub)),'mean_abs_up_move':float(sub.abs_move.mean()) if len(sub) else 0.0,'median_abs_up_move':float(sub.abs_move.median()) if len(sub) else 0.0})
    actual_no_data_holds = all(
        (df.loc[df.data_status.eq('No Data'), 'movement'].fillna(0).abs() == 0).all()
        for df in [pd.DataFrame(d) for _, d in days]
    )
    summary = {
        'project': meta['config'],
        'legacy_movement': legacy_stats,
        'persistence_movement': new_stats,
        'persistence_streak_effect': streak_summary,
        'daily_rank_quality_means': {
            'legacy': {k: (None if not np.isfinite(v) else float(v)) for k,v in eval_df[eval_df.method=='Legacy movement'][['spearman','top10','top20','top100','rank_error']].mean().items()},
            'persistence': {k: (None if not np.isfinite(v) else float(v)) for k,v in eval_df[eval_df.method=='Persistence movement'][['spearman','top10','top20','top100','rank_error']].mean().items()},
        },
        'assertions': {
            'pipeline_days_present': len(days) == 30,
            'all_new_positions_unique_each_day': all(df.policy_pos.nunique() == len(df) for df in new),
            'no_missing_data_moves': actual_no_data_holds,
            'persistence_moves_monotonic_by_streak': not streak_summary or all(streak_summary[i]['mean_abs_up_move'] <= streak_summary[i+1]['mean_abs_up_move'] + 1e-9 for i in range(len(streak_summary)-1)),
        }
    }
    out=DATA/'persistence_movement_test'
    out.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out/'persistence_vs_legacy_daily.csv',index=False)
    pd.DataFrame(streak_summary).to_csv(out/'persistence_streak_effect.csv',index=False)
    (out/'persistence_movement_test_summary.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    readme='''Persistence movement test\n\nThis test compares the legacy controlled-movement policy with the new persistence-aware policy using the same stored daily scores, ideal ranks, evidence strength and actual ranks. It also checks position uniqueness, no-data holds, and whether longer upward persistence permits larger movement on average.\n'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
