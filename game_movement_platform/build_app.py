import json, re, math
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

BASE=Path('/mnt/data/game_movement_platform')
JP=Path('/mnt/data/hollywoodbets_allgames_2026-09-11.json')
CP=Path('/mnt/data/game_performance.csv')
BASE.mkdir(parents=True, exist_ok=True)

with open(JP, encoding='utf-8') as f:
    pos_raw=json.load(f)
pos=pd.DataFrame(pos_raw)
perf=pd.read_csv(CP)
pos=pos.sort_values('position').drop_duplicates('title', keep='first').copy()

def clean_name(s):
    s=str(s or '').lower().replace('\u2019',"'").replace('&',' and ')
    s=re.sub(r'\([^)]*\)',' ',s)
    s=re.sub(r'^\s*[a-z0-9]{2,10}\s*[-:]\s*','',s)
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

perf['match_key']=perf['TournamentName'].map(clean_name)
pos['match_key']=pos['title'].map(clean_name)

# Aggregate once at date + normalized game level.
daily=(perf.groupby(['EventDateID','match_key'], as_index=False)[['UNIQUE_CLIENTS','BetCount','Stakes','GGR']].sum())
all_dates=sorted(daily['EventDateID'].unique().tolist())
key_stats=daily.groupby('match_key').agg(days=('EventDateID','size')).reset_index()
choices=key_stats['match_key'].tolist()

# Exact matches first. Fuzzy only for unmatched names.
key_set=set(choices)
match_map={}
score_map={}
for _,r in pos.iterrows():
    k=r['match_key']
    if k in key_set:
        match_map[r['title']]=k; score_map[r['title']]=100

unmatched=pos[~pos['title'].isin(match_map)].copy()
for _,r in unmatched.iterrows():
    k=r['match_key']
    if not k: continue
    cand=process.extract(k, choices, scorer=fuzz.token_sort_ratio, limit=3, score_cutoff=84)
    if cand:
        best=cand[0]
        second=cand[1] if len(cand)>1 else None
        # High-confidence fuzzy, or a clear lead.
        if best[1] >= 94 or (best[1] >= 88 and (second is None or best[1]-second[1] >= 5)):
            match_map[r['title']]=best[0]; score_map[r['title']]=round(float(best[1]),1)

# Avoid re-filtering the full daily dataframe for every game.
groups={k:g.set_index('EventDateID') for k,g in daily.groupby('match_key', sort=False)}
records=[]
for _,r in pos.iterrows():
    title=r['title']; sel=match_map.get(title)
    series=[]
    if sel is not None:
        sub=groups[sel]
        for d in all_dates:
            if d in sub.index:
                x=sub.loc[d]
                # loc is a Series because date is unique per match_key.
                series.append({'d':str(d),'players':int(x['UNIQUE_CLIENTS']),'bets':int(x['BetCount']),'stakes':float(x['Stakes']),'ggr':float(x['GGR'])})
            else:
                series.append({'d':str(d),'players':None,'bets':None,'stakes':None,'ggr':None})
    else:
        series=[{'d':str(d),'players':None,'bets':None,'stakes':None,'ggr':None} for d in all_dates]
    records.append({'name':title,'position':int(r['position']),'image':r.get('src',''),'matchScore':score_map.get(title),'series':series})

meta={'dates':all_dates,'positionsCount':len(pos),'matchedCount':len(match_map),'performanceGames':int(perf['TournamentName'].nunique()),'rows':int(len(perf))}
(BASE/'data.js').write_text('window.GAME_DATA='+json.dumps({'meta':meta,'games':records},separators=(',',':'))+';', encoding='utf-8')
print(meta)
print('fuzzy matches', sum(1 for s in score_map.values() if s<100))
print('unmatched', len(pos)-len(match_map))
