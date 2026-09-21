@echo off
call .venv\Scripts\activate.bat
python build_ranking_pipeline.py "merged_slot_games.xlsx"
pause
