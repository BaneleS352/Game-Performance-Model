HOLLYWOODBETS GAME MOVEMENT PLATFORM
====================================

Included:
- index.html        Main dashboard
- data.js           Embedded data generated from the supplied JSON + CSV
- start_server.py   One-click local server (Python required)
- start_windows.bat Windows launcher

FASTEST WAY
1. Unzip this folder.
2. Open index.html in a browser.

RECOMMENDED WAY
1. Make sure Python 3 is installed.
2. Double-click start_windows.bat on Windows, or run:
   python start_server.py
3. Your browser will open automatically.

WHAT IT DOES
- Uses the supplied current game positions.
- Uses the supplied 60-day daily performance data.
- Lets users search and filter games.
- Ranks by Most Movement, Momentum, current position, or metric change.
- Shows current position beside every game.
- Shows Rising / Falling / Stable movement.
- Click a game to inspect its trend chart.
- Toggle Stakes, GGR, Bets, Players, or all four.
- Change the comparison window (7d, 14d, 30d).

MATCHING NOTE
Performance names were normalized against position titles. High-confidence fuzzy matches were used for naming differences. Games without a reliable performance match remain visible as "position only" and are not given a fabricated performance trend.

SOURCE SNAPSHOT
- Position entries: 3,877 unique game titles after de-duplicating repeated titles to the top-most position.
- Performance rows: 227,522.
- Performance dates: 60.
- Matched position titles: 3,734.
