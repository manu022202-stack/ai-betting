# MLB Props & Parlay Phone App

This version adds player-prop market support.

## What it scans

Game markets:
- Moneyline
- Run line
- Totals

Player prop markets, if available on your Odds API plan:
- Batter hits
- Batter total bases
- Batter RBIs
- Batter runs
- Batter home runs
- Batter strikeouts
- Pitcher strikeouts
- Pitcher hits allowed
- Pitcher earned runs
- Pitcher outs

## Upload to GitHub

Upload:
- streamlit_app.py
- requirements.txt
- README.md

Then deploy on Streamlit Cloud.

## Important

Player props may require a paid Odds API plan or specific market access.

## Next advanced upgrades

The current version has a framework for advanced context scoring, but the live advanced feeds are not fully connected yet.

The next model upgrades would add:
- Baseball Savant / Statcast hitter data
- Pitcher handedness and pitch mix
- Batter vs pitch type strengths
- Pull/oppo/center hit charts
- Stadium park factors
- Weather and wind
- Confirmed lineup spot
- Bullpen fatigue
- Umpire tendencies
- Recent player form
- Rolling xwOBA, hard-hit %, barrel %, K%, BB%, chase %, whiff %

This app is designed so those can be added to the `context_score` section.
