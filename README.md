# 🏉 International Rugby Elo Rankings

A self-updating Elo rating system for tier-1 men's international rugby union,
with an interactive dashboard and a win-probability calculator.

**Live site:** https://tommykilbourn.github.io/rugby-rankings/

## What it does

- Rates the 10 tier-1 nations with a home-adjusted, prequentially-tuned Elo model
  (`K≈43`, home advantage `≈110` Elo points).
- One-step-ahead (out-of-sample) accuracy of ~76% on decisive matches since 2000,
  well calibrated.
- Interactive dashboard: ranking table with 12-month movement and recent form,
  rating-history chart, recent results, and a **win-probability calculator**
  (win / draw / loss + predicted margin for any two teams and venue).

## Data

- Historical base (`data/rugby_results.csv`): tier-1 tests 1871–2023.
- Live extension: pulled from ESPN's public scoreboard API (`espn_rugby.py`),
  covering World Cup, Six Nations, The Rugby Championship, and all summer/autumn
  tests. `update_data.py` merges the two into `data/rugby_results_master.csv`.

## How updates work

GitHub Actions (`.github/workflows/deploy.yml`) runs daily: it re-pulls results
from ESPN, rebuilds the dashboard, and deploys it to GitHub Pages. No manual
steps once it's set up.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
python update_data.py    # refresh data
python build_site.py     # rebuild site/index.html
python run_baseline.py   # (optional) re-tune + evaluate + plots
```

## Files

| File | Purpose |
|------|---------|
| `rugby_elo.py` | Elo engine (tunable, prequential evaluation) |
| `espn_rugby.py` | ESPN ingestion client |
| `update_data.py` | Merge historical + live into the master file |
| `build_site.py` | Generate the self-contained dashboard |
| `run_baseline.py` | Tune, evaluate, plot rating history |
| `refresh.py` | One-shot refresh (used by the scheduler / CI) |
