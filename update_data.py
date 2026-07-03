"""
update_data.py
==============
Build / refresh the master international results file.

    master = historical CSV (<= 2023-07-15, tier-1, Wikipedia-sourced)
           + ESPN internationals (> 2023-07-15, all nations, live)

Idempotent: the master is regenerated from source each run, so running it again
after new matches are played simply appends them -- no drift, no double-counting.
Past calendar years are cached on disk; only the current year is re-fetched.

    python update_data.py            # refresh through today
"""
import os
from datetime import date

import pandas as pd

import espn_rugby as espn

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORICAL = os.path.join(HERE, "data", "rugby_results.csv")
MASTER = os.path.join(HERE, "data", "rugby_results_master.csv")

COLS = ["date", "home_team", "away_team", "home_score", "away_score",
        "competition", "stadium", "city", "country", "neutral", "world_cup"]

TIER1 = {"Argentina", "Australia", "England", "France", "Ireland", "Italy",
         "New Zealand", "Scotland", "South Africa", "Wales"}

# Pull ESPN back to 2015 so tier-2 sides (Fiji, Japan, ...) get real history.
# The curated tier-1 base (1871-2023) stays authoritative for tier-1-v-tier-1;
# from ESPN we take everything after the base cutoff, plus tier-2-involving
# matches before it (World Cups back to 2015, regular Tests from 2015).
ESPN_START_YEAR = 2015


def _key(df):
    return df.apply(lambda r: (r["date"],
                               frozenset((r["home_team"], r["away_team"]))),
                    axis=1)


def main():
    # ---- historical base -------------------------------------------------- #
    hist = pd.read_csv(HISTORICAL, encoding="utf-8", encoding_errors="replace")
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    hist = hist[COLS].copy()
    hist["source"] = "historical"
    cutoff = hist["date"].max()
    print(f"Historical base: {len(hist)} matches, ends {cutoff}")

    # previous master (for a 'what's new' diff)
    prev_keys = set()
    if os.path.exists(MASTER):
        prev = pd.read_csv(MASTER)
        prev["date"] = prev["date"].astype(str)
        prev_keys = set(_key(prev))

    # ---- ESPN extension --------------------------------------------------- #
    this_year = date.today().year
    print(f"Fetching ESPN internationals {ESPN_START_YEAR}-{this_year} ...")
    esp = espn.fetch_internationals(ESPN_START_YEAR, this_year)
    # keep everything after the base cutoff, plus tier-2-involving games before
    # it (the curated tier-1-v-tier-1 history stays from the base file)
    involves_t2 = (~esp["home_team"].isin(TIER1)) | (~esp["away_team"].isin(TIER1))
    esp = esp[(esp["date"] > cutoff) | involves_t2].copy()
    esp["source"] = "espn"
    n_pre = int(((esp["date"] <= cutoff)).sum())
    print(f"ESPN matches kept: {len(esp)}  ({n_pre} tier-2 games before {cutoff})")

    # ---- merge & dedupe --------------------------------------------------- #
    master = pd.concat([hist, esp[COLS + ['source']]], ignore_index=True)
    master["_k"] = _key(master)
    master = (master.drop_duplicates("_k", keep="first")   # historical wins ties
              .drop(columns="_k")
              .sort_values("date")
              .reset_index(drop=True))
    master.to_csv(MASTER, index=False)

    # ---- summary ---------------------------------------------------------- #
    new_keys = set(_key(master)) - prev_keys
    print("\n" + "=" * 60)
    print(f"MASTER written: {MASTER}")
    print(f"  total matches : {len(master)}  ({master.date.min()} -> {master.date.max()})")
    print(f"  from ESPN     : {(master.source == 'espn').sum()}")
    if prev_keys:
        print(f"  new this run  : {len(new_keys)}")
    t1 = master[master.home_team.isin(TIER1) & master.away_team.isin(TIER1)]
    print(f"  tier-1 v tier-1: {len(t1)}")

    # diagnostics: RWC neutral handling across tournaments
    rwc = esp[esp.world_cup]
    if len(rwc):
        yrs = sorted(rwc.date.str[:4].unique().tolist())
        print(f"\n  RWC matches: {len(rwc)} across {yrs}; "
              f"neutral share {rwc.neutral.mean():.2f} (hosts play at home)")

    print("\n  Latest 8 tier-1 internationals ingested:")
    show = t1[t1.source == "espn"].tail(8)
    for _, r in show.iterrows():
        v = " (N)" if r.neutral else ""
        print(f"    {r.date}  {r.home_team} {r.home_score}-{r.away_score} "
              f"{r.away_team}{v}   [{r.competition}]")


if __name__ == "__main__":
    main()
