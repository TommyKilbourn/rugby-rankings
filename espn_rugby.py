"""
espn_rugby.py
=============
Live ingestion of international rugby results from ESPN's public JSON API.

Why ESPN (not rugbypy): rugbypy reads static parquet snapshots from a GitHub
repo whose per-match score files are patchy (2025 tests had no scores). ESPN's
scoreboard API is the authoritative, real-time source -- one call per
(league, year) returns every match with scores, venue, and status, so the whole
history backfills in ~two dozen calls and daily updates are near-instant.

Endpoint:
  https://site.api.espn.com/apis/site/v2/sports/rugby/{league}/scoreboard
      ?dates=YYYY&limit=1000
`dates=YYYY` returns the league's full calendar year; `limit=1000` defeats the
silent 100-event cap that otherwise drops November autumn tests.

Neutral venues: ESPN's `neutralSite` is unreliable for World Cups (it labels the
neutral-venue final as non-neutral). We override for the RWC using a host table:
the host nation plays at home; every other RWC match is neutral.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "espn_cache")
os.makedirs(CACHE, exist_ok=True)

# Men's senior international competitions on ESPN (league_id -> label).
INTERNATIONAL_LEAGUES = {
    "164205": "Rugby World Cup",
    "180659": "Six Nations",
    "244293": "The Rugby Championship",
    "289234": "International Test Match",     # summer tours + autumn tests + misc
    "17567":  "Nations Championship",
    "268565": "British and Irish Lions Tour",
    "289274": "Tri Nations",                 # one-off 2020
}
WORLD_CUP_LEAGUE = "164205"

# Rugby World Cup host nations (extend as new hosts are confirmed).
RWC_HOSTS = {
    2023: {"France"},
    2027: {"Australia"},
    2031: {"United States"},
}

# Normalise a few ESPN team names to a canonical form (tier-1 already match).
TEAM_NORMALISE = {
    "United States of America": "United States",
    "Czechia": "Czech Republic",
}

BASE = ("https://site.api.espn.com/apis/site/v2/sports/rugby/"
        "{lid}/scoreboard?dates={d}&limit=1000")
HEADERS = {"User-Agent": "Mozilla/5.0 (rugby-rankings research)"}


# --------------------------------------------------------------------------- #
def _get_json(url: str, retries: int = 4, backoff: float = 1.5) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:  # network hiccup
            last = repr(e)
        time.sleep(backoff ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def _canon(name: str | None) -> str | None:
    if name is None:
        return None
    return TEAM_NORMALISE.get(name, name)


def _score(competitor: dict):
    s = competitor.get("score")
    try:
        return int(s)
    except (TypeError, ValueError):
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None


def _parse_event(ev: dict, league_id: str, league_name: str) -> dict | None:
    comp = (ev.get("competitions") or [{}])[0]
    status = (ev.get("status") or {}).get("type") or {}
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_team = _canon((home.get("team") or {}).get("displayName"))
    away_team = _canon((away.get("team") or {}).get("displayName"))
    hs, as_ = _score(home), _score(away)
    d = (ev.get("date") or "")[:10]
    venue = comp.get("venue") or {}
    addr = venue.get("address") or {}
    world_cup = league_id == WORLD_CUP_LEAGUE
    neutral = bool(comp.get("neutralSite"))

    # RWC neutral correction: host at home, everyone else neutral. If the host
    # is listed as the away side, swap so it carries home advantage.
    if world_cup and d:
        hosts = RWC_HOSTS.get(int(d[:4]), set())
        if home_team in hosts:
            neutral = False
        elif away_team in hosts:
            home_team, away_team = away_team, home_team
            hs, as_ = as_, hs
            neutral = False
        else:
            neutral = True

    return {
        "date": d,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": hs,
        "away_score": as_,
        "competition": league_name,
        "stadium": venue.get("fullName") or "",
        "city": addr.get("city") or "",
        "country": addr.get("country") or "",
        "neutral": neutral,
        "world_cup": world_cup,
        "completed": bool(status.get("completed")),
        "espn_event_id": ev.get("id"),
    }


def fetch_league_year(league_id: str, year: int, use_cache: bool = True) -> list[dict]:
    """Return parsed events for one league in one calendar year.

    Past years are cached to disk (immutable); the current year is always
    re-fetched so newly played matches are picked up.
    """
    cache_path = os.path.join(CACHE, f"{league_id}_{year}.json")
    is_past = year < date.today().year
    if use_cache and is_past and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            events = json.load(f)
    else:
        url = BASE.format(lid=league_id, d=year)
        events = _get_json(url).get("events", [])
        if is_past:  # only persist completed, immutable years
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(events, f)

    name = INTERNATIONAL_LEAGUES[league_id]
    out = []
    for ev in events:
        row = _parse_event(ev, league_id, name)
        if row is not None:
            out.append(row)
    return out


def fetch_internationals(start_year: int, end_year: int,
                         use_cache: bool = True) -> pd.DataFrame:
    """All men's senior internationals across leagues for [start_year, end_year].

    Keeps only completed matches with numeric scores; dedupes by (date, team-pair).
    """
    rows: list[dict] = []
    for league_id in INTERNATIONAL_LEAGUES:
        for year in range(start_year, end_year + 1):
            try:
                rows.extend(fetch_league_year(league_id, year, use_cache))
            except Exception as e:
                print(f"  [warn] {INTERNATIONAL_LEAGUES[league_id]} {year}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["completed"]
            & df["home_score"].notna() & df["away_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # dedupe: same day + same pair of teams (order-independent)
    df["_key"] = df.apply(
        lambda r: (r["date"], frozenset((r["home_team"], r["away_team"]))), axis=1)
    df = df.drop_duplicates("_key").drop(columns="_key")
    return df.sort_values("date").reset_index(drop=True)
