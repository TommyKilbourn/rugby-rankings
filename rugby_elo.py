"""
rugby_elo.py
============
A tunable Elo rating engine for international (tier-1) rugby union.

Design goals
------------
* **Prequential ("predict-then-update")**: every match is first *predicted*
  using only information available before kick-off, then used to update the
  ratings. The time-ordered average of those pre-update prediction errors is a
  genuine out-of-sample (prospective) score -- no look-ahead. This is the
  natural way to evaluate online learners like Elo.
* **Configurable** via :class:`EloConfig`: step size ``K``, home-field
  advantage ``hfa`` (Elo points, applied only at non-neutral venues), logistic
  scale, base rating, an optional margin-of-victory multiplier, and optional
  between-season regression to the mean.
* **Extensible**: the update is isolated in :func:`EloModel.update_match`, so
  Glicko/TrueSkill-style uncertainty or a Bayesian state-space variant can be
  dropped in alongside without touching the evaluation harness.

Scoring
-------
Outcomes are encoded from the home team's perspective as
``S in {1.0 win, 0.5 draw, 0.0 loss}``. The Elo expectation ``E`` is the
probability-like expected score. We score predictions with proper rules that
accept the half-outcome of a draw:

* **log loss**: ``-(S*ln E + (1-S)*ln(1-E))``  (minimised at E=S; =ln2 at E=0.5)
* **Brier**:    ``(E - S)^2``
* **accuracy**: on decisive games only, did sign(E-0.5) match the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
TIER1 = {"Argentina", "Australia", "England", "France", "Ireland", "Italy",
         "New Zealand", "Scotland", "South Africa", "Wales"}


def load_results(path: str, tier1_only: bool = False) -> pd.DataFrame:
    """Load and normalise the rugby results CSV.

    Returns a frame sorted by date with columns:
    date, home_team, away_team, home_score, away_score, neutral, world_cup,
    margin (home - away), s_home (1/0.5/0).

    tier1_only keeps only matches where BOTH teams are tier-1 nations -- use it
    to reproduce the tier-1 baseline on the extended master file (whose post-2023
    rows also include tier-1-vs-tier-2 and tier-2 internationals).
    """
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    for col in ("neutral", "world_cup"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )

    if tier1_only:
        df = df[df["home_team"].isin(TIER1) & df["away_team"].isin(TIER1)]
        df = df.reset_index(drop=True)

    df["margin"] = df["home_score"] - df["away_score"]
    df["s_home"] = np.where(df["margin"] > 0, 1.0,
                            np.where(df["margin"] < 0, 0.0, 0.5))
    return df


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class EloConfig:
    k: float = 32.0                 # base step size
    hfa: float = 60.0              # home-field advantage in Elo points
    scale: float = 400.0           # logistic scale (a 'scale' gap => ~76% exp.)
    base: float = 1500.0           # initial rating for every team
    # --- optional enhancements (off by default for the clean baseline) ---
    use_mov: bool = False          # margin-of-victory multiplier (538-style)
    mov_cap: float = 2.2           # autocorrelation-correction constant
    regress: float = 0.0           # fraction pulled to mean at each new season
    world_cup_mult: float = 1.0    # extra weight on World Cup matches
    # --- newcomer seeding (for extending beyond the established pool) ---
    # Teams NOT in `established_teams` are seeded at `new_team_base` on their
    # first appearance instead of `base` -- a lower prior reflecting that a side
    # joining a mature pool is typically below its average, not at it.
    established_teams: Optional[frozenset] = None
    new_team_base: Optional[float] = None

    def describe(self) -> str:
        bits = [f"K={self.k:g}", f"HFA={self.hfa:g}"]
        if self.use_mov:
            bits.append("MOV=on")
        if self.regress:
            bits.append(f"regress={self.regress:g}")
        if self.world_cup_mult != 1.0:
            bits.append(f"wc_mult={self.world_cup_mult:g}")
        return ", ".join(bits)


# --------------------------------------------------------------------------- #
# Elo model
# --------------------------------------------------------------------------- #
class EloModel:
    """Sequential Elo engine with a prequential prediction log."""

    def __init__(self, config: EloConfig):
        self.cfg = config
        self.ratings: dict[str, float] = {}
        self.games_played: dict[str, int] = {}

    # -- seed-on-first-use -------------------------------------------------- #
    def _rating(self, team: str) -> float:
        r = self.ratings.get(team)
        if r is None:
            if (self.cfg.new_team_base is not None
                    and self.cfg.established_teams is not None
                    and team not in self.cfg.established_teams):
                r = self.cfg.new_team_base
            else:
                r = self.cfg.base
            self.ratings[team] = r
        return r

    # -- expectation -------------------------------------------------------- #
    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        rh = self._rating(home)
        ra = self._rating(away)
        adj = 0.0 if neutral else self.cfg.hfa
        return 1.0 / (1.0 + 10.0 ** (-((rh + adj) - ra) / self.cfg.scale))

    # -- margin-of-victory multiplier (only used if cfg.use_mov) ------------ #
    def _mov_mult(self, margin: float, elo_diff_winner: float) -> float:
        """538-style: ln(|margin|+1) damped by an autocorrelation correction
        that shrinks the multiplier when a strong favourite wins. Returns 1.0
        for draws so the base S-E update still applies."""
        m = abs(margin)
        if m == 0:
            return 1.0
        return np.log(m + 1.0) * (self.cfg.mov_cap /
                                  (0.001 * elo_diff_winner + self.cfg.mov_cap))

    # -- single match update ----------------------------------------------- #
    def update_match(self, home: str, away: str, s_home: float,
                     margin: float, neutral: bool, world_cup: bool) -> float:
        """Predict (return E_home), then update both ratings in place."""
        rh = self._rating(home)
        ra = self._rating(away)
        e_home = self.expected_home(home, away, neutral)

        k = self.cfg.k
        if world_cup:
            k *= self.cfg.world_cup_mult
        if self.cfg.use_mov:
            adj = 0.0 if neutral else self.cfg.hfa
            # rating gap from the winner's perspective (after home adjustment)
            eff_h, eff_a = rh + adj, ra
            if s_home >= 0.5:
                elo_diff_winner = eff_h - eff_a
            else:
                elo_diff_winner = eff_a - eff_h
            k *= self._mov_mult(margin, elo_diff_winner)

        delta = k * (s_home - e_home)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1
        return e_home


# --------------------------------------------------------------------------- #
# Run over a schedule (prequential)
# --------------------------------------------------------------------------- #
def run_elo(df: pd.DataFrame, config: EloConfig,
            track_history: bool = False):
    """Process matches in date order. Returns (predictions_df, model, history).

    predictions_df has one row per match with the *pre-update* expectation and
    the realised outcome, so it can be scored as a prospective forecast.
    """
    model = EloModel(config)
    e_home = np.empty(len(df))
    rh_pre = np.empty(len(df))
    ra_pre = np.empty(len(df))
    history: list[dict] = []

    cur_year: Optional[int] = None
    dates = df["date"].dt.year.to_numpy()
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    s_home = df["s_home"].to_numpy()
    margin = df["margin"].to_numpy()
    neutral = df["neutral"].to_numpy()
    world_cup = df["world_cup"].to_numpy()

    for i in range(len(df)):
        # optional between-season regression to the mean
        if config.regress and dates[i] != cur_year:
            if cur_year is not None:
                for t in model.ratings:
                    model.ratings[t] += config.regress * (
                        config.base - model.ratings[t])
            cur_year = dates[i]

        rh_pre[i] = model.ratings.get(home[i], config.base)
        ra_pre[i] = model.ratings.get(away[i], config.base)
        e_home[i] = model.update_match(
            home[i], away[i], s_home[i], margin[i],
            bool(neutral[i]), bool(world_cup[i]))

        if track_history:
            history.append({
                "date": df["date"].iloc[i],
                **{t: r for t, r in model.ratings.items()},
            })

    pred = df[["date", "home_team", "away_team", "home_score", "away_score",
               "neutral", "world_cup", "margin", "s_home"]].copy()
    pred["rating_home_pre"] = rh_pre
    pred["rating_away_pre"] = ra_pre
    pred["e_home"] = e_home

    hist_df = pd.DataFrame(history) if track_history else None
    return pred, model, hist_df


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pred: pd.DataFrame, start_year: Optional[int] = None,
             eps: float = 1e-12) -> dict:
    """Prequential scores on the chosen window (burn-in via start_year)."""
    p = pred
    if start_year is not None:
        p = p[p["date"].dt.year >= start_year]
    e = np.clip(p["e_home"].to_numpy(), eps, 1 - eps)
    s = p["s_home"].to_numpy()

    log_loss = float(np.mean(-(s * np.log(e) + (1 - s) * np.log(1 - e))))
    brier = float(np.mean((e - s) ** 2))

    decisive = s != 0.5
    if decisive.sum():
        pred_home = e[decisive] > 0.5
        actual_home = s[decisive] == 1.0
        accuracy = float(np.mean(pred_home == actual_home))
    else:
        accuracy = float("nan")

    return {
        "n": int(len(p)),
        "log_loss": log_loss,
        "brier": brier,
        "accuracy": accuracy,
    }


def baselines(pred: pd.DataFrame, start_year: Optional[int] = None) -> dict:
    """Reference scores: uninformed (E=0.5) and fixed home-win-rate prior."""
    p = pred
    if start_year is not None:
        p = p[p["date"].dt.year >= start_year]
    s = p["s_home"].to_numpy()

    def score(e_vec):
        e = np.clip(e_vec, 1e-12, 1 - 1e-12)
        return {
            "log_loss": float(np.mean(-(s * np.log(e) + (1 - s) * np.log(1 - e)))),
            "brier": float(np.mean((e - s) ** 2)),
        }

    p_home = float(np.mean(s))  # in-sample home expected score (incl. draws)
    return {
        "uninformed (E=0.5)": score(np.full_like(s, 0.5)),
        f"home-rate prior (E={p_home:.3f})": score(np.full_like(s, p_home)),
    }


def calibration_table(pred: pd.DataFrame, start_year: Optional[int] = None,
                      bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted vs realised home expected score by bin."""
    p = pred
    if start_year is not None:
        p = p[p["date"].dt.year >= start_year]
    e = p["e_home"].to_numpy()
    s = p["s_home"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(e, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({
                "bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})",
                "n": int(m.sum()),
                "pred_mean": float(e[m].mean()),
                "actual_mean": float(s[m].mean()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def tune_grid(df: pd.DataFrame, k_grid, hfa_grid,
              eval_start_year: int, metric: str = "log_loss",
              base_config: Optional[EloConfig] = None) -> pd.DataFrame:
    """Grid-search K and HFA minimising a prequential metric on the eval window.

    Ratings are built from the full history (1871 on) for every grid point;
    only the *scoring* is restricted to eval_start_year onward.
    """
    base = base_config or EloConfig()
    rows = []
    for k in k_grid:
        for hfa in hfa_grid:
            cfg = EloConfig(k=k, hfa=hfa, scale=base.scale, base=base.base,
                            use_mov=base.use_mov, mov_cap=base.mov_cap,
                            regress=base.regress,
                            world_cup_mult=base.world_cup_mult)
            pred, _, _ = run_elo(df, cfg)
            m = evaluate(pred, start_year=eval_start_year)
            rows.append({"k": k, "hfa": hfa, **m})
    out = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
    return out


def current_ranking(model: EloModel) -> pd.DataFrame:
    """Final ratings as a ranked table."""
    rows = [{"team": t, "rating": r, "games": model.games_played.get(t, 0)}
            for t, r in model.ratings.items()]
    return (pd.DataFrame(rows)
            .sort_values("rating", ascending=False)
            .reset_index(drop=True))
