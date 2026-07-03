"""
run_baseline.py
===============
Fit, tune and evaluate the baseline (binary, home-adjusted) Elo model on
international tier-1 rugby, then emit the current ranking, a calibration table
and a rating-history plot.

Usage:  python run_baseline.py
"""
import os
import numpy as np
import pandas as pd

import rugby_elo as re

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "data", "rugby_results_master.csv")
DATA = MASTER if os.path.exists(MASTER) else os.path.join(HERE, "data", "rugby_results.csv")
TIER1_ONLY = DATA == MASTER      # master mixes tiers post-2023; filter to tier-1
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

EVAL_START = 2000   # window used to tune & headline-evaluate (modern era)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def main():
    df = re.load_results(DATA, tier1_only=TIER1_ONLY)
    print(f"Loaded {len(df)} matches, "
          f"{df.date.min().date()} -> {df.date.max().date()} "
          f"({'tier-1 only from master' if TIER1_ONLY else 'historical'})\n")

    # ---- Stage 1: coarse grid search on K, HFA --------------------------- #
    coarse = re.tune_grid(
        df,
        k_grid=np.arange(8, 49, 4),
        hfa_grid=np.arange(0, 101, 10),
        eval_start_year=EVAL_START,
        metric="log_loss",
    )
    b = coarse.iloc[0]
    print(f"Coarse best: K={b.k:g}, HFA={b.hfa:g}  "
          f"(log_loss={b.log_loss:.4f})")

    # ---- Stage 2: refine around the coarse optimum ----------------------- #
    fine = re.tune_grid(
        df,
        k_grid=np.arange(max(2, b.k - 3), b.k + 3.1, 1),
        hfa_grid=np.arange(max(0, b.hfa - 10), b.hfa + 10.1, 2.5),
        eval_start_year=EVAL_START,
        metric="log_loss",
    )
    best = fine.iloc[0]
    cfg = re.EloConfig(k=float(best.k), hfa=float(best.hfa))
    print(f"Refined best: {cfg.describe()}  "
          f"(log_loss={best.log_loss:.4f}, "
          f"brier={best.brier:.4f}, acc={best.accuracy:.3f})\n")
    coarse.to_csv(os.path.join(OUT, "tune_grid_coarse.csv"), index=False)
    fine.to_csv(os.path.join(OUT, "tune_grid_fine.csv"), index=False)

    # ---- Final fit with history ------------------------------------------ #
    pred, model, hist = re.run_elo(df, cfg, track_history=True)

    # ---- Evaluation vs baselines ----------------------------------------- #
    print("=" * 64)
    print(f"PREQUENTIAL EVALUATION (out-of-sample, {EVAL_START}+)")
    print("=" * 64)
    elo_m = re.evaluate(pred, start_year=EVAL_START)
    base_m = re.baselines(pred, start_year=EVAL_START)
    print(f"{'model':<34}{'log_loss':>10}{'brier':>9}{'acc':>8}")
    for name, m in base_m.items():
        print(f"{name:<34}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{'':>8}")
    print(f"{'Elo (' + cfg.describe() + ')':<34}"
          f"{elo_m['log_loss']:>10.4f}{elo_m['brier']:>9.4f}"
          f"{elo_m['accuracy']:>8.3f}")
    print(f"(n = {elo_m['n']} matches)\n")

    for yr in (2010, 2015):
        m = re.evaluate(pred, start_year=yr)
        print(f"  Elo on {yr}+ : log_loss={m['log_loss']:.4f}, "
              f"brier={m['brier']:.4f}, acc={m['accuracy']:.3f} (n={m['n']})")
    print()

    # ---- Calibration ----------------------------------------------------- #
    cal = re.calibration_table(pred, start_year=EVAL_START, bins=10)
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False)
    print("CALIBRATION (predicted vs realised home expected score, 2000+):")
    print(cal.to_string(index=False), "\n")

    # ---- Current ranking ------------------------------------------------- #
    rank = re.current_ranking(model)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    rank["rating"] = rank["rating"].round(1)
    rank.to_csv(os.path.join(OUT, "current_ratings.csv"), index=False)
    print("=" * 64)
    print(f"ELO RANKING as of {df.date.max().date()}")
    print("=" * 64)
    print(rank.to_string(index=False))

    # ---- Rating history plot --------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
        h = hist.set_index("date").sort_index()
        h = h.reindex(columns=sorted(teams)).ffill()
        # year-end snapshot for a readable line chart
        h_annual = h.resample("YE").last()

        fig, ax = plt.subplots(figsize=(13, 7))
        for t in sorted(teams):
            ax.plot(h_annual.index, h_annual[t], label=t, linewidth=1.4)
        ax.axhline(1500, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"International rugby Elo, {df.date.min().year}"
                     f"-{df.date.max().year}  ({cfg.describe()})")
        ax.set_ylabel("Elo rating")
        ax.set_xlabel("Year")
        ax.legend(ncol=2, fontsize=9, loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = os.path.join(OUT, "rating_history.png")
        fig.savefig(p, dpi=130)
        # modern-era zoom
        fig2, ax2 = plt.subplots(figsize=(13, 7))
        hz = h_annual[h_annual.index.year >= 1995]
        for t in sorted(teams):
            ax2.plot(hz.index, hz[t], label=t, linewidth=1.6)
        ax2.axhline(1500, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax2.set_title(f"International rugby Elo, 1995-{df.date.max().year}")
        ax2.set_ylabel("Elo rating"); ax2.set_xlabel("Year")
        ax2.legend(ncol=2, fontsize=9, loc="lower left")
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        p2 = os.path.join(OUT, "rating_history_modern.png")
        fig2.savefig(p2, dpi=130)
        print(f"\nSaved plots: {p}\n            {p2}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[plot skipped: {exc}]")


if __name__ == "__main__":
    main()
