"""Stage 4: diagnostic rankings and the shortlist.

Reads cache/pool.csv and the cached matrices; writes results/rankings_*.csv.

1. Round-robin win rate against the pool, two ways:
   - family weighting (main): each source family gets equal total weight, each
     subfamily an equal share of its family, rows within a subfamily by duplicate count
   - raw weighting: every submission counts once (Riddler-dominated; for comparison)
   plus the same with draws counted as 0, and mean score (family weighting).
2. Iterated elimination: rank by family-weighted win rate, keep the top half, recompute
   the family weights among survivors only, repeat. Every strategy's rank in every round
   is recorded, so "bullies" (strong only against weak opponents) show up as falling ranks.
3. Shortlist for robustness.py: the top SHORTLIST_TOP by round-robin win rate, plus the
   top SHORTLIST_TOP by elimination depth (rounds survived, then rank in the last round),
   plus the top SHORTLIST_PER_FAMILY of every source family, so strategies that might do
   best in other field scenarios (e.g. the best AI and hand-built ones) also get tested.

Held-out origins (pool.is_holdout) are excluded everywhere: they are neither opponents
nor candidates. Tests check that changing their matrix entries changes nothing here.

Run from the project root (after pool.py and matrix.py):
    .venv/bin/python rankings.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from matrix import build_matrices
from pool import BATTLEFIELD_COLUMNS, is_holdout, load_pool, origin_family, origin_subfamily

RESULTS = Path(__file__).resolve().parent / "results"
SHORTLIST_TOP = 100  # from each of the two lists; the union is the shortlist
SHORTLIST_PER_FAMILY = 5  # plus the best few of every source family
ELIMINATION_KEEP = 0.5


# --- Weights ----------------------------------------------------------------


def family_weights(pool: pd.DataFrame, include: np.ndarray | None = None) -> np.ndarray:
    """Opponent weights summing to 1: equal per family, equal per subfamily within a
    family, then by duplicate count. Held-out rows, and rows not in `include` (a boolean
    mask), get 0."""
    eligible = ~pool["origin"].map(is_holdout).to_numpy()
    if include is not None:
        eligible &= include
    weights = np.zeros(len(pool))
    families = pool["origin"].map(origin_family).to_numpy()
    subfamilies = pool["origin"].map(origin_subfamily).to_numpy()
    counts = pool["weight"].to_numpy(dtype=float)
    present_families = sorted(set(families[eligible]))
    for family in present_families:
        in_family = eligible & (families == family)
        present_subfamilies = sorted(set(subfamilies[in_family]))
        for subfamily in present_subfamilies:
            rows = in_family & (subfamilies == subfamily)
            weights[rows] = counts[rows] / counts[rows].sum() / len(present_subfamilies) / len(present_families)
    return weights


def raw_weights(pool: pd.DataFrame) -> np.ndarray:
    """Opponent weights proportional to duplicate count (every submission once).
    Held-out rows get 0."""
    weights = pool["weight"].to_numpy(dtype=float) * ~pool["origin"].map(is_holdout).to_numpy()
    return weights / weights.sum()


# --- Win rates --------------------------------------------------------------


def weighted_rates(payoff: np.ndarray, weights: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """For each row index in `rows`: the weighted mean of payoff[row, :], skipping the
    row's own column (you never play yourself). payoff is any (N, N) array, e.g. W."""
    totals = payoff[rows].astype(np.float64) @ weights
    own = payoff[rows, rows].astype(np.float64) * weights[rows]
    return (totals - own) / (1.0 - weights[rows])


def round_robin(pool: pd.DataFrame, W2: np.ndarray, S200: np.ndarray) -> pd.DataFrame:
    """Win rates and mean score for every non-held-out strategy."""
    candidates = np.flatnonzero(~pool["origin"].map(is_holdout).to_numpy())
    family = family_weights(pool)
    raw = raw_weights(pool)
    table = pool.iloc[candidates][["id", "origin", *BATTLEFIELD_COLUMNS]].copy()
    table["family"] = table["origin"].map(origin_family)
    table["win_rate"] = weighted_rates(W2, family, candidates) / 2
    table["win_rate_raw"] = weighted_rates(W2, raw, candidates) / 2
    table["win_rate_draw0"] = weighted_rates((W2 == 2).astype(np.int8), family, candidates)
    table["mean_score"] = weighted_rates(S200, family, candidates) / 200
    table["rank"] = rank_order(table["win_rate"], table["mean_score"], table["id"])
    table["rank_raw"] = rank_order(table["win_rate_raw"], table["mean_score"], table["id"])
    return table.sort_values("rank").reset_index(drop=True)


def rank_order(win_rate: pd.Series, mean_score: pd.Series, ids: pd.Series) -> np.ndarray:
    """1 = best. Ties broken by mean score, then id (so ranks are deterministic)."""
    order = pd.DataFrame({"w": win_rate.to_numpy(), "s": mean_score.to_numpy(), "id": ids.to_numpy()})
    order = order.sort_values(["w", "s", "id"], ascending=[False, False, True])
    ranks = np.empty(len(order), dtype=int)
    ranks[order.index.to_numpy()] = np.arange(1, len(order) + 1)
    return ranks


# --- Iterated elimination ---------------------------------------------------


def iterated_elimination(pool: pd.DataFrame, W2: np.ndarray, S200: np.ndarray, keep: float = ELIMINATION_KEEP) -> pd.DataFrame:
    """One row per non-held-out strategy: its rank in each round (blank after it is
    eliminated), rounds survived, and its rank in the last round it played."""
    alive = ~pool["origin"].map(is_holdout).to_numpy()
    ids = pool["id"].to_numpy()
    table = pd.DataFrame({"id": ids[alive], "origin": pool["origin"].to_numpy()[alive]}).set_index("id")
    table["rounds_survived"] = 0
    round_number = 0
    while alive.sum() >= 2:
        rows = np.flatnonzero(alive)
        weights = family_weights(pool, include=alive)
        win = weighted_rates(W2, weights, rows) / 2
        score = weighted_rates(S200, weights, rows) / 200
        ranks = rank_order(pd.Series(win), pd.Series(score), pd.Series(ids[rows]))
        table.loc[ids[rows], f"rank_round_{round_number}"] = ranks
        table.loc[ids[rows], "last_rank"] = ranks
        survivors = rows[ranks <= math.ceil(len(rows) * keep)]
        table.loc[ids[survivors], "rounds_survived"] += 1
        alive = np.zeros_like(alive)
        alive[survivors] = True
        round_number += 1
    table["last_rank"] = table["last_rank"].astype(int)
    for column in [c for c in table.columns if c.startswith("rank_round_")]:
        table[column] = table[column].astype("Int64")
    return table.reset_index().sort_values(["rounds_survived", "last_rank"], ascending=[False, True]).reset_index(drop=True)


# --- Shortlist --------------------------------------------------------------


def shortlist(
    round_robin_table: pd.DataFrame,
    elimination_table: pd.DataFrame,
    top: int = SHORTLIST_TOP,
    per_family: int = SHORTLIST_PER_FAMILY,
) -> pd.DataFrame:
    """Union of the top `top` by round-robin rank, the top `top` by elimination depth, and
    the top `per_family` of each source family, with the reason(s) each was included."""
    by_round_robin = set(round_robin_table.nsmallest(top, "rank")["id"])
    by_elimination = set(elimination_table.head(top)["id"])
    by_family = set(round_robin_table.sort_values("rank").groupby("family").head(per_family)["id"])
    reasons = (("round_robin", by_round_robin), ("elimination", by_elimination), ("family_top", by_family))
    chosen = round_robin_table[round_robin_table["id"].isin(by_round_robin | by_elimination | by_family)].copy()
    chosen["reason"] = ["+".join(r for r, ids in reasons if i in ids) for i in chosen["id"]]
    depth = elimination_table.set_index("id")[["rounds_survived", "last_rank"]]
    return chosen.join(depth, on="id").sort_values("rank").reset_index(drop=True)


# --- Main -------------------------------------------------------------------


def main() -> None:
    pool, meta = load_pool()
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    matrices, _ = build_matrices(pool["id"].to_numpy(dtype=str), allocations, meta["pool_hash"])

    rr = round_robin(pool, matrices.W2, matrices.S200)
    elimination = iterated_elimination(pool, matrices.W2, matrices.S200)
    short = shortlist(rr, elimination)

    RESULTS.mkdir(exist_ok=True)
    rr.to_csv(RESULTS / "rankings_round_robin.csv", index=False)
    elimination.to_csv(RESULTS / "rankings_elimination.csv", index=False)
    short.to_csv(RESULTS / "rankings_shortlist.csv", index=False)
    (RESULTS / "rankings_meta.json").write_text(
        json.dumps(
            {
                "pool_hash": meta["pool_hash"],
                "lut_hash": matrices.lut_hash,
                "candidates": len(rr),
                "excluded_holdout_origins": meta["holdout_origins"],
                "shortlist_top": SHORTLIST_TOP,
                "shortlist_per_family": SHORTLIST_PER_FAMILY,
                "elimination_keep": ELIMINATION_KEEP,
                "shortlist_size": len(short),
            },
            indent=2,
        )
        + "\n"
    )

    pd.set_option("display.width", 160)
    show = ["rank", "id", "origin", *BATTLEFIELD_COLUMNS, "win_rate", "win_rate_raw", "mean_score"]
    print("Top 15 by family-weighted win rate:\n", rr[show].head(15).to_string(index=False))
    print("\nFamily medians:\n", rr.groupby("family")[["win_rate", "win_rate_raw", "mean_score"]].median().round(3).to_string())
    print(f"\nElimination: {elimination['rounds_survived'].max()} rounds; last survivor "
          f"{elimination.iloc[0]['id']} ({elimination.iloc[0]['origin']})")
    print(f"Shortlist: {len(short)} strategies; by origin family:\n", short["family"].value_counts().to_string())


if __name__ == "__main__":
    main()
