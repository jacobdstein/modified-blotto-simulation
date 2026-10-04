"""Pipeline step 3: diagnostic rankings, the Nash mix, and the shortlist.

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
4. Nash mix of the meta-game (payoff W, non-held-out pool), solved exactly as a linear
   program with scipy. robustness.py samples its "Nash" scenario fields from it.

Held-out origins (pool.is_holdout) are excluded everywhere: they are neither opponents
nor candidates. Opponent-only origins (pool.is_opponent_only, the final counterattack
round) are opponents in the round-robin weights but are never candidates, and take no
part in elimination. Tests check that changing their matrix entries changes nothing here.

Run from the project root (after pool.py and matrix.py):
    .venv/bin/python rankings.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from matrix import build_matrices
from pool import BATTLEFIELD_COLUMNS, is_holdout, is_opponent_only, load_pool, origin_family, origin_subfamily

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


def weighted_rates(payoff: np.ndarray, weights: np.ndarray, rows: np.ndarray, copies: np.ndarray) -> np.ndarray:
    """For each row index in `rows`: the weighted mean of payoff[row, :] over opponents.

    A row with copies = k stands for k identical submissions and carries their combined
    weight. A player never plays itself, so only 1/k of the row's own weight is removed;
    the other k - 1 copies stay in as opponents, and payoff[row, row] is exactly the
    result against an identical allocation (a 0-0 draw). payoff is any (N, N) array."""
    own_weight = weights[rows] / copies
    totals = payoff[rows].astype(np.float64) @ weights
    own = payoff[rows, rows].astype(np.float64) * own_weight
    return (totals - own) / (1.0 - own_weight)


def round_robin(pool: pd.DataFrame, W2: np.ndarray, S200: np.ndarray) -> pd.DataFrame:
    """Win rates and mean score for every non-held-out strategy."""
    candidates = np.flatnonzero(candidate_mask(pool))
    copies = pool["weight"].to_numpy(dtype=float)[candidates]
    family = family_weights(pool)
    raw = raw_weights(pool)
    table = pool.iloc[candidates][["id", "origin", *BATTLEFIELD_COLUMNS]].copy()
    table["family"] = table["origin"].map(origin_family)
    table["win_rate"] = weighted_rates(W2, family, candidates, copies) / 2
    table["win_rate_raw"] = weighted_rates(W2, raw, candidates, copies) / 2
    table["win_rate_draw0"] = weighted_rates((W2 == 2).astype(np.int8), family, candidates, copies)
    table["mean_score"] = weighted_rates(S200, family, candidates, copies) / 200
    table["rank"] = rank_order(table["win_rate"], table["mean_score"], table["id"])
    table["rank_raw"] = rank_order(table["win_rate_raw"], table["mean_score"], table["id"])
    return table.sort_values("rank").reset_index(drop=True)


def candidate_mask(pool: pd.DataFrame) -> np.ndarray:
    """Rows that may be ranked and picked: not held out, not opponent-only."""
    origins = pool["origin"]
    return (~origins.map(is_holdout) & ~origins.map(is_opponent_only)).to_numpy()


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
    alive = candidate_mask(pool)
    ids = pool["id"].to_numpy()
    table = pd.DataFrame({"id": ids[alive], "origin": pool["origin"].to_numpy()[alive]}).set_index("id")
    table["rounds_survived"] = 0
    round_number = 0
    while alive.sum() >= 2:
        rows = np.flatnonzero(alive)
        copies = pool["weight"].to_numpy(dtype=float)[rows]
        weights = family_weights(pool, include=alive)
        win = weighted_rates(W2, weights, rows, copies) / 2
        score = weighted_rates(S200, weights, rows, copies) / 200
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


# --- Nash mix ---------------------------------------------------------------


def nash_mix(pool: pd.DataFrame, W2: np.ndarray) -> pd.DataFrame:
    """Equilibrium mix of the meta-game over non-held-out strategies.

    The game is symmetric and constant-sum (W[i, j] + W[j, i] = 1), so with
    A = W - 1/2 we find a probability vector x maximizing v subject to
    (x^T A)_j >= v for every strategy j. Its value is 0: no single strategy in the pool
    beats the mix on average. Returns id, origin, probability (support only) and the
    exploitability (the best any strategy does against the mix, minus 1/2)."""
    rows = np.flatnonzero(~pool["origin"].map(is_holdout).to_numpy())
    A = W2[np.ix_(rows, rows)].astype(np.float64) / 2 - 0.5
    n = len(rows)
    cost = np.zeros(n + 1)
    cost[-1] = -1.0  # maximize v
    upper = np.hstack([-A.T, np.ones((n, 1))])  # v - (x^T A)_j <= 0
    equality = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
    bounds = [(0, None)] * n + [(None, None)]
    result = linprog(cost, A_ub=upper, b_ub=np.zeros(n), A_eq=equality, b_eq=[1.0], bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Nash LP failed: {result.message}")
    x = np.clip(result.x[:n], 0, None)
    x /= x.sum()
    exploitability = float((A @ x).max())  # best strategy's payoff against the mix, minus 1/2
    support = x > 1e-9
    table = pool.iloc[rows[support]][["id", "origin", *BATTLEFIELD_COLUMNS]].copy()
    table["probability"] = x[support]
    table.attrs["exploitability"] = exploitability
    table.attrs["value"] = float(result.x[-1])
    return table.sort_values("probability", ascending=False).reset_index(drop=True)


# --- Main -------------------------------------------------------------------


def main() -> None:
    pool, meta = load_pool()
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    matrices, _ = build_matrices(pool["id"].to_numpy(dtype=str), allocations, meta["pool_hash"])

    rr = round_robin(pool, matrices.W2, matrices.S200)
    elimination = iterated_elimination(pool, matrices.W2, matrices.S200)
    short = shortlist(rr, elimination)
    nash = nash_mix(pool, matrices.W2)

    RESULTS.mkdir(exist_ok=True)
    rr.to_csv(RESULTS / "rankings_round_robin.csv", index=False)
    elimination.to_csv(RESULTS / "rankings_elimination.csv", index=False)
    short.to_csv(RESULTS / "rankings_shortlist.csv", index=False)
    nash.to_csv(RESULTS / "rankings_nash.csv", index=False)
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
                "nash_support": len(nash),
                "nash_value": nash.attrs["value"],
                "nash_exploitability": nash.attrs["exploitability"],
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
    print(f"\nNash mix: {len(nash)} strategies in support, value {nash.attrs['value']:.2e}, "
          f"exploitability {nash.attrs['exploitability']:.2e}")
    print(nash.head(8)[["id", "origin", *BATTLEFIELD_COLUMNS, "probability"]].to_string(index=False))


if __name__ == "__main__":
    main()
