"""Why does a strategy work? Battlefield-by-battlefield analysis against the baseline field.

For one allocation (default: the selection-rule pick), plays every non-held-out pool
strategy, weighted as in the expected baseline field (scenario shares × within-group
weights; no Dirichlet randomization), and reports:
1. per battlefield: how often it wins / ties / loses that battlefield, overall and by
   opponent group
2. by opponent group: match win rate, mean score for and against, how often each bonus
   applies
3. the most common won-sets: how often, the score they give, and the match win rate
   when that is what it won

Run from the project root:
    .venv/bin/python analysis/pick_analysis.py [pool_id ...]
Defaults to the top of results/robustness_final.csv and the original leader P01229.
Writes results/pick_analysis_<id>_{battlefields,groups,wonsets}.csv.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from matrix import BIT_VALUES, build_lut200, won_from_mask
from pool import BATTLEFIELD_COLUMNS, load_pool
from robustness import GROUPS, SCENARIO_SHARES, group_vectors, scenario_mix
from scoring import BONUS_BATTLEFIELD, NUM_BATTLEFIELDS

RESULTS = ROOT / "results"
DEFAULT_COMPARISON = "P01229"
TOP_WONSETS = 15


def format_set(battlefields) -> str:
    return "{" + " ".join(map(str, sorted(battlefields))) + "}"


def matchups(allocation: np.ndarray, opponents: np.ndarray, lut: np.ndarray) -> dict[str, np.ndarray]:
    """Per opponent: which battlefields are won/tied/lost, both won-masks, both scores
    (×200), and the match result (1 / 0.5 / 0)."""
    won = allocation[None, :] > opponents
    lost = allocation[None, :] < opponents
    my_mask, their_mask = won @ BIT_VALUES, lost @ BIT_VALUES
    my_score, their_score = lut[my_mask], lut[their_mask]
    return {
        "won": won, "lost": lost, "tied": ~won & ~lost,
        "my_mask": my_mask, "my_score": my_score, "their_score": their_score,
        "result": (1 + np.sign(my_score - their_score)) / 2,
    }


def battlefield_table(m: dict, weights: np.ndarray, group_weights: dict[str, np.ndarray], allocation: np.ndarray) -> pd.DataFrame:
    rows = []
    for b in range(NUM_BATTLEFIELDS):
        row = {
            "battlefield": b + 1,
            "troops": int(allocation[b]),
            "p_win": float(weights @ m["won"][:, b]),
            "p_tie": float(weights @ m["tied"][:, b]),
            "p_lose": float(weights @ m["lost"][:, b]),
        }
        for group, w in group_weights.items():
            row[f"p_win_vs_{group}"] = float(w @ m["won"][:, b])
        rows.append(row)
    return pd.DataFrame(rows)


def group_table(m: dict, weights: np.ndarray, group_weights: dict[str, np.ndarray], field_shares: dict[str, float]) -> pd.DataFrame:
    base = np.array([sum(won_from_mask(int(mask))) for mask in range(1024)])
    rule2 = (base % 5 == 0) & (base > 0)
    rule3 = np.array([BONUS_BATTLEFIELD in won_from_mask(int(mask)) for mask in range(1024)])
    adjacency = np.array([any(((mask >> i) & 3) == 3 for i in range(NUM_BATTLEFIELDS - 1)) for mask in range(1024)])
    rows = []
    for group, w in [("whole field", weights), *group_weights.items()]:
        rows.append({
            "opponents": group,
            "share_of_field": 1.0 if group == "whole field" else field_shares[group],
            "win_rate": float(w @ m["result"]),
            "mean_score": float(w @ m["my_score"]) / 200,
            "mean_opponent_score": float(w @ m["their_score"]) / 200,
            "battlefields_won": float(w @ m["won"].sum(axis=1)),
            "p_rule1_adjacency": float(w @ adjacency[m["my_mask"]]),
            "p_rule2_x1.4": float(w @ rule2[m["my_mask"]]),
            "p_rule3_battlefield6": float(w @ rule3[m["my_mask"]]),
        })
    return pd.DataFrame(rows)


def wonset_table(m: dict, weights: np.ndarray, lut: np.ndarray, top: int = TOP_WONSETS) -> pd.DataFrame:
    frame = pd.DataFrame({"mask": m["my_mask"], "w": weights, "win": m["result"] * weights})
    grouped = frame.groupby("mask").agg(share=("w", "sum"), wins=("win", "sum"))
    grouped = grouped[grouped["share"] > 0].sort_values("share", ascending=False).head(top)
    return pd.DataFrame({
        "won_set": [format_set(won_from_mask(int(mask))) for mask in grouped.index],
        "share_of_matches": grouped["share"].to_numpy(),
        "score": lut[grouped.index.to_numpy()] / 200,
        "base": [sum(won_from_mask(int(mask))) for mask in grouped.index],
        "match_win_rate_given_this_set": (grouped["wins"] / grouped["share"]).to_numpy(),
    })


def field_weights(pool: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Expected baseline field over pool rows, each group's own weights, and shares."""
    vectors = group_vectors(pool)
    names, shares = scenario_mix(SCENARIO_SHARES["baseline"], vectors)
    weights = sum(s * vectors[n] for s, n in zip(shares, names))
    return weights, {n: vectors[n] for n in GROUPS if n in names}, dict(zip(names, shares))


def analyze(pool_id: str, pool: pd.DataFrame, lut: np.ndarray) -> dict[str, pd.DataFrame]:
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    allocation = allocations[pd.Index(pool["id"]).get_loc(pool_id)]
    weights, group_weights, shares = field_weights(pool)
    m = matchups(allocation, allocations, lut)
    return {
        "battlefields": battlefield_table(m, weights, group_weights, allocation),
        "groups": group_table(m, weights, group_weights, shares),
        "wonsets": wonset_table(m, weights, lut),
    }


def main() -> None:
    ids = sys.argv[1:]
    if not ids:
        top = pd.read_csv(RESULTS / "robustness_final.csv", keep_default_na=False).iloc[0]["label"]
        ids = [top, DEFAULT_COMPARISON]
    pool, _ = load_pool()
    lut = build_lut200()
    pd.set_option("display.width", 220)
    for pool_id in ids:
        tables = analyze(pool_id, pool, lut)
        allocation = pool.loc[pool["id"] == pool_id, BATTLEFIELD_COLUMNS].iloc[0].tolist()
        print(f"\n===== {pool_id} {allocation} vs the expected baseline field =====")
        bf = tables["battlefields"]
        print(bf[["battlefield", "troops", "p_win", "p_tie", "p_lose"]].round(3).to_string(index=False))
        print("\n", tables["groups"].round(3).to_string(index=False))
        print("\nMost common won-sets:\n", tables["wonsets"].round(3).to_string(index=False))
        for name, table in tables.items():
            table.to_csv(RESULTS / f"pick_analysis_{pool_id}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
