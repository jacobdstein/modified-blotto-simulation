"""Pipeline step 4: simulated contests and the selection rule.

Reads cache/pool.csv, the cached matrices, results/rankings_shortlist.csv and
results/rankings_nash.csv. Writes results/robustness_*.csv.

One simulated contest: sample a field of 100–225 players from a scenario, add the
candidate, compute every player's total wins (draw = 0.5) and total points against all
the others, and record the candidate's rank / players ("percentile"; smaller is better).
Every candidate faces the same sampled fields.

Runs (all seeded; see CLAUDE.md "Objective and selection rule"):
1. selection — every shortlisted strategy, SELECTION_CONTESTS per scenario
2. final     — the top FINALISTS, FINAL_CONTESTS per scenario on a fresh seed
3. hold-out  — finalists vs the baseline scenario with Riddler groups drawn from round 4
4. neighbors — every one-troop move of each finalist, NEIGHBOR_CONTESTS per scenario

Run from the project root (after pool.py, matrix.py, rankings.py):
    .venv/bin/python robustness.py
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from matrix import Matrices, build_lut200, build_matrices, payoff_block
from pool import BATTLEFIELD_COLUMNS, is_holdout, load_pool, origin_family, origin_subfamily
from scoring import NUM_BATTLEFIELDS, validate_allocation

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# --- Settings (fixed before seeing results; mirrored in CLAUDE.md) ----------

GROUPS = (
    "naive_rules", "naive_blotto", "experienced_blotto", "clumsy",
    "ai_gemini", "ai_chatgpt", "ai_claude", "random", "counter",
)
SCENARIO_SHARES = {  # % of the field per group
    "baseline": dict(naive_rules=7, naive_blotto=7, experienced_blotto=20, clumsy=15,
                     ai_gemini=10, ai_chatgpt=18, ai_claude=13, random=3, counter=7),
    "naive_heavy": dict(naive_rules=30, naive_blotto=20, experienced_blotto=5, clumsy=15,
                        ai_gemini=15, ai_chatgpt=10, ai_claude=5, random=0, counter=0),
    "sophisticated_heavy": dict(naive_rules=5, naive_blotto=0, experienced_blotto=25, clumsy=5,
                                ai_gemini=5, ai_chatgpt=20, ai_claude=25, random=0, counter=15),
    "ai_heavy": dict(naive_rules=10, naive_blotto=5, experienced_blotto=10, clumsy=5,
                     ai_gemini=20, ai_chatgpt=25, ai_claude=20, random=0, counter=5),
}
# The score: pooled P(top 20%) over the realistic scenarios (weights are relative).
POOLED_WEIGHTS = {"baseline": 40, "sophisticated_heavy": 24, "ai_heavy": 19, "naive_heavy": 7}
# Nash is run and reported, and breaks ties, but is not in the score or the flags:
# no strategy can reliably reach the top 20% of an equilibrium field.
TIEBREAK_SCENARIO = "nash"
SCENARIOS = ("baseline", "sophisticated_heavy", "ai_heavy", "nash", "naive_heavy")
TIE_DECIMALS = 2  # pooled P(top 20%) equal to the nearest percentage point counts as a tie

DIRICHLET_CONCENTRATION = 20
FIELD_SIZE = (100, 225)  # inclusive
TOP_SHARE = 0.20
FLAG_BELOW = 0.50
AI_FIRST_ANSWER_WEIGHT = 3

SELECTION_CONTESTS = 2_000
FINAL_CONTESTS = 10_000
FINALISTS = 10
HOLDOUT_CONTESTS = 10_000
HOLDOUT_MIN = 0.50
HOLDOUT_MAX_DROP = 0.15
NEIGHBOR_CONTESTS = 2_000
NEIGHBOR_CLOSE = 0.10

SEED = 20261004
RUN_SEEDS = {"selection": 1, "final": 2, "holdout": 3, "neighbors": 4}


# --- Groups -----------------------------------------------------------------


def groups_of(origin: str, holdout: bool = False) -> list[str]:
    """The field group(s) a pool row belongs to. Normally held-out rows belong to none.
    In the hold-out check, raw round 4 fills both Riddler-human groups, adjusted round 4
    the clumsy group, and rounds 1–3 Riddler rows are left out."""
    family = origin_family(origin)
    if family in ("riddler", "riddler_adj"):
        if holdout != is_holdout(origin):
            return []
        if family == "riddler_adj":
            return ["clumsy"]
        if holdout:
            return ["naive_blotto", "experienced_blotto"]
        return ["naive_blotto"] if origin == "riddler:r1" else ["experienced_blotto"]
    return {
        "hand": ["naive_rules"],
        "llm:gemini": ["ai_gemini"],
        "llm:chatgpt": ["ai_chatgpt"],
        "llm:claude": ["ai_claude"],
        "random_uniform": ["random"],
        "counter": ["counter"],
    }[family]


def within_group_weights(pool: pd.DataFrame, rows: np.ndarray, group: str) -> np.ndarray:
    """Weights over `rows` (summing to 1). AI groups: each answer (original + variants)
    gets 1 share, the first-listed answer AI_FIRST_ANSWER_WEIGHT shares; other groups:
    equal per subfamily. Within an answer or subfamily, rows weigh by duplicate count."""
    subfamilies = pool["origin"].to_numpy()[rows]
    subfamilies = np.array([origin_subfamily(o) for o in subfamilies])
    counts = pool["weight"].to_numpy(dtype=float)[rows]
    weights = np.zeros(len(rows))
    for subfamily in sorted(set(subfamilies)):
        members = subfamilies == subfamily
        if group.startswith("ai_"):
            share = AI_FIRST_ANSWER_WEIGHT if subfamily.split(":")[2] == "1" else 1
        else:
            share = 1
        weights[members] = share * counts[members] / counts[members].sum()
    return weights / weights.sum()


def group_vectors(pool: pd.DataFrame, holdout: bool = False) -> dict[str, np.ndarray]:
    """For each non-empty group: a probability vector over all pool rows."""
    membership = [groups_of(o, holdout) for o in pool["origin"]]
    vectors = {}
    for group in GROUPS:
        rows = np.array([i for i, gs in enumerate(membership) if group in gs], dtype=int)
        if len(rows):
            vector = np.zeros(len(pool))
            vector[rows] = within_group_weights(pool, rows, group)
            vectors[group] = vector
    return vectors


def scenario_mix(shares: dict[str, float], vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Groups with a positive share and at least one row, and their shares renormalized
    to sum to 1 (empty groups' shares spread proportionally over the rest)."""
    names = [g for g in GROUPS if shares.get(g, 0) > 0 and g in vectors]
    values = np.array([shares[g] for g in names], dtype=float)
    return names, values / values.sum()


# --- Candidates -------------------------------------------------------------


@dataclass
class Candidates:
    """Each candidate's results against every pool row.
    W2[c, j]: 2/1/0 for candidate c vs pool row j; S200[c, j]: c's score;
    S200_opp[c, j]: pool row j's score against c."""

    labels: list[str]
    allocations: np.ndarray
    W2: np.ndarray
    S200: np.ndarray
    S200_opp: np.ndarray

    @classmethod
    def from_pool(cls, matrices: Matrices, pool: pd.DataFrame, ids: list[str]) -> "Candidates":
        index = pd.Index(pool["id"]).get_indexer(ids)
        if (index < 0).any():
            raise KeyError(f"unknown pool ids: {[i for i, k in zip(ids, index) if k < 0]}")
        allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)[index]
        return cls(list(ids), allocations, matrices.W2[index], matrices.S200[index], matrices.S200[:, index].T.copy())

    @classmethod
    def from_allocations(cls, allocations: np.ndarray, pool_allocations: np.ndarray, labels: list[str]) -> "Candidates":
        """For allocations that may not be in the pool (e.g. ±1 neighbors)."""
        for allocation in allocations:
            validate_allocation(allocation.tolist())
        lut = build_lut200()
        S200, W2 = payoff_block(allocations, pool_allocations, lut)
        S200_opp, _ = payoff_block(pool_allocations, allocations, lut)
        return cls(list(labels), allocations, W2, S200, S200_opp.T.copy())


def one_troop_neighbors(allocation: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Every allocation reachable by moving one troop from battlefield a to b, with a
    label like '6→7'."""
    neighbors = []
    for source in range(NUM_BATTLEFIELDS):
        if allocation[source] == 0:
            continue
        for target in range(NUM_BATTLEFIELDS):
            if target != source:
                moved = allocation.copy()
                moved[source] -= 1
                moved[target] += 1
                neighbors.append((moved, f"{source + 1}→{target + 1}"))
    return neighbors


# --- Contests ---------------------------------------------------------------


def rank_fraction(field_totals: np.ndarray, candidate_totals: np.ndarray) -> np.ndarray:
    """Candidate rank / players, ties sharing the average rank (1 = best).
    field_totals: (C, n); candidate_totals: (C,)."""
    ahead = (field_totals > candidate_totals[:, None]).sum(axis=1)
    tied = (field_totals == candidate_totals[:, None]).sum(axis=1)
    rank = 1 + ahead + tied / 2
    return rank / (field_totals.shape[1] + 1)


def contest(field: np.ndarray, matrices: Matrices, candidates: Candidates) -> dict[str, np.ndarray]:
    """One contest: each candidate's rank fraction by wins, by points, by wins with draws
    counted as 0, and its mean score. `field` holds pool row indices (repeats allowed:
    repeated rows are separate players with identical allocations)."""
    W_field = matrices.W2[np.ix_(field, field)].astype(np.int32)
    S_field = matrices.S200[np.ix_(field, field)].astype(np.int64)
    # Each field player's totals against the rest of the field; the diagonal is the
    # player itself (W2 = 1, S200 = 0), which it never plays.
    wins_vs_field = W_field.sum(axis=1) - 1
    points_vs_field = S_field.sum(axis=1)
    pure_wins_vs_field = (W_field == 2).sum(axis=1)

    cand_W = matrices_slice(candidates.W2, field).astype(np.int32)  # (C, n)
    field_wins = wins_vs_field[None, :] + (2 - cand_W)
    field_points = points_vs_field[None, :] + matrices_slice(candidates.S200_opp, field)
    field_pure_wins = pure_wins_vs_field[None, :] + (cand_W == 0)

    cand_points = matrices_slice(candidates.S200, field).astype(np.int64).sum(axis=1)
    return {
        "wins": rank_fraction(field_wins, cand_W.sum(axis=1)),
        "points": rank_fraction(field_points, cand_points),
        "draw0": rank_fraction(field_pure_wins, (cand_W == 2).sum(axis=1)),
        "mean_score": cand_points / len(field) / 200,
    }


def matrices_slice(rows: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Columns `field` of a (C, N) candidate array."""
    return rows[:, field]


def field_probability(rng: np.random.Generator, names: list[str], shares: np.ndarray, vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Pool-row probabilities for one contest: group shares drawn from a Dirichlet around
    the scenario's shares, then each group's within-group weights."""
    drawn = rng.dirichlet(DIRICHLET_CONCENTRATION * shares)
    return sum(share * vectors[name] for share, name in zip(drawn, names))


def run_scenario(
    scenario: str,
    matrices: Matrices,
    candidates: Candidates,
    vectors: dict[str, np.ndarray],
    nash: np.ndarray,
    contests: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """`contests` simulated contests; returns arrays of shape (contests, C)."""
    n_pool = matrices.W2.shape[0]
    if scenario == "nash":
        names, shares = [], np.array([])
    else:
        names, shares = scenario_mix(SCENARIO_SHARES[scenario], vectors)
    out = {k: np.empty((contests, len(candidates.labels))) for k in ("wins", "points", "draw0", "mean_score")}
    for c in range(contests):
        p = nash if scenario == "nash" else field_probability(rng, names, shares, vectors)
        size = int(rng.integers(FIELD_SIZE[0], FIELD_SIZE[1] + 1))
        field = rng.choice(n_pool, size=size, p=p / p.sum())
        for key, value in contest(field, matrices, candidates).items():
            out[key][c] = value
    return out


def run(
    matrices: Matrices,
    candidates: Candidates,
    vectors: dict[str, np.ndarray],
    nash: np.ndarray,
    contests: int,
    run_name: str,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> dict[str, dict[str, np.ndarray]]:
    """Every scenario, each with its own seeded random stream."""
    return {
        s: run_scenario(s, matrices, candidates, vectors, nash, contests,
                        np.random.default_rng([SEED, RUN_SEEDS[run_name], i]))
        for i, s in enumerate(scenarios)
    }


# --- Summaries --------------------------------------------------------------


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def summarize(results: dict[str, dict[str, np.ndarray]], candidates: Candidates) -> pd.DataFrame:
    """Per candidate: P(top 20%) per scenario, pooled P(top 20%) by wins / points /
    draw 0 over the scored scenarios, pooled median percentile and mean score, flags, and
    the rank: pooled P(top 20%) to the nearest percentage point, then the tiebreak
    scenario's P(top 20%), then median percentile, then mean score."""
    scored = {s: r for s, r in results.items() if s in POOLED_WEIGHTS}
    total_weight = sum(POOLED_WEIGHTS[s] for s in scored)
    table = pd.DataFrame({"label": candidates.labels})
    for k in range(NUM_BATTLEFIELDS):
        table[BATTLEFIELD_COLUMNS[k]] = candidates.allocations[:, k]
    for scenario, r in results.items():
        table[f"p_top20_{scenario}"] = (r["wins"] <= TOP_SHARE).mean(axis=0)
    pooled = {"p_top20": 0.0, "p_top20_points": 0.0, "p_top20_draw0": 0.0, "mean_score": 0.0}
    for scenario, r in scored.items():
        w = POOLED_WEIGHTS[scenario] / total_weight
        p = table[f"p_top20_{scenario}"].to_numpy()
        pooled["p_top20"] = pooled["p_top20"] + w * p
        pooled["p_top20_points"] = pooled["p_top20_points"] + w * (r["points"] <= TOP_SHARE).mean(axis=0)
        pooled["p_top20_draw0"] = pooled["p_top20_draw0"] + w * (r["draw0"] <= TOP_SHARE).mean(axis=0)
        pooled["mean_score"] = pooled["mean_score"] + w * r["mean_score"].mean(axis=0)
    for key, value in pooled.items():
        table[key] = value
    # Pooled median percentile: every scored contest weighted by its scenario's share.
    values = np.concatenate([r["wins"] for r in scored.values()])
    weights = np.concatenate([np.full(len(r["wins"]), POOLED_WEIGHTS[s] / total_weight / len(r["wins"])) for s, r in scored.items()])
    table["median_percentile"] = [weighted_median(values[:, c], weights) for c in range(values.shape[1])]
    scenario_columns = [f"p_top20_{s}" for s in scored]
    table["worst_scenario_p_top20"] = table[scenario_columns].min(axis=1)
    table["flags"] = [
        "; ".join(
            f
            for f, bad in (
                ("scenario<50%", row.worst_scenario_p_top20 < FLAG_BELOW),
                ("points<50%", row.p_top20_points < FLAG_BELOW),
                ("draw0<50%", row.p_top20_draw0 < FLAG_BELOW),
            )
            if bad
        )
        for row in table.itertuples()
    ]
    tiebreak = table[f"p_top20_{TIEBREAK_SCENARIO}"] if f"p_top20_{TIEBREAK_SCENARIO}" in table else 0.0
    keys = table.assign(_rounded=table["p_top20"].round(TIE_DECIMALS), _tiebreak=tiebreak)
    order = keys.sort_values(["_rounded", "_tiebreak", "median_percentile", "mean_score"], ascending=[False, False, True, False]).index
    table.loc[order, "rank"] = np.arange(1, len(table) + 1)
    table["rank"] = table["rank"].astype(int)
    return table.sort_values("rank").reset_index(drop=True)


def holdout_check(final: pd.DataFrame, holdout_results: dict[str, np.ndarray]) -> pd.DataFrame:
    """Hold-out P(top 20%) and pass/fail per finalist."""
    p = (holdout_results["wins"] <= TOP_SHARE).mean(axis=0)
    out = final.copy()
    out["p_top20_holdout"] = p
    out["holdout_pass"] = (p >= HOLDOUT_MIN) & (p >= out["p_top20_baseline"] - HOLDOUT_MAX_DROP)
    return out


# --- Main -------------------------------------------------------------------


def unique_allocations(shortlist: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct allocation (the same allocation can enter the pool from
    several origins). Keeps the first row in shortlist order and lists the others' ids
    in same_allocation_ids."""
    key = shortlist[BATTLEFIELD_COLUMNS].astype(str).agg(",".join, axis=1)
    others = shortlist.groupby(key)["id"].apply(lambda ids: " ".join(ids.iloc[1:]))
    unique = shortlist[~key.duplicated()].copy()
    unique["same_allocation_ids"] = key[~key.duplicated()].map(others).to_numpy()
    return unique.reset_index(drop=True)


def nash_vector(pool: pd.DataFrame, nash_table: pd.DataFrame) -> np.ndarray:
    vector = np.zeros(len(pool))
    index = pd.Index(pool["id"]).get_indexer(nash_table["id"])
    if (index < 0).any():
        raise ValueError("rankings_nash.csv refers to ids not in the pool; re-run rankings.py")
    vector[index] = nash_table["probability"].to_numpy()
    return vector / vector.sum()


def main() -> None:
    started = time.perf_counter()
    pool, meta = load_pool()
    pool_allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    matrices, _ = build_matrices(pool["id"].to_numpy(dtype=str), pool_allocations, meta["pool_hash"])
    rankings_meta = json.loads((RESULTS / "rankings_meta.json").read_text())
    if rankings_meta["pool_hash"] != meta["pool_hash"]:
        raise ValueError("rankings results are for a different pool; re-run rankings.py")
    shortlist = unique_allocations(pd.read_csv(RESULTS / "rankings_shortlist.csv", keep_default_na=False))
    nash = nash_vector(pool, pd.read_csv(RESULTS / "rankings_nash.csv"))
    vectors = group_vectors(pool)
    holdout_vectors = group_vectors(pool, holdout=True)
    if any(is_holdout(o) for o in shortlist["origin"]):
        raise ValueError("held-out strategy in the shortlist")

    # 1. Selection run over the whole shortlist.
    candidates = Candidates.from_pool(matrices, pool, list(shortlist["id"]))
    selection = summarize(run(matrices, candidates, vectors, nash, SELECTION_CONTESTS, "selection"), candidates)
    selection = selection.merge(shortlist[["id", "origin", "reason", "same_allocation_ids"]], left_on="label", right_on="id").drop(columns="id")

    # 2. Final run: top finalists on a fresh seed.
    finalist_ids = list(selection.nsmallest(FINALISTS, "rank")["label"])
    finalists = Candidates.from_pool(matrices, pool, finalist_ids)
    final = summarize(run(matrices, finalists, vectors, nash, FINAL_CONTESTS, "final"), finalists)
    final = final.merge(shortlist[["id", "origin", "same_allocation_ids"]], left_on="label", right_on="id").drop(columns="id")

    # 3. Hold-out check (baseline shares, Riddler groups from round 4).
    names, shares = scenario_mix(SCENARIO_SHARES["baseline"], holdout_vectors)
    if "naive_blotto" not in names or "clumsy" not in names:
        raise ValueError("hold-out groups are empty")
    final_by_label = Candidates.from_pool(matrices, pool, list(final["label"]))
    holdout_results = run_scenario("baseline", matrices, final_by_label, holdout_vectors, nash, HOLDOUT_CONTESTS,
                                   np.random.default_rng([SEED, RUN_SEEDS["holdout"]]))
    final = holdout_check(final, holdout_results)

    # 4. ±1 neighborhoods: each finalist and its neighbors on the same fields.
    neighbor_rows, labels, allocations = [], [], []
    for label, allocation in zip(final["label"], final[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)):
        labels.append(f"{label}|self")
        allocations.append(allocation)
        for moved, move in one_troop_neighbors(allocation):
            labels.append(f"{label}|{move}")
            allocations.append(moved)
    neighbors = Candidates.from_allocations(np.array(allocations), pool_allocations, labels)
    neighbor_table = summarize(run(matrices, neighbors, vectors, nash, NEIGHBOR_CONTESTS, "neighbors"), neighbors)
    neighbor_table[["finalist", "move"]] = neighbor_table["label"].str.split("|", expand=True)
    for label in final["label"]:
        group = neighbor_table[neighbor_table["finalist"] == label]
        own = group.loc[group["move"] == "self", "p_top20"].item()
        others = group[group["move"] != "self"]
        close = (others["p_top20"] - own).abs() <= NEIGHBOR_CLOSE
        best = others.loc[others["p_top20"].idxmax()]
        i = final.index[final["label"] == label][0]
        final.loc[i, "neighbors"] = len(others)
        final.loc[i, "share_neighbors_close"] = close.mean()
        final.loc[i, "best_neighbor_move"] = best["move"]
        final.loc[i, "best_neighbor_gain"] = best["p_top20"] - own
    final["neighbors"] = final["neighbors"].astype(int)
    extra_flags = [
        "; ".join(f for f, bad in (("holdout_fail", not row.holdout_pass),
                                   ("neighbors_not_close", row.share_neighbors_close < FLAG_BELOW)) if bad)
        for row in final.itertuples()
    ]
    final["flags"] = ["; ".join(x for x in (a, b) if x) for a, b in zip(final["flags"], extra_flags)]

    RESULTS.mkdir(exist_ok=True)
    selection.to_csv(RESULTS / "robustness_selection.csv", index=False)
    final.to_csv(RESULTS / "robustness_final.csv", index=False)
    neighbor_table.drop(columns="label").to_csv(RESULTS / "robustness_neighbors.csv", index=False)
    (RESULTS / "robustness_meta.json").write_text(json.dumps({
        "pool_hash": meta["pool_hash"],
        "lut_hash": matrices.lut_hash,
        "seed": SEED,
        "run_seeds": RUN_SEEDS,
        "contests": {"selection": SELECTION_CONTESTS, "final": FINAL_CONTESTS,
                     "holdout": HOLDOUT_CONTESTS, "neighbors": NEIGHBOR_CONTESTS},
        "pooled_weights": POOLED_WEIGHTS,
        "tiebreak_scenario": TIEBREAK_SCENARIO,
        "tie_decimals": TIE_DECIMALS,
        "scenario_shares": SCENARIO_SHARES,
        "groups_available": sorted(vectors),
        "groups_missing": sorted(set(GROUPS) - set(vectors)),
        "dirichlet_concentration": DIRICHLET_CONCENTRATION,
        "field_size": list(FIELD_SIZE),
        "finalists": FINALISTS,
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n")

    pd.set_option("display.width", 200)
    show = ["rank", "label", "origin", *BATTLEFIELD_COLUMNS, "p_top20", "p_top20_nash", "worst_scenario_p_top20",
            "p_top20_holdout", "share_neighbors_close", "median_percentile", "mean_score", "flags"]
    print(final[show].round(3).to_string(index=False))
    pick = final.iloc[0]
    print(f"\nSelection rule pick: {pick['label']} {[int(x) for x in pick[BATTLEFIELD_COLUMNS]]} "
          f"P(top 20%) = {pick['p_top20']:.3f}; flags: {pick['flags'] or 'none'}")
    print(f"Done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
