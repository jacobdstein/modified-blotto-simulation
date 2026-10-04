"""Pipeline step 5: counter strategies built to beat the current leaders.

A counter strategy models a clever applicant: it maximizes
    objective = 1/2 × mean win rate against our current finalists
              + 1/2 × win rate against the baseline field (expected group shares)
found by local search from 40 seeded starts (half the top Riddler and AI strategies,
half random). Each step tries moving troops between two battlefields so that one of them
lands exactly on a "threshold" (0, or one more than some target's count there — anything
in between is wasted) and takes the best improving move; it stops when none improves.
The best distinct results are written to data/counter/round_<N>.csv, which pool.py
appends to the pool as origin counter:r<N>.

One round, from the project root (after robustness.py):
    .venv/bin/python counterattack.py
    then re-run pool.py, matrix.py, rankings.py, robustness.py
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from matrix import build_lut200, payoff_block
from pool import BATTLEFIELD_COLUMNS, DATA, OPPONENT_ONLY_COUNTER_ROUNDS, counter_files, is_holdout, load_pool, random_uniform
from robustness import SCENARIO_SHARES, group_vectors, scenario_mix
from scoring import NUM_BATTLEFIELDS, validate_allocation

RESULTS = Path(__file__).resolve().parent / "results"
COUNTER_DIR = DATA / "counter"

FINALIST_WEIGHT = 0.5  # the rest is the baseline field
STARTS_FROM_POOL = 20  # top Riddler/Riddler-adjusted (half) and AI (half) strategies
STARTS_RANDOM = 20
KEEP = 20
THRESHOLDS_PER_SIDE = 3  # nearest thresholds considered above/below a battlefield's count
MAX_STEPS = 300
SEED = 20261004


# --- Objective --------------------------------------------------------------


class Objective:
    """Win credit (draw = 0.5) of allocations against a weighted set of targets."""

    def __init__(self, targets: np.ndarray, weights: np.ndarray):
        keep = weights > 0
        self.targets = targets[keep]
        self.weights = weights[keep] / weights[keep].sum()
        self.lut = build_lut200()

    def __call__(self, allocations: np.ndarray) -> np.ndarray:
        _, W2 = payoff_block(np.atleast_2d(allocations), self.targets, self.lut)
        return (W2 / 2) @ self.weights


def build_objective(finalists: np.ndarray, field: np.ndarray, field_weights: np.ndarray) -> Objective:
    """FINALIST_WEIGHT spread equally over the finalists, the rest over the field."""
    targets = np.concatenate([finalists, field])
    weights = np.concatenate([
        np.full(len(finalists), FINALIST_WEIGHT / len(finalists)),
        (1 - FINALIST_WEIGHT) * field_weights / field_weights.sum(),
    ])
    return Objective(targets, weights)


# --- Local search -----------------------------------------------------------


def battlefield_thresholds(targets: np.ndarray) -> list[np.ndarray]:
    """Per battlefield: the troop counts worth having — 0 and one more than any
    target's count there (strictly more troops wins)."""
    return [np.unique(np.concatenate([[0], targets[:, b] + 1])) for b in range(NUM_BATTLEFIELDS)]


def candidate_moves(allocation: np.ndarray, thresholds: list[np.ndarray], per_side: int = THRESHOLDS_PER_SIDE) -> np.ndarray:
    """Allocations one move away: shift k troops from battlefield a to b, where k makes b
    reach one of its next `per_side` thresholds, or drops a to one of its nearest lower
    thresholds (0 included), or k = 1."""
    moves = set()
    for a in range(NUM_BATTLEFIELDS):
        if allocation[a] == 0:
            continue
        lower = thresholds[a][thresholds[a] < allocation[a]][-per_side:]
        for b in range(NUM_BATTLEFIELDS):
            if b == a:
                continue
            upper = thresholds[b][thresholds[b] > allocation[b]][:per_side]
            amounts = {1, *(upper - allocation[b]), *(allocation[a] - lower)}
            for k in amounts:
                if 1 <= k <= allocation[a]:
                    moved = allocation.copy()
                    moved[a] -= k
                    moved[b] += k
                    moves.add(tuple(moved))
    return np.array(sorted(moves), dtype=np.int64)


def local_search(start: np.ndarray, objective: Objective, thresholds: list[np.ndarray], max_steps: int = MAX_STEPS) -> tuple[np.ndarray, float, int]:
    """Steepest ascent from `start`. Returns (allocation, objective value, steps taken).
    Ties between equally good moves go to the first in sorted order (deterministic)."""
    current = start.copy()
    value = float(objective(current)[0])
    for step in range(max_steps):
        moves = candidate_moves(current, thresholds)
        values = objective(moves)
        best = int(np.argmax(values))
        if values[best] <= value + 1e-12:
            return current, value, step
        current, value = moves[best], float(values[best])
    return current, value, max_steps


# --- Starts and rounds ------------------------------------------------------


def starting_points(round_robin: pd.DataFrame, rng: np.random.Generator) -> list[tuple[np.ndarray, str]]:
    """(allocation, parent id or '') for every start: the best distinct Riddler-family and
    AI strategies by round-robin rank (half each), then random allocations."""
    starts, seen = [], set()
    half = STARTS_FROM_POOL // 2
    for prefixes in (("riddler",), ("llm",)):
        rows = round_robin[round_robin["origin"].str.startswith(prefixes)].sort_values("rank")
        taken = 0
        for row in rows.itertuples():
            allocation = tuple(int(getattr(row, c)) for c in BATTLEFIELD_COLUMNS)
            if allocation not in seen:
                seen.add(allocation)
                starts.append((np.array(allocation), row.id))
                taken += 1
            if taken == half:
                break
    for _ in range(STARTS_RANDOM):
        starts.append((np.array(random_uniform(rng)), ""))
    return starts


def run_round(
    starts: list[tuple[np.ndarray, str]], objective: Objective, finalist_objective: Objective, field_objective: Objective
) -> pd.DataFrame:
    """Search from every start; keep the KEEP best distinct results."""
    thresholds = battlefield_thresholds(objective.targets)
    found = []
    for start, parent in starts:
        result, value, steps = local_search(start, objective, thresholds)
        validate_allocation(result.tolist())
        found.append({"parent": parent, **dict(zip(BATTLEFIELD_COLUMNS, result.tolist())), "objective": value, "steps": steps})
    table = pd.DataFrame(found).sort_values("objective", ascending=False, kind="stable")
    table = table.drop_duplicates(subset=BATTLEFIELD_COLUMNS).head(KEEP).reset_index(drop=True)
    allocations = table[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    table["win_vs_finalists"] = finalist_objective(allocations)
    table["win_vs_field"] = field_objective(allocations)
    return table


def baseline_field(pool: pd.DataFrame) -> np.ndarray:
    """Expected baseline field: scenario shares (empty groups spread over the rest) times
    each group's within-group weights. Held-out rows get 0."""
    vectors = group_vectors(pool)
    names, shares = scenario_mix(SCENARIO_SHARES["baseline"], vectors)
    return sum(share * vectors[name] for share, name in zip(shares, names))


def main() -> None:
    started = time.perf_counter()
    pool, meta = load_pool()
    robustness_meta = json.loads((RESULTS / "robustness_meta.json").read_text())
    if robustness_meta["pool_hash"] != meta["pool_hash"]:
        raise ValueError("robustness results are for a different pool; re-run rankings.py and robustness.py first")
    round_number = len(counter_files()) + 1
    if round_number > max(OPPONENT_ONLY_COUNTER_ROUNDS):
        raise SystemExit(f"counterattack is complete: round {max(OPPONENT_ONLY_COUNTER_ROUNDS)} was the final, opponent-only round")
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)

    final = pd.read_csv(RESULTS / "robustness_final.csv", keep_default_na=False)
    finalists = final[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    field_weights = baseline_field(pool)
    assert field_weights[pool["origin"].map(is_holdout).to_numpy()].sum() == 0

    objective = build_objective(finalists, allocations, field_weights)
    finalist_objective = Objective(finalists, np.ones(len(finalists)))
    field_objective = Objective(allocations, field_weights)

    round_robin = pd.read_csv(RESULTS / "rankings_round_robin.csv")
    rng = np.random.default_rng([SEED, round_number])
    table = run_round(starting_points(round_robin, rng), objective, finalist_objective, field_objective)

    COUNTER_DIR.mkdir(parents=True, exist_ok=True)
    path = COUNTER_DIR / f"round_{round_number}.csv"
    table.insert(0, "index", np.arange(1, len(table) + 1))
    table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    (COUNTER_DIR / f"round_{round_number}_meta.json").write_text(json.dumps({
        "round": round_number,
        "pool_hash": meta["pool_hash"],
        "finalists": list(final["label"]),
        "finalist_weight": FINALIST_WEIGHT,
        "starts": {"from_pool": STARTS_FROM_POOL, "random": STARTS_RANDOM},
        "keep": KEEP,
        "seed": [SEED, round_number],
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print(table[["index", "parent", *BATTLEFIELD_COLUMNS, "objective", "win_vs_finalists", "win_vs_field", "steps"]].round(3).to_string(index=False))
    print(f"\nRound {round_number}: wrote {len(table)} counter strategies to {path} in {time.perf_counter() - started:.0f}s")
    print("Next: re-run pool.py, matrix.py, rankings.py, robustness.py")


if __name__ == "__main__":
    main()
