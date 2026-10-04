"""Tests for pipeline step 5: counterattack.py (and pool.py loading counter rounds)."""

import numpy as np
import pandas as pd
import pytest

import counterattack
import pool as pool_module
from counterattack import (
    Objective,
    battlefield_thresholds,
    build_objective,
    candidate_moves,
    local_search,
    run_round,
    starting_points,
)
from pool import BATTLEFIELD_COLUMNS, build_pool, counter_files, random_uniform
from scoring import match, validate_allocation

EVEN = np.array([10] * 10)


# --- Objective --------------------------------------------------------------


def test_objective_matches_scoring_match():
    rng = np.random.default_rng(4)
    targets = np.array([random_uniform(rng) for _ in range(6)])
    weights = np.array([1, 2, 0, 1, 3, 1], dtype=float)
    candidates = np.array([random_uniform(rng) for _ in range(4)])
    values = Objective(targets, weights)(candidates)
    for value, candidate in zip(values, candidates):
        expected = sum(w * match(candidate.tolist(), t.tolist()).result_a for w, t in zip(weights, targets)) / weights.sum()
        assert value == pytest.approx(expected)


def test_build_objective_splits_weight_half_and_half():
    finalists = np.array([EVEN, [0] * 9 + [100]])
    field = np.array([[100] + [0] * 9, [50, 50] + [0] * 8, [0, 0, 100] + [0] * 7])
    objective = build_objective(finalists, field, np.array([1.0, 1.0, 2.0]))
    assert objective.weights.sum() == pytest.approx(1)
    assert objective.weights[:2].tolist() == pytest.approx([0.25, 0.25])
    assert objective.weights[2:].tolist() == pytest.approx([0.125, 0.125, 0.25])


# --- Moves and search -------------------------------------------------------


def test_candidate_moves_are_valid_and_reach_thresholds():
    targets = np.array([[10] * 10, [0, 0, 0, 0, 0, 0, 25, 25, 25, 25]])
    thresholds = battlefield_thresholds(targets)
    assert thresholds[9].tolist() == [0, 11, 26]
    moves = candidate_moves(EVEN, thresholds)
    assert len(moves) == len({tuple(m) for m in moves})
    for move in moves:
        validate_allocation(move.tolist())
        assert np.abs(move - EVEN).sum() > 0
    # Moving 1 troop from battlefield 1 to 10 reaches 10's threshold 11; moving all 10
    # off battlefield 1 drops it to threshold 0.
    as_set = {tuple(m) for m in moves}
    assert (9, 10, 10, 10, 10, 10, 10, 10, 10, 11) in as_set
    assert (0, 10, 10, 10, 10, 10, 10, 10, 10, 20) in as_set


def test_local_search_beats_a_single_target_and_never_gets_worse():
    objective = Objective(np.array([EVEN]), np.array([1.0]))
    start_value = objective(EVEN)[0]
    result, value, steps = local_search(EVEN, objective, battlefield_thresholds(objective.targets))
    validate_allocation(result.tolist())
    assert start_value == 0.5 and value == 1.0  # from a draw to a guaranteed win
    assert match(result.tolist(), EVEN.tolist()).result_a == 1.0
    assert steps >= 1


def test_local_search_is_deterministic():
    rng = np.random.default_rng(9)
    targets = np.array([random_uniform(rng) for _ in range(30)])
    objective = Objective(targets, np.ones(30))
    thresholds = battlefield_thresholds(targets)
    start = np.array(random_uniform(rng))
    a = local_search(start, objective, thresholds)
    b = local_search(start, objective, thresholds)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert a[1] >= objective(start)[0]


# --- Starts and rounds ------------------------------------------------------


def test_starting_points():
    rows = []
    for n in range(30):
        origin = "riddler:r2" if n % 2 == 0 else "llm:claude:1:jitter"
        allocation = [10] * 10 if n < 4 else list(random_uniform(np.random.default_rng(n)))  # some duplicates
        rows.append({"id": f"P{n:05d}", "origin": origin, "rank": n + 1, **dict(zip(BATTLEFIELD_COLUMNS, allocation))})
    starts = starting_points(pd.DataFrame(rows), np.random.default_rng(0))
    assert len(starts) == counterattack.STARTS_FROM_POOL + counterattack.STARTS_RANDOM
    from_pool = starts[: counterattack.STARTS_FROM_POOL]
    assert len({tuple(a) for a, _ in from_pool}) == len(from_pool)  # distinct
    assert sum(p.startswith("P") and int(p[1:]) % 2 == 0 for _, p in from_pool) == counterattack.STARTS_FROM_POOL // 2
    assert all(parent == "" for _, parent in starts[counterattack.STARTS_FROM_POOL:])


def test_run_round_keeps_best_distinct(monkeypatch):
    monkeypatch.setattr(counterattack, "KEEP", 3)
    objective = Objective(np.array([EVEN]), np.array([1.0]))
    starts = [(EVEN.copy(), "P00001"), (EVEN.copy(), "P00002"), (np.array([100] + [0] * 9), ""), (np.array([0] * 9 + [100]), "")]
    table = run_round(starts, objective, objective, objective)
    assert len(table) <= 3
    assert len(table.drop_duplicates(subset=BATTLEFIELD_COLUMNS)) == len(table)
    assert table["objective"].is_monotonic_decreasing


# --- pool.py loads counter rounds -------------------------------------------


def data_dir_with_counters(tmp_path, rounds):
    """A data directory that reuses the real inputs and adds counter round files."""
    (tmp_path / "riddler").symlink_to(pool_module.DATA / "riddler")
    (tmp_path / "llm_strategies.csv").symlink_to(pool_module.DATA / "llm_strategies.csv")
    (tmp_path / "counter").mkdir()
    for number, rows in rounds.items():
        frame = pd.DataFrame([{"index": i + 1, "parent": parent, **dict(zip(BATTLEFIELD_COLUMNS, a))} for i, (a, parent) in enumerate(rows)])
        frame.to_csv(tmp_path / "counter" / f"round_{number}.csv", index=False)
    return tmp_path


def test_pool_appends_counter_rounds_last(tmp_path):
    (tmp_path / "none").mkdir()
    (tmp_path / "some").mkdir()
    plain, _ = build_pool(data_dir=data_dir_with_counters(tmp_path / "none", {}))  # same inputs, no counters
    rounds = {1: [([0, 6, 1, 11, 1, 14, 28, 2, 34, 3], "P01229"), ([10] * 10, "")], 2: [([9] * 9 + [19], "")]}
    pool, meta = build_pool(data_dir=data_dir_with_counters(tmp_path / "some", rounds))
    assert len(pool) == len(plain) + 3
    pd.testing.assert_frame_equal(pool.iloc[: len(plain)], plain)  # existing rows untouched
    tail = pool.iloc[len(plain):]
    assert list(tail["origin"]) == ["counter:r1", "counter:r1", "counter:r2"]
    assert list(tail["parent"]) == ["P01229", "", ""]
    assert meta["counter_rounds"] == {"r1": 2, "r2": 1}


def test_counter_rounds_must_not_have_gaps(tmp_path):
    data_dir = data_dir_with_counters(tmp_path, {2: [([10] * 10, "")]})
    with pytest.raises(ValueError, match="no gaps"):
        counter_files(data_dir)


def test_local_search_stops_at_a_local_optimum():
    rng = np.random.default_rng(12)
    targets = np.array([random_uniform(rng) for _ in range(40)])
    objective = Objective(targets, np.ones(40))
    thresholds = battlefield_thresholds(targets)
    result, value, steps = local_search(np.array(random_uniform(rng)), objective, thresholds)
    assert steps < counterattack.MAX_STEPS  # stopped because nothing improves
    assert value == pytest.approx(objective(result)[0])
    assert objective(candidate_moves(result, thresholds)).max() <= value + 1e-12
