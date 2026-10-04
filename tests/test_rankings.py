"""Tests for stage 4: rankings.py."""

import numpy as np
import pandas as pd
import pytest

from matrix import build_lut200, compute_matrices
from pool import BATTLEFIELD_COLUMNS, POOL_COLUMNS, is_holdout, random_uniform
from rankings import (
    family_weights,
    iterated_elimination,
    nash_mix,
    rank_order,
    raw_weights,
    round_robin,
    shortlist,
    weighted_rates,
)

A = [10] * 10
B = [0, 0, 0, 0, 0, 0, 25, 25, 25, 25]
C = [0, 0, 0, 0, 20, 20, 20, 20, 20, 0]


def make_pool(rows: list[tuple[str, list[int], int]]) -> pd.DataFrame:
    """A pool DataFrame from (origin, allocation, weight) tuples."""
    records = []
    for n, (origin, allocation, weight) in enumerate(rows, start=1):
        records.append({"id": f"P{n:05d}", "origin": origin, **dict(zip(BATTLEFIELD_COLUMNS, allocation)), "weight": weight, "parent": ""})
    return pd.DataFrame(records, columns=POOL_COLUMNS)


def matrices_for(pool: pd.DataFrame):
    return compute_matrices(pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64), build_lut200())


@pytest.fixture(scope="module")
def mixed_pool():
    """Several families and subfamilies, duplicates, and held-out rows."""
    rng = np.random.default_rng(5)
    rows = [
        ("hand:even", A, 1),
        ("hand:even:noisy", list(random_uniform(rng)), 1),
        ("hand:even:noisy", list(random_uniform(rng)), 1),
        ("llm:claude:1", B, 1),
        ("llm:chatgpt:1", C, 1),
        ("riddler:r1", list(random_uniform(rng)), 3),
        ("riddler:r1", list(random_uniform(rng)), 1),
        ("riddler:r2", list(random_uniform(rng)), 2),
        ("riddler_adj:all:r1", list(random_uniform(rng)), 1),
        ("riddler_adj:rule3:r2", list(random_uniform(rng)), 1),
        ("random_uniform", list(random_uniform(rng)), 1),
        ("riddler:r4", [0] * 9 + [100], 5),  # held out
        ("riddler_adj:all:r4", list(random_uniform(rng)), 2),  # held out
    ]
    rows += [("random_uniform", list(random_uniform(rng)), 1) for _ in range(20)]
    return make_pool(rows)


# --- Weights ----------------------------------------------------------------


def test_family_weights_structure(mixed_pool):
    w = family_weights(mixed_pool)
    assert w.sum() == pytest.approx(1)
    families = mixed_pool["origin"].map(lambda o: o.split(":")[0] if not o.startswith("llm") else o[:o.rindex(":")])
    held = mixed_pool["origin"].map(is_holdout).to_numpy()
    assert (w[held] == 0).all()
    per_family = pd.Series(w).groupby(families.to_numpy()).sum()
    per_family = per_family[per_family > 0]
    assert np.allclose(per_family, 1 / len(per_family))  # equal per family
    # hand: base and noisy subfamilies split the family's weight equally
    assert w[0] == pytest.approx(w[1] + w[2])
    # riddler: r1 and r2 split equally; within r1, by duplicate count (3 : 1)
    assert w[5] + w[6] == pytest.approx(w[7])
    assert w[5] == pytest.approx(3 * w[6])


def test_family_weights_respects_include(mixed_pool):
    include = np.zeros(len(mixed_pool), dtype=bool)
    include[[0, 3, 4]] = True
    w = family_weights(mixed_pool, include=include)
    assert np.allclose(w[[0, 3, 4]], 1 / 3) and w.sum() == pytest.approx(1)


def test_raw_weights(mixed_pool):
    w = raw_weights(mixed_pool)
    assert w.sum() == pytest.approx(1)
    assert w[5] == pytest.approx(3 * w[6])
    assert (w[mixed_pool["origin"].map(is_holdout).to_numpy()] == 0).all()


# --- Win rates --------------------------------------------------------------


def test_weighted_rates_skip_your_own_row():
    # A/B/C: B beats A and C, C beats A. Equal weights; each faces the other two.
    pool = make_pool([("hand:a", A, 1), ("hand:b", B, 1), ("hand:c", C, 1)])
    _, W2 = matrices_for(pool)
    rates = weighted_rates(W2, np.full(3, 1 / 3), np.arange(3), np.ones(3)) / 2
    assert rates.tolist() == pytest.approx([0.0, 1.0, 0.5])


def test_weighted_rates_count_your_identical_copies_as_draws():
    # A stands for 3 identical submissions, B for 1: A's opponents are 2 copies of A
    # (draws) and B (A loses to B). Brute force: (0.5 + 0.5 + 0) / 3.
    pool = make_pool([("hand:a", A, 3), ("hand:b", B, 1)])
    S200, W2 = matrices_for(pool)
    weights = np.array([0.75, 0.25])
    rate_a = weighted_rates(W2, weights, np.array([0]), np.array([3.0]))[0] / 2
    assert rate_a == pytest.approx(1 / 3)
    rate_b = weighted_rates(W2, weights, np.array([1]), np.array([1.0]))[0] / 2
    assert rate_b == pytest.approx(1.0)  # B beats all three copies of A


def test_round_robin_columns_and_draw0(mixed_pool):
    S200, W2 = matrices_for(mixed_pool)
    table = round_robin(mixed_pool, W2, S200)
    assert set(table["rank"]) == set(range(1, len(table) + 1))
    assert (table["win_rate_draw0"] <= table["win_rate"] + 1e-12).all()
    assert table["win_rate"].between(0, 1).all()
    assert not table["origin"].map(is_holdout).any()


def test_rank_order_breaks_ties_by_score_then_id():
    ranks = rank_order(pd.Series([0.5, 0.5, 0.5, 0.9]), pd.Series([1.0, 2.0, 2.0, 0.0]), pd.Series(["P3", "P2", "P1", "P4"]))
    assert ranks.tolist() == [4, 3, 2, 1]


# --- Iterated elimination ---------------------------------------------------


def test_elimination_halves_and_records_ranks(mixed_pool):
    S200, W2 = matrices_for(mixed_pool)
    table = iterated_elimination(mixed_pool, W2, S200)
    candidates = (~mixed_pool["origin"].map(is_holdout)).sum()
    assert len(table) == candidates
    alive = candidates
    for column in sorted((c for c in table.columns if c.startswith("rank_round_")), key=lambda c: int(c.split("_")[-1])):
        assert table[column].notna().sum() == alive
        alive = -(-alive // 2)  # ceil(alive / 2)
    assert (table["rounds_survived"] == table["rounds_survived"].max()).sum() == 1  # one last survivor


def test_elimination_exposes_a_bully():
    # Eight real allocations (found by search). P00007 ranks #1 against everyone, but its
    # wins come from strategies cut in round 1; among survivors it falls to #3 and is cut.
    # P00003 ranks only #4 at first and is the last survivor.
    allocations = [
        [3, 8, 0, 13, 15, 20, 1, 3, 37, 0], [6, 30, 5, 4, 2, 19, 0, 4, 27, 3],
        [1, 2, 3, 20, 0, 2, 30, 4, 8, 30], [1, 16, 3, 3, 22, 16, 13, 2, 1, 23],
        [16, 7, 25, 2, 5, 25, 19, 0, 1, 0], [3, 1, 42, 28, 7, 0, 11, 0, 8, 0],
        [19, 5, 2, 3, 3, 3, 36, 6, 18, 5], [17, 19, 1, 14, 0, 4, 3, 22, 8, 12],
    ]
    pool = make_pool([("random_uniform", a, 1) for a in allocations])
    S200, W2 = matrices_for(pool)
    table = iterated_elimination(pool, W2, S200).set_index("id")
    assert table.loc["P00007", "rank_round_0"] == 1
    assert table.loc["P00007", "rank_round_1"] == 3 and table.loc["P00007", "rounds_survived"] == 1
    assert table.loc["P00003", "rank_round_0"] == 4
    assert table.loc["P00003", "rounds_survived"] == table["rounds_survived"].max()


# --- Hold-out safeguards ----------------------------------------------------


def test_holdout_entries_cannot_change_any_ranking(mixed_pool):
    S200, W2 = matrices_for(mixed_pool)
    held = np.flatnonzero(mixed_pool["origin"].map(is_holdout).to_numpy())
    S_changed, W_changed = S200.copy(), W2.copy()
    rng = np.random.default_rng(0)
    for h in held:  # scramble every match involving a held-out row (keeping invariants)
        for j in range(len(mixed_pool)):
            if j != h:
                w = rng.integers(0, 3)
                W_changed[h, j], W_changed[j, h] = w, 2 - w
                S_changed[h, j], S_changed[j, h] = rng.integers(0, 25000, size=2)
    pd.testing.assert_frame_equal(round_robin(mixed_pool, W2, S200), round_robin(mixed_pool, W_changed, S_changed))
    pd.testing.assert_frame_equal(
        iterated_elimination(mixed_pool, W2, S200), iterated_elimination(mixed_pool, W_changed, S_changed)
    )


def test_shortlist_has_no_holdout_and_explains_reasons(mixed_pool):
    S200, W2 = matrices_for(mixed_pool)
    rr = round_robin(mixed_pool, W2, S200)
    elimination = iterated_elimination(mixed_pool, W2, S200)
    short = shortlist(rr, elimination, top=5, per_family=1)
    assert not short["origin"].map(is_holdout).any()
    allowed = {"round_robin", "elimination", "family_top"}
    assert all(set(reason.split("+")) <= allowed for reason in short["reason"])
    assert set(rr.nsmallest(5, "rank")["id"]) <= set(short["id"])
    assert set(elimination.head(5)["id"]) <= set(short["id"])


def test_shortlist_includes_best_of_every_family(mixed_pool):
    S200, W2 = matrices_for(mixed_pool)
    rr = round_robin(mixed_pool, W2, S200)
    short = shortlist(rr, iterated_elimination(mixed_pool, W2, S200), top=1, per_family=2)
    for family, group in rr.groupby("family"):
        best = group.nsmallest(2, "rank")["id"]
        assert set(best) <= set(short["id"]), family
    assert len(short) <= 1 + 1 + 2 * rr["family"].nunique()


# --- Nash mix ---------------------------------------------------------------


def test_nash_of_a_cycle_is_uniform():
    # X beats Y, Y beats Z, Z beats X (real allocations found earlier).
    cycle = [[8, 6, 1, 14, 30, 8, 24, 4, 5, 0], [12, 13, 21, 8, 2, 1, 20, 13, 2, 8], [0, 2, 45, 5, 21, 11, 7, 0, 7, 2]]
    pool = make_pool([("hand:x", cycle[0], 1), ("hand:y", cycle[1], 1), ("hand:z", cycle[2], 1)])
    _, W2 = matrices_for(pool)
    assert W2[0, 1] == 2 and W2[1, 2] == 2 and W2[2, 0] == 2
    nash = nash_mix(pool, W2)
    assert sorted(nash["probability"]) == pytest.approx([1 / 3] * 3)
    assert nash.attrs["exploitability"] == pytest.approx(0, abs=1e-9)


def test_nash_puts_all_weight_on_a_dominant_strategy():
    pool = make_pool([("hand:a", A, 1), ("hand:b", B, 1), ("hand:c", C, 1)])  # B beats both
    _, W2 = matrices_for(pool)
    nash = nash_mix(pool, W2)
    assert list(nash["id"]) == ["P00002"] and nash["probability"].iloc[0] == pytest.approx(1)


def test_nash_excludes_holdout_and_is_unexploitable(mixed_pool):
    _, W2 = matrices_for(mixed_pool)
    nash = nash_mix(mixed_pool, W2)
    assert not nash["origin"].map(is_holdout).any()
    assert nash["probability"].sum() == pytest.approx(1)
    assert nash.attrs["exploitability"] <= 1e-9
