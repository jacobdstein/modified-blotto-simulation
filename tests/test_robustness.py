"""Tests for pipeline step 4: robustness.py."""

import numpy as np
import pandas as pd
import pytest

import robustness
from matrix import Matrices, build_lut200, compute_matrices
from pool import BATTLEFIELD_COLUMNS, POOL_COLUMNS, is_holdout, random_uniform
from robustness import (
    POOLED_WEIGHTS,
    SCENARIO_SHARES,
    Candidates,
    contest,
    group_vectors,
    groups_of,
    one_troop_neighbors,
    rank_fraction,
    run_scenario,
    scenario_mix,
    summarize,
    unique_allocations,
    weighted_median,
)
from scoring import match


def make_pool(rows):
    records = [
        {"id": f"P{n:05d}", "origin": origin, **dict(zip(BATTLEFIELD_COLUMNS, allocation)), "weight": weight, "parent": ""}
        for n, (origin, allocation, weight) in enumerate(rows, start=1)
    ]
    return pd.DataFrame(records, columns=POOL_COLUMNS)


def matrices_for(pool):
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    S200, W2 = compute_matrices(allocations, build_lut200())
    return Matrices(pool["id"].to_numpy(), S200, W2, "test", "test")


@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(3)
    r = lambda: list(random_uniform(rng))  # noqa: E731
    rows = [("hand:even", [10] * 10, 1), ("hand:even:noisy", r(), 1), ("hand:even:round5", r(), 1)]
    for tool in ("gemini", "chatgpt", "claude"):
        for index in (1, 2, 3):
            rows.append((f"llm:{tool}:{index}", r(), 1))
            rows.append((f"llm:{tool}:{index}:jitter", r(), 1))
            rows.append((f"llm:{tool}:{index}:jitter", r(), 2))  # a merged duplicate variant
    rows += [("riddler:r1", r(), 2), ("riddler:r2", r(), 1), ("riddler:r3", r(), 1),
             ("riddler_adj:all:r1", r(), 1), ("riddler_adj:rule3:r2", r(), 1),
             ("random_uniform", r(), 1),
             ("riddler:r4", r(), 3), ("riddler_adj:rule1:r4", r(), 1)]
    return make_pool(rows)


@pytest.fixture(scope="module")
def matrices(pool):
    return matrices_for(pool)


# --- Groups -----------------------------------------------------------------


def test_groups_of_normal_and_holdout():
    assert groups_of("hand:even:noisy") == ["naive_rules"]
    assert groups_of("riddler:r1") == ["naive_blotto"]
    assert groups_of("riddler:r3") == ["experienced_blotto"]
    assert groups_of("riddler_adj:rule2:r2") == ["clumsy"]
    assert groups_of("llm:claude:4:jitter") == ["ai_claude"]
    assert groups_of("riddler:r4") == [] and groups_of("riddler_adj:all:r4") == []
    assert groups_of("riddler:r4", holdout=True) == ["naive_blotto", "experienced_blotto"]
    assert groups_of("riddler_adj:all:r4", holdout=True) == ["clumsy"]
    assert groups_of("riddler:r2", holdout=True) == [] and groups_of("riddler_adj:all:r1", holdout=True) == []
    assert groups_of("hand:even", holdout=True) == ["naive_rules"]  # non-Riddler groups unchanged


def test_group_vectors_sum_to_one_and_skip_holdout(pool):
    vectors = group_vectors(pool)
    held = pool["origin"].map(is_holdout).to_numpy()
    for group, vector in vectors.items():
        assert vector.sum() == pytest.approx(1), group
        assert (vector[held] == 0).all(), group
    assert "counter" not in vectors  # no counter rows yet


def test_ai_first_answer_gets_a_quarter_when_there_are_ten_answers():
    rows = []
    for index in range(1, 11):
        rows.append((f"llm:claude:{index}", [10] * 10, 1))
        rows.append((f"llm:claude:{index}:jitter", [11, 9] + [10] * 8, 3))  # 3 merged variants
    pool = make_pool(rows)
    vector = group_vectors(pool)["ai_claude"]
    first = pool["origin"].str.startswith("llm:claude:1:") | (pool["origin"] == "llm:claude:1")
    assert vector[first.to_numpy()].sum() == pytest.approx(3 / 12)
    # Within an answer, rows weigh by duplicate count: original 1, merged variants 3.
    assert vector[1] == pytest.approx(3 * vector[0])


def test_naive_rules_subfamilies_share_equally(pool):
    vector = group_vectors(pool)["naive_rules"]
    assert vector[:3].tolist() == pytest.approx([1 / 3] * 3)


def test_holdout_vectors_use_only_round_4_for_riddler_groups(pool):
    vectors = group_vectors(pool, holdout=True)
    origins = pool["origin"]
    for group in ("naive_blotto", "experienced_blotto", "clumsy"):
        used = origins[vectors[group] > 0]
        assert used.map(is_holdout).all(), group
    assert vectors["naive_blotto"][(origins == "riddler:r4").to_numpy()].sum() == pytest.approx(1)


def test_scenario_mix_spreads_missing_counter_share(pool):
    vectors = group_vectors(pool)
    names, shares = scenario_mix(SCENARIO_SHARES["baseline"], vectors)
    assert "counter" not in names and shares.sum() == pytest.approx(1)
    # 7% counter spread proportionally: each remaining share scales by 100 / 93.
    assert shares[names.index("experienced_blotto")] == pytest.approx(20 / 93)
    names, _ = scenario_mix(SCENARIO_SHARES["sophisticated_heavy"], vectors)
    assert "naive_blotto" not in names and "random" not in names  # 0% stays 0


def test_scored_weights_and_shares():
    assert POOLED_WEIGHTS == {"baseline": 40, "sophisticated_heavy": 24, "ai_heavy": 19, "naive_heavy": 7}
    assert "nash" not in POOLED_WEIGHTS and robustness.TIEBREAK_SCENARIO == "nash"
    assert set(robustness.SCENARIOS) == set(POOLED_WEIGHTS) | {"nash"}
    for shares in SCENARIO_SHARES.values():
        assert sum(shares.values()) == 100


# --- Contests ---------------------------------------------------------------


def brute_force_contest(field_allocations, candidate):
    """Every player plays every other player with scoring.match; returns the candidate's
    rank fraction by wins, points, wins-with-draws-as-0, and its mean score."""
    players = [candidate] + field_allocations
    wins = np.zeros(len(players))
    points = np.zeros(len(players))
    pure = np.zeros(len(players))
    for i in range(len(players)):
        for j in range(len(players)):
            if i != j:
                result = match(np.asarray(players[i]).tolist(), np.asarray(players[j]).tolist())
                wins[i] += result.result_a
                points[i] += result.score200_a
                pure[i] += result.result_a == 1.0

    def fraction(totals):
        ahead = (totals[1:] > totals[0]).sum()
        tied = (totals[1:] == totals[0]).sum()
        return (1 + ahead + tied / 2) / len(players)

    return fraction(wins), fraction(points), fraction(pure), points[0] / len(field_allocations) / 200


def test_contest_matches_brute_force(pool, matrices):
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    field = np.array([0, 0, 3, 5, 5, 5, 9, 12, 20, 33, 34, 7])  # repeats = identical players
    candidate_rows = [0, 5, 20]  # including candidates that also appear in the field
    candidates = Candidates.from_pool(matrices, pool, list(pool["id"].iloc[candidate_rows]))
    result = contest(field, matrices, candidates)
    for k, row in enumerate(candidate_rows):
        wins, points, pure, mean_score = brute_force_contest([allocations[f] for f in field], allocations[row])
        assert result["wins"][k] == pytest.approx(wins)
        assert result["points"][k] == pytest.approx(points)
        assert result["draw0"][k] == pytest.approx(pure)
        assert result["mean_score"][k] == pytest.approx(mean_score)


def test_candidates_from_allocations_match_from_pool(pool, matrices):
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    from_pool = Candidates.from_pool(matrices, pool, list(pool["id"].iloc[[2, 7]]))
    outside = Candidates.from_allocations(allocations[[2, 7]], allocations, ["a", "b"])
    for name in ("W2", "S200", "S200_opp"):
        assert np.array_equal(getattr(from_pool, name), getattr(outside, name)), name


def test_rank_fraction_ties_share_the_average_rank():
    field = np.array([[5, 3, 3, 1]])
    assert rank_fraction(field, np.array([3]))[0] == pytest.approx((1 + 1 + 1) / 5)  # 1 ahead, 2 tied
    assert rank_fraction(field, np.array([9]))[0] == pytest.approx(1 / 5)


def test_top20_threshold_is_inclusive():
    # rank 30 of 150 → exactly 0.20 → counts as top 20%.
    field = np.zeros((1, 149))
    field[0, :29] = 10
    assert rank_fraction(field, np.array([5]))[0] <= robustness.TOP_SHARE
    field[0, 29] = 10
    assert rank_fraction(field, np.array([5]))[0] > robustness.TOP_SHARE


# --- Scenarios and runs -----------------------------------------------------


def test_normal_scenarios_never_sample_holdout_rows(pool, matrices, monkeypatch):
    vectors = group_vectors(pool)
    held = np.flatnonzero(pool["origin"].map(is_holdout).to_numpy())
    sampled = []
    real_contest = robustness.contest
    monkeypatch.setattr(robustness, "contest", lambda field, m, c: sampled.append(field) or real_contest(field, m, c))
    candidates = Candidates.from_pool(matrices, pool, ["P00001"])
    nash = np.full(len(pool), 1 / len(pool))
    for scenario in SCENARIO_SHARES:
        run_scenario(scenario, matrices, candidates, vectors, nash, 30, np.random.default_rng(1))
    assert not np.isin(np.concatenate(sampled), held).any()


def test_nash_scenario_samples_from_the_nash_vector(pool, matrices, monkeypatch):
    sampled = []
    real_contest = robustness.contest
    monkeypatch.setattr(robustness, "contest", lambda field, m, c: sampled.append(field) or real_contest(field, m, c))
    nash = np.zeros(len(pool))
    nash[[4, 9]] = [0.25, 0.75]
    candidates = Candidates.from_pool(matrices, pool, ["P00001"])
    run_scenario("nash", matrices, candidates, group_vectors(pool), nash, 20, np.random.default_rng(2))
    assert set(np.concatenate(sampled)) <= {4, 9}


def test_runs_are_reproducible(pool, matrices):
    vectors = group_vectors(pool)
    candidates = Candidates.from_pool(matrices, pool, ["P00001", "P00010"])
    nash = np.full(len(pool), 1 / len(pool))
    a = run_scenario("baseline", matrices, candidates, vectors, nash, 25, np.random.default_rng(7))
    b = run_scenario("baseline", matrices, candidates, vectors, nash, 25, np.random.default_rng(7))
    for key in a:
        assert np.array_equal(a[key], b[key])


def make_results(wins_by_scenario):
    """results dict with the same values for every measure, from scenario → (contests, C)."""
    return {s: {"wins": w, "points": w, "draw0": w, "mean_score": np.ones_like(w)} for s, w in wins_by_scenario.items()}


def test_summarize_pools_scored_scenarios_and_flags():
    candidates = Candidates(["x", "y"], np.array([[10] * 10, [10] * 10]), None, None, None)
    wins = {s: np.tile([0.1, 0.9], (4, 1)) for s in robustness.SCENARIOS}  # x always top 20%, y never
    wins["nash"] = np.full((4, 2), 0.9)  # nobody top 20% in Nash
    wins["naive_heavy"] = np.tile([0.9, 0.9], (4, 1))  # x fails only in naive-heavy (weight 7 of 90)
    table = summarize(make_results(wins), candidates).set_index("label")
    assert table.loc["x", "p_top20"] == pytest.approx(83 / 90)  # Nash not in the score
    assert table.loc["x", "p_top20_nash"] == 0  # but reported
    assert "scenario<50%" in table.loc["x", "flags"]  # from naive-heavy, not Nash
    assert table.loc["y", "p_top20"] == 0 and table.loc["x", "rank"] == 1


def test_nash_alone_never_flags():
    candidates = Candidates(["x"], np.array([[10] * 10]), None, None, None)
    wins = {s: np.full((4, 1), 0.1) for s in robustness.SCENARIOS}
    wins["nash"] = np.full((4, 1), 0.9)
    table = summarize(make_results(wins), candidates)
    assert table.loc[0, "p_top20"] == 1 and table.loc[0, "flags"] == ""


def test_nash_breaks_ties_at_the_nearest_percentage_point():
    candidates = Candidates(["x", "y", "z"], np.array([[10] * 10] * 3), None, None, None)
    contests = 1000
    wins = {}
    for s in robustness.SCENARIOS:
        w = np.full((contests, 3), 0.1)
        w[:3, 1] = 0.9  # y: 99.7% top 20% → rounds to 100%, tied with x
        w[:50, 2] = 0.9  # z: 95% → not tied
        wins[s] = w
    nash = np.full((contests, 3), 0.9)
    nash[:600, 1] = 0.1  # y does much better than x in Nash
    nash[:900, 2] = 0.1  # z does best in Nash, but isn't tied on the score
    wins["nash"] = nash
    table = summarize(make_results(wins), candidates).set_index("label")
    assert table.loc["y", "rank"] == 1 and table.loc["x", "rank"] == 2 and table.loc["z", "rank"] == 3


def test_weighted_median():
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.2, 0.6])) == 3.0
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]) / 3) == 2.0


# --- Neighbors --------------------------------------------------------------


def test_one_troop_neighbors():
    allocation = np.array([10] * 10)
    neighbors = one_troop_neighbors(allocation)
    assert len(neighbors) == 90
    assert len({tuple(n) for n, _ in neighbors}) == 90
    for moved, label in neighbors:
        assert moved.sum() == 100 and (moved >= 0).all() and np.abs(moved - allocation).sum() == 2
    assert len(one_troop_neighbors(np.array([100] + [0] * 9))) == 9
    moved, label = one_troop_neighbors(np.array([100] + [0] * 9))[0]
    assert label == "1→2" and moved[0] == 99 and moved[1] == 1


def test_unique_allocations_keeps_first_and_lists_the_rest():
    shortlist = make_pool([("riddler_adj:rule3:r3", [3, 4, 7, 1, 0, 20, 27, 1, 36, 1], 1),
                           ("hand:even", [10] * 10, 1),
                           ("riddler_adj:all:r3", [3, 4, 7, 1, 0, 20, 27, 1, 36, 1], 1),
                           ("riddler_adj:rule1:r3", [3, 4, 7, 1, 0, 20, 27, 1, 36, 1], 1)])
    unique = unique_allocations(shortlist)
    assert list(unique["id"]) == ["P00001", "P00002"]
    assert unique.loc[0, "same_allocation_ids"] == "P00003 P00004"
    assert unique.loc[1, "same_allocation_ids"] == ""
