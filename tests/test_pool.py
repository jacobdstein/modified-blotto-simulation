"""Tests for stage 2: pool.py."""

import numpy as np
import pandas as pd
import pytest

import pool as pool_module
from pool import (
    BATTLEFIELD_COLUMNS,
    POOL_COLUMNS,
    Entry,
    adjacent_count,
    adjust_riddler,
    apportion,
    battlefield6_heavy,
    build_pool,
    hand_strategies,
    is_holdout,
    load_pool,
    load_riddler_round,
    merge_exact_duplicates,
    merge_near_duplicates,
    origin_family,
    origin_subfamily,
    perturb,
    pool_hash,
    random_uniform,
    round_to_5,
    sample_submissions,
    save_pool,
)
from scoring import validate_allocation


@pytest.fixture(scope="module")
def built():
    return build_pool()


@pytest.fixture(scope="module")
def pool(built):
    return built[0]


@pytest.fixture(scope="module")
def meta(built):
    return built[1]


def allocations(df: pd.DataFrame) -> list[tuple[int, ...]]:
    return [tuple(int(x) for x in row) for row in df[BATTLEFIELD_COLUMNS].to_numpy()]


# --- Allocation helpers -----------------------------------------------------


def test_apportion_sums_and_is_proportional():
    assert apportion([1] * 10) == (10,) * 10
    assert apportion(list(range(1, 11))) == (2, 4, 5, 7, 9, 11, 13, 15, 16, 18)
    assert sum(apportion([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])) == 100


def test_round_to_5():
    rounded = round_to_5((2, 4, 5, 7, 9, 11, 13, 15, 16, 18))
    assert sum(rounded) == 100 and all(t % 5 == 0 for t in rounded)


def test_battlefield6_heavy_puts_exactly_35_on_6():
    allocation = battlefield6_heavy()
    assert allocation[5] == 35 and sum(allocation) == 100


def test_perturb_moves_troops_and_stays_valid():
    rng = np.random.default_rng(0)
    start = (10,) * 10
    moved = perturb(start, rng, 5)
    validate_allocation(list(moved))
    assert sum(abs(a - b) for a, b in zip(start, moved)) <= 10


def test_perturb_never_takes_from_empty_battlefield():
    rng = np.random.default_rng(1)
    for _ in range(200):
        validate_allocation(list(perturb((100,) + (0,) * 9, rng, 8)))


def test_random_uniform_is_valid_and_seeded():
    a = [random_uniform(np.random.default_rng(7)) for _ in range(3)]
    b = [random_uniform(np.random.default_rng(7)) for _ in range(3)]
    assert a == b
    rng = np.random.default_rng(3)
    for _ in range(500):
        validate_allocation(list(random_uniform(rng)))


def test_hand_strategies_are_valid_and_uniquely_named():
    strategies = hand_strategies()
    names = [name for name, _ in strategies]
    assert len(names) == len(set(names))
    for _, allocation in strategies:
        validate_allocation(list(allocation))


# --- Merging ----------------------------------------------------------------


def test_exact_duplicates_merge_within_origin_only():
    a, b = (10,) * 10, (100,) + (0,) * 9
    entries = [Entry("riddler:r1", a), Entry("riddler:r1", a), Entry("riddler:r4", a), Entry("riddler:r1", b)]
    merge_exact_duplicates(entries)
    assert entries[0].weight == 2 and entries[1].merged_into == 0
    assert entries[2].merged_into is None and entries[2].weight == 1  # other origin kept apart
    assert entries[3].merged_into is None


def test_near_duplicate_merge_is_noop_at_zero():
    entries = [Entry("random_uniform", (10,) * 10), Entry("random_uniform", (11, 9) + (10,) * 8)]
    merge_near_duplicates(entries, 0)
    assert all(e.merged_into is None for e in entries)


def test_near_duplicate_merge_skips_protected_origins():
    close = [(10,) * 10, (11, 9) + (10,) * 8]  # L1 distance 2
    unprotected = [Entry("random_uniform", a) for a in close]
    protected = [Entry("hand:even", a) for a in close]
    merge_near_duplicates(unprotected, 2)
    merge_near_duplicates(protected, 2)
    assert unprotected[1].merged_into == 0 and unprotected[0].weight == 2
    assert all(e.merged_into is None for e in protected)


# --- The built pool ---------------------------------------------------------


def test_columns_and_ids(pool):
    assert list(pool.columns) == POOL_COLUMNS
    assert pool["id"].is_unique
    assert list(pool["id"]) == sorted(pool["id"])
    assert pool["id"].str.fullmatch(r"P\d{5}").all()


def test_every_row_is_a_valid_allocation(pool):
    for allocation in allocations(pool):
        validate_allocation(list(allocation))


def test_no_exact_duplicates_within_an_origin(pool):
    keys = list(zip(pool["origin"], allocations(pool)))
    assert len(keys) == len(set(keys))


def test_weights_account_for_every_entry(pool, meta):
    assert (pool["weight"] >= 1).all()
    assert pool["weight"].sum() == meta["entries_before_merging"] == meta["total_weight"]


def test_riddler_raw_weight_is_the_sample_size(pool, meta):
    for round_number in pool_module.RIDDLER_FILES:
        available = meta["riddler_submissions"][f"r{round_number}"]
        rows = pool[pool["origin"] == f"riddler:r{round_number}"]
        assert rows["weight"].sum() == min(pool_module.RIDDLER_SAMPLE, available)


def test_riddler_sample_comes_from_its_round(pool):
    for round_number, filename in pool_module.RIDDLER_FILES.items():
        submitted, _ = load_riddler_round(pool_module.DATA / "riddler" / filename)
        sampled = allocations(pool[pool["origin"] == f"riddler:r{round_number}"])
        assert set(sampled) <= set(submitted)


def test_adjusted_rows_follow_their_parent(pool):
    by_id = pool.set_index("id")
    adjusted = pool[pool["origin"].str.startswith("riddler_adj:")]
    assert len(adjusted) > 0
    for _, row in adjusted.iterrows():
        version, round_tag = row["origin"].split(":")[1:]
        parent = by_id.loc[row["parent"]]
        assert parent["origin"] == f"riddler:{round_tag}"
        parent_allocation = tuple(int(parent[c]) for c in BATTLEFIELD_COLUMNS)
        assert adjust_riddler(parent_allocation, version) == tuple(int(row[c]) for c in BATTLEFIELD_COLUMNS)


def test_parents_point_to_existing_rows(pool):
    ids = set(pool["id"])
    parents = pool.loc[pool["parent"] != "", "parent"]
    assert set(parents) <= ids
    assert pool.loc[pool["origin"].str.endswith(":noisy"), "parent"].ne("").all()


def test_all_30_llm_strategies_present(pool):
    llm = pool[pool["origin"].str.startswith("llm:") & ~pool["origin"].str.endswith(":jitter")]
    assert len(llm) == 30
    assert set(llm["origin"].map(origin_family)) == {"llm:claude", "llm:chatgpt", "llm:gemini"}


def test_llm_jitter_variants(pool):
    by_id = pool.set_index("id")
    jitter = pool[pool["origin"].str.endswith(":jitter")]
    assert jitter["weight"].sum() == 30 * pool_module.LLM_JITTER_COPIES  # every variant accounted for
    for _, row in jitter.iterrows():
        parent = by_id.loc[row["parent"]]
        assert row["origin"] == parent["origin"] + ":jitter"
        moved = sum(abs(int(row[c]) - int(parent[c])) for c in BATTLEFIELD_COLUMNS)
        assert 0 < moved <= 2 * pool_module.LLM_JITTER_TROOPS[1]  # never identical to the original
    assert set(jitter["origin"].map(origin_family)) == {"llm:claude", "llm:chatgpt", "llm:gemini"}


def test_holdout_covers_round_4_and_its_adjusted_versions(pool, meta):
    expected = ["riddler:r4"] + [f"riddler_adj:{v}:r4" for v in sorted(pool_module.ADJUST_VERSIONS)]
    assert meta["holdout_origins"] == sorted(expected)
    assert all(not is_holdout(o) for o in pool["origin"] if not o.endswith(":r4"))


def test_is_holdout():
    assert is_holdout("riddler:r4") and is_holdout("riddler_adj:rule3:r4")
    assert not is_holdout("riddler:r1") and not is_holdout("riddler_adj:all:r2")
    assert not is_holdout("llm:claude:4")


def test_same_seed_same_pool(pool, meta):
    again, again_meta = build_pool()
    pd.testing.assert_frame_equal(pool, again)
    assert again_meta["pool_hash"] == meta["pool_hash"]


def test_different_seed_changes_only_random_parts(pool):
    other, _ = build_pool(seed=pool_module.SEED + 1)

    def fixed(df):  # rows that don't depend on the seed
        origin = df["origin"]
        hand_fixed = origin.str.startswith("hand:") & ~origin.str.endswith(":noisy")
        llm_fixed = origin.str.startswith("llm:") & ~origin.str.endswith(":jitter")
        return df[hand_fixed | llm_fixed]

    assert allocations(fixed(pool)) == allocations(fixed(other))
    assert pool_hash(pool) != pool_hash(other)


def test_sample_submissions():
    submissions = [(i,) for i in range(1000)]
    a = sample_submissions(submissions, 300, seed=1, round_number=2)
    assert a == sample_submissions(submissions, 300, seed=1, round_number=2)
    assert a != sample_submissions(submissions, 300, seed=1, round_number=3)
    assert len(a) == len(set(a)) == 300 and a == sorted(a)  # no repeats, file order kept
    assert sample_submissions(submissions[:50], 300, seed=1, round_number=2) == submissions[:50]


# --- Saving and loading -----------------------------------------------------


def test_save_and_load_round_trip(tmp_path, built):
    pool, meta = built
    save_pool(pool, meta, tmp_path)
    loaded, loaded_meta = load_pool(tmp_path)
    pd.testing.assert_frame_equal(pool, loaded)
    assert loaded_meta == meta


def test_load_rejects_edited_pool(tmp_path, built):
    pool, meta = built
    save_pool(pool, meta, tmp_path)
    edited = pool.copy()
    edited.loc[0, "b1"] += 1
    edited.loc[0, "b2"] -= 1
    edited.to_csv(tmp_path / "pool.csv", index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_pool(tmp_path)


def test_riddler_loader_drops_invalid_rows(tmp_path):
    header = ",".join(f"Castle {i}" for i in range(1, 11)) + ",Why?\n"
    rows = [
        "10,10,10,10,10,10,10,10,10,10,fine\n",
        "10,10,10,10,10,10,10,10,10,9,sums to 99\n",
        '-1,11,10,10,10,10,10,10,10,10,"negative, with comma"\n',
        "x,10,10,10,10,10,10,10,10,10,not a number\n",
    ]
    path = tmp_path / "round.csv"
    path.write_text(header + "".join(rows))
    allocations_, dropped = load_riddler_round(path)
    assert allocations_ == [(10,) * 10]
    assert dropped == 3


def test_origin_family_and_subfamily():
    cases = {
        "hand:even": ("hand", "hand:base"),
        "hand:even:noisy": ("hand", "hand:noisy"),
        "hand:even:round5": ("hand", "hand:round5"),
        "llm:claude:3": ("llm:claude", "llm:claude:3"),
        "llm:claude:3:jitter": ("llm:claude", "llm:claude:3"),
        "riddler:r2": ("riddler", "riddler:r2"),
        "riddler_adj:rule2:r1": ("riddler_adj", "riddler_adj:rule2"),
        "random_uniform": ("random_uniform", "random_uniform"),
        "counter:r1": ("counter", "counter:r1"),
    }
    for origin, (family, subfamily) in cases.items():
        assert origin_family(origin) == family
        assert origin_subfamily(origin) == subfamily


# --- Bonus-aware Riddler adjustment -----------------------------------------

EXAMPLE = (52, 2, 2, 2, 2, 2, 2, 12, 12, 12)  # contests {1, 8, 9, 10}, base 31


@pytest.mark.parametrize(
    "version, expected",
    [
        ("all", (0, 2, 54, 2, 2, 2, 2, 12, 12, 12)),  # 1 → 3: base 30, ×1.4, best score
        ("rule1", (0, 2, 2, 2, 2, 2, 54, 12, 12, 12)),  # 1 → 7: run 7–10
        ("rule2", (0, 2, 54, 2, 2, 2, 2, 12, 12, 12)),  # 1 → 3: smallest swap to base 30
        ("rule3", (52, 2, 2, 2, 2, 14, 2, 0, 12, 12)),  # 8 → 6: closest swap adding 6
    ],
)
def test_adjust_example(version, expected):
    assert adjust_riddler(EXAMPLE, version) == expected


def test_adjust_single_battlefield():
    everything_on_1 = (100,) + (0,) * 9
    assert adjust_riddler(everything_on_1, "all") == (0,) * 9 + (100,)  # base 10 → ×1.4
    assert adjust_riddler(everything_on_1, "rule2") == (0,) * 4 + (100,) + (0,) * 5  # 1 → 5
    assert adjust_riddler(everything_on_1, "rule3") == (0,) * 5 + (100,) + (0,) * 4  # 1 → 6
    assert adjust_riddler(everything_on_1, "rule1") is None  # one battlefield can't be adjacent


def test_adjust_returns_none_when_nothing_to_fix():
    already_has_all_bonuses = (0, 0, 0, 0, 0, 25, 25, 25, 25, 0)  # {6,7,8,9}: run, base 30, has 6
    for version in pool_module.ADJUST_VERSIONS:
        assert adjust_riddler(already_has_all_bonuses, version) is None
    assert adjust_riddler((10,) * 10, "all") is None  # contests all ten


def test_adjust_ignores_dust_troops():
    # Dust (< 5 troops) is neither contested nor moved.
    allocation = (3, 3, 3, 3, 3, 0, 0, 30, 30, 25)  # contests {8, 9, 10}, base 27
    adjusted = adjust_riddler(allocation, "rule2")
    assert adjusted[:5] == allocation[:5]


def test_adjust_moves_troops_as_is():
    for version in pool_module.ADJUST_VERSIONS:
        adjusted = adjust_riddler(EXAMPLE, version)
        changed = [i for i in range(10) if adjusted[i] != EXAMPLE[i]]
        assert len(changed) == 2 and sum(adjusted) == 100
        dropped, added = sorted(changed, key=lambda i: adjusted[i])
        assert adjusted[dropped] == 0
        assert adjusted[added] == EXAMPLE[added] + EXAMPLE[dropped]


def test_adjust_always_gains_the_noticed_bonus(pool):
    from pool import bonuses_gained, contested

    raw = pool[pool["origin"].str.startswith("riddler:")]
    for allocation in allocations(raw)[:400]:
        for version in pool_module.ADJUST_VERSIONS:
            adjusted = adjust_riddler(allocation, version)
            if adjusted is None:
                continue
            gained = bonuses_gained(contested(allocation), contested(adjusted))
            assert any(gained.values()) if version == "all" else gained[version]


def test_adjust_rejects_unknown_version():
    with pytest.raises(ValueError):
        adjust_riddler(EXAMPLE, "rule4")


def test_adjacent_count():
    assert adjacent_count(frozenset({1, 2, 3})) == 3
    assert adjacent_count(frozenset({1, 3, 5})) == 0
    assert adjacent_count(frozenset({1, 10})) == 0  # no wrap-around


def test_existing_rows_keep_their_ids_and_allocations(pool):
    # New sources (e.g. counterattack rows) may only be appended; rows that existed when
    # tests/pool_snapshot.json was taken must be unchanged, IDs included.
    import json
    from pathlib import Path

    snapshot = json.loads((Path(__file__).parent / "pool_snapshot.json").read_text())
    assert len(pool) >= snapshot["rows"]
    assert pool_hash(pool.iloc[: snapshot["rows"]]) == snapshot["prefix_hash"]
    assert list(pool["id"].iloc[: snapshot["rows"]]) == [f"P{n:05d}" for n in range(1, snapshot["rows"] + 1)]
