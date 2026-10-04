"""Tests for analysis/lut_analysis.py."""

import numpy as np
import pytest

from analysis.lut_analysis import (
    battlefield_prices,
    best_opponent_score,
    best_subset_score,
    harmful_additions_table,
    won_sets_table,
)
from matrix import ALL_WON_MASK, NUM_MASKS, build_lut200, mask_from_won


@pytest.fixture(scope="module")
def lut():
    return build_lut200()


@pytest.fixture(scope="module")
def sets(lut):
    return won_sets_table(lut).set_index("mask")


def test_best_subset_score_matches_brute_force(lut):
    best = best_subset_score(lut)
    for mask in [0, 1, 116, 300, 777, ALL_WON_MASK]:
        subsets = [s for s in range(NUM_MASKS) if s & mask == s]
        assert best[mask] == max(lut[s] for s in subsets)


def test_opponent_best_can_beat_full_complement(lut):
    # Score isn't monotone (Rule 2), so the opponent's best is sometimes a strict
    # subset of the complement. If I win only {1}, the complement {2..10} (base 54,
    # no ×1.4) scores 89.1, but its subset {2,3,5,6,7,8,9,10} (base 50, ×1.4) scores 115.5.
    opponent = best_opponent_score(lut)
    complement_scores = lut[ALL_WON_MASK ^ np.arange(NUM_MASKS)]
    assert (opponent >= complement_scores).all()
    mask = mask_from_won({1})
    assert lut[ALL_WON_MASK ^ mask] / 200 == pytest.approx(89.1)
    assert opponent[mask] / 200 == pytest.approx(115.5)


def test_known_guaranteed_set(sets):
    row = sets.loc[mask_from_won({2, 6, 8, 9})]
    assert row["score"] == pytest.approx(51.59)
    assert row["best_opponent_score"] == pytest.approx(50.4)
    assert row["guaranteed_win"] and row["minimal_guaranteed"]


def test_no_set_of_three_or_fewer_guarantees_a_win(sets):
    assert not sets.loc[sets["count"] <= 3, "guaranteed_win"].any()


def test_winning_nothing_never_guaranteed_and_everything_always(sets):
    assert not sets.loc[0, "guaranteed_win"]
    assert sets.loc[ALL_WON_MASK, "guaranteed_win"]


def test_adding_5_or_10_never_lowers_score(lut):
    # Adding 5 or 10 keeps base % 5 unchanged, so Rule 2 can't switch off.
    harmful = harmful_additions_table(lut).set_index("battlefield_added")
    assert harmful.loc[5, "share_of_sets_where_it_lowers_score"] == 0
    assert harmful.loc[10, "share_of_sets_where_it_lowers_score"] == 0


def test_adding_battlefield_1_can_cost_many_points(lut):
    # {5..10}: base 45 → ×1.4. Adding 1 makes base 46 and loses the bonus.
    before = lut[mask_from_won({5, 6, 7, 8, 9, 10})]
    after = lut[mask_from_won({1, 5, 6, 7, 8, 9, 10})]
    assert (before - after) / 200 == pytest.approx(28.6)


def test_battlefield_prices():
    field = np.array([[10] * 10, [0] * 9 + [100], [20] * 5 + [0] * 5])
    prices = battlefield_prices(field, percentile=50)
    assert prices.tolist() == [11] * 5 + [1] * 4 + [11]
