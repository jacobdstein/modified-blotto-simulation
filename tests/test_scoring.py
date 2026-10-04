"""Tests for the reference scoring module."""

from fractions import Fraction
from itertools import combinations

import pytest

import scoring
from scoring import (
    battlefields_won,
    match,
    score200,
    total_score,
    validate_allocation,
)

EVEN = [10] * 10


def all_won_sets():
    """Every subset of battlefields 1..10 (1024 sets)."""
    for size in range(11):
        for combo in combinations(range(1, 11), size):
            yield frozenset(combo)


def oracle_score(won):
    """Independent float scorer written from the spec, using runs of consecutive wins."""
    score = 0.0
    run = []
    for battlefield in range(1, 12):  # 11 is a sentinel that closes the last run
        if battlefield in won:
            run.append(battlefield)
            continue
        multiplier = 1.5 if len(run) >= 2 else 1.0
        score += multiplier * sum(run)
        run = []
    if sum(won) % 5 == 0:
        score *= 1.4
    if 6 in won:
        score *= 1.1
    return score


# --- Spec test cases --------------------------------------------------------


@pytest.mark.parametrize(
    "won, expected",
    [
        ({3, 5, 6, 7}, 33.0),
        (set(range(1, 11)), 127.05),
        (set(), 0.0),
    ],
)
def test_spec_examples(won, expected):
    assert float(total_score(won)) == pytest.approx(expected)


# --- Rule 1: adjacency ------------------------------------------------------


def test_isolated_battlefields_get_no_bonus():
    assert total_score({1, 3}) == 4


def test_adjacent_pair_both_get_bonus():
    assert total_score({1, 2}) == Fraction(9, 2)  # 1.5 + 3


def test_run_of_three_bonus_applied_once_each():
    # 1.5 + 3 + 4.5; battlefield 2 has two neighbors but is multiplied only once
    assert total_score({1, 2, 3}) == 9


def test_no_wraparound_between_1_and_10():
    assert total_score({1, 10}) == 11


# --- Rule 2: multiple of five, checked on base ------------------------------


def test_rule2_applies_when_base_is_multiple_of_5():
    # base 5, rule 1 gives 7.5, rule 2 gives 10.5
    assert total_score({2, 3}) == Fraction(21, 2)


def test_rule2_checks_base_not_rule1_score():
    # base 12 (not a multiple of 5); rule 1 score is 15 (a multiple of 5).
    # Correct: 15 × 1.1 = 16.5. Checking the rule-1 score would give 23.1.
    assert total_score({1, 2, 3, 6}) == Fraction(33, 2)


# --- Rule 3: battlefield 6 --------------------------------------------------


def test_rule3_battlefield_6_alone():
    assert total_score({6}) == Fraction(33, 5)  # 6 × 1.1


def test_rule2_and_rule3_both_apply():
    assert total_score({4, 6}) == Fraction(77, 5)  # 10 × 1.4 × 1.1


# --- Exhaustive checks ------------------------------------------------------


def test_all_1024_sets_match_independent_oracle():
    for won in all_won_sets():
        assert float(total_score(won)) == pytest.approx(oracle_score(won)), sorted(won)


def test_score200_is_exact_integer_for_all_sets():
    for won in all_won_sets():
        assert score200(won) == total_score(won) * 200


def test_won_set_rejects_out_of_range_battlefields():
    for bad in ({0}, {11}, {-1}):
        with pytest.raises(ValueError):
            total_score(bad)
    with pytest.raises(TypeError):
        total_score({1.0})


# --- Battlefield outcomes ---------------------------------------------------


def test_strictly_more_troops_wins_and_ties_win_nothing():
    a = [11, 9, 10, 10, 10, 10, 10, 10, 10, 10]
    b = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    assert battlefields_won(a, b) == {1}
    assert battlefields_won(b, a) == {2}


def test_index_5_is_battlefield_6():
    a = [10, 10, 10, 10, 10, 11, 10, 10, 10, 9]
    assert battlefields_won(a, EVEN) == {6}


# --- Matches ----------------------------------------------------------------


def test_identical_allocations_draw():
    result = match(EVEN, EVEN)
    assert result.won_a == result.won_b == frozenset()
    assert result.result_a == result.result_b == 0.5


def test_win_and_loss_are_symmetric():
    a = [0, 0, 0, 0, 0, 0, 25, 25, 25, 25]
    b = EVEN
    assert match(a, b).result_a == 1.0
    assert match(b, a).result_a == 0.0
    assert match(a, b).result_b == 0.0


def test_spec_example_through_match():
    # A wins exactly battlefields 3, 5, 6, 7; B wins 1, 2, 4, 8, 9, 10.
    a = [0, 0, 20, 0, 20, 20, 40, 0, 0, 0]
    b = [1, 1, 0, 1, 0, 0, 0, 30, 33, 34]
    result = match(a, b)
    assert result.won_a == {3, 5, 6, 7}
    assert result.score_a == pytest.approx(33.0)


def test_equal_scores_draw_even_with_different_battlefield_counts():
    # A wins {9} → 9. B wins {2, 7} → 9. No tiebreak: this is a draw.
    a = [10, 9, 10, 10, 10, 10, 9, 10, 12, 10]
    b = [10, 11, 10, 10, 10, 10, 11, 10, 8, 10]
    result = match(a, b)
    assert result.won_a == {9}
    assert result.won_b == {2, 7}
    assert result.score200_a == result.score200_b == 9 * 200
    assert result.result_a == result.result_b == 0.5


# --- Validation -------------------------------------------------------------


def test_valid_allocation_returns_tuple():
    assert validate_allocation(EVEN) == tuple(EVEN)
    assert validate_allocation([100] + [0] * 9) == (100,) + (0,) * 9


@pytest.mark.parametrize(
    "bad",
    [
        [10] * 9,  # too short
        [10] * 9 + [5, 5],  # too long
        [10] * 9 + [9],  # sums to 99
        [10] * 9 + [11],  # sums to 101
        [-1, 11] + [10] * 8,  # negative entry
    ],
)
def test_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        validate_allocation(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [10.0] + [10] * 9,  # float entry
        [True] + [11] * 9,  # bool entry (True == 1, but not a troop count)
        ["10"] + [10] * 9,  # string entry
        [None] + [10] * 9,  # None entry
        "1234567890",  # string container
        {i: 10 for i in range(10)},  # dict container
        iter(EVEN),  # iterator
        None,
    ],
)
def test_rejects_bad_types(bad):
    with pytest.raises(TypeError):
        validate_allocation(bad)


def test_match_validates_both_sides():
    with pytest.raises(ValueError):
        match(EVEN, [10] * 9)
    with pytest.raises(ValueError):
        match([10] * 9, EVEN)


def test_constants_match_spec():
    assert list(scoring.BATTLEFIELDS) == list(range(1, 11))
    assert scoring.TOTAL_TROOPS == 100
