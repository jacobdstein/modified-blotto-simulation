"""Tests for the stage-1 score lookup table in matrix.py."""

from itertools import combinations

import numpy as np
import pytest

from matrix import (
    NUM_MASKS,
    build_lut200,
    lut_hash,
    mask_from_won,
    verify_lut200,
    won_from_mask,
)
from scoring import battlefields_won, score200

# Pinned on purpose: if scoring.py changes, this fails and every cached matrix is stale.
EXPECTED_LUT_HASH = "eeb7f122bccccd9be02878ca14e709cea90b5547a0ba637d4f24ee62d53d6fb4"


@pytest.fixture(scope="module")
def lut():
    return build_lut200()


# --- Masks ------------------------------------------------------------------


@pytest.mark.parametrize(
    "won, mask",
    [
        (set(), 0),
        ({1}, 1),
        ({2}, 2),
        ({10}, 512),
        ({3, 5, 6, 7}, 116),
        (set(range(1, 11)), 1023),
    ],
)
def test_mask_examples(won, mask):
    assert mask_from_won(won) == mask
    assert won_from_mask(mask) == won


def test_mask_round_trip_for_all_1024():
    for mask in range(NUM_MASKS):
        assert mask_from_won(won_from_mask(mask)) == mask


def test_all_won_sets_give_distinct_masks():
    masks = {
        mask_from_won(set(won))
        for size in range(11)
        for won in combinations(range(1, 11), size)
    }
    assert masks == set(range(NUM_MASKS))


def test_mask_bit_i_matches_allocation_index_i():
    # Win only battlefield 6 (index 5) → bit 5 → mask 32.
    mine = [10, 10, 10, 10, 10, 11, 10, 10, 10, 9]
    theirs = [10] * 10
    assert mask_from_won(battlefields_won(mine, theirs)) == 1 << 5


@pytest.mark.parametrize("bad", [-1, 1024, 5000])
def test_won_from_mask_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        won_from_mask(bad)


@pytest.mark.parametrize("bad", [1.0, "3", None, True])
def test_won_from_mask_rejects_non_int(bad):
    with pytest.raises(TypeError):
        won_from_mask(bad)


def test_mask_from_won_rejects_bad_battlefields():
    with pytest.raises(ValueError):
        mask_from_won({0})
    with pytest.raises(ValueError):
        mask_from_won({11})


# --- Lookup table -----------------------------------------------------------


def test_lut_shape_and_dtype(lut):
    assert lut.shape == (1024,)
    assert lut.dtype == np.int64


def test_lut_spec_examples(lut):
    assert lut[0] == 0  # win none → 0
    assert lut[116] == 33 * 200  # win {3,5,6,7} → 33.0
    assert lut[1023] == 25410  # win all → 127.05 × 200


def test_lut_equals_scoring_for_every_won_set(lut):
    for size in range(11):
        for won in combinations(range(1, 11), size):
            assert lut[mask_from_won(set(won))] == score200(set(won)), won


def test_verify_accepts_correct_table(lut):
    verify_lut200(lut)  # does not raise


def test_verify_rejects_one_wrong_entry(lut):
    bad = lut.copy()
    bad[116] += 1
    with pytest.raises(ValueError, match="1 won-sets"):
        verify_lut200(bad)


def test_verify_rejects_bit_reversed_table(lut):
    # Bit i ↔ bit 9 - i: what you'd get by treating the 1s place as battlefield 10.
    reversed_order = [int(f"{mask:010b}"[::-1], 2) for mask in range(NUM_MASKS)]
    with pytest.raises(ValueError):
        verify_lut200(lut[reversed_order])


def test_verify_rejects_wrong_shape_or_dtype(lut):
    with pytest.raises(ValueError):
        verify_lut200(lut[:1000])
    with pytest.raises(ValueError):
        verify_lut200(lut.astype(np.float64))


# --- Hash -------------------------------------------------------------------


def test_lut_hash_is_pinned(lut):
    assert lut_hash(lut) == EXPECTED_LUT_HASH


def test_lut_hash_is_stable_across_builds(lut):
    assert lut_hash(build_lut200()) == lut_hash(lut)


def test_lut_hash_changes_when_table_changes(lut):
    bad = lut.copy()
    bad[0] = 1
    assert lut_hash(bad) != lut_hash(lut)
