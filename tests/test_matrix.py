"""Tests for stage 3: the W and S matrices in matrix.py."""

import numpy as np
import pytest

import matrix
from matrix import (
    build_lut200,
    build_matrices,
    check_against_scoring,
    check_invariants,
    compute_matrices,
    extend_matrices,
    matrix_path,
    payoff_block,
    win_masks,
)
from pool import random_uniform
from scoring import match


@pytest.fixture(scope="module")
def lut():
    return build_lut200()


@pytest.fixture(scope="module")
def small_pool():
    """40 allocations: a few hand-picked (including duplicates and ties) plus random ones."""
    rng = np.random.default_rng(11)
    hand = [
        [10] * 10,
        [10] * 10,  # duplicate → self-like draws off the diagonal
        [0, 0, 0, 0, 0, 0, 25, 25, 25, 25],
        [0, 0, 0, 0, 20, 20, 20, 20, 20, 0],
        [10, 9, 10, 10, 10, 10, 9, 10, 12, 10],  # draws 9–9 with the next one
        [10, 11, 10, 10, 10, 10, 11, 10, 8, 10],
        [100] + [0] * 9,
    ]
    random_rows = [list(random_uniform(rng)) for _ in range(33)]
    return np.array(hand + random_rows, dtype=np.int64)


@pytest.fixture(scope="module")
def small_matrices(small_pool, lut):
    return compute_matrices(small_pool, lut)


# --- Masks and blocks -------------------------------------------------------


def test_win_masks_bit_i_is_battlefield_i_plus_1():
    mine = np.array([[10, 10, 10, 10, 10, 11, 10, 10, 10, 9]])  # wins only battlefield 6
    theirs = np.array([[10] * 10])
    my_mask, their_mask = win_masks(mine, theirs)
    assert my_mask[0, 0] == 1 << 5
    assert their_mask[0, 0] == 1 << 9  # they win battlefield 10


def test_payoff_block_on_spec_example(lut):
    a = np.array([[0, 0, 20, 0, 20, 20, 40, 0, 0, 0]])  # wins {3,5,6,7} → 33.0
    b = np.array([[1, 1, 0, 1, 0, 0, 0, 30, 33, 34]])
    S200, W2 = payoff_block(a, b, lut)
    assert S200[0, 0] == 33 * 200
    assert S200.dtype == np.int32 and W2.dtype == np.int8


# --- Agreement with scoring.py ----------------------------------------------


def test_every_pair_matches_scoring_match(small_pool, small_matrices):
    S200, W2 = small_matrices
    n = len(small_pool)
    for i in range(n):
        for j in range(n):
            result = match(small_pool[i].tolist(), small_pool[j].tolist())
            assert S200[i, j] == result.score200_a, (i, j)
            assert W2[i, j] == round(2 * result.result_a), (i, j)


def test_known_draw_off_the_diagonal(small_matrices):
    S200, W2 = small_matrices
    assert W2[4, 5] == W2[5, 4] == 1  # 9 vs 9 with different battlefield counts
    assert W2[0, 1] == 1 and S200[0, 1] == 0  # identical allocations


def test_check_against_scoring_passes_and_catches_tampering(small_pool, small_matrices):
    S200, W2 = small_matrices
    check_against_scoring(small_pool, S200, W2, num_pairs=500)
    tampered = S200.copy()
    tampered[:, :] += 1  # every entry wrong, so any sampled pair catches it
    with pytest.raises(ValueError, match="disagrees with scoring.match"):
        check_against_scoring(small_pool, tampered, W2, num_pairs=10)


def test_sample_check_catches_a_bit_reversed_table(small_pool, lut):
    # If mask_from_won and won_from_mask were BOTH reversed, the table would pass its own
    # check (it agrees with itself) but be stored in reversed bit order, while win_masks
    # builds masks with bit i = battlefield i + 1. Only scoring.match can catch that.
    reversed_index = np.array([int(f"{m:010b}"[::-1], 2) for m in range(1024)])
    S200, W2 = compute_matrices(small_pool, lut[reversed_index])
    with pytest.raises(ValueError):
        check_against_scoring(small_pool, S200, W2, num_pairs=500)


def test_reversing_both_table_and_masks_cancels_out(small_pool, small_matrices, lut, monkeypatch):
    # Consistent reversal everywhere is harmless: same matrices.
    reversed_index = np.array([int(f"{m:010b}"[::-1], 2) for m in range(1024)])
    monkeypatch.setattr(matrix, "BIT_VALUES", matrix.BIT_VALUES[::-1].copy())
    S200, W2 = compute_matrices(small_pool, lut[reversed_index])
    assert np.array_equal(S200, small_matrices[0]) and np.array_equal(W2, small_matrices[1])


# --- Invariants -------------------------------------------------------------


def test_invariants_hold(small_matrices):
    check_invariants(*small_matrices)


def test_invariants_catch_a_one_sided_change(small_matrices):
    S200, W2 = small_matrices
    broken = W2.copy()
    broken[2, 3] = (broken[2, 3] + 1) % 3  # change one side only
    with pytest.raises(ValueError):
        check_invariants(S200, broken)


def test_invariants_catch_bad_diagonal(small_matrices):
    S200, W2 = small_matrices
    broken = S200.copy()
    broken[0, 0] = 5
    with pytest.raises(ValueError):
        check_invariants(broken, W2)


# --- Blocking and extending -------------------------------------------------


@pytest.mark.parametrize("block_rows", [1, 7, 1000])
def test_block_size_does_not_change_the_result(small_pool, small_matrices, lut, block_rows):
    S200, W2 = compute_matrices(small_pool, lut, block_rows=block_rows)
    assert np.array_equal(S200, small_matrices[0]) and np.array_equal(W2, small_matrices[1])


def test_extend_equals_full_recompute(small_pool, small_matrices, lut):
    old, new = small_pool[:30], small_pool[30:]
    S_old, W_old = compute_matrices(old, lut)
    S_ext, W_ext = extend_matrices(S_old, W_old, old, new, lut)
    assert np.array_equal(S_ext, small_matrices[0]) and np.array_equal(W_ext, small_matrices[1])


# --- Caching ----------------------------------------------------------------


def test_build_saves_then_reuses(tmp_path, small_pool, small_matrices, monkeypatch):
    ids = np.array([f"P{i:05d}" for i in range(1, len(small_pool) + 1)])
    built, reused = build_matrices(ids, small_pool, "a" * 64, cache_dir=tmp_path)
    assert not reused and matrix_path("a" * 64, tmp_path).exists()
    assert np.array_equal(built.S200, small_matrices[0])

    def fail(*args, **kwargs):
        raise AssertionError("should have reused the cache")

    monkeypatch.setattr(matrix, "compute_matrices", fail)
    again, reused = build_matrices(ids, small_pool, "a" * 64, cache_dir=tmp_path)
    assert reused and np.array_equal(again.W2, built.W2)


def test_build_rebuilds_when_lut_hash_differs(tmp_path, small_pool):
    ids = np.array([f"P{i:05d}" for i in range(1, len(small_pool) + 1)])
    build_matrices(ids, small_pool, "b" * 64, cache_dir=tmp_path)
    path = matrix_path("b" * 64, tmp_path)
    with np.load(path) as data:
        contents = dict(data)
    contents["lut_hash"] = np.array("stale")
    np.savez_compressed(path, **contents)
    _, reused = build_matrices(ids, small_pool, "b" * 64, cache_dir=tmp_path)
    assert not reused


def test_different_pool_hash_uses_a_different_file(tmp_path, small_pool):
    ids = np.array([f"P{i:05d}" for i in range(1, len(small_pool) + 1)])
    build_matrices(ids, small_pool, "c" * 64, cache_dir=tmp_path)
    _, reused = build_matrices(ids[:20], small_pool[:20], "d" * 64, cache_dir=tmp_path)
    assert not reused
    assert matrix_path("c" * 64, tmp_path) != matrix_path("d" * 64, tmp_path)
