"""Payoff matrices for the pool, built from a verified score lookup table.

Run from the project root (after pool.py):
    .venv/bin/python matrix.py
Writes cache/matrix_<pool_hash>.npz, or reuses it if the pool and table are unchanged.

Stage 1: the lookup table.
A player's score depends only on which battlefields they won. There are 2**10 = 1024
possible won-sets, so we score each one once with scoring.py and store the results.

Mask convention: bit i of the mask is set if battlefield i + 1 was won.
    {1}          → 0b0000000001 = 1
    {10}         → 0b1000000000 = 512
    {3, 5, 6, 7} → 0b0001110100 = 116
This matches allocation indexing: allocation index i is battlefield i + 1, so the
mask is sum(1 << i for every index i where I have strictly more troops).

LUT200[mask] = scoring.score200(won-set for that mask), the exact integer 200 × score.
"""

import hashlib
import time
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from scoring import NUM_BATTLEFIELDS, match, score200, validate_won

NUM_MASKS = 1 << NUM_BATTLEFIELDS  # 1024
ALL_WON_MASK = NUM_MASKS - 1  # 1023


# --- Masks ------------------------------------------------------------------


def mask_from_won(won: AbstractSet[int]) -> int:
    """Encode a set of won battlefields (numbers 1..10) as a 10-bit mask."""
    won = validate_won(won)
    return sum(1 << (battlefield - 1) for battlefield in won)


def won_from_mask(mask: int) -> frozenset[int]:
    """Decode a 10-bit mask into the set of won battlefields (numbers 1..10)."""
    if isinstance(mask, bool) or not isinstance(mask, int):
        raise TypeError(f"mask must be an int, got {type(mask).__name__}")
    if not 0 <= mask < NUM_MASKS:
        raise ValueError(f"mask must be in 0..{ALL_WON_MASK}, got {mask}")
    return frozenset(i + 1 for i in range(NUM_BATTLEFIELDS) if mask & (1 << i))


# --- Lookup table -----------------------------------------------------------


def build_lut200() -> np.ndarray:
    """Return LUT200 (int64[1024]), verified against scoring.py before returning."""
    lut = np.array([score200(won_from_mask(mask)) for mask in range(NUM_MASKS)], dtype=np.int64)
    verify_lut200(lut)
    return lut


def verify_lut200(lut: np.ndarray) -> None:
    """Raise ValueError unless lut matches scoring.py exactly for every won-set.

    Enumerates won-sets with itertools.combinations (not by decoding masks), encodes
    each with mask_from_won, and checks that every mask is reached exactly once and
    holds scoring.py's value. A bit-order mistake in either direction is caught.
    """
    if lut.shape != (NUM_MASKS,):
        raise ValueError(f"lookup table must have shape ({NUM_MASKS},), got {lut.shape}")
    if lut.dtype != np.int64:
        raise ValueError(f"lookup table must be int64, got {lut.dtype}")
    seen = set()
    mismatches = []
    for size in range(NUM_BATTLEFIELDS + 1):
        for won in combinations(range(1, NUM_BATTLEFIELDS + 1), size):
            mask = mask_from_won(set(won))
            seen.add(mask)
            if int(lut[mask]) != score200(set(won)):
                mismatches.append((list(won), int(lut[mask]), score200(set(won))))
    if len(seen) != NUM_MASKS:
        raise ValueError(f"won-sets map to {len(seen)} distinct masks, expected {NUM_MASKS}")
    if mismatches:
        examples = "; ".join(f"won {w}: table {got}, scoring.py {exp}" for w, got, exp in mismatches[:5])
        raise ValueError(f"lookup table disagrees with scoring.py on {len(mismatches)} won-sets: {examples}")


def lut_hash(lut: np.ndarray) -> str:
    """SHA-256 of the table's bytes. Stored with every cached matrix; a change in the
    table (e.g. from a change to scoring.py) invalidates the cache."""
    return hashlib.sha256(lut.astype("<i8").tobytes()).hexdigest()


# --- Stage 3: W and S for every pair ----------------------------------------
#
# For strategies i (row) and j (column):
#   S200[i, j] = 200 × i's total score when i plays j          (int32)
#   W2[i, j]   = 2 if i wins, 1 if draw, 0 if i loses           (int8; W = W2 / 2)
# Each pair: compare troops on every battlefield, turn each side's wins into a mask,
# look both masks up in LUT200, compare the two scores.

CACHE = Path(__file__).resolve().parent / "cache"
BLOCK_ROWS = 256  # rows computed at once; limits memory to ~BLOCK_ROWS × N × 10 booleans
SAMPLE_CHECKS = 2000  # random pairs re-checked against scoring.match() on every build
CHECK_SEED = 20261004

BIT_VALUES = 1 << np.arange(NUM_BATTLEFIELDS, dtype=np.int64)  # bit i = battlefield i + 1


@dataclass
class Matrices:
    """W2 and S200 for a pool, with the fingerprints they were built from."""

    ids: np.ndarray
    S200: np.ndarray
    W2: np.ndarray
    pool_hash: str
    lut_hash: str


def win_masks(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks for every (row, col) pair: battlefields the row player wins, and battlefields
    the column player wins. Inputs are int arrays of shape (R, 10) and (C, 10)."""
    mine = rows[:, None, :] > cols[None, :, :]
    theirs = rows[:, None, :] < cols[None, :, :]
    return mine @ BIT_VALUES, theirs @ BIT_VALUES


def payoff_block(rows: np.ndarray, cols: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S200, W2) for every row strategy against every column strategy."""
    my_mask, their_mask = win_masks(rows, cols)
    my_score = lut[my_mask]
    their_score = lut[their_mask]
    W2 = (1 + np.sign(my_score - their_score)).astype(np.int8)
    return my_score.astype(np.int32), W2


def compute_matrices(allocations: np.ndarray, lut: np.ndarray, block_rows: int = BLOCK_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """S200 and W2 for all pairs, computed BLOCK_ROWS rows at a time."""
    n = len(allocations)
    S200 = np.empty((n, n), dtype=np.int32)
    W2 = np.empty((n, n), dtype=np.int8)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        S200[start:stop], W2[start:stop] = payoff_block(allocations[start:stop], allocations, lut)
    return S200, W2


def extend_matrices(
    S200: np.ndarray, W2: np.ndarray, old: np.ndarray, new: np.ndarray, lut: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Add rows/columns for `new` strategies without recomputing old pairs.
    Result equals compute_matrices on old followed by new."""
    combined = np.concatenate([old, new])
    n_old, n = len(old), len(old) + len(new)
    S_big = np.empty((n, n), dtype=np.int32)
    W_big = np.empty((n, n), dtype=np.int8)
    S_big[:n_old, :n_old], W_big[:n_old, :n_old] = S200, W2
    for start in range(0, n_old, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n_old)
        S_big[start:stop, n_old:], W_big[start:stop, n_old:] = payoff_block(old[start:stop], new, lut)
    for start in range(n_old, n, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n)
        S_big[start:stop], W_big[start:stop] = payoff_block(combined[start:stop], combined, lut)
    return S_big, W_big


def check_invariants(S200: np.ndarray, W2: np.ndarray) -> None:
    """Raise ValueError unless W2 + W2.T == 2, diag(W2) == 1, diag(S200) == 0, and W2
    agrees with comparing S200 to its transpose."""
    if not np.array_equal(W2.astype(np.int16) + W2.T, np.full(W2.shape, 2, dtype=np.int16)):
        raise ValueError("W2 + W2.T is not 2 everywhere (a win must be the other side's loss)")
    if not (np.diagonal(W2) == 1).all() or not (np.diagonal(S200) == 0).all():
        raise ValueError("a strategy against itself must be a 0-0 draw")
    if not np.array_equal(W2, (1 + np.sign(S200.astype(np.int64) - S200.T)).astype(np.int8)):
        raise ValueError("W2 disagrees with comparing S200[i, j] to S200[j, i]")


def check_against_scoring(
    allocations: np.ndarray, S200: np.ndarray, W2: np.ndarray, num_pairs: int = SAMPLE_CHECKS, seed: int = CHECK_SEED
) -> None:
    """Re-play a seeded random sample of pairs with scoring.match() and raise ValueError
    on any disagreement. scoring.match never uses masks, so this also catches bit-order
    mistakes that the lookup-table check alone cannot."""
    rng = np.random.default_rng(seed)
    n = len(allocations)
    for i, j in rng.integers(0, n, size=(num_pairs, 2)):
        result = match(allocations[i].tolist(), allocations[j].tolist())
        expected_w2 = int(round(2 * result.result_a))
        if S200[i, j] != result.score200_a or S200[j, i] != result.score200_b or W2[i, j] != expected_w2:
            raise ValueError(
                f"matrix disagrees with scoring.match for rows {i} and {j}: "
                f"S200 {S200[i, j]}/{S200[j, i]} vs {result.score200_a}/{result.score200_b}, "
                f"W2 {W2[i, j]} vs {expected_w2}"
            )


def matrix_path(pool_hash: str, cache_dir: Path = CACHE) -> Path:
    """Where the matrices for this pool are cached."""
    return cache_dir / f"matrix_{pool_hash[:16]}.npz"


def load_cached(path: Path, pool_hash: str, current_lut_hash: str) -> Matrices | None:
    """The cached matrices if the file exists and both fingerprints match, else None."""
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["pool_hash"]) != pool_hash or str(data["lut_hash"]) != current_lut_hash:
            return None
        return Matrices(data["ids"], data["S200"], data["W2"], pool_hash, current_lut_hash)


def build_matrices(
    ids: np.ndarray, allocations: np.ndarray, pool_hash: str, cache_dir: Path = CACHE, force: bool = False
) -> tuple[Matrices, bool]:
    """Return (matrices, reused). Reuses the cache when pool_hash and lut_hash match;
    otherwise computes, checks, and saves. Every build is checked; a reused cache is
    re-checked against the invariants."""
    lut = build_lut200()
    current_lut_hash = lut_hash(lut)
    path = matrix_path(pool_hash, cache_dir)
    cached = None if force else load_cached(path, pool_hash, current_lut_hash)
    if cached is not None and np.array_equal(cached.ids, ids):
        check_invariants(cached.S200, cached.W2)
        return cached, True

    S200, W2 = compute_matrices(allocations, lut)
    check_invariants(S200, W2)
    check_against_scoring(allocations, S200, W2)
    matrices = Matrices(np.asarray(ids), S200, W2, pool_hash, current_lut_hash)
    cache_dir.mkdir(exist_ok=True)
    np.savez_compressed(path, ids=matrices.ids, S200=S200, W2=W2, pool_hash=pool_hash, lut_hash=current_lut_hash)
    return matrices, False


def main() -> None:
    from pool import BATTLEFIELD_COLUMNS, load_pool

    pool, meta = load_pool()
    allocations = pool[BATTLEFIELD_COLUMNS].to_numpy(dtype=np.int64)
    started = time.perf_counter()
    matrices, reused = build_matrices(pool["id"].to_numpy(dtype=str), allocations, meta["pool_hash"])
    elapsed = time.perf_counter() - started
    n = len(matrices.ids)
    draws = (matrices.W2 == 1).sum() - n
    print(f"{'Reused' if reused else 'Built'} {n} × {n} matrices in {elapsed:.1f}s → {matrix_path(meta['pool_hash'])}")
    print(f"Draws (excluding self-play): {draws / (n * n - n):.2%} of pairs")
    if not reused:
        print(f"Checked invariants and {SAMPLE_CHECKS} random pairs against scoring.match()")


if __name__ == "__main__":
    main()
