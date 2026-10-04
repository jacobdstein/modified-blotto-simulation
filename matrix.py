"""Payoff matrices for the pool, built from a verified score lookup table.

Stage 1 (this section): the lookup table.
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
from collections.abc import Set as AbstractSet
from itertools import combinations

import numpy as np

from scoring import NUM_BATTLEFIELDS, score200, validate_won

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
