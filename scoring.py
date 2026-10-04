"""Reference scoring for the modified Colonel Blotto challenge.

This module is the source of truth: every faster implementation (e.g. the lookup
table in matrix.py) must agree with it exactly. It favors readability over speed.

Conventions:
- An allocation is a list or tuple of 10 non-negative ints summing to 100.
  Index i (0-based) holds the troops sent to battlefield i + 1.
- Battlefield n is worth n points, so "battlefield number" and "value" coincide.
- A set of won battlefields uses battlefield numbers 1..10 (not indices).
- Scores are exact Fractions; score200() gives the exact integer 200 × score.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import AbstractSet, Sequence

NUM_BATTLEFIELDS = 10
TOTAL_TROOPS = 100
BATTLEFIELDS = range(1, NUM_BATTLEFIELDS + 1)  # battlefield n is worth n points

ADJACENCY_MULTIPLIER = Fraction(3, 2)  # Rule 1
MULTIPLE_OF_FIVE_MULTIPLIER = Fraction(7, 5)  # Rule 2
BONUS_BATTLEFIELD = 6
BONUS_BATTLEFIELD_MULTIPLIER = Fraction(11, 10)  # Rule 3

SCORE_SCALE = 200  # 200 × score is always an integer


# --- Validation -------------------------------------------------------------


def validate_allocation(allocation: Sequence[int]) -> tuple[int, ...]:
    """Return the allocation as a tuple, or raise if it is not a legal submission.

    Raises TypeError unless it is a list/tuple of plain ints (bools rejected).
    Raises ValueError unless it has 10 entries, all >= 0, summing to 100.
    """
    if not isinstance(allocation, (list, tuple)):
        raise TypeError(f"allocation must be a list or tuple, got {type(allocation).__name__}")
    if len(allocation) != NUM_BATTLEFIELDS:
        raise ValueError(f"allocation must have {NUM_BATTLEFIELDS} entries, got {len(allocation)}")
    for index, troops in enumerate(allocation):
        if isinstance(troops, bool) or not isinstance(troops, int):
            raise TypeError(f"entry {index} must be an int, got {type(troops).__name__}")
        if troops < 0:
            raise ValueError(f"entry {index} is negative: {troops}")
    if sum(allocation) != TOTAL_TROOPS:
        raise ValueError(f"allocation must sum to {TOTAL_TROOPS}, got {sum(allocation)}")
    return tuple(allocation)


def validate_won(won: AbstractSet[int]) -> frozenset[int]:
    """Return the won set as a frozenset, or raise if it holds anything but 1..10."""
    for battlefield in won:
        if isinstance(battlefield, bool) or not isinstance(battlefield, int):
            raise TypeError(f"battlefield must be an int, got {type(battlefield).__name__}")
        if battlefield not in BATTLEFIELDS:
            raise ValueError(f"battlefield must be in 1..{NUM_BATTLEFIELDS}, got {battlefield}")
    return frozenset(won)


# --- Battlefield outcomes ---------------------------------------------------


def battlefields_won(mine: Sequence[int], theirs: Sequence[int]) -> frozenset[int]:
    """Battlefield numbers where `mine` has strictly more troops. Ties win nothing."""
    mine = validate_allocation(mine)
    theirs = validate_allocation(theirs)
    return frozenset(
        battlefield
        for battlefield, my_troops, their_troops in zip(BATTLEFIELDS, mine, theirs)
        if my_troops > their_troops
    )


# --- Scoring rules ----------------------------------------------------------


def base_score(won: AbstractSet[int]) -> int:
    """Sum of the values of the won battlefields."""
    return sum(won)


def has_won_neighbor(battlefield: int, won: AbstractSet[int]) -> bool:
    """True if battlefield - 1 or battlefield + 1 is also won. No wrap-around."""
    return (battlefield - 1) in won or (battlefield + 1) in won


def apply_rule1_adjacency(won: AbstractSet[int]) -> Fraction:
    """Rule 1: each won battlefield with a won neighbor counts 1.5×, at most once."""
    score = Fraction(0)
    for battlefield in won:
        if has_won_neighbor(battlefield, won):
            score += battlefield * ADJACENCY_MULTIPLIER
        else:
            score += battlefield
    return score


def rule2_multiplier(base: int) -> Fraction:
    """Rule 2: 1.4× if the *base* score is a multiple of 5, else 1×."""
    return MULTIPLE_OF_FIVE_MULTIPLIER if base % 5 == 0 else Fraction(1)


def rule3_multiplier(won: AbstractSet[int]) -> Fraction:
    """Rule 3: 1.1× if battlefield 6 is won, else 1×."""
    return BONUS_BATTLEFIELD_MULTIPLIER if BONUS_BATTLEFIELD in won else Fraction(1)


def total_score(won: AbstractSet[int]) -> Fraction:
    """Exact total score for a set of won battlefields: Rule 1, then 2, then 3."""
    won = validate_won(won)
    score = apply_rule1_adjacency(won)
    score *= rule2_multiplier(base_score(won))
    score *= rule3_multiplier(won)
    return score


def score200(won: AbstractSet[int]) -> int:
    """Exact integer 200 × total_score(won). Use this for comparisons."""
    scaled = total_score(won) * SCORE_SCALE
    assert scaled.denominator == 1, f"200 × score is not an integer for {sorted(won)}"
    return int(scaled)


# --- Matches ----------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one head-to-head match between players A and B."""

    won_a: frozenset[int]
    won_b: frozenset[int]
    score200_a: int
    score200_b: int

    @property
    def score_a(self) -> float:
        """A's total score as a float, for display only."""
        return self.score200_a / SCORE_SCALE

    @property
    def score_b(self) -> float:
        """B's total score as a float, for display only."""
        return self.score200_b / SCORE_SCALE

    @property
    def result_a(self) -> float:
        """A's win credit: 1 for a win, 0.5 for equal scores, 0 for a loss."""
        if self.score200_a > self.score200_b:
            return 1.0
        if self.score200_a == self.score200_b:
            return 0.5
        return 0.0

    @property
    def result_b(self) -> float:
        """B's win credit; always 1 - result_a."""
        return 1.0 - self.result_a


def match(a: Sequence[int], b: Sequence[int]) -> MatchResult:
    """Play allocation `a` against `b`. Each side is scored independently."""
    won_a = battlefields_won(a, b)
    won_b = battlefields_won(b, a)
    return MatchResult(won_a, won_b, score200(won_a), score200(won_b))
