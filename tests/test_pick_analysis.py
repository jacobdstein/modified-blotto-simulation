"""Tests for analysis/pick_analysis.py."""

import numpy as np
import pytest

from analysis.pick_analysis import field_weights, matchups
from matrix import build_lut200
from pool import build_pool, random_uniform
from scoring import match


def test_matchups_agree_with_scoring_match():
    rng = np.random.default_rng(8)
    me = np.array([1, 5, 5, 3, 2, 5, 28, 6, 36, 9])
    opponents = np.array([random_uniform(rng) for _ in range(50)] + [me.tolist()])
    m = matchups(me, opponents, build_lut200())
    for k, opponent in enumerate(opponents):
        result = match(me.tolist(), opponent.tolist())
        assert m["result"][k] == result.result_a
        assert m["my_score"][k] == result.score200_a and m["their_score"][k] == result.score200_b
        assert set(np.flatnonzero(m["won"][k]) + 1) == result.won_a
    assert m["tied"][-1].all()  # against itself every battlefield ties


def test_field_weights_sum_to_one_and_match_group_shares():
    pool, _ = build_pool()
    weights, groups, shares = field_weights(pool)
    assert weights.sum() == pytest.approx(1)
    assert sum(shares.values()) == pytest.approx(1)
    for group, vector in groups.items():
        assert vector.sum() == pytest.approx(1), group
