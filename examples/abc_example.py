"""Walk through the A/B/C example: masks, the S and W matrices, field win rates,
and a few steps of replicator dynamics.

Run from the project root:
    .venv/bin/python examples/abc_example.py

Edit STRATEGIES and FIELD below to try your own allocations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import scoring` works

from scoring import match, score200

# --- Edit these -------------------------------------------------------------

STRATEGIES = {
    "A": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # even spread
    "B": [0, 0, 0, 0, 0, 0, 25, 25, 25, 25],  # top four battlefields
    "C": [0, 0, 0, 0, 20, 20, 20, 20, 20, 0],  # middle block
}

FIELD = {"A": 100, "B": 50, "C": 50}  # weights: how many copies of each in the field

REPLICATOR_STEPS = 10

# ----------------------------------------------------------------------------

NAMES = list(STRATEGIES)


def mask_of(won: frozenset[int]) -> int:
    """Battlefield n won → bit n-1 set. This is the index into the 1024-entry table."""
    return sum(1 << (battlefield - 1) for battlefield in won)


def print_table(title: str, rows: dict[str, list[float]], fmt: str) -> None:
    """Print a small labeled matrix: rows = me, columns = opponent."""
    print(f"\n{title}  (row = me, column = opponent)")
    print("      " + "".join(f"{name:>9}" for name in NAMES))
    for name in NAMES:
        print(f"  {name:<4}" + "".join(f"{value:>9{fmt}}" for value in rows[name]))


def show_matches() -> None:
    """Every pair: who wins which battlefields, the masks, and the scores."""
    print("=== Matches ===")
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1 :]:
            result = match(STRATEGIES[a], STRATEGIES[b])
            print(f"\n{a} vs {b}")
            for name, won, s200 in (
                (a, result.won_a, result.score200_a),
                (b, result.won_b, result.score200_b),
            ):
                mask = mask_of(won)
                print(
                    f"  {name} wins {sorted(won)!s:<20} "
                    f"mask {mask:>4} = {mask:010b}   "
                    f"table[{mask}] = {s200:>5}  →  score {s200 / 200:g}"
                )
            outcome = {1.0: f"{a} wins", 0.5: "draw", 0.0: f"{b} wins"}[result.result_a]
            print(f"  → {outcome}")


def build_matrices() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """S[me][opp] = my score; W[me][opp] = 1 win, 0.5 draw, 0 loss."""
    S = {me: [match(STRATEGIES[me], STRATEGIES[opp]).score_a for opp in NAMES] for me in NAMES}
    W = {me: [match(STRATEGIES[me], STRATEGIES[opp]).result_a for opp in NAMES] for me in NAMES}
    return S, W


def check_invariants(S: dict[str, list[float]], W: dict[str, list[float]]) -> None:
    """W[i][j] + W[j][i] = 1, diagonal of W = 0.5, diagonal of S = 0."""
    for i, a in enumerate(NAMES):
        assert S[a][i] == 0 and W[a][i] == 0.5
        for j, b in enumerate(NAMES):
            assert W[a][j] + W[b][i] == 1
    print("\nInvariants hold: W[i][j] + W[j][i] = 1, diag(W) = 0.5, diag(S) = 0")


def field_win_rates(W: dict[str, list[float]], S: dict[str, list[float]]) -> None:
    """Weighted win rate and mean score vs the field. You never play yourself,
    so you face one fewer copy of your own strategy."""
    print(f"\n=== Against the field {FIELD} ===")
    for i, me in enumerate(NAMES):
        opponents = {opp: FIELD[opp] - (1 if opp == me else 0) for opp in NAMES}
        total = sum(opponents.values())
        wins = sum(opponents[opp] * W[me][j] for j, opp in enumerate(NAMES))
        points = sum(opponents[opp] * S[me][j] for j, opp in enumerate(NAMES))
        terms = " + ".join(f"{opponents[opp]}×{W[me][j]:g}" for j, opp in enumerate(NAMES))
        print(
            f"  {me}: ({terms}) / {total} = {wins / total:6.1%} win rate, "
            f"mean score {points / total:.2f}"
        )


def replicator(W: dict[str, list[float]]) -> None:
    """Each step, a strategy's share grows in proportion to how well it does
    against the current population (its 'fitness') relative to the average."""
    print(f"\n=== Replicator dynamics ({REPLICATOR_STEPS} steps, equal starting shares) ===")
    shares = {name: 1 / len(NAMES) for name in NAMES}
    for step in range(REPLICATOR_STEPS + 1):
        print(f"  step {step:>2}: " + "  ".join(f"{n} {shares[n]:6.1%}" for n in NAMES))
        fitness = {me: sum(W[me][j] * shares[opp] for j, opp in enumerate(NAMES)) for me in NAMES}
        average = sum(shares[n] * fitness[n] for n in NAMES)
        shares = {n: shares[n] * fitness[n] / average for n in NAMES}


def main() -> None:
    show_matches()
    S, W = build_matrices()
    print_table("S matrix: my score", S, ".2f")
    print_table("W matrix: 1 = win, 0.5 = draw, 0 = loss", W, "g")
    check_invariants(S, W)
    field_win_rates(W, S)
    replicator(W)
    print(f"\n(Table check: won-set {{3,5,6,7}} → mask {mask_of(frozenset({3,5,6,7}))}, "
          f"table value {score200({3, 5, 6, 7})} = 33.0 × 200)")


if __name__ == "__main__":
    main()
