# Blotto Simulation

## Purpose
Evaluate troop allocations for a modified Colonel Blotto challenge.
Metrics: head-to-head win rate and mean total score.

## Scoring rules (source of truth — do not reinterpret)
- 10 battlefields, values 1–10. Allocations: 10 non-negative ints summing to 100.
- Win a battlefield only with strictly more troops. Ties: nobody gets it.
- Base = sum of values of won battlefields.
- Rule 1: each won battlefield with ≥1 won neighbor (|i−j|=1, no wrap) counts 1.5×. Applied once per battlefield.
- Rule 2: if base % 5 == 0, multiply score by 1.4. Check uses base, not score.
- Rule 3: if battlefield 6 is won, multiply score by 1.1.
- Order: Rule 1 → Rule 2 (checked on base) → Rule 3. In a run of consecutive won
  battlefields, every battlefield in the run gets 1.5×, never more than once.
- Indexing: allocation index i (0-based) is battlefield i+1 with value i+1. Battlefield 6 is index 5.
- Every player is scored the same way, independently, from the battlefields they won.
- Test cases: win {3,5,6,7} → 33.0; win all → 127.05; win none → 0.

## Decisions
- Tied total scores: the player who won more battlefields (count, not value) wins.
  If total score and battlefield count are both equal, it's a draw (0.5 win).
- Exact arithmetic: compare scores as integers (score × 200 is always an integer).
  Convert to float only for display; tests use pytest.approx on floats.
- Simulation: 200 players, each with their own strategy (duplicates allowed).
  Every player plays every other player (round-robin).
- Primary metric: win rate. Secondary: mean score.

## Layout
- scoring.py — scoring and match functions
- opponents.py — opponent population generators
- tournament.py — evaluation against populations
- search.py — optimization methods
- tests/ — pytest tests
- results/ — saved outputs (CSV)

## Commands
- Tests: `python -m pytest -v`
- Run: `python tournament.py`

## Working rules
- Run tests after every change.
- Ask before modifying scoring.py; all results depend on it.
- Set random seeds so runs are reproducible.
- Save results to results/ with descriptive filenames.
- Only use numpy, pandas, matplotlib unless I approve others. pytest is approved as a dev-only dependency.
