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
- Equal total scores are a draw (0.5 win), matching the problem statement. No tiebreak.
- Exact arithmetic: compare scores as integers (score × 200 is always an integer).
  Convert to float only for display; tests use pytest.approx on floats.
- Pool = every strategy we have considered. A "field" (e.g. 200 players, duplicates
  allowed) is a weight vector over pool IDs. A strategy's win rate against a field is
  the weighted mean of its row of W; a copy of yourself counts as a draw, and you
  never play yourself.
- Deduplication merges exact duplicates only (summing weight). An L1 threshold
  setting exists and defaults to 0; never apply it to candidate strategies.
- Meta-game payoff for Nash, replicator dynamics, and elimination is W (win rate).
  S is reported but not optimized unless stated.
- Iterated elimination: repeatedly cut the bottom 50% by win rate and re-rank among
  survivors; record every strategy's rank in every round.
- Primary metric: win rate. Secondary: mean score.

## Pipeline (each stage reads only the previous stage's saved output)
1. pool.py — generate strategies → cache/pool.csv + cache/pool_meta.json
2. matrix.py — lookup table + W, S for every pair → cache/matrix_<pool_hash>.npz
3. rankings.py — round-robin win rate, iterated elimination, replicator dynamics,
   Nash of the meta-game → results/rankings_*.csv
4. robustness.py — weighted opponent fields, ±1-troop neighborhoods → results/robustness_*.csv
5. counterattack.py — best-response loop against top strategies; appends to the pool,
   then stages 2–3 re-run

## Layout
- scoring.py — reference scoring and match functions (source of truth)
- pool.py, matrix.py, rankings.py, robustness.py, counterattack.py — pipeline stages
- tests/ — pytest tests
- results/ — saved outputs (CSV, committed)
- cache/ — pool and matrices (gitignored, regenerable)

## Data formats
- cache/pool.csv: id (str, stable, e.g. "P00042"), origin (str, e.g. "random_uniform",
  "hand:proportional", "counter:r3"), b1..b10 (int, sum 100), weight (int ≥1, number
  of merged duplicates), parent (id or empty). Rows sorted by id; row order = matrix
  index order.
- cache/pool_meta.json: seed, generator settings, dedup threshold, pool_hash
  (sha256 of b1..b10 in row order).
- cache/matrix_<pool_hash>.npz:
  - ids (str[N])
  - S200 (int32[N,N]): S200[i,j] = 200 × i's total score when i plays j
  - W2 (int8[N,N]): 2 = i wins, 1 = draw (S200[i,j] == S200[j,i]), 0 = i loses; W = W2 / 2
  - lut_hash
  - Invariants: W2 + W2.T == 2; diag(W2) == 1; diag(S200) == 0.
  - The cache is valid only if pool_hash and lut_hash both match; otherwise rebuild.
- Lookup table: LUT200[mask] (int64[1024]); bit i of mask = battlefield i+1 won.
  Built from scoring.py's integer score, never by rounding floats.

## Lookup-table rule
- LUT200 must equal scoring.py exactly for all 1024 masks. A test checks this, and
  matrix.py re-checks it every time it builds the table. Any change to scoring.py or
  the table requires both checks to pass and invalidates every cached matrix.
- matrix.py must also agree with scoring.match() on a random sample of pairs every build.

## Commands
- Setup: `uv venv && uv pip install pytest numpy pandas matplotlib scipy`
- Tests: `.venv/bin/python -m pytest -v`
- Run: `python pool.py && python matrix.py && python rankings.py` (then robustness.py,
  counterattack.py as needed)

## Working rules
- Run tests after every change.
- Ask before modifying scoring.py; all results depend on it.
- Set random seeds so runs are reproducible; record seeds in saved metadata.
- Save results to results/ with descriptive filenames.
- Only use numpy, pandas, matplotlib, scipy unless I approve others. pytest is approved
  as a dev-only dependency.
