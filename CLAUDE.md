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
- Deduplication merges exact duplicates only, and only within the same origin (summing
  weight); identical allocations from different origins stay separate rows so per-source
  counts and the held-out round survive. An L1 threshold setting exists and defaults to
  0; never apply it to candidate strategies (origins hand:, llm:, counter:).
- Meta-game payoff for Nash, replicator dynamics, and elimination is W (win rate).
  S is reported but not optimized unless stated.
- Rankings weighting (diagnostic only): each source family (hand, llm:claude,
  llm:chatgpt, llm:gemini, riddler, riddler_adj, random_uniform, counter) gets equal total
  weight; within a family each subfamily (pool.origin_subfamily, e.g. riddler:r2,
  riddler_adj:rule1, hand:noisy) gets an equal share; within a subfamily, rows weigh by
  duplicate count. Held-out origins are excluded. Also report raw pool weighting.
- Iterated elimination: repeatedly cut the bottom 50% by win rate and re-rank among
  survivors; record every strategy's rank in every round.
- Per-match metrics: win credit (W) first, mean score (S) second. Also compute rankings
  with draw = 0 to check sensitivity; W2 supports both.

## Objective and selection rule (fixed before seeing results)
- The challenge rewards finishing in roughly the top 25–30% of entries. Treat 25% as the
  cutoff.
  Our objective is to maximize the probability of finishing in the top 20% (a safety
  margin), not to maximize expected win rate. Consistency beats peak results; finishing
  above the cutoff earns nothing extra.
- Among candidates with near-equal P(top 20%), the final submission is a judgment call,
  including how clearly the choice can be justified.
- Simulated contest: sample a field (size 100–225) from a scenario's weights (seeded), add
  the candidate as one more player, compute every player's wins and points against the
  rest, rank everyone, record the candidate's percentile. Repeat over many fields.
- Scenarios: level-based field (most weight on level 0–2, with the level mix itself
  randomized), naive-heavy, sophisticated-heavy, and the Nash mix.
- Primary: P(top 20%), ranked by wins (draw = 0.5), pooled over all simulated fields and
  scenarios. Tiebreakers: median percentile, then mean score.
- Reported checks (flag a candidate that collapses): P(top 20%) when ranked by total
  points, P(top 20%) with draw = 0, and how many ±1 neighbors stay close.
- rankings.py is diagnostic (builds the shortlist and explains it); robustness.py
  applies this rule and picks the submission.

## Field model
- Build the pool in reasoning levels: level 0 = naive human-style allocations (round
  numbers, value-proportional, runs, battlefield 6, base multiple of 5) plus
  LLM-suggested allocations (origin "llm"); level 1 = best responses to level 0;
  level 2 = best responses to level 1. Uniform random allocations are a small minority
  or a separate scenario, never the bulk of a field.
- Counterattack strategies belong in fields as a minority, not as the main target.
- Riddler data (approved): FiveThirtyEight `riddler-castles`, rounds 1–4 only (round 5
  has 13 castles). Same 10 castles / 100 soldiers, but no bonus rules and ties split
  points, so it models how people allocate, not which battlefields they'd target here.
  Measured under our rules, Riddler entries are not junk (rounds 2–3 rank just behind the
  Claude/ChatGPT strategies), so there is no quality filter on opponents: weak players
  are part of the real field. Use a seeded sample of 300 submissions per round (repeats
  kept in proportion). Drop rows that fail validate_allocation and report how many.
- Bonus-adjusted Riddler ("riddler_adj:<version>:r<N>", parent = raw row): a Riddler
  player who noticed the bonuses makes ONE swap — stop contesting one battlefield (≥5
  troops = contested) and move its troops, unchanged, to an uncontested one. The swap
  must gain a bonus the plan lacked. Versions: all (any bonus; highest-scoring swap),
  rule1 / rule2 / rule3 (only that bonus; smallest swap by value distance, ties → higher
  score). No row if that player has nothing to fix.
- Measured: the swap usually makes an entry weaker than its parent (median win-rate
  change −4 to −7 points; rule3 ≈ 0), likely because moved troops are sized for the old
  battlefield. Decision: keep as is, treat riddler_adj as "bonus-aware but clumsy"
  players with a modest field weight; strong opponents come from llm:claude,
  llm:chatgpt, and counterattack.
- LLM strategies: I collect them from ChatGPT and Gemini (and a fresh claude.ai chat with
  no project context), origin "llm:<tool>". Claude Code does not invent "LLM-style"
  answers itself.
- data/llm_strategies.csv: 10 each from Claude, ChatGPT, Gemini, one chat per tool. Each
  got the PDF, my answers to its clarifying questions, and the goal "top 25%", so they
  model AI-assisted applicants. Each tool's 10 are one correlated cluster: in fields,
  give each tool's first-listed strategy extra weight (Claude labeled #1 its top pick;
  ChatGPT/Gemini order may not be a ranking). Assign reasoning levels by behavior, not
  by tool (Gemini ≈ level 0, ChatGPT ≈ 0–1, Claude ≈ 1–2, which ran its own simulations).
- Held-out check: Riddler round 4 and its adjusted versions are in the pool (listed in
  pool_meta.json holdout_origins) so the matrix covers them, but they are never used for
  building or selecting. After the
  shortlist is chosen, it must still reach top 20% against a field that includes it.
- Re-test: choose the shortlist on one set of seeds, make the final comparison on fresh
  seeds.
- Field size per simulated contest: drawn uniformly from 100–225.

## Scope (deadline: submission closes Tuesday 2026-10-06 at midnight)
- Minimum version first: stages 1–3 (win rate, elimination), the simulated-contest
  selection with ±1 neighborhoods and 2–3 scenarios, and 1–2 counterattack rounds.
- Replicator dynamics and Nash are optional extras if time allows. If time is short,
  cut optional analysis before cutting writeup time.
- After stage 1, analyze the lookup table (points per troop cost of won-sets) to suggest
  hand-built candidates.
- Counterattack local search jumps to threshold values (0 or opponent count + 1) rather
  than only ±1 moves.

## Pipeline (each stage reads only the previous stage's saved output)
1. pool.py — generate strategies → cache/pool.csv + cache/pool_meta.json
2. matrix.py — lookup table + W, S for every pair → cache/matrix_<pool_hash>.npz
3. rankings.py — round-robin win rate, iterated elimination, replicator dynamics,
   Nash of the meta-game; diagnostic, produces the shortlist → results/rankings_*.csv
4. robustness.py — simulated contests over sampled fields, ±1-troop neighborhoods,
   applies the selection rule → results/robustness_*.csv
5. counterattack.py — best-response loop against top strategies; appends to the pool,
   then stages 2–3 re-run

## Layout
- scoring.py — reference scoring and match functions (source of truth)
- pool.py, matrix.py, rankings.py, robustness.py, counterattack.py — pipeline stages
- tests/ — pytest tests
- analysis/ — exploratory scripts that write to results/ (e.g. lut_analysis.py); not pipeline stages
- examples/ — small standalone teaching scripts (e.g. abc_example.py); not part of the pipeline
- data/ — committed input data (llm_strategies.csv; riddler/ rounds 1–4, downloaded
  from github.com/fivethirtyeight/data/tree/master/riddler-castles)
- results/ — saved outputs (CSV, committed)
- cache/ — pool and matrices (gitignored, regenerable)

## Data formats
- cache/pool.csv: id (str, stable, e.g. "P00042"), origin (str: "hand:<name>",
  "hand:<name>:round5", "hand:<name>:noisy", "llm:<tool>:<index>", "riddler:r<N>",
  "riddler_adj:<version>:r<N>", "random_uniform", later "counter:r<N>"), b1..b10 (int, sum 100), weight (int ≥1, number
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
