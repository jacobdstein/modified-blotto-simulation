# Blotto Simulation

## Purpose
Choose one troop allocation for a modified Colonel Blotto challenge. Objective: maximize
the probability of finishing in the top 20% of a simulated field of applicants (see
"Objective and selection rule"). Per-match metrics: win credit first, score second.

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
  riddler_adj:rule1, hand:noisy, llm:claude:3 = one AI answer plus its variants) gets an
  equal share; within a subfamily, rows weigh by duplicate count. Held-out origins are
  excluded. Also report raw pool weighting.
- A row with weight k stands for k identical submissions. A player never plays itself,
  but its k − 1 identical copies are opponents and count as draws.
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
- Simulated contest: sample a field (size 100–225, uniform) from a scenario's weights
  (seeded, with replacement), add the candidate as one more player, compute every
  player's wins and points against all the others, rank everyone, record the candidate's
  rank. Repeat over many fields. The same sampled fields are used for every candidate.
- Top 20%: with P = field size + 1 players and ties in the ranking measure sharing the
  average rank, the candidate is in the top 20% if rank / P ≤ 0.20 (e.g. rank ≤ 30 of
  150). "Percentile" in results means rank / P (smaller is better).
- Within a group, rows are weighted as in rankings (equal per subfamily, then by
  duplicate count), except AI groups: each AI answer = original + its jitter variants,
  shared by the pool's weight column; the first-listed answer gets 3× the weight of each
  other answer (25% of the tool's share).
- Groups with no rows yet (counter, before counterattack exists) have their share spread
  proportionally over the other groups before the Dirichlet step; groups with 0% share
  stay at 0.
- Scenarios: baseline, naive-heavy, sophisticated-heavy, AI-heavy (group shares: see
  "Field scenarios"), and the Nash mix of the non-held-out pool. In every contest the
  group shares are randomized around the scenario's values (Dirichlet, concentration 20).
- Primary (the score): P(top 20%), ranked by wins (draw = 0.5; ties in total wins share
  the average rank), pooled over the four realistic scenarios with relative weights
  baseline 40, sophisticated-heavy 24, AI-heavy 19, naive-heavy 7. Lower naive weight
  because the challenge is optional, so low-effort applicants mostly skip it.
- Ranking order: score rounded to the nearest percentage point (equal = tied); within a
  tie, unflagged candidates before flagged ones, then the Nash scenario's P(top 20%),
  then median percentile, then mean score. Nash is run and reported for every candidate
  but is not in the score or the flags (see Decision log). Finalists are re-ranked after
  the hold-out and neighbor flags are added.
- Also report P(top 20%) per scenario; flag any candidate below 50% in any single
  realistic scenario.
- Runs: selection = 2,000 contests per scenario for every shortlisted strategy (one
  seed); final = top 10 re-run with 10,000 contests per scenario on a fresh seed.
  Candidates are distinct allocations: when the same allocation appears under several
  ids (different origins), it is evaluated once and the other ids are listed.
- Hold-out check: baseline scenario with the Riddler groups drawn only from round 4
  (raw r4 fills both Riddler-human slots, adjusted r4 the clumsy slot). Pass: P(top 20%)
  ≥ 50% and no more than 15 points below the finalist's baseline value.
- Hold-out mapping: the normal group weighting always excludes round 4, so the hold-out
  check uses its own mapping (riddler:r4 → both Riddler-human groups, riddler_adj:*:r4 →
  the clumsy group; rounds 1–3 Riddler rows are not used in that check).
- Reported checks; a candidate is flagged if any of these is below 50%: P(top 20%) in
  any single realistic scenario, pooled P(top 20%) when ranked by total points, pooled P(top 20%)
  with draws counted as 0. ±1 neighbors: every valid one-troop move (≤ 90); a neighbor is
  "close" if its pooled P(top 20%) is within 10 points of the candidate's (same fields);
  flag if fewer than half are close. A neighbor that beats the candidate is reported,
  not auto-selected.
- rankings.py is diagnostic (builds the shortlist and explains it); robustness.py
  applies this rule and picks the submission.
- Shortlist (rankings.py): union of the top 100 by family-weighted round-robin rank, the
  top 100 by elimination depth (rounds survived, then last-round rank), and the top 5 of
  every source family. Elimination depth in the last few rounds is noisy (a handful of
  strategies playing rock-paper-scissors), so it is one input, not a verdict.

## Decision log
- Nash scenario (2026-10-04). Added as a worst-case field (the equilibrium mix: no
  strategy beats it on average), originally with 10% of the pooled weight and subject to
  the 50% per-scenario flag. First full run: the best of all 177 distinct shortlisted
  candidates reached the top 20% in only 22.5% of Nash contests, so the flag fired for
  every candidate; Nash results were essentially uncorrelated with the four realistic
  scenarios (correlation −0.09); and because the top candidates were at 99–100% in all
  realistic scenarios, the Nash term decided 6 of the 10 finalist slots. Decision: remove
  Nash from the score and the flags, keep reporting it, and use it as an explicit
  tiebreaker between candidates with equal (rounded) scores.
- Counterattack round 1 (2026-10-04). All 20 counters beat all 10 finalists (many by
  putting 7 instead of 6 troops on battlefields 9–10) while winning 76–87% against the
  baseline field. After adding them, the previous pick P01229 fell from 100% to 89.9%
  pooled (72.8% in sophisticated-heavy, where counters are 15% of the field); every
  candidate still at ≥ 99% is a counter strategy.
- Counterattack round 2 (2026-10-04). All 20 round-2 counters beat all 10 round-1
  finalists; the new top 10 are all round-2 counters. But the round-1 finalists, now
  targeted, barely dropped (P04443 100% → 99.0%, P04447 → 98.7%), unlike the original
  leaders after round 1 (P01229 100% → 89.9%). Pattern: the newest round always leads,
  because no counters aimed at it exist yet; being robust to a counterattack aimed at
  you is the more informative test.
- Final opponent-only round (2026-10-04, decided before round 3 ran). Because the newest
  counters always lead untested, round 3 counters are opponents only, so the pick must
  hold up against a counterattack aimed at it.
- Counterattack round 3 result (2026-10-04). The round-2 leader P04460
  [1,5,5,3,2,5,28,6,36,9] stayed #1 after a round of counters aimed at it: pooled
  P(top 20%) 1.000 → 0.997 (sophisticated-heavy 0.989), points 0.922, draw0 0.997,
  hold-out 0.996, all ±1 neighbors close, Nash 0.184, no flags. For comparison after
  round 3: round-1 leader P04443 0.976, original leader P01229 0.925.
- Flags before the Nash tiebreaker (2026-10-04). With Nash as the first tiebreaker and a
  1-point tie window, the rule picked P01353 (score 99.5%, Nash 20.1%) over P01229
  (score 100%, Nash 19.3%), although P01353 fails the points check and P01229 has no
  flags; the Nash gap (< 1 point) is about two standard errors. Decision: within a tie,
  unflagged candidates rank first; keep the 1-point window.

## Field scenarios (share of the field, %)
| Group (pool source)                        | Baseline | Naive-heavy | Sophisticated-heavy | AI-heavy |
|--------------------------------------------|----------|-------------|---------------------|----------|
| Naive rules (hand:*)                       | 7        | 30          | 5                   | 10       |
| Naive Blotto humans (riddler:r1)           | 7        | 20          | 0                   | 5        |
| Experienced Blotto humans (riddler:r2, r3) | 20       | 5           | 25                  | 10       |
| Bonus-aware but clumsy (riddler_adj, r1–3) | 15       | 15          | 5                   | 5        |
| AI: Gemini (llm:gemini, incl. jitter)      | 10       | 15          | 5                   | 20       |
| AI: ChatGPT (llm:chatgpt, incl. jitter)    | 18       | 10          | 20                  | 25       |
| AI: Claude (llm:claude, incl. jitter)      | 13       | 5           | 25                  | 20       |
| Random (random_uniform)                    | 3        | 0           | 0                   | 0        |
| Counter (counter:*, from step 5)           | 7        | 0           | 15                  | 5        |
- Baseline has 14% naive players: the challenge is optional, so low-effort applicants
  mostly skip it. Naive-heavy covers the case where they don't.
- Fifth scenario: the Nash mix of the non-held-out pool (reported and tiebreaker only).

## Field model
- Reasoning levels come from the sources, by behavior: level 0 = naive rules (hand:*),
  first-time Blotto humans (riddler:r1), Gemini; level 0–1 = ChatGPT, bonus-aware but
  clumsy (riddler_adj); level 1–2 = experienced Blotto humans (riddler:r2, r3), Claude;
  level 2+ = counterattack strategies built to beat the current leaders. Uniform random
  allocations are a small minority, never the bulk of a field.
- Counterattack strategies belong in fields as a minority, not as the main target.
- Counterattack (counterattack.py, step 5): each counter maximizes 50% mean win rate
  against the current finalists (results/robustness_final.csv) + 50% win rate against
  the expected baseline field. Steepest-ascent local search; moves shift troops between
  two battlefields so one lands on a threshold (0 or one more than some target's count;
  nearest 3 above/below, plus 1-troop moves) and stop when nothing improves. 40 seeded
  starts per round: the best 10 distinct Riddler-family and 10 distinct AI strategies by
  round-robin rank, plus 20 random. Keep the 20 best distinct results as
  data/counter/round_<N>.csv (origin counter:r<N>, parent = start id or empty). After
  each round: pool.py, matrix.py, rankings.py, robustness.py. Counter strategies from
  rounds 1–2 are allowed as candidates for our own submission.
- Round 3 is the final, opponent-only round (pool.OPPONENT_ONLY_COUNTER_ROUNDS): its
  counters, built against the round-2 finalists, are opponents in every field (part of
  the counter group) but are never ranked, shortlisted, or picked. So every eligible
  candidate has faced a counterattack aimed at it. counterattack.py refuses a round 4.
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
  must gain a bonus the plan lacked. Versions: all (any bonus, and the swap must raise
  the score of winning the contested set; highest-scoring swap), rule1 / rule2 / rule3 (only that bonus; smallest swap by value distance, ties → higher
  score). No row if that player has nothing to fix.
- Measured: the swap usually makes an entry weaker than its parent (median win-rate
  change −4 to −7 points; rule3 ≈ 0), likely because moved troops are sized for the old
  battlefield. Decision: keep as is, treat riddler_adj as "bonus-aware but clumsy"
  players with a modest field weight; strong opponents come from llm:claude,
  llm:chatgpt, and counterattack.
- LLM strategies are collected by me from the chatbots (origin "llm:<tool>:<index>");
  Claude Code does not invent "LLM-style" answers itself.
- data/llm_strategies.csv: 10 each from Claude, ChatGPT, Gemini, one chat per tool. Each
  got the PDF, my answers to its clarifying questions, and the goal "top 25%", so they
  model AI-assisted applicants. Each tool's 10 are one correlated cluster: in fields,
  give each tool's first-listed strategy extra weight (Claude labeled #1 its top pick;
  ChatGPT/Gemini order may not be a ranking). Assign reasoning levels by behavior, not
  by tool (Gemini ≈ level 0, ChatGPT ≈ 0–1, Claude ≈ 1–2, which ran its own simulations).
- Held-out data: Riddler round 4 and its adjusted versions are in the pool (listed in
  pool_meta.json holdout_origins) so the matrix covers them, but they are never used for
  building or selecting — only in the hold-out check (see "Objective and selection rule").
- AI-answer variation: each of the 30 AI strategies has 10 seeded variants moving 1–4
  troops, never identical to the original (origin "llm:<tool>:<index>:jitter", parent =
  original), so AI players in a field aren't all exact copies.
- Pool IDs are stable: new sources are always appended in pool.py, and
  tests/pool_snapshot.json pins every existing row's ID and allocation. Update the
  snapshot only deliberately, when an existing source is intentionally changed.

## Scope (deadline: submission closes Tuesday 2026-10-06 at midnight)
- Done: scoring.py, lookup table, pool (step 1), matrices (step 2), rankings + Nash mix +
  shortlist (step 3), lookup-table analysis.
- Done: robustness.py (step 4); counterattack.py (step 5) rounds 1–2.
- Done: counterattack rounds 1–3 (round 3 opponent-only). Selection-rule pick: P04460.
- Next: the final pick (my judgment among near-ties) and the writeup.
- Optional if time allows: replicator dynamics; re-running the lookup-table price
  analysis with Riddler as the reference field. If time is short, cut optional analysis
  before cutting writeup time.
- Counterattack local search jumps to threshold values (0 or opponent count + 1) rather
  than only ±1 moves.

## Pipeline (each step reads only earlier steps' saved output; code docstrings use these numbers)
0. scoring.py — reference scoring (foundation; not a pipeline step)
1. pool.py — generate strategies → cache/pool.csv + cache/pool_meta.json
2. matrix.py — lookup table + W, S for every pair → cache/matrix_<16 chars of pool_hash>.npz
3. rankings.py — round-robin win rate, iterated elimination, Nash mix of the meta-game
   (diagnostic) → results/rankings_*.csv, including the shortlist and rankings_nash.csv
4. robustness.py — simulated contests over sampled fields, ±1-troop neighborhoods,
   hold-out check, applies the selection rule → results/robustness_*.csv
5. counterattack.py — best-response loop against top strategies; appends to the pool,
   then steps 2–4 re-run

## Layout
- scoring.py — reference scoring and match functions (source of truth)
- pool.py, matrix.py, rankings.py, robustness.py, counterattack.py — pipeline steps 1–5
- tests/ — pytest tests
- analysis/ — exploratory scripts that write to results/ (e.g. lut_analysis.py); not pipeline steps
- examples/ — small standalone teaching scripts (e.g. abc_example.py); not part of the pipeline
- data/ — committed input data (llm_strategies.csv; riddler/ rounds 1–4, downloaded
  from github.com/fivethirtyeight/data/tree/master/riddler-castles; counter/ is
  generated by counterattack.py, committed so the pool can be rebuilt without re-running
  the search)
- results/ — saved outputs (CSV, committed)
- cache/ — pool and matrices (gitignored, regenerable)

## Data formats
- cache/pool.csv: id (str, stable, e.g. "P00042"), origin (str: "hand:<name>",
  "hand:<name>:round5", "hand:<name>:noisy", "llm:<tool>:<index>",
  "llm:<tool>:<index>:jitter", "riddler:r<N>",
  "riddler_adj:<version>:r<N>", "random_uniform", later "counter:r<N>"), b1..b10 (int, sum 100), weight (int ≥1, number
  of merged duplicates), parent (id or empty). Rows sorted by id; row order = matrix
  index order.
- cache/pool_meta.json: seed, generator settings, dedup threshold, pool_hash
  (sha256 of b1..b10 in row order).
- cache/matrix_<first 16 chars of pool_hash>.npz (compressed; ~20 MB and ~6 s to build
  for ~4,400 rows):
  - ids (str[N]), pool_hash (full), lut_hash
  - S200 (int32[N,N]): S200[i,j] = 200 × i's total score when i plays j
  - W2 (int8[N,N]): 2 = i wins, 1 = draw (S200[i,j] == S200[j,i]), 0 = i loses; W = W2 / 2
  - Invariants: W2 + W2.T == 2; diag(W2) == 1; diag(S200) == 0; W2 matches comparing
    S200[i,j] with S200[j,i]. Checked on every build and every cache reuse.
  - matrix.extend_matrices adds rows for new strategies (counterattack) without
    recomputing old pairs; callers must re-run check_invariants and
    check_against_scoring on the result.
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
- Run: `.venv/bin/python pool.py && .venv/bin/python matrix.py && .venv/bin/python rankings.py`
  (then robustness.py, counterattack.py as needed). Plain `python` works only with the
  venv activated.

## Working rules
- Run tests after every change.
- Ask before modifying scoring.py; all results depend on it.
- Set random seeds so runs are reproducible; record seeds in saved metadata.
- Save results to results/ with descriptive filenames.
- Only use numpy, pandas, matplotlib, scipy unless I approve others. pytest is approved
  as a dev-only dependency.
