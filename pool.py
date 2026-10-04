"""Pipeline step 1: build the strategy pool.

Sources, in the order rows are created (which fixes the IDs; new sources are always
added at the end so existing IDs never shift):
1. hand:<name>          naive, human-style allocations (level 0), built by rules
2. hand:<name>:round5   the same, rounded to multiples of 5 (people like round numbers)
3. hand:<name>:noisy    seeded random perturbations of each base rule (1); parent = the base row
4. llm:<tool>:<index>   strategies collected from chatbots (data/llm_strategies.csv)
   llm:<tool>:<index>:jitter
                        small seeded variants of each (people asking the same chatbot
                        get varied answers); parent = the original
5. riddler:r<N>         a seeded sample of FiveThirtyEight Riddler submissions
                        (data/riddler/), raw; Riddler had no bonus rules
6. riddler_adj:<version>:r<N>
                        each sampled submission after one bonus-aware swap, for four
                        kinds of player (see adjust_riddler); parent = the raw row
7. random_uniform       uniformly random allocations, kept a small minority
8. llm:<tool>:<index>:jitter   (see 4; created here so earlier IDs don't shift)
9. counter:r<N>         strategies built by counterattack.py to beat the leaders
                        (data/counter/round_<N>.csv); parent = the pool row the search
                        started from, if any

Exact duplicates are merged only within the same origin (weights add up), so per-source
counts survive and the held-out Riddler round never merges into other rows.

Run from the project root:
    .venv/bin/python pool.py
Writes cache/pool.csv and cache/pool_meta.json.
"""

import csv
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import BONUS_BATTLEFIELD, NUM_BATTLEFIELDS, TOTAL_TROOPS, score200, validate_allocation

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"

SEED = 20261004
NOISY_COPIES = 5  # per hand-built base allocation
NOISE_TROOPS = (2, 8)  # each noisy copy moves between 2 and 8 troops
NUM_RANDOM = 200
LLM_JITTER_COPIES = 10  # variants per AI strategy
LLM_JITTER_TROOPS = (1, 4)  # each variant moves between 1 and 4 troops
L1_THRESHOLD = 0  # near-duplicate merging; 0 = exact duplicates only
PROTECTED_PREFIXES = ("hand:", "llm:", "counter:")  # never near-duplicate merged

RIDDLER_FILES = {
    1: "castle-solutions.csv",
    2: "castle-solutions-2.csv",
    3: "castle-solutions-3.csv",
    4: "castle-solutions-4.csv",
}
RIDDLER_SAMPLE = 300  # submissions sampled per round (with their repeats), seeded
HOLDOUT_ROUND = 4  # Riddler round never used for building or selecting
OPPONENT_ONLY_COUNTER_ROUNDS = (3,)  # final counterattack round: opponents, never candidates

CONTESTED_MIN = 5  # a battlefield with at least this many troops counts as contested
ADJUST_VERSIONS = ("all", "rule1", "rule2", "rule3")  # which bonus(es) the player noticed

BATTLEFIELD_COLUMNS = [f"b{i}" for i in range(1, NUM_BATTLEFIELDS + 1)]
POOL_COLUMNS = ["id", "origin", *BATTLEFIELD_COLUMNS, "weight", "parent"]

Allocation = tuple[int, ...]


@dataclass
class Entry:
    """A pool row before IDs are assigned. `parent` is an index into the entry list;
    `parent_id` is used instead when the parent is already known by its pool id
    (counter strategies, whose parents come from an earlier build of the pool)."""

    origin: str
    allocation: Allocation
    weight: int = 1
    parent: int | None = None
    parent_id: str = ""
    merged_into: int | None = field(default=None, repr=False)


# --- Allocation helpers -----------------------------------------------------


def apportion(shares: list[float], total: int = TOTAL_TROOPS) -> Allocation:
    """Split `total` troops in proportion to `shares` (largest-remainder rounding).
    Ties in the remainder go to the higher battlefield."""
    shares = np.asarray(shares, dtype=float)
    exact = shares / shares.sum() * total
    troops = np.floor(exact).astype(int)
    leftover = total - troops.sum()
    order = sorted(range(len(shares)), key=lambda i: (-(exact[i] - troops[i]), -i))
    for i in order[:leftover]:
        troops[i] += 1
    return tuple(int(t) for t in troops)


def on_battlefields(battlefields: list[int], weights: list[float] | None = None) -> Allocation:
    """All 100 troops on the given battlefields (numbers 1..10), split by `weights`
    (equal if None); zero elsewhere."""
    weights = weights or [1.0] * len(battlefields)
    shares = [0.0] * NUM_BATTLEFIELDS
    for battlefield, weight in zip(battlefields, weights):
        shares[battlefield - 1] = weight
    return apportion(shares)


def round_to_5(allocation: Allocation) -> Allocation:
    """The allocation rounded to multiples of 5, still summing to 100."""
    fives = apportion(list(allocation), total=TOTAL_TROOPS // 5)
    return tuple(5 * f for f in fives)


def perturb(allocation: Allocation, rng: np.random.Generator, troops: int) -> Allocation:
    """Move `troops` single troops, each from a random non-empty battlefield to a random
    different battlefield."""
    result = list(allocation)
    for _ in range(troops):
        source = rng.choice([i for i in range(NUM_BATTLEFIELDS) if result[i] > 0])
        target = rng.choice([i for i in range(NUM_BATTLEFIELDS) if i != source])
        result[source] -= 1
        result[target] += 1
    return tuple(result)


def random_uniform(rng: np.random.Generator) -> Allocation:
    """Uniformly random allocation (every allocation equally likely): choose 9 cut
    points among 109 positions ("stars and bars")."""
    cuts = np.sort(rng.choice(TOTAL_TROOPS + NUM_BATTLEFIELDS - 1, NUM_BATTLEFIELDS - 1, replace=False))
    edges = np.concatenate(([-1], cuts, [TOTAL_TROOPS + NUM_BATTLEFIELDS - 1]))
    return tuple(int(x) for x in np.diff(edges) - 1)


# --- Sources ----------------------------------------------------------------


def battlefield6_heavy(troops_on_6: int = 35) -> Allocation:
    """`troops_on_6` on battlefield 6; the rest split in proportion to value."""
    others = [v if v != 6 else 0 for v in range(1, NUM_BATTLEFIELDS + 1)]
    rest = apportion(others, total=TOTAL_TROOPS - troops_on_6)
    return tuple(troops_on_6 if b == 6 else t for b, t in zip(range(1, NUM_BATTLEFIELDS + 1), rest))


def hand_strategies() -> list[tuple[str, Allocation]]:
    """Naive, rule-of-thumb allocations a person might write down (level 0)."""
    values = list(range(1, NUM_BATTLEFIELDS + 1))
    strategies = [
        ("even", on_battlefields(values)),
        ("proportional", on_battlefields(values, values)),
        ("battlefield6_heavy", battlefield6_heavy()),
    ]
    for k in range(3, 7):
        top = values[-k:]
        strategies.append((f"top{k}_equal", on_battlefields(top)))
        strategies.append((f"top{k}_proportional", on_battlefields(top, top)))
    for skip in range(2, 6):
        rest = values[skip:]
        strategies.append((f"skip_low{skip}", on_battlefields(rest, rest)))
    for length in range(3, 7):
        for start in range(1, NUM_BATTLEFIELDS - length + 2):
            block = list(range(start, start + length))
            strategies.append((f"block{start}-{start + length - 1}", on_battlefields(block, block)))
    # Bonus-aware: four battlefields, every one with a won neighbor (Rule 1), base a
    # multiple of 5 (Rule 2), split in proportion to value.
    for chosen in combinations(values, 4):
        all_adjacent = all((b - 1) in chosen or (b + 1) in chosen for b in chosen)
        if all_adjacent and sum(chosen) % 5 == 0:
            name = "bonus4_" + "-".join(map(str, chosen))
            strategies.append((name, on_battlefields(list(chosen), list(chosen))))
    return strategies


def load_llm_strategies(path: Path) -> list[tuple[str, Allocation]]:
    """(origin, allocation) for each row of data/llm_strategies.csv."""
    with open(path, newline="") as f:
        return [
            (f"{row['source']}:{row['index']}", validate_allocation([int(row[c]) for c in BATTLEFIELD_COLUMNS]))
            for row in csv.DictReader(f)
        ]


def load_riddler_round(path: Path) -> tuple[list[Allocation], int]:
    """Valid allocations from one Riddler CSV, and how many rows were dropped."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = [f"Castle {i}" for i in range(1, NUM_BATTLEFIELDS + 1)]
    allocations, dropped = [], 0
    for values in df[columns].itertuples(index=False):
        try:
            allocations.append(validate_allocation([int(v) for v in values]))
        except (TypeError, ValueError):
            dropped += 1
    return allocations, dropped


def sample_submissions(allocations: list[Allocation], size: int, seed: int, round_number: int) -> list[Allocation]:
    """A seeded random sample of `size` submissions, without replacement, keeping file
    order. Popular allocations appear in proportion to how often they were submitted.
    Each round has its own random stream, so changing other settings never changes it."""
    if len(allocations) <= size:
        return list(allocations)
    rng = np.random.default_rng([seed, round_number])
    chosen = np.sort(rng.choice(len(allocations), size, replace=False))
    return [allocations[i] for i in chosen]


# --- Bonus-aware Riddler adjustment -----------------------------------------
#
# A Riddler player never saw the bonus rules. Each adjusted version imagines that player
# noticing some of them and making ONE swap: stop contesting one battlefield and move its
# troops, unchanged, to a battlefield they weren't contesting. Every swap must gain a
# bonus the original plan lacked.
#   all    noticed every bonus: among swaps that gain any bonus and raise the score of
#          winning the contested set, the highest-scoring one
#   rule1  noticed only adjacency: the smallest swap that adds adjacent battlefields
#   rule2  noticed only "base is a multiple of 5": the smallest swap that achieves it
#   rule3  noticed only battlefield 6: the smallest swap that adds battlefield 6
# "Smallest" = the two battlefields' values are closest; ties go to the higher score.


def contested(allocation: Allocation) -> frozenset[int]:
    """Battlefield numbers with at least CONTESTED_MIN troops."""
    return frozenset(b for b, troops in enumerate(allocation, start=1) if troops >= CONTESTED_MIN)


def adjacent_count(battlefields: frozenset[int]) -> int:
    """How many battlefields in the set have a neighbor in the set (Rule 1 applies)."""
    return sum(1 for b in battlefields if (b - 1) in battlefields or (b + 1) in battlefields)


def bonuses_gained(before: frozenset[int], after: frozenset[int]) -> dict[str, bool]:
    """Which bonuses the target set `after` gains over `before`."""
    return {
        "rule1": adjacent_count(after) > adjacent_count(before),
        "rule2": sum(after) % 5 == 0 and sum(before) % 5 != 0,
        "rule3": BONUS_BATTLEFIELD in after and BONUS_BATTLEFIELD not in before,
    }


@lru_cache(maxsize=None)
def set_score200(battlefields: frozenset[int]) -> int:
    """200 × score for winning exactly these battlefields (cached scoring.score200)."""
    return score200(battlefields)


def adjust_riddler(allocation: Allocation, version: str) -> Allocation | None:
    """The allocation after the version's one bonus-aware swap, or None if that player
    has nothing to fix (or contests no battlefields, or all ten)."""
    if version not in ADJUST_VERSIONS:
        raise ValueError(f"unknown version {version!r}; expected one of {ADJUST_VERSIONS}")
    before = contested(allocation)
    if not before or len(before) == NUM_BATTLEFIELDS:
        return None
    uncontested = [b for b in range(1, NUM_BATTLEFIELDS + 1) if b not in before]
    options = []
    for drop, add in product(sorted(before), uncontested):
        after = (before - {drop}) | {add}
        gained = bonuses_gained(before, after)
        score = set_score200(after)
        if version == "all":
            if any(gained.values()) and score > set_score200(before):
                options.append(((-score, abs(add - drop)), drop, add))
        elif gained[version]:
            options.append(((abs(add - drop), -score), drop, add))
    if not options:
        return None
    _, drop, add = min(options)
    result = list(allocation)
    result[add - 1] += result[drop - 1]
    result[drop - 1] = 0
    return tuple(result)


def is_opponent_only(origin: str) -> bool:
    """True for counter strategies from the final test round: they appear in fields as
    opponents but are never ranked, shortlisted, or picked."""
    return origin in {f"counter:r{n}" for n in OPPONENT_ONLY_COUNTER_ROUNDS}


def is_holdout(origin: str) -> bool:
    """True for every row derived from the held-out Riddler round."""
    return origin.startswith(("riddler:", "riddler_adj:")) and origin.endswith(f":r{HOLDOUT_ROUND}")


# --- Merging ----------------------------------------------------------------


def merge_exact_duplicates(entries: list[Entry]) -> None:
    """Within each origin, fold repeated allocations into the first one (adding weight)."""
    first_seen: dict[tuple[str, Allocation], int] = {}
    for index, entry in enumerate(entries):
        key = (entry.origin, entry.allocation)
        if key in first_seen:
            keeper = entries[first_seen[key]]
            keeper.weight += entry.weight
            entry.merged_into = first_seen[key]
        else:
            first_seen[key] = index


def merge_near_duplicates(entries: list[Entry], threshold: int) -> None:
    """Within each unprotected origin, fold an allocation into an earlier kept one if
    their L1 distance is <= threshold. No-op when threshold is 0."""
    if threshold <= 0:
        return
    kept_by_origin: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        if entry.merged_into is not None or entry.origin.startswith(PROTECTED_PREFIXES):
            continue
        kept = kept_by_origin.setdefault(entry.origin, [])
        for keeper_index in kept:
            distance = sum(abs(a - b) for a, b in zip(entry.allocation, entries[keeper_index].allocation))
            if distance <= threshold:
                entries[keeper_index].weight += entry.weight
                entry.merged_into = keeper_index
                break
        else:
            kept.append(index)


# --- Building and saving ----------------------------------------------------


def collect_entries(seed: int = SEED, data_dir: Path = DATA) -> tuple[list[Entry], dict]:
    """Every source in a fixed order, before merging. Also returns per-source notes."""
    rng = np.random.default_rng(seed)
    entries: list[Entry] = []
    notes: dict = {"riddler_dropped_rows": {}}

    bases = hand_strategies()
    base_index = {}
    for name, allocation in bases:
        base_index[name] = len(entries)
        entries.append(Entry(f"hand:{name}", allocation))
    for name, allocation in bases:
        rounded = round_to_5(allocation)
        if rounded != allocation:  # already in multiples of 5 → nothing new to add
            entries.append(Entry(f"hand:{name}:round5", rounded, parent=base_index[name]))
    for name, allocation in bases:
        for _ in range(NOISY_COPIES):
            troops = int(rng.integers(NOISE_TROOPS[0], NOISE_TROOPS[1] + 1))
            entries.append(Entry(f"hand:{name}:noisy", perturb(allocation, rng, troops), parent=base_index[name]))

    llm = load_llm_strategies(data_dir / "llm_strategies.csv")
    llm_index = {}
    for origin, allocation in llm:
        llm_index[origin] = len(entries)
        entries.append(Entry(origin, allocation))

    notes["riddler_submissions"] = {}
    for round_number, filename in RIDDLER_FILES.items():
        allocations, dropped = load_riddler_round(data_dir / "riddler" / filename)
        notes["riddler_dropped_rows"][f"r{round_number}"] = dropped
        notes["riddler_submissions"][f"r{round_number}"] = len(allocations)
        sampled = sample_submissions(allocations, RIDDLER_SAMPLE, seed, round_number)
        raw_index = []
        for allocation in sampled:
            raw_index.append(len(entries))
            entries.append(Entry(f"riddler:r{round_number}", allocation))
        for version in ADJUST_VERSIONS:
            for allocation, parent in zip(sampled, raw_index):
                adjusted = adjust_riddler(allocation, version)
                if adjusted is not None:
                    entries.append(Entry(f"riddler_adj:{version}:r{round_number}", adjusted, parent=parent))

    entries.extend(Entry("random_uniform", random_uniform(rng)) for _ in range(NUM_RANDOM))

    # Added last so that adding a source never shifts the IDs of existing rows.
    jitter_rng = np.random.default_rng([seed, 1])  # own stream: doesn't change other sources
    for origin, allocation in llm:
        for _ in range(LLM_JITTER_COPIES):
            variant = allocation
            while variant == allocation:  # moves can cancel out; redraw until it differs
                troops = int(jitter_rng.integers(LLM_JITTER_TROOPS[0], LLM_JITTER_TROOPS[1] + 1))
                variant = perturb(allocation, jitter_rng, troops)
            entries.append(Entry(f"{origin}:jitter", variant, parent=llm_index[origin]))

    notes["counter_rounds"] = {}
    for round_number, path in counter_files(data_dir):
        allocations = load_counter_round(path)
        notes["counter_rounds"][f"r{round_number}"] = len(allocations)
        entries.extend(Entry(f"counter:r{round_number}", a, parent_id=parent) for a, parent in allocations)
    return entries, notes


def counter_files(data_dir: Path = DATA) -> list[tuple[int, Path]]:
    """(round number, path) for every data/counter/round_<N>.csv, in round order. Rounds
    must be numbered 1, 2, ... with no gaps."""
    paths = sorted((int(p.stem.split("_")[1]), p) for p in (data_dir / "counter").glob("round_*.csv"))
    if [n for n, _ in paths] != list(range(1, len(paths) + 1)):
        raise ValueError(f"counter rounds must be 1..N with no gaps, found {[n for n, _ in paths]}")
    return paths


def load_counter_round(path: Path) -> list[tuple[Allocation, str]]:
    """(allocation, parent pool id or '') for each row of a counter round file."""
    with open(path, newline="") as f:
        return [
            (validate_allocation([int(row[c]) for c in BATTLEFIELD_COLUMNS]), row["parent"])
            for row in csv.DictReader(f)
        ]


def build_pool(seed: int = SEED, l1_threshold: int = L1_THRESHOLD, data_dir: Path = DATA) -> tuple[pd.DataFrame, dict]:
    """The pool as a DataFrame in POOL_COLUMNS order, plus metadata."""
    entries, notes = collect_entries(seed, data_dir)
    merge_exact_duplicates(entries)
    merge_near_duplicates(entries, l1_threshold)

    survivors = [i for i, e in enumerate(entries) if e.merged_into is None]
    ids = {index: f"P{n:05d}" for n, index in enumerate(survivors, start=1)}
    rows = []
    for index in survivors:
        entry = entries[index]
        validate_allocation(list(entry.allocation))
        rows.append(
            {
                "id": ids[index],
                "origin": entry.origin,
                **dict(zip(BATTLEFIELD_COLUMNS, entry.allocation)),
                "weight": entry.weight,
                "parent": entry.parent_id or (ids[resolve(entries, entry.parent)] if entry.parent is not None else ""),
            }
        )
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS)
    meta = {
        "seed": seed,
        "settings": {
            "noisy_copies": NOISY_COPIES,
            "noise_troops": list(NOISE_TROOPS),
            "num_random": NUM_RANDOM,
            "llm_jitter_copies": LLM_JITTER_COPIES,
            "llm_jitter_troops": list(LLM_JITTER_TROOPS),
            "riddler_sample_per_round": RIDDLER_SAMPLE,
            "contested_min": CONTESTED_MIN,
            "adjust_versions": list(ADJUST_VERSIONS),
            "protected_prefixes": list(PROTECTED_PREFIXES),
        },
        "dedup_l1_threshold": l1_threshold,
        "holdout_origins": sorted(o for o in pool["origin"].unique() if is_holdout(o)),
        "entries_before_merging": len(entries),
        "rows": len(pool),
        "total_weight": int(pool["weight"].sum()),
        **notes,
        "pool_hash": pool_hash(pool),
    }
    return pool, meta


def resolve(entries: list[Entry], index: int) -> int:
    """Follow merges to the surviving entry."""
    while entries[index].merged_into is not None:
        index = entries[index].merged_into
    return index


def pool_hash(pool: pd.DataFrame) -> str:
    """SHA-256 of b1..b10 in row order (what the matrices depend on)."""
    return hashlib.sha256(pool[BATTLEFIELD_COLUMNS].to_numpy(dtype="<i8").tobytes()).hexdigest()


def origin_family(origin: str) -> str:
    """The source family used for equal-weight rankings:
    hand, llm:<tool>, riddler, riddler_adj, random_uniform, counter."""
    parts = origin.split(":")
    if parts[0] == "llm":
        return ":".join(parts[:2])
    return parts[0]


def origin_subfamily(origin: str) -> str:
    """The group that shares a family's weight equally:
    'hand:even:noisy' → 'hand:noisy', 'hand:even' → 'hand:base',
    'riddler:r2' → 'riddler:r2', 'riddler_adj:rule2:r1' → 'riddler_adj:rule2',
    'llm:claude:3' and 'llm:claude:3:jitter' → 'llm:claude:3' (one AI answer plus its
    variants)."""
    parts = origin.split(":")
    if parts[0] == "hand":
        return "hand:" + (parts[2] if len(parts) > 2 else "base")
    if parts[0] == "llm":
        return ":".join(parts[:3])
    return ":".join(parts[:2])


def save_pool(pool: pd.DataFrame, meta: dict, cache_dir: Path = CACHE) -> None:
    cache_dir.mkdir(exist_ok=True)
    pool.to_csv(cache_dir / "pool.csv", index=False)
    (cache_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_pool(cache_dir: Path = CACHE) -> tuple[pd.DataFrame, dict]:
    """Read the saved pool and check it against its metadata hash."""
    pool = pd.read_csv(cache_dir / "pool.csv", dtype={"parent": str}, keep_default_na=False)
    meta = json.loads((cache_dir / "pool_meta.json").read_text())
    if pool_hash(pool) != meta["pool_hash"]:
        raise ValueError("cache/pool.csv does not match the hash in pool_meta.json; rebuild the pool")
    return pool, meta


def main() -> None:
    pool, meta = build_pool()
    save_pool(pool, meta)
    groups = pool["origin"].map(lambda o: origin_family(o) if o.startswith("llm:") else origin_subfamily(o))
    summary = pool.assign(group=groups).groupby("group").agg(
        rows=("id", "size"), weight=("weight", "sum")
    )
    print(summary.to_string())
    print(f"\n{meta['rows']} rows (from {meta['entries_before_merging']} entries), "
          f"total weight {meta['total_weight']}, pool_hash {meta['pool_hash'][:12]}…")
    print(f"Riddler rows dropped as invalid: {meta['riddler_dropped_rows']}")


if __name__ == "__main__":
    main()
