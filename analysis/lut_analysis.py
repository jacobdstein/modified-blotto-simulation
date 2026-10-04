"""Structural analysis of the 1024-entry score lookup table.

Questions it answers:
1. How do scores grow with the number of battlefields won?
2. Which won-sets guarantee a match win no matter which of the remaining
   battlefields the opponent wins? Which are minimal (no smaller subset guarantees it)?
3. When does winning one *more* battlefield lower your score? (Rule 2 can switch off.)
4. Given a reference field, roughly how many troops does each won-set cost?

Run from the project root:
    .venv/bin/python analysis/lut_analysis.py [reference_field.csv]

The reference field defaults to data/llm_strategies.csv. Outputs go to results/.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import matrix` works when run as a script

import numpy as np
import pandas as pd

from matrix import ALL_WON_MASK, NUM_MASKS, build_lut200, won_from_mask
from scoring import BONUS_BATTLEFIELD, NUM_BATTLEFIELDS

RESULTS = ROOT / "results"
DEFAULT_FIELD = ROOT / "data" / "llm_strategies.csv"
PRICE_PERCENTILE = 75


def format_set(battlefields) -> str:
    """Set notation: '{3 5 6 7}', and '{}' for the empty set, so a set is never written
    as a blank cell (which a CSV can't tell apart from missing data)."""
    return "{" + " ".join(map(str, sorted(battlefields))) + "}"


def parse_set(text: str) -> list[int]:
    """Inverse of format_set."""
    return [int(b) for b in text.strip("{}").split()]


def battlefield_count(mask: int) -> int:
    """Number of battlefields won in a mask."""
    return bin(mask).count("1")


def best_subset_score(lut: np.ndarray) -> np.ndarray:
    """best[m] = highest table score over all subsets of mask m (including m itself).

    Built one battlefield at a time: a subset of m either drops battlefield b or not.
    """
    best = lut.copy()
    for b in range(NUM_BATTLEFIELDS):
        bit = 1 << b
        for mask in range(NUM_MASKS):
            if mask & bit:
                best[mask] = max(best[mask], best[mask ^ bit])
    return best


def best_opponent_score(lut: np.ndarray) -> np.ndarray:
    """opp[m] = the most an opponent can score if I win exactly m.

    The opponent can only win battlefields I didn't win, so they win some subset of
    the complement. Score is not monotone (Rule 2), so the full complement is not
    necessarily their best case.
    """
    best = best_subset_score(lut)
    return np.array([best[ALL_WON_MASK ^ mask] for mask in range(NUM_MASKS)], dtype=np.int64)


def won_sets_table(lut: np.ndarray) -> pd.DataFrame:
    """One row per won-set with its score, bonuses, and guaranteed-win status.
    Note: the empty set has rule2 = True (0 is a multiple of 5); harmless, 1.4 × 0 = 0."""
    opponent = best_opponent_score(lut)
    guaranteed = lut > opponent
    rows = []
    for mask in range(NUM_MASKS):
        won = sorted(won_from_mask(mask))
        rows.append(
            {
                "mask": mask,
                "battlefields": format_set(won),
                "count": len(won),
                "base": sum(won),
                "score": lut[mask] / 200,
                "rule2": sum(won) % 5 == 0,
                "rule3": BONUS_BATTLEFIELD in won,
                "best_opponent_score": opponent[mask] / 200,
                "guaranteed_win": bool(guaranteed[mask]),
                "minimal_guaranteed": bool(guaranteed[mask])
                and not any(guaranteed[mask ^ (1 << b)] for b in range(NUM_BATTLEFIELDS) if mask >> b & 1),
            }
        )
    return pd.DataFrame(rows)


def by_count_table(sets: pd.DataFrame) -> pd.DataFrame:
    """Score range for each number of battlefields won, with the best set."""
    rows = []
    for count, group in sets.groupby("count"):
        best = group.loc[group["score"].idxmax()]
        rows.append(
            {
                "count": count,
                "num_sets": len(group),
                "min_score": group["score"].min(),
                "median_score": group["score"].median(),
                "max_score": group["score"].max(),
                "best_set": best["battlefields"],
                "share_guaranteed_win": group["guaranteed_win"].mean(),
            }
        )
    return pd.DataFrame(rows)


def harmful_additions_table(lut: np.ndarray) -> pd.DataFrame:
    """For each battlefield: how often winning it in addition lowers your score.
    worst_case_set is genuinely missing (blank in the CSV, NaN when read back) when
    adding that battlefield never lowers the score; worst_drop is then 0.0."""
    rows = []
    for b in range(NUM_BATTLEFIELDS):
        bit = 1 << b
        without = [mask for mask in range(NUM_MASKS) if not mask & bit]
        drops = [(int(lut[mask | bit] - lut[mask]), mask) for mask in without]
        harmful = [(d, mask) for d, mask in drops if d < 0]
        worst_drop, worst_mask = min(drops)
        rows.append(
            {
                "battlefield_added": b + 1,
                "share_of_sets_where_it_lowers_score": len(harmful) / len(without),
                "worst_drop": -worst_drop / 200 if worst_drop < 0 else 0.0,
                "worst_case_set": format_set(won_from_mask(worst_mask)) if worst_drop < 0 else None,
            }
        )
    return pd.DataFrame(rows)


def load_field(path: Path) -> np.ndarray:
    """Allocations (rows of b1..b10) from a CSV with columns b1..b10."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return np.array([[int(row[f"b{i}"]) for i in range(1, NUM_BATTLEFIELDS + 1)] for row in reader])


def battlefield_prices(field: np.ndarray, percentile: int = PRICE_PERCENTILE) -> np.ndarray:
    """Troops needed to beat `percentile`% of the field on each battlefield (one more
    than the field's percentile value). A rough per-battlefield price, not a guarantee
    of beating that share of players overall."""
    return (np.percentile(field, percentile, axis=0, method="higher") + 1).astype(int)


def prices_table(sets: pd.DataFrame, prices: np.ndarray) -> pd.DataFrame:
    """Each won-set's price at the given per-battlefield prices, if affordable (≤ 100)."""
    priced = sets.copy()
    priced["price"] = [
        int(sum(prices[b - 1] for b in parse_set(s))) for s in priced["battlefields"]
    ]
    affordable = priced[priced["price"] <= 100].sort_values(["score", "price"], ascending=[False, True])
    return affordable[["battlefields", "count", "base", "score", "price", "rule2", "rule3", "guaranteed_win"]]


def save_results(tables: dict[str, pd.DataFrame], results_dir: Path = RESULTS) -> None:
    """Write each table to results_dir/<name>.csv."""
    results_dir.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)


def main() -> None:
    field_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIELD
    lut = build_lut200()
    sets = won_sets_table(lut)
    by_count = by_count_table(sets)
    harmful = harmful_additions_table(lut)
    prices = battlefield_prices(load_field(field_path))
    priced = prices_table(sets, prices)

    save_results(
        {
            "lut_won_sets": sets,
            "lut_by_battlefield_count": by_count,
            "lut_harmful_additions": harmful,
            f"lut_affordable_sets_p{PRICE_PERCENTILE}_{field_path.stem}": priced,
        }
    )

    pd.set_option("display.width", 120)
    print("Scores by number of battlefields won:\n", by_count.to_string(index=False))
    print(f"\nGuaranteed-win sets: {sets['guaranteed_win'].sum()} of {NUM_MASKS}; "
          f"minimal: {sets['minimal_guaranteed'].sum()}; smallest has "
          f"{sets.loc[sets['minimal_guaranteed'], 'count'].min()} battlefields")
    print("\nWinning one more battlefield can lower your score:\n", harmful.to_string(index=False))
    print(f"\nPer-battlefield price ({PRICE_PERCENTILE}th percentile of {field_path.name} + 1): {prices.tolist()}")
    print("Top affordable sets:\n", priced.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
