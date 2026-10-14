from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs" / "results" / "retention" / "formal_retention_summary_tc.csv"
OUTFILE = ROOT / "outputs" / "results" / "retention" / "formal_retention_vs_c28_statistics.csv"

CONDITIONS = ["C1", "C2", "C3", "C5", "C10", "C16", "C19"]
REFERENCE = "C28"
N_BOOT = 20000
BOOT_SEED = 236

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (m - np.arange(m)) * p[order])
    )
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

def paired_bootstrap_ci(diff, rng):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = diff[idx].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])

def main():
    df = pd.read_csv(INFILE)
    required = {"Condition", "Algorithm", "Seed", "Evaluator", "Mean_Discounted_Return"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    if len(df) != 640:
        raise SystemExit(f"Expected 640 summary rows, found {len(df)}")

    expected = {(c, a, e, seed) for c in CONDITIONS + [REFERENCE]
                for a in ("IQL", "CQL") for e in ("ARX", "XGBoost")
                for seed in range(1, 21)}
    keys = list(zip(df["Condition"].astype(str), df["Algorithm"].astype(str),
                    df["Evaluator"].astype(str), df["Seed"].astype(int)))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise SystemExit("Formal summary has missing, duplicated, or unexpected policy/evaluator keys.")
    if not np.isfinite(pd.to_numeric(df["Mean_Discounted_Return"], errors="coerce")).all():
        raise SystemExit("Nonfinite formal returns.")
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for algorithm in sorted(df["Algorithm"].unique()):
        for evaluator in sorted(df["Evaluator"].unique()):
            sub = df[(df["Algorithm"] == algorithm) & (df["Evaluator"] == evaluator)]
            ref = (
                sub[sub["Condition"] == REFERENCE][["Seed", "Mean_Discounted_Return"]]
                .rename(columns={"Mean_Discounted_Return": "Reference_Return"})
            )
            family_rows = []

            for condition in CONDITIONS:
                cur = (
                    sub[sub["Condition"] == condition][["Seed", "Mean_Discounted_Return"]]
                    .rename(columns={"Mean_Discounted_Return": "Condition_Return"})
                )
                paired = cur.merge(ref, on="Seed", how="inner", validate="one_to_one").sort_values("Seed")
                if len(paired) != 20:
                    raise SystemExit(
                        f"{algorithm}/{evaluator}/{condition}: expected 20 matched seeds, found {len(paired)}"
                    )

                diff = (paired["Condition_Return"] - paired["Reference_Return"]).to_numpy(float)
                lo, hi = paired_bootstrap_ci(diff, rng)
                sd = diff.std(ddof=1)
                dz = diff.mean() / sd if sd > 0 else np.nan

                # Compute the paired differences explicitly before Wilcoxon.
                # SciPy recommends this form to avoid subtraction/ranking roundoff issues.
                w = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")

                family_rows.append({
                    "Algorithm": algorithm,
                    "Evaluator": evaluator,
                    "Condition": condition,
                    "Reference": REFERENCE,
                    "N_Matched_Seeds": len(diff),
                    "Mean_Difference_Ck_minus_C28": diff.mean(),
                    "Median_Difference_Ck_minus_C28": np.median(diff),
                    "Bootstrap_95CI_Lower": lo,
                    "Bootstrap_95CI_Upper": hi,
                    "Wilcoxon_Statistic": float(w.statistic),
                    "Wilcoxon_P_Raw": float(w.pvalue),
                    "Cohen_dz": dz,
                    "Seeds_Ck_Greater_C28": int((diff > 0).sum()),
                    "Seeds_Ck_Less_C28": int((diff < 0).sum()),
                    "Seeds_Tied": int((diff == 0).sum()),
                })

            adjusted = holm_adjust([r["Wilcoxon_P_Raw"] for r in family_rows])
            for r, p_adj in zip(family_rows, adjusted):
                r["Holm_P_Adjusted"] = p_adj
                r["Holm_Family"] = f"{algorithm} x {evaluator}: seven Ck-vs-C28 comparisons"
                rows.append(r)

    out = pd.DataFrame(rows)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTFILE, index=False)
    provenance = {
        "source": str(INFILE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(INFILE.read_bytes()).hexdigest(),
        "statistics_sha256": hashlib.sha256(OUTFILE.read_bytes()).hexdigest(),
        "comparison": "Ck minus C28, matched seeds",
        "bootstrap_repetitions": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "holm_family": "seven comparisons within each algorithm/evaluator",
    }
    OUTFILE.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote: {OUTFILE}")
    print(f"Rows: {len(out)}")
    print("Design: matched seeds; Ck-C28; 20,000 paired bootstrap; two-sided Wilcoxon; Cohen dz")
    print("Holm: seven Ck-vs-C28 tests separately within each Algorithm x Evaluator family")

if __name__ == "__main__":
    main()
