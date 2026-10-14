from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_GAMMA = 0.99
SENSITIVITY_GAMMA = 0.95
FORMAL = ["C1", "C2", "C3", "C5", "C10", "C16", "C19", "C28"]
COMPARE = FORMAL[:-1]
REFERENCE = "C28"
N_BOOT = 20000
BOOT_SEED = 236

INPUT = ROOT / "outputs" / "results" / "retention" / "formal_retention_summary_tc.csv"
PRIMARY_STATS = ROOT / "outputs" / "results" / "retention" / "formal_retention_vs_c28_statistics.csv"
OUT = ROOT / "outputs" / "results" / "sensitivity" / "eval_gamma_095"
OUT.mkdir(parents=True, exist_ok=True)

print("Run role: evaluation-discount sensitivity at gamma=0.95")
print("Policies and evaluator rollouts are unchanged; only the 2-hour return aggregation is changed.")

df = pd.read_csv(INPUT)
source_sha256 = hashlib.sha256(INPUT.read_bytes()).hexdigest()
required = {
    "Condition", "Algorithm", "Seed", "Evaluator",
    "Mean_Discounted_Return", "Mean_Reward", "Horizon_Hours"
}
missing = required.difference(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")
if len(df) != 640:
    raise SystemExit(f"Expected 640 final rows, found {len(df)}")
if set(pd.to_numeric(df["Horizon_Hours"], errors="raise").astype(int).unique()) != {2}:
    raise SystemExit("Gamma sensitivity is defined only for the final 2-hour panel.")
if set(df["Condition"].astype(str)) != set(FORMAL):
    raise SystemExit("Unexpected formal conditions.")
if not (df.groupby(["Condition", "Algorithm", "Evaluator"])["Seed"].nunique() == 20).all():
    raise SystemExit("Expected 20 seeds in every formal cell.")
expected = {(c, a, e, seed) for c in FORMAL for a in ("IQL", "CQL")
            for e in ("ARX", "XGBoost") for seed in range(1, 21)}
keys = list(zip(df["Condition"].astype(str), df["Algorithm"].astype(str),
                df["Evaluator"].astype(str), df["Seed"].astype(int)))
if len(keys) != len(set(keys)) or set(keys) != expected:
    raise SystemExit("Unexpected, duplicated, or missing formal policy/evaluator keys.")
if not np.isfinite(df[["Mean_Discounted_Return", "Mean_Reward"]].to_numpy(dtype=float)).all():
    raise SystemExit("Nonfinite primary returns or rewards.")

# For two rewards r0,r1:
# Mean_Reward = (r0+r1)/2 and G_0.99 = r0 + 0.99*r1.
denom = 1.0 - PRIMARY_GAMMA
out = df.copy()
out["Mean_Reward_Step2_Recovered"] = (
    2.0 * out["Mean_Reward"] - out["Mean_Discounted_Return"]
) / denom
out["Mean_Reward_Step1_Recovered"] = (
    2.0 * out["Mean_Reward"] - out["Mean_Reward_Step2_Recovered"]
)
reconstructed = (
    out["Mean_Reward_Step1_Recovered"]
    + PRIMARY_GAMMA * out["Mean_Reward_Step2_Recovered"]
)
max_error = float(np.max(np.abs(reconstructed - out["Mean_Discounted_Return"])))
if max_error > 1e-10:
    raise SystemExit(f"Primary-return reconstruction failed: max abs error={max_error}")

out["Mean_Discounted_Return_Gamma095"] = (
    out["Mean_Reward_Step1_Recovered"]
    + SENSITIVITY_GAMMA * out["Mean_Reward_Step2_Recovered"]
)
out["Evaluation_Gamma_Primary"] = PRIMARY_GAMMA
out["Evaluation_Gamma_Sensitivity"] = SENSITIVITY_GAMMA
out["Policy_Training_Gamma"] = PRIMARY_GAMMA
out.to_csv(OUT / "gamma_095_policy_values.csv", index=False)

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted

rng = np.random.default_rng(BOOT_SEED)
rows = []
for algorithm in sorted(out["Algorithm"].unique()):
    for evaluator in sorted(out["Evaluator"].unique()):
        sub = out[(out["Algorithm"] == algorithm) & (out["Evaluator"] == evaluator)]
        ref = sub[sub["Condition"] == REFERENCE][
            ["Seed", "Mean_Discounted_Return_Gamma095"]
        ].rename(columns={"Mean_Discounted_Return_Gamma095": "Reference_Return"})
        family = []
        for condition in COMPARE:
            cur = sub[sub["Condition"] == condition][
                ["Seed", "Mean_Discounted_Return_Gamma095"]
            ].rename(columns={"Mean_Discounted_Return_Gamma095": "Condition_Return"})
            paired = cur.merge(ref, on="Seed", validate="one_to_one").sort_values("Seed")
            if len(paired) != 20:
                raise SystemExit(f"{algorithm}/{evaluator}/{condition}: expected 20 matched seeds.")
            diff = (paired["Condition_Return"] - paired["Reference_Return"]).to_numpy(float)
            idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
            lo, hi = np.quantile(diff[idx].mean(axis=1), [0.025, 0.975])
            sd = diff.std(ddof=1)
            w = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
            family.append({
                "Algorithm": algorithm,
                "Evaluator": evaluator,
                "Condition": condition,
                "Reference": REFERENCE,
                "N_Matched_Seeds": len(diff),
                "Mean_Difference_Ck_minus_C28": float(diff.mean()),
                "Median_Difference_Ck_minus_C28": float(np.median(diff)),
                "Bootstrap_95CI_Lower": float(lo),
                "Bootstrap_95CI_Upper": float(hi),
                "Wilcoxon_Statistic": float(w.statistic),
                "Wilcoxon_P_Raw": float(w.pvalue),
                "Cohen_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
                "Seeds_Ck_Greater_C28": int((diff > 0).sum()),
                "Seeds_Ck_Less_C28": int((diff < 0).sum()),
                "Seeds_Tied": int((diff == 0).sum()),
            })
        adjusted = holm_adjust([r["Wilcoxon_P_Raw"] for r in family])
        for row, p_adj in zip(family, adjusted):
            row["Holm_P_Adjusted"] = float(p_adj)
            row["Holm_Family"] = f"{algorithm} x {evaluator}: seven Ck-vs-C28 comparisons"
            rows.append(row)

stats095 = pd.DataFrame(rows)
stats095.to_csv(OUT / "gamma_095_retention_vs_c28_statistics.csv", index=False)

condition_summary = (
    out.groupby(["Algorithm", "Evaluator", "Condition"], as_index=False)
    .agg(
        Mean_Value=("Mean_Discounted_Return_Gamma095", "mean"),
        Median_Value=("Mean_Discounted_Return_Gamma095", "median"),
        SD_Value=("Mean_Discounted_Return_Gamma095", "std"),
    )
)
condition_summary.to_csv(OUT / "gamma_095_condition_summary.csv", index=False)

agreement_path = OUT / "gamma_095_direction_agreement.csv"
if PRIMARY_STATS.exists():
    primary_provenance_path = PRIMARY_STATS.with_suffix(".provenance.json")
    if not primary_provenance_path.exists():
        raise SystemExit("Missing primary statistics provenance; run script 05 first.")
    primary_provenance = json.loads(primary_provenance_path.read_text(encoding="utf-8"))
    if primary_provenance.get("source_sha256") != source_sha256:
        raise SystemExit("Primary statistics are stale; rerun script 05 before gamma sensitivity.")
    if primary_provenance.get("statistics_sha256") != hashlib.sha256(PRIMARY_STATS.read_bytes()).hexdigest():
        raise SystemExit("Primary statistics changed since provenance; rerun script 05.")
    primary = pd.read_csv(PRIMARY_STATS)
    key = ["Algorithm", "Evaluator", "Condition"]
    merged = primary[key + ["Mean_Difference_Ck_minus_C28"]].merge(
        stats095[key + ["Mean_Difference_Ck_minus_C28"]],
        on=key, validate="one_to_one", suffixes=("_Gamma099", "_Gamma095")
    )
    merged["Direction_Agreement"] = (
        np.sign(merged["Mean_Difference_Ck_minus_C28_Gamma099"])
        == np.sign(merged["Mean_Difference_Ck_minus_C28_Gamma095"])
    )
    merged.to_csv(agreement_path, index=False)
else:
    pd.DataFrame().to_csv(agreement_path, index=False)

protocol = {
    "analysis_role": "evaluation-discount sensitivity",
    "source_summary_sha256": source_sha256,
    "primary_policy_training_gamma": PRIMARY_GAMMA,
    "primary_evaluation_gamma": PRIMARY_GAMMA,
    "sensitivity_evaluation_gamma": SENSITIVITY_GAMMA,
    "policy_retraining": False,
    "evaluator_refitting": False,
    "rollout_recomputation": False,
    "horizon_hours": 2,
    "comparison_design": "each earlier retention condition Ck versus full retention C28",
    "bootstrap_repetitions": N_BOOT,
    "bootstrap_unit": "matched policy-training seed",
    "max_abs_primary_reconstruction_error": max_error,
    "interpretation_guardrail": (
        "This changes evaluation discounting only; it does not retrain IQL/CQL at gamma=0.95."
    ),
}
(OUT / "gamma_095_protocol.json").write_text(
    json.dumps(protocol, indent=2), encoding="utf-8"
)

print(f"Primary reconstruction max abs error: {max_error:.3e}")
print(stats095[
    ["Algorithm", "Evaluator", "Condition", "Mean_Difference_Ck_minus_C28",
     "Bootstrap_95CI_Lower", "Bootstrap_95CI_Upper", "Holm_P_Adjusted"]
].to_string(index=False))
print("\nEVALUATION-DISCOUNT SENSITIVITY COMPLETED")
