from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results" / "retention"
FIGURES = ROOT / "outputs" / "results" / "retention"
CONDITIONS_PATH = ROOT / "outputs" / "fragments" / "formal_cumulative_conditions.csv"
SUMMARY_PATH = RESULTS / "formal_retention_summary_tc.csv"
STATS_PATH = RESULTS / "formal_retention_vs_c28_statistics.csv"

FORMAL = ["C1", "C2", "C3", "C5", "C10", "C16", "C19", "C28"]
ORDER = {condition: i for i, condition in enumerate(FORMAL)}


def main():
    provenance_path = STATS_PATH.with_suffix(".provenance.json")
    if not provenance_path.exists():
        raise SystemExit("Missing statistics provenance; run script 05 first.")

    summary = pd.read_csv(SUMMARY_PATH)
    stats = pd.read_csv(STATS_PATH)
    conditions = pd.read_csv(CONDITIONS_PATH)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    if provenance.get("source_sha256") != hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest():
        raise SystemExit("Statistics are stale relative to the formal OPE summary.")

    if provenance.get("statistics_sha256") != hashlib.sha256(STATS_PATH.read_bytes()).hexdigest():
        raise SystemExit("Statistics CSV differs from its provenance.")

    required_summary = {
        "Condition", "Algorithm", "Seed", "Evaluator",
        "Mean_Discounted_Return",
    }
    missing = required_summary - set(summary.columns)
    if missing:
        raise SystemExit(f"Missing summary columns: {sorted(missing)}")

    if len(summary) != 640:
        raise SystemExit(f"Expected 640 policy/evaluator rows, found {len(summary)}")

    if set(summary["Condition"].astype(str)) != set(FORMAL):
        raise SystemExit("Unexpected formal retention conditions.")

    counts = summary.groupby(
        ["Condition", "Algorithm", "Evaluator"]
    )["Seed"].nunique()

    if len(counts) != 32 or not (counts == 20).all():
        raise SystemExit("Expected 20 seeds in each of 32 evaluation groups.")

    required_stats = {
        "Algorithm", "Evaluator", "Condition", "Reference",
        "Mean_Difference_Ck_minus_C28",
        "Bootstrap_95CI_Lower", "Bootstrap_95CI_Upper",
        "Wilcoxon_P_Raw", "Holm_P_Adjusted", "Cohen_dz",
    }
    missing = required_stats - set(stats.columns)
    if missing:
        raise SystemExit(f"Missing statistics columns: {sorted(missing)}")

    required_conditions = {"Condition", "Cumulative_Eligible_Transitions"}
    missing = required_conditions - set(conditions.columns)
    if missing:
        raise SystemExit(f"Missing retention columns: {sorted(missing)}")

    transitions = conditions.set_index("Condition")[
        "Cumulative_Eligible_Transitions"
    ].to_dict()

    condition_summary = (
        summary.groupby(
            ["Algorithm", "Evaluator", "Condition"], as_index=False
        )
        .agg(Mean_Discounted_Return=("Mean_Discounted_Return", "mean"))
    )

    condition_summary["Condition_Order"] = (
        condition_summary["Condition"].map(ORDER)
    )
    condition_summary["Transitions"] = (
        condition_summary["Condition"].map(transitions)
    )

    if condition_summary["Transitions"].isna().any():
        raise SystemExit("Missing transition counts for formal conditions.")

    FIGURES.mkdir(parents=True, exist_ok=True)

    for evaluator in ["ARX", "XGBoost"]:
        data = condition_summary[
            condition_summary["Evaluator"] == evaluator
        ]

        if data.empty:
            raise SystemExit(f"Missing evaluator results: {evaluator}")

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        for algorithm in ["IQL", "CQL"]:
            values = data[
                data["Algorithm"] == algorithm
            ].sort_values("Condition_Order")

            if len(values) != len(FORMAL):
                raise SystemExit(
                    f"Incomplete retention curve: {evaluator}, {algorithm}"
                )

            ax.plot(
                values["Transitions"],
                values["Mean_Discounted_Return"],
                marker="o",
                label=algorithm,
            )

            for row in values.itertuples(index=False):
                ax.annotate(
                    row.Condition,
                    (row.Transitions, row.Mean_Discounted_Return),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

        ax.set_xlabel("Eligible transitions retained")
        ax.set_ylabel("Mean discounted return")
        ax.set_title(f"Natural-fragment retention curve: {evaluator}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        output = FIGURES / f"fragment_retention_curve_{evaluator.lower()}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Created:", output)

    print("Formal result validation and figure generation completed.")


if __name__ == "__main__":
    main()
