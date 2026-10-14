from pathlib import Path
import json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.config import load_config

EXPECTED_REPRESENTATION="80-D B+U state; 13 physical successor predictions and 1 derived underheating quantity (Comfort_U)"
REQUIRED_CHANNELS=("ARX","XGBoost")
QUALIFICATION_TARGETS=("Tin_Mean","RTP_Total")

def main():
    cfg=load_config(ROOT/"config"/"experiment.yaml")
    if int(cfg["evaluation"]["horizon_hours"])!=2:
        raise SystemExit("Formal evaluator acceptance requires horizon_hours=2.")

    src=ROOT/"outputs"/"results"/"evaluator"/"evaluator_validation.json"
    if not src.exists():
        raise SystemExit("Missing evaluator_validation.json. Run 03b first.")
    data=json.loads(src.read_text(encoding="utf-8"))

    if data.get("formal_horizon_hours")!=2:
        raise SystemExit("Refusing validation artifact with a non-2 h formal horizon.")
    if data.get("representation")!=EXPECTED_REPRESENTATION:
        raise SystemExit("Refusing validation artifact with a stale representation.")

    coverage=data.get("coverage",{})
    recursive=data.get("recursive_summary",[])

    decisions={}
    accepted=[]
    rejected=[]

    for ev in REQUIRED_CHANNELS:
        cov=coverage.get(ev,{})
        coverage_complete=bool(cov.get("complete",False))

        matches=[
            r for r in recursive
            if r.get("Evaluator")==ev and int(r.get("Horizon_Step",-1))==2
        ]
        if len(matches)!=1:
            raise SystemExit(f"Expected exactly one 2 h recursive summary row for {ev}.")
        row=matches[0]

        skills={}
        for target in QUALIFICATION_TARGETS:
            key=f"{target}_RMSE_Skill_vs_Persistence"
            if key not in row:
                raise SystemExit(f"Missing qualification metric {key} for {ev}.")
            val=float(row[key])
            if not np.isfinite(val):
                raise SystemExit(f"Non-finite qualification metric {key} for {ev}.")
            skills[target]=val

        passed=bool(
            coverage_complete
            and skills["Tin_Mean"]>0.0
            and skills["RTP_Total"]>0.0
        )

        decisions[ev]={
            "accepted_for_formal_2h_channel":passed,
            "coverage_complete":coverage_complete,
            "two_hour_tin_mean_rmse_skill_vs_persistence":skills["Tin_Mean"],
            "two_hour_rtp_total_rmse_skill_vs_persistence":skills["RTP_Total"],
            "qualification":(
                "PASS" if passed else
                "FAIL: requires complete LOFO coverage and positive 2 h RMSE skill versus "
                "matched persistence for both Tin_Mean and RTP_Total."
            ),
        }
        (accepted if passed else rejected).append(ev)

    # The current formal design specifies ARX and XGBoost as two independent
    # evaluator channels, so both must qualify before formal policy evaluation.
    gate_passed=(set(accepted)==set(REQUIRED_CHANNELS))

    payload={
        "accepted":gate_passed,
        "formal_horizon_hours":2,
        "representation":EXPECTED_REPRESENTATION,
        "acceptance_rule":(
            "Each evaluator is assessed independently under leave-one-natural-fragment-out "
            "validation. A formal 2 h channel requires complete fragment coverage and positive "
            "recursive RMSE skill versus matched persistence for both Tin_Mean and RTP_Total. "
            "The 13 physical-target metrics and derived underheating metric are reported as diagnostics and do not introduce "
            "a post hoc all-target-positive threshold."
        ),
        "required_channels":list(REQUIRED_CHANNELS),
        "accepted_channels":accepted,
        "rejected_channels":rejected,
        "decisions":decisions,
        "target_policy_support_requirement":(
            "State-action support distances must be recorded during every formal policy rollout "
            "and reported separately by evaluator."
        ),
    }

    out=ROOT/"outputs"/"results"/"evaluator"/"evaluator_validation_acceptance.json"
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

    if not gate_passed:
        raise SystemExit("Evaluator validation gate failed.")

if __name__=="__main__":
    main()
