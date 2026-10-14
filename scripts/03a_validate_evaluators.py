from pathlib import Path
import argparse, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.config import load_config
from src.pipeline import FormalPipeline
from src.utils import ensure_dir
from src.evaluator_protocol import (
    EvaluatorFold, _historical_dynamics_indices,
    fit_arx_ridge_bundle, fit_xgboost_default_bundle,
    validate_one_step, validate_recursive_behavior_rollout,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evaluator",choices=["ARX","XGBoost"],required=True)
    ap.add_argument("--fragment",default=None)
    args=ap.parse_args()

    cfg=load_config(ROOT/"config"/"experiment.yaml")
    if int(cfg["evaluation"]["horizon_hours"])!=2:
        raise SystemExit("Formal evaluator validation requires horizon_hours=2.")
    pipe=FormalPipeline(cfg).build()
    pipe.assert_frozen_qa()

    historical_dynamics_n = sum(
        len(_historical_dynamics_indices(pipe, f.name))
        for f in pipe.fragments
    )
    if historical_dynamics_n != 1098:
        raise AssertionError(
            f"Historical evaluator dynamics transition count mismatch: "
            f"{historical_dynamics_n} != 1098"
        )

    names=[f.name for f in pipe.fragments if len(_historical_dynamics_indices(pipe,f.name))>0]
    if args.fragment is not None:
        if args.fragment not in names:
            raise SystemExit(f"Unknown/noncontributing fragment: {args.fragment}")
        names=[args.fragment]

    outdir=ensure_dir(ROOT/"outputs"/"results"/"evaluator"/"evaluator_folds")
    for held in names:
        train=tuple(f.name for f in pipe.fragments
                    if f.name!=held and len(_historical_dynamics_indices(pipe,f.name))>0)
        fold=EvaluatorFold(held_out_fragment=held,train_fragments=train)
        bundle=(fit_arx_ridge_bundle(pipe,fold) if args.evaluator=="ARX"
                else fit_xgboost_default_bundle(pipe,fold,random_state=int(cfg["project"]["seed"])))
        bundles={args.evaluator:{held:bundle}}
        one=validate_one_step(pipe,bundles)
        rec=validate_recursive_behavior_rollout(pipe,bundles)
        tag=f"{args.evaluator}_{held}"
        pd.DataFrame(one).to_csv(outdir/f"{tag}_one_step.csv",index=False)
        pd.DataFrame(rec).to_csv(outdir/f"{tag}_recursive_2h.csv",index=False)
        print(f"{tag}: one_step={len(one)} recursive_rows={len(rec)}")

if __name__=="__main__":
    main()
