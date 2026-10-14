from pathlib import Path
import json, re, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.config import load_config
from src.pipeline import FormalPipeline
from src.utils import ensure_dir
from src.evaluator_protocol import summarize_validation, _historical_dynamics_indices

TARGETS=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]
EVALUATORS=("ARX","XGBoost")
PSCALE=1178.965672
ALPHA=1.500824

def fragment_key(name):
    m=re.fullmatch(r"F(\d+)",str(name))
    if not m:
        raise RuntimeError(f"Unexpected fragment name: {name}")
    return int(m.group(1))

def expected_fragments(pipe):
    return sorted(
        [f.name for f in pipe.fragments if len(_historical_dynamics_indices(pipe,f.name))>0],
        key=fragment_key,
    )

def read_canonical_parts(folder, suffix, expected):
    paths=[]
    missing=[]
    for ev in EVALUATORS:
        for frag in expected:
            p=folder/f"{ev}_{frag}_{suffix}.csv"
            if p.exists():
                paths.append(p)
            else:
                missing.append(p.name)
    if missing:
        raise RuntimeError("Missing canonical evaluator fold files: "+", ".join(missing))
    frames=[]
    for p in paths:
        df=pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames,ignore_index=True), paths

def required_columns(one_step):
    id_cols=["Evaluator","Fragment","Time","Support_Distance"] if one_step else ["Evaluator","Fragment","Origin_Time","Horizon_Step"]
    value_cols=[]
    for n in TARGETS:
        value_cols += [n+"_True",n+"_Pred"]
    return id_cols+value_cols

def validate_rows(df, one_step, expected):
    req=required_columns(one_step)
    missing=[c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError("Missing validation columns: "+", ".join(missing))
    if df.empty:
        raise RuntimeError("Validation rows are empty.")
    numeric=[c for c in req if c not in {"Evaluator","Fragment","Time","Origin_Time"}]
    vals=df[numeric].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise RuntimeError("NaN/Inf detected in evaluator validation rows.")
    bad_ev=sorted(set(df["Evaluator"].astype(str))-set(EVALUATORS))
    if bad_ev:
        raise RuntimeError(f"Unexpected evaluator labels: {bad_ev}")
    expected_set=set(expected)
    for ev in EVALUATORS:
        got=set(df.loc[df["Evaluator"]==ev,"Fragment"].astype(str))
        if got != expected_set:
            raise RuntimeError(f"Incomplete {ev} fragment coverage: expected={sorted(expected_set,key=fragment_key)}, got={sorted(got,key=fragment_key)}")
    if not one_step:
        hs=set(pd.to_numeric(df["Horizon_Step"]).astype(int))
        if hs != {1,2}:
            raise RuntimeError(f"Recursive validation must contain exactly horizon steps 1 and 2, got {sorted(hs)}")

def endo(rec):
    return np.asarray([
        rec["tin_mean"],rec["tin_min"],rec["tin_std"],rec["comfort_u"],
        *np.asarray(rec["rtp_channels"],dtype=float).tolist(),rec["rtp_total"]
    ],dtype=float)

def matched_persistence(pipe,model_rows,one_step):
    rows=[]
    for r in model_rows.to_dict("records"):
        key="Time" if one_step else "Origin_Time"
        t=pd.Timestamp(r[key])
        cur=endo(pipe.data[t])
        z={"Evaluator":f"Persistence_for_{r['Evaluator']}","Fragment":r["Fragment"]}
        if one_step:
            z["Time"]=r["Time"]
        else:
            z["Origin_Time"]=r["Origin_Time"]
            z["Horizon_Step"]=int(r["Horizon_Step"])
        for j,n in enumerate(TARGETS):
            z[n+"_True"]=float(r[n+"_True"])
            z[n+"_Pred"]=float(cur[j])
        rows.append(z)
    return pd.DataFrame(rows)

def exact_summary(model,pers,one_step):
    ms=summarize_validation(model.to_dict("records"),one_step)
    ps=summarize_validation(pers.to_dict("records"),one_step)
    hk=None if one_step else "Horizon_Step"
    base={}
    for r in ps:
        ev=str(r["Evaluator"]).replace("Persistence_for_","",1)
        base[(ev,r.get(hk) if hk else None)]=r
    out=[]
    for r in ms:
        k=(r["Evaluator"],r.get(hk) if hk else None)
        if k not in base:
            raise RuntimeError(f"Missing matched persistence summary: {k}")
        b=base[k]
        if int(r["N"])!=int(b["N"]):
            raise RuntimeError(f"Matched N mismatch: {k}")
        q=dict(r)
        q["Persistence_N"]=int(b["N"])
        for n in TARGETS:
            den=float(b[n+"_RMSE"])
            q["Persistence_"+n+"_RMSE"]=den
            q[n+"_RMSE_Skill_vs_Persistence"]=np.nan if den<=0 else 1.0-float(r[n+"_RMSE"])/den
        out.append(q)
    return out,ps

def assert_finite_summary(rows,label):
    for r in rows:
        for k,v in r.items():
            if k in {"Evaluator","Horizon_Step","N","Persistence_N"}:
                continue
            if isinstance(v,(int,float,np.integer,np.floating)) and not np.isfinite(float(v)):
                raise RuntimeError(f"Non-finite {label} metric: {r.get('Evaluator')} {k}={v}")

def main():
    cfg=load_config(ROOT/"config"/"experiment.yaml")
    if int(cfg["evaluation"]["horizon_hours"])!=2:
        raise SystemExit("Formal evaluator aggregation requires horizon_hours=2.")
    pipe=FormalPipeline(cfg).build()
    pipe.assert_frozen_qa()

    folder=ROOT/"outputs"/"results"/"evaluator"/"evaluator_folds"
    expected=expected_fragments(pipe)
    one,one_paths=read_canonical_parts(folder,"one_step",expected)
    rec,rec_paths=read_canonical_parts(folder,"recursive_2h",expected)
    validate_rows(one,True,expected)
    validate_rows(rec,False,expected)

    coverage={}
    for ev in EVALUATORS:
        got=sorted(one.loc[one.Evaluator==ev,"Fragment"].unique().tolist(),key=fragment_key)
        coverage[ev]={"expected":expected,"observed":got,"complete":got==expected}

    po=matched_persistence(pipe,one,True)
    pr=matched_persistence(pipe,rec,False)
    os,ops=exact_summary(one,po,True)
    rs,rps=exact_summary(rec,pr,False)
    assert_finite_summary(os,"one-step")
    assert_finite_summary(rs,"recursive")

    td=ensure_dir(ROOT/"outputs"/"results"/"evaluator")
    qd=ensure_dir(ROOT/"outputs"/"results"/"evaluator")
    pd.DataFrame(os).to_csv(td/"evaluator_one_step_summary.csv",index=False)
    pd.DataFrame(rs).to_csv(td/"evaluator_recursive_2h_summary.csv",index=False)
    po.to_csv(td/"persistence_matched_one_step_rows.csv",index=False)
    pr.to_csv(td/"persistence_matched_recursive_2h_rows.csv",index=False)
    pd.DataFrame(ops).to_csv(td/"persistence_matched_one_step_summary.csv",index=False)
    pd.DataFrame(rps).to_csv(td/"persistence_matched_recursive_2h_summary.csv",index=False)

    artifact={
        "formal_horizon_hours":2,
        "crossfit_scheme":cfg["evaluation"]["crossfit_scheme"],
        "representation":"80-D B+U state; 13 physical successor predictions and 1 derived underheating quantity (Comfort_U)",
        "comparison_rule":"Persistence is recomputed on exactly the same evaluator-specific fragment/origin/horizon rows.",
        "canonical_fold_files":{"one_step":len(one_paths),"recursive_2h":len(rec_paths)},
        "coverage":coverage,
        "one_step_summary":os,
        "recursive_summary":rs,
        "persistence_one_step":ops,
        "persistence_recursive":rps,
        "formal_acceptance":"PENDING_80D_METHOD_REVIEW",
        "note":"This artifact qualifies the corrected 2 h evaluator representation."
    }
    payload=json.dumps(artifact,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))
    (qd/"evaluator_validation.json").write_text(payload,encoding="utf-8")
    print(payload)

if __name__=="__main__":
    main()
