from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .evaluation import DynamicsEvaluator, EvaluationOrigin, _next_state_from_prediction
from .transitions import _state_at


@dataclass(frozen=True)
class EvaluatorFold:
    """Leave-one-fragment-out evaluator fold.

    Each natural fragment is treated as one held-out group. Evaluators fitted for
    a held-out fragment never use transitions from that fragment. This avoids
    transition-target leakage without introducing an arbitrary number of folds.
    """

    held_out_fragment: str
    train_fragments: Tuple[str, ...]



class StandardizedSupportIndex:
    """Nearest-neighbor support diagnostic in evaluator input space.

    The transform is fitted only on the evaluator training fold. Distances are
    therefore fold-local diagnostics rather than globally normalized scores.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray, train_x: np.ndarray):
        self.mean = mean
        self.scale = scale
        self.train_z = (train_x - mean) / scale

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardizedSupportIndex":
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean, scale=scale, train_x=x)

    def nearest_distance(self, x: np.ndarray) -> float:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        d = np.linalg.norm(self.train_z - z.reshape(1, -1), axis=1)
        return float(np.min(d))


@dataclass
class CrossFittedDynamicsBundle:
    evaluator: DynamicsEvaluator
    support_index: StandardizedSupportIndex
    fold: EvaluatorFold


def build_leave_one_fragment_out_folds(fragment_names: Iterable[str]) -> List[EvaluatorFold]:
    """Create one evaluator fold per natural fragment.

    This is a study-specific group cross-fitting rule aligned with the natural
    fragmentation structure. It avoids choosing an arbitrary K for GroupKFold.
    """
    names = tuple(fragment_names)
    folds = []
    for held_out in names:
        train = tuple(name for name in names if name != held_out)
        folds.append(EvaluatorFold(held_out_fragment=held_out, train_fragments=train))
    return folds



def _historical_dynamics_indices(pipeline, fragment_name: str):
    """Return physically observed evaluator indices without applying the policy envelope.

    Historical dynamics validity and policy action admissibility are different
    concepts. Real logged FTS values are retained for evaluator fitting and
    validation when the telemetry, lag history, and PM requirements are valid,
    even if a logged FTS lies outside the policy's 30-50 C action envelope.
    """
    fragment = next(f for f in pipeline.fragments if f.name == fragment_name)
    ft = list(fragment.times)
    cfg = pipeline.cfg

    scalar_lags = int(cfg["state"]["scalar_lags"])
    fts_history = int(cfg["state"]["fts_history_hours"])
    pm_indices = [int(x) for x in cfg["state"]["pm_required_indices"]]
    pm_value = float(cfg["state"]["pm_required_value"])
    pm_tol = float(cfg["state"]["pm_tolerance"])
    max_history = max(
        scalar_lags,
        fts_history,
        max(-x for x in pm_indices if x < 0),
    )

    valid = []
    for i in range(max_history, len(ft) - 1):
        pm_ok = True
        for offset in pm_indices:
            j = i + offset
            if j < 0 or j >= len(ft):
                pm_ok = False
                break
            pm = pipeline.data[ft[j]]["pm"]
            if not np.allclose(pm, pm_value, atol=pm_tol, rtol=0.0):
                pm_ok = False
                break
        if pm_ok:
            valid.append(i)
    return valid


def _historical_fragment_xy(pipeline,fragment_name):
    fragment=next(f for f in pipeline.fragments if f.name==fragment_name); ft=list(fragment.times); cfg=pipeline.cfg
    lags=int(cfg["state"]["scalar_lags"]); hist=int(cfg["state"]["fts_history_hours"]); xs=[]; ys=[]; times=[]
    for i in _historical_dynamics_indices(pipeline,fragment_name):
        state=_state_at(ft,i,pipeline.data,lags,hist); nxt=_state_at(ft,i+1,pipeline.data,lags,hist); action=pipeline.data[ft[i]]["fts"].copy()
        xs.append(np.concatenate([state,action])); ys.append(nxt[:14].copy()); times.append(ft[i])
    sdim=(lags+1)*16+hist*len(cfg["signals"]["action_controllers"]); adim=len(cfg["signals"]["action_controllers"])
    return np.asarray(xs,dtype=np.float64).reshape(-1,sdim+adim),np.asarray(ys,dtype=np.float64).reshape(-1,14),np.asarray(times,dtype=object)


def _concat_historical_fragment_xy(pipeline,fragment_names):
    parts=[_historical_fragment_xy(pipeline,n) for n in fragment_names]; parts=[x for x in parts if len(x[1])>0]
    if not parts: raise ValueError("No evaluator training transitions")
    return np.concatenate([x[0] for x in parts]),np.concatenate([x[1] for x in parts])


class TCStructuredRidgeModel:
    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept_ = float(intercept)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return np.asarray([x @ self.coef_ + self.intercept_], dtype=np.float64)
        return x @ self.coef_ + self.intercept_


def _tc_penalty_matrix(scale, lam=3.0, rho=0.99):
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (88,):
        raise RuntimeError(f"TC ARX requires 88 input scales, got {scale.shape}")

    ages = np.arange(1, 6, dtype=np.float64)
    kernel = float(rho) ** np.maximum(ages[:, None], ages[None, :])
    precision = np.linalg.inv(kernel + 1e-10 * np.eye(5, dtype=np.float64))
    chol = np.linalg.cholesky(precision).T

    rows = []
    for controller in range(8):
        indices = [
            80 + controller,
            48 + controller,
            56 + controller,
            64 + controller,
            72 + controller,
        ]
        for qrow in chol:
            row = np.zeros(88, dtype=np.float64)
            for value, idx in zip(qrow, indices):
                row[idx] = np.sqrt(float(lam)) * value / scale[idx]
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _fit_tc_arx_models(x, y, base_ridge_alpha=1.0, tc_lambda=3.0, tc_rho=0.99):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 88:
        raise RuntimeError(f"TC ARX requires N x 88 inputs, got {x.shape}")
    if y.ndim != 2 or y.shape[1] != 14:
        raise RuntimeError(f"TC ARX requires N x 14 targets, got {y.shape}")

    mu = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    z = (x - mu) / scale
    p = z.shape[1]

    penalty = _tc_penalty_matrix(scale, lam=float(tc_lambda), rho=float(tc_rho))
    regularizer = float(base_ridge_alpha) * np.eye(p, dtype=np.float64) + penalty.T @ penalty
    lhs = z.T @ z + regularizer

    y_mean = y.mean(axis=0)
    rhs = z.T @ (y - y_mean)
    theta = np.linalg.solve(lhs, rhs)

    physical_coef = theta / scale[:, None]
    intercept = y_mean - mu @ physical_coef

    return tuple(
        TCStructuredRidgeModel(physical_coef[:, j], intercept[j])
        for j in range(y.shape[1])
    )


def fit_arx_ridge_bundle(pipeline, fold: EvaluatorFold) -> CrossFittedDynamicsBundle:
    x, y = _concat_historical_fragment_xy(pipeline, fold.train_fragments)
    evaluation_cfg = pipeline.cfg["evaluation"]
    models = _fit_tc_arx_models(
        x,
        y,
        base_ridge_alpha=float(evaluation_cfg["arx_base_ridge_alpha"]),
        tc_lambda=float(evaluation_cfg["arx_tc_lambda"]),
        tc_rho=float(evaluation_cfg["arx_tc_rho"]),
    )
    evaluator = DynamicsEvaluator(name="ARX_TC_RIDGE_LOFO", models=models)
    return CrossFittedDynamicsBundle(
        evaluator=evaluator,
        support_index=StandardizedSupportIndex.fit(x),
        fold=fold,
    )


def fit_xgboost_default_bundle(pipeline,fold,random_state:int=0):
    x,y=_concat_historical_fragment_xy(pipeline,fold.train_fragments); common={"objective":"reg:squarederror","random_state":int(random_state)}
    models=tuple(XGBRegressor(**common).fit(x,y[:,j]) for j in range(14))
    return CrossFittedDynamicsBundle(DynamicsEvaluator("XGBOOST_DEFAULT_LOFO",models),StandardizedSupportIndex.fit(x),fold)


def build_cross_fitted_evaluators(pipeline, random_state: int = 0) -> Dict[str, Dict[str, CrossFittedDynamicsBundle]]:
    """Fit ARX and XGBoost evaluators for every held-out natural fragment."""
    names = [
        f.name for f in pipeline.fragments
        if len(_historical_dynamics_indices(pipeline, f.name)) > 0
    ]
    out: Dict[str, Dict[str, CrossFittedDynamicsBundle]] = {"ARX": {}, "XGBoost": {}}
    for fold in build_leave_one_fragment_out_folds(names):
        out["ARX"][fold.held_out_fragment] = fit_arx_ridge_bundle(pipeline, fold)
        out["XGBoost"][fold.held_out_fragment] = fit_xgboost_default_bundle(
            pipeline, fold, random_state=random_state
        )
    return out



def build_selected_cross_fitted_evaluators(
    pipeline,
    channels,
    random_state: int = 0,
    progress_callback=None,
) -> Dict[str, Dict[str, CrossFittedDynamicsBundle]]:
    """Fit only formally accepted evaluator channels.

    This avoids rebuilding rejected evaluator families during formal evaluation.
    The returned structure matches `build_cross_fitted_evaluators`.
    """
    channels = list(channels)
    supported = {"ARX", "XGBoost"}
    unknown = sorted(set(channels).difference(supported))
    if unknown:
        raise ValueError(f"Unsupported evaluator channels: {unknown}")

    names = [
        f.name for f in pipeline.fragments
        if len(_historical_dynamics_indices(pipeline, f.name)) > 0
    ]
    folds = build_leave_one_fragment_out_folds(names)
    total = len(folds) * len(channels)
    completed = 0
    out = {name: {} for name in channels}

    for channel in channels:
        for fold_index, fold in enumerate(folds, start=1):
            if channel == "ARX":
                bundle = fit_arx_ridge_bundle(pipeline, fold)
            else:
                bundle = fit_xgboost_default_bundle(
                    pipeline,
                    fold,
                    random_state=random_state,
                )
            out[channel][fold.held_out_fragment] = bundle
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    channel,
                    fold_index,
                    len(folds),
                    completed,
                    total,
                )
    return out



def validate_one_step(pipeline,bundles):
    rows=[]; names=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]
    for evaluator_name,by_fragment in bundles.items():
        for fragment_name,bundle in by_fragment.items():
            x,y,times=_historical_fragment_xy(pipeline,fragment_name)
            for i,time_value in enumerate(times):
                pred=bundle.evaluator.predict_next(x[i,:-8],x[i,-8:])
                rec={"Evaluator":evaluator_name,"Fragment":fragment_name,"Time":time_value,"Support_Distance":bundle.support_index.nearest_distance(x[i])}
                for j,name in enumerate(names): rec[name+"_True"]=float(y[i,j]); rec[name+"_Pred"]=float(pred[j])
                rows.append(rec)
    return rows


def _fragment_time_lookup(pipeline, fragment_name: str):
    fragment = next(f for f in pipeline.fragments if f.name == fragment_name)
    times = list(fragment.times)
    index = {t: i for i, t in enumerate(times)}
    return times, index


def validate_recursive_behavior_rollout(pipeline,bundles):
    horizon=int(pipeline.cfg["evaluation"]["horizon_hours"]); rows=[]; names=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]
    for evaluator_name,by_fragment in bundles.items():
        for fragment_name,bundle in by_fragment.items():
            ds=pipeline.fragment_datasets[fragment_name]
            if len(ds)==0: continue
            times,time_index=_fragment_time_lookup(pipeline,fragment_name); state_by_time={x:ds.S[i].copy() for i,x in enumerate(ds.time)}
            for origin in ds.time:
                i=time_index[origin]
                if i+horizon>=len(times): continue
                state=state_by_time[origin].copy()
                for h in range(1,horizon+1):
                    action=pipeline.data[times[i+h-1]]["fts"].copy(); nxt=pipeline.data[times[i+h]]
                    pred=bundle.evaluator.predict_next(state,action)
                    true=np.asarray([nxt["tin_mean"],nxt["tin_min"],nxt["tin_std"],nxt["comfort_u"],*nxt["rtp_channels"].tolist(),nxt["rtp_total"]],dtype=np.float64)
                    row={"Evaluator":evaluator_name,"Fragment":fragment_name,"Origin_Time":origin,"Horizon_Step":h}
                    for j,name in enumerate(names): row[name+"_True"]=float(true[j]); row[name+"_Pred"]=float(pred[j])
                    rows.append(row)
                    state=_next_state_from_prediction(state,action,np.asarray([nxt["tout_mean"],nxt["tout_std"]]),pred,pipeline.cfg)
    return rows


def summarize_validation(rows,one_step:bool):
    if not rows: return []
    import pandas as pd
    df=pd.DataFrame(rows); groups=["Evaluator"] if one_step else ["Evaluator","Horizon_Step"]; names=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]; out=[]
    for keys,g in df.groupby(groups):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(groups,keys)); row["N"]=int(len(g))
        for name in names:
            err=g[name+"_Pred"].to_numpy()-g[name+"_True"].to_numpy()
            row[name+"_MAE"]=float(np.mean(np.abs(err))); row[name+"_RMSE"]=float(np.sqrt(np.mean(err**2)))
        out.append(row)
    return out


def evaluate_policy_cross_fitted(
    policy,
    bundles: Dict[str, Dict[str, CrossFittedDynamicsBundle]],
    origins,
    reward_fn,
    action_space,
    cfg,
):
    """Evaluate a policy using the evaluator fold that excludes each origin fragment."""
    from .evaluation import rollout_policy

    rows = []
    for evaluator_name, by_fragment in bundles.items():
        for origin in origins:
            if origin.fragment not in by_fragment:
                continue
            bundle = by_fragment[origin.fragment]
            result = rollout_policy(
                policy=policy,
                evaluator=bundle.evaluator,
                origin=origin,
                reward_fn=reward_fn,
                action_space=action_space,
                cfg=cfg,
                support_index=bundle.support_index,
            )
            # Re-run the action path only for support diagnostics would duplicate
            # rollout work. Per-step support logging is added to the formal result
            # runner after evaluator validation is accepted.
            rows.append(
                {
                    "Evaluator": evaluator_name,
                    "Fragment": origin.fragment,
                    "Origin_Time": origin.time,
                    "Discounted_Return": result["discounted_return"],
                    "Mean_Reward": result["mean_reward"],
                    "Mean_Energy_Component": result["mean_energy_component"],
                    "Mean_Underheat_Component": result["mean_underheat_component"],
                    "Mean_Support_Distance": result["mean_support_distance"],
                    "Max_Support_Distance": result["max_support_distance"],
                }
            )
    return rows


def validate_persistence_baseline(pipeline):
    horizon=int(pipeline.cfg["evaluation"]["horizon_hours"]); one=[]; recursive=[]; names=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]
    for fragment in pipeline.fragments:
        ds=pipeline.fragment_datasets[fragment.name]
        if len(ds)==0: continue
        times,time_index=_fragment_time_lookup(pipeline,fragment.name)
        for j,time_value in enumerate(ds.time):
            current=ds.S[j,:14].copy(); true1=ds.SN[j,:14].copy()
            row={"Evaluator":"Persistence","Fragment":fragment.name,"Time":time_value,"Support_Distance":0.0}
            for k,name in enumerate(names): row[name+"_True"]=float(true1[k]); row[name+"_Pred"]=float(current[k])
            one.append(row); i=time_index[time_value]
            if i+horizon>=len(times): continue
            for h in range(1,horizon+1):
                nxt=pipeline.data[times[i+h]]
                true=np.asarray([nxt["tin_mean"],nxt["tin_min"],nxt["tin_std"],nxt["comfort_u"],*nxt["rtp_channels"].tolist(),nxt["rtp_total"]],dtype=np.float64)
                rr={"Evaluator":"Persistence","Fragment":fragment.name,"Origin_Time":time_value,"Horizon_Step":h}
                for k,name in enumerate(names): rr[name+"_True"]=float(true[k]); rr[name+"_Pred"]=float(current[k])
                recursive.append(rr)
    return one,recursive


def add_skill_against_persistence(model_summary,persistence_summary,horizon_key=None):
    names=["Tin_Mean","Tin_Min","Tin_Std","Comfort_U"]+[f"RTP_{i:02d}" for i in range(1,10)]+["RTP_Total"]; baseline={(r.get(horizon_key) if horizon_key else None):r for r in persistence_summary}; out=[]
    for row in model_summary:
        key=row.get(horizon_key) if horizon_key else None; b=baseline[key]; rec=dict(row)
        for name in names:
            den=float(b[name+"_RMSE"]); rec[name+"_RMSE_Skill_vs_Persistence"]=np.nan if den<=0 else 1.0-float(row[name+"_RMSE"])/den
        out.append(rec)
    return out


