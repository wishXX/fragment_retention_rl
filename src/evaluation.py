from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class EvaluationOrigin:
    fragment: str
    time: object
    state_physical: np.ndarray
    future_exogenous: np.ndarray


def _project_endogenous_to_physical_domain(y):
    """Project 14 outputs to nonnegative U, std and coherent RTP channels.

    The nine power channels and their independently predicted total are
    projected jointly; Tin_Mean and Tin_Min are left unchanged.
    """
    y=np.asarray(y,dtype=np.float64).copy()
    if not np.isfinite(y).all():
        raise ValueError("Non-finite endogenous predictions")
    if y.shape!=(14,): raise ValueError(f"Expected 14 endogenous predictions, got {y.shape}")
    y[2]=max(y[2],0.0)
    y[3]=max(y[3],0.0)
    b=y[4:13].copy(); total_base=float(y[13]); active=np.ones(9,dtype=bool)
    while True:
        idx=np.flatnonzero(active)
        if len(idx)==0:
            x=np.zeros(9,dtype=np.float64); break
        lam=(float(np.sum(b[idx]))-total_base)/(1.0+len(idx))
        candidate=b[idx]-lam; bad=candidate<=0.0
        if not np.any(bad):
            x=np.zeros(9,dtype=np.float64); x[idx]=candidate; break
        active[idx[bad]]=False
    y[4:13]=x
    y[13]=float(np.sum(x))
    return y


# Compatibility alias for existing diagnostic scripts; remove after migration.
_project_arx_endogenous_to_physical_domain = _project_endogenous_to_physical_domain


@dataclass
class DynamicsEvaluator:
    name: str
    models: tuple
    def predict_next(self,state_physical,action_physical):
        x=np.concatenate([np.asarray(state_physical,dtype=np.float64),np.asarray(action_physical,dtype=np.float64)]).reshape(1,-1)
        y=np.asarray([float(m.predict(x)[0]) for m in self.models],dtype=np.float64)
        if y.shape!=(14,): raise RuntimeError(f"Expected 14 predictions, got {y.shape}")
        if self.name in ("ARX_TC_RIDGE_LOFO", "XGBOOST_DEFAULT_LOFO"):
            y=_project_endogenous_to_physical_domain(y)
        return y


def _next_state_from_prediction(state_physical,action_physical,next_exogenous,next_endogenous,cfg,support_index=None):
    lags=int(cfg["state"]["scalar_lags"]); hist=int(cfg["state"]["fts_history_hours"])
    adim=len(cfg["signals"]["action_controllers"]); state=np.asarray(state_physical,dtype=np.float64)
    pcount=(lags+1)*16; expected=pcount+hist*adim
    if state.shape!=(expected,): raise ValueError(f"Expected state ({expected},), got {state.shape}")
    physical=state[:pcount].reshape(lags+1,16); fts=state[pcount:].reshape(hist,adim)
    endo=np.asarray(next_endogenous,dtype=np.float64); exo=np.asarray(next_exogenous,dtype=np.float64)
    if endo.shape!=(14,) or exo.shape!=(2,): raise ValueError("Bad successor component shape")
    return np.concatenate([np.vstack([np.concatenate([endo,exo]),physical[:lags]]).reshape(-1),
                           np.vstack([np.asarray(action_physical,dtype=np.float64),fts[:hist-1]]).reshape(-1)])


def rollout_policy(policy,evaluator,origin,reward_fn,action_space,cfg,support_index=None):
    horizon=int(cfg["evaluation"]["horizon_hours"]); gamma=float(cfg["evaluation"]["gamma"])
    state=np.asarray(origin.state_physical,dtype=np.float64).copy()
    rewards=[]; energy_terms=[]; underheat_terms=[]; actions=[]; distances=[]
    for h in range(horizon):
        action=np.asarray(policy(state),dtype=np.float64)
        if support_index is not None: distances.append(float(support_index.nearest_distance(np.concatenate([state,action]))))
        if not action_space.contains(action): raise ValueError("Policy action outside frozen envelope")
        endo=evaluator.predict_next(state,action)
        energy,underheat=reward_fn.components(float(max(endo[3],0.0)),float(endo[13]))
        reward=float(-(energy+underheat))
        rewards.append(reward); energy_terms.append(float(energy)); underheat_terms.append(float(underheat)); actions.append(action)
        state=_next_state_from_prediction(state,action,origin.future_exogenous[h],endo,cfg)
    return {"discounted_return":float(sum((gamma**i)*r for i,r in enumerate(rewards))),
            "mean_reward":float(np.mean(rewards)),"mean_energy_component":float(np.mean(energy_terms)),
            "mean_underheat_component":float(np.mean(underheat_terms)),"actions":np.asarray(actions),
            "mean_support_distance":float(np.mean(distances)) if distances else None,
            "max_support_distance":float(np.max(distances)) if distances else None}


def build_common_origins(pipeline,stride:int=1):
    if stride<1: raise ValueError("stride must be at least 1")
    horizon=int(pipeline.cfg["evaluation"]["horizon_hours"]); origins=[]; counter=0
    for fragment in pipeline.fragments:
        ds=pipeline.fragment_datasets[fragment.name]
        if len(ds)==0: continue
        state_by_time={x:ds.S[i] for i,x in enumerate(ds.time)}
        ft=list(fragment.times); idx={x:i for i,x in enumerate(ft)}
        for x in ds.time:
            i=idx[x]
            if i+horizon>=len(ft): continue
            if counter%stride!=0: counter+=1; continue
            future=np.asarray([[pipeline.data[ft[i+h]]["tout_mean"],pipeline.data[ft[i+h]]["tout_std"]] for h in range(1,horizon+1)],dtype=np.float64)
            origins.append(EvaluationOrigin(fragment.name,x,state_by_time[x].copy(),future)); counter+=1
    return origins


def evaluate_policy_on_origins(
    policy,
    evaluators: dict[str, DynamicsEvaluator],
    origins: Iterable[EvaluationOrigin],
    reward_fn,
    action_space,
    cfg,
):
    rows = []
    for evaluator_name, evaluator in evaluators.items():
        for origin in origins:
            result = rollout_policy(
                policy,
                evaluator,
                origin,
                reward_fn,
                action_space,
                cfg,
            )
            rows.append(
                {
                    "Evaluator": evaluator_name,
                    "Fragment": origin.fragment,
                    "Origin_Time": origin.time,
                    "Discounted_Return": result["discounted_return"],
                    "Mean_Reward": result["mean_reward"],
                    "Mean_Energy_Component": result["mean_energy_component"],
                    "Mean_Underheat_Component": result["mean_underheat_component"],
                }
            )
    return rows
