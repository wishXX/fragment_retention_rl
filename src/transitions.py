from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TransitionDataset:
    S: np.ndarray
    A_physical: np.ndarray
    A_normalized: np.ndarray
    SN: np.ndarray
    R: np.ndarray
    terminal: np.ndarray
    truncated: np.ndarray
    fragment: np.ndarray
    time: np.ndarray

    def __len__(self):
        return len(self.R)


def _physical_block(rec):
    return [
        rec["tin_mean"], rec["tin_min"], rec["tin_std"], rec["comfort_u"],
        *rec["rtp_channels"].tolist(), rec["rtp_total"],
        rec["tout_mean"], rec["tout_std"],
    ]


def _state_at(fragment_times, i, data, scalar_lags: int, fts_history_hours: int):
    physical = []
    for lag in range(0, scalar_lags + 1):
        physical.extend(_physical_block(data[fragment_times[i - lag]]))
    fts_hist = []
    for lag in range(1, fts_history_hours + 1):
        fts_hist.extend(data[fragment_times[i - lag]]["fts"].tolist())
    return np.asarray(physical + fts_hist, dtype=np.float64)


def eligible_indices(fragment, data, cfg, action_space):
    scalar_lags = int(cfg["state"]["scalar_lags"])
    fts_history = int(cfg["state"]["fts_history_hours"])
    pm_indices = [int(x) for x in cfg["state"]["pm_required_indices"]]
    pm_value = float(cfg["state"]["pm_required_value"])
    pm_tol = float(cfg["state"]["pm_tolerance"])
    max_history = max(scalar_lags, fts_history, max(-x for x in pm_indices if x < 0))
    ft = list(fragment.times)
    out = []
    for i in range(max_history, len(ft) - 1):
        pm_ok = True
        for offset in pm_indices:
            j = i + offset
            if j < 0 or j >= len(ft):
                pm_ok = False
                break
            pm = data[ft[j]]["pm"]
            if not np.allclose(pm, pm_value, atol=pm_tol, rtol=0.0):
                pm_ok = False
                break
        if not pm_ok:
            continue
        action = data[ft[i]]["fts"]
        if cfg["action"]["exclude_outside_primary"] and not action_space.contains(action):
            continue
        out.append(i)
    return out


def compute_fragment_next_rtp(fragment, data, cfg, action_space):
    idx = eligible_indices(fragment, data, cfg, action_space)
    ft = list(fragment.times)
    return np.asarray([data[ft[i + 1]]["rtp"] for i in idx], dtype=np.float64)


def build_fragment_transitions(fragment, data, cfg, action_space, reward_fn):
    scalar_lags = int(cfg["state"]["scalar_lags"])
    fts_history = int(cfg["state"]["fts_history_hours"])
    S, A_phys, A_norm, SN, R = [], [], [], [], []
    fragment_names, times = [], []
    ft = list(fragment.times)

    for i in eligible_indices(fragment, data, cfg, action_space):
        action = data[ft[i]]["fts"].copy()
        s = _state_at(ft, i, data, scalar_lags, fts_history)
        sn = _state_at(ft, i + 1, data, scalar_lags, fts_history)
        r = float(reward_fn(data[ft[i + 1]]["comfort_u"], data[ft[i + 1]]["rtp_total"]))

        S.append(s)
        A_phys.append(action)
        A_norm.append(action_space.encode(action))
        SN.append(sn)
        R.append(r)
        fragment_names.append(fragment.name)
        times.append(ft[i])

    n = len(R)
    physical_dim = 16
    state_dim = (scalar_lags + 1) * physical_dim + fts_history * len(cfg["signals"]["action_controllers"])
    action_dim = len(cfg["signals"]["action_controllers"])
    return TransitionDataset(
        S=np.asarray(S, dtype=np.float64).reshape(n, state_dim),
        A_physical=np.asarray(A_phys, dtype=np.float64).reshape(n, action_dim),
        A_normalized=np.asarray(A_norm, dtype=np.float64).reshape(n, action_dim),
        SN=np.asarray(SN, dtype=np.float64).reshape(n, state_dim),
        R=np.asarray(R, dtype=np.float64),
        terminal=np.zeros(n, dtype=np.float64),
        truncated=np.zeros(n, dtype=np.float64),
        fragment=np.asarray(fragment_names, dtype=object),
        time=np.asarray(times, dtype=object),
    )


def concatenate_datasets(parts):
    if not parts:
        raise ValueError("No transition datasets to concatenate")
    return TransitionDataset(
        S=np.concatenate([p.S for p in parts], axis=0),
        A_physical=np.concatenate([p.A_physical for p in parts], axis=0),
        A_normalized=np.concatenate([p.A_normalized for p in parts], axis=0),
        SN=np.concatenate([p.SN for p in parts], axis=0),
        R=np.concatenate([p.R for p in parts], axis=0),
        terminal=np.concatenate([p.terminal for p in parts], axis=0),
        truncated=np.concatenate([p.truncated for p in parts], axis=0),
        fragment=np.concatenate([p.fragment for p in parts], axis=0),
        time=np.concatenate([p.time for p in parts], axis=0),
    )


def transition_inventory(fragment_datasets, fragments):
    rows = []
    frag_map = {f.name: f for f in fragments}
    for name, ds in fragment_datasets.items():
        f = frag_map[name]
        rows.append({
            "Fragment": name,
            "Start": f.start,
            "End": f.end,
            "Observed_Hours": f.observed_hours,
            "Eligible_Transitions": len(ds),
            "First_Eligible_t": ds.time[0] if len(ds) else None,
            "Last_Eligible_t": ds.time[-1] if len(ds) else None,
        })
    return pd.DataFrame(rows)
