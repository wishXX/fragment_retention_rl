from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from .action_space import AffineActionSpace
from .data_loading import load_raw_signals, select_modeling_signals, make_joint_valid_hourly_data
from .fragmentation import build_natural_fragments, order_fragments_primary, order_fragments_temporal_proximity, fragment_inventory
from .reward import RewardFunction
from .transitions import compute_fragment_next_rtp, build_fragment_transitions, concatenate_datasets, transition_inventory
from .preprocessing import StateScaler


class FormalPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_path = None
        self.data = None
        self.valid_times = None
        self.fragments = None
        self.primary_order = None
        self.temporal_order = None
        self.action_space = None
        self.reward_fn = None
        self.fragment_datasets = None
        self.rtp_scale = None

    def build(self):
        path, raw, all_times = load_raw_signals(self.cfg)
        selected = select_modeling_signals(raw, self.cfg)
        valid_times, data = make_joint_valid_hourly_data(all_times, selected, self.cfg)
        fragments = build_natural_fragments(valid_times, int(self.cfg["data"]["sampling_hours"]))
        primary_order = order_fragments_primary(fragments)
        temporal_order = order_fragments_temporal_proximity(fragments)

        action_space = AffineActionSpace(
            self.cfg["action"]["primary_lower_c"],
            self.cfg["action"]["primary_upper_c"],
        )
        longest_rtp = compute_fragment_next_rtp(primary_order[0], data, self.cfg, action_space)
        rtp_scale = float(np.mean(longest_rtp))
        reward_fn = RewardFunction(
            self.cfg["reward"]["target_temp_c"],
            self.cfg["reward"]["underheat_alpha"],
            rtp_scale,
        )

        fragment_datasets = {
            f.name: build_fragment_transitions(f, data, self.cfg, action_space, reward_fn)
            for f in fragments
        }

        self.raw_path = path
        self.data = data
        self.valid_times = valid_times
        self.fragments = fragments
        self.primary_order = primary_order
        self.temporal_order = temporal_order
        self.action_space = action_space
        self.rtp_scale = rtp_scale
        self.reward_fn = reward_fn
        self.fragment_datasets = fragment_datasets
        return self

    def condition_dataset(self, condition: str, path: str = "primary"):
        if not condition.startswith("C"):
            raise ValueError(f"Invalid condition: {condition}")
        k = int(condition[1:])
        order = self.primary_order if path == "primary" else self.temporal_order
        if k < 1 or k > len(order):
            raise ValueError(f"Condition {condition} outside 1..{len(order)}")
        return concatenate_datasets([self.fragment_datasets[f.name] for f in order[:k]])

    def all_dataset(self):
        return concatenate_datasets([self.fragment_datasets[f.name] for f in self.fragments])

    def state_scaler_for(self, condition: str, path: str = "primary"):
        return StateScaler.fit(self.condition_dataset(condition, path).S)

    def training_updates(self, condition: str, path: str = "primary"):
        n = len(self.condition_dataset(condition, path))
        batch = int(self.cfg["offline_rl"]["batch_size"])
        exposures = float(self.cfg["offline_rl"]["benchmark_exposures_per_transition"])
        return int(math.ceil(exposures * n / batch))

    def qa_tables(self):
        inv = transition_inventory(self.fragment_datasets, self.fragments)
        ordered_names = [f.name for f in self.primary_order]
        temporal_names = [f.name for f in self.temporal_order]
        total_n = int(sum(len(ds) for ds in self.fragment_datasets.values()))
        total_h = int(sum(f.observed_hours for f in self.fragments))
        n1 = len(self.fragment_datasets[self.primary_order[0].name])
        h1 = self.primary_order[0].observed_hours

        rows = []
        cum_n = 0
        cum_h = 0
        for i, f in enumerate(self.primary_order, start=1):
            ds = self.fragment_datasets[f.name]
            cum_n += len(ds)
            cum_h += f.observed_hours
            rows.append({
                "Condition": f"C{i}",
                "Added_Fragment": f.name,
                "Fragments_Retained": i,
                "Cumulative_Observed_Hours": cum_h,
                "Cumulative_Eligible_Transitions": cum_n,
                "Added_Transitions_This_Step": len(ds),
                "Total_Transition_Retention_pct": 100.0 * cum_n / total_n,
                "Additional_Fragment_Retention_pct": 100.0 * (cum_n - n1) / (total_n - n1) if total_n > n1 else 0.0,
                "Total_Hours_Retention_pct": 100.0 * cum_h / total_h,
            })
        conditions = pd.DataFrame(rows)
        return inv, conditions, ordered_names, temporal_names

    def assert_frozen_qa(self):
        expected_scale = float(self.cfg["reward"]["expected_rtp_scale"])
        tol = float(self.cfg["reward"]["rtp_scale_tolerance"])
        if abs(self.rtp_scale - expected_scale) > tol:
            raise AssertionError(f"F1 RTP scale mismatch: got {self.rtp_scale}, expected {expected_scale}")

        f1_n = len(self.fragment_datasets["F1"])
        all_n = sum(len(ds) for ds in self.fragment_datasets.values())
        if f1_n != 335:
            raise AssertionError(f"F1 eligible transition count mismatch: {f1_n} != 335")
        if all_n != 1098:
            raise AssertionError(f"Total eligible transition count mismatch: {all_n} != 1098")

        state_dim = next(ds.S.shape[1] for ds in self.fragment_datasets.values() if len(ds))
        if state_dim != 80:
            raise AssertionError(f"State dimension mismatch: {state_dim} != 80")
        if any(np.any(ds.terminal != 0.0) for ds in self.fragment_datasets.values()):
            raise AssertionError("Telemetry fragments must not be encoded as terminals")
        return True
