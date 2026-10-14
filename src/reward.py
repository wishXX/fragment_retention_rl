from __future__ import annotations
import numpy as np


class RewardFunction:
    def __init__(self, target_temp_c: float, underheat_alpha: float, rtp_scale: float):
        self.target_temp_c = float(target_temp_c)
        self.underheat_alpha = float(underheat_alpha)
        self.rtp_scale = float(rtp_scale)
        if self.rtp_scale <= 0:
            raise ValueError("rtp_scale must be positive")

    def components(self, next_comfort_u, next_rtp):
        comfort_u = np.asarray(next_comfort_u, dtype=np.float64)
        rtp = np.asarray(next_rtp, dtype=np.float64)
        energy = rtp / self.rtp_scale
        underheat = self.underheat_alpha * np.maximum(comfort_u, 0.0)
        return energy, underheat

    def __call__(self, next_comfort_u, next_rtp):
        energy, underheat = self.components(next_comfort_u, next_rtp)
        return -(energy + underheat)
