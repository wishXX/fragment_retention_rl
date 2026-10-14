from __future__ import annotations
import numpy as np


class AffineActionSpace:
    def __init__(self, lower_c: float, upper_c: float):
        if upper_c <= lower_c:
            raise ValueError("upper_c must exceed lower_c")
        self.lower_c = float(lower_c)
        self.upper_c = float(upper_c)

    def encode(self, physical):
        x = np.asarray(physical, dtype=np.float64)
        return 2.0 * (x - self.lower_c) / (self.upper_c - self.lower_c) - 1.0

    def decode(self, normalized):
        a = np.asarray(normalized, dtype=np.float64)
        return self.lower_c + 0.5 * (a + 1.0) * (self.upper_c - self.lower_c)

    def contains(self, physical, atol: float = 1e-12):
        x = np.asarray(physical, dtype=np.float64)
        return bool(np.all(x >= self.lower_c - atol) and np.all(x <= self.upper_c + atol))

    def clip_normalized(self, normalized):
        return np.clip(np.asarray(normalized), -1.0, 1.0)
