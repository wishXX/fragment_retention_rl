from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class StateScaler:
    scaler: StandardScaler

    @classmethod
    def fit(cls, S):
        return cls(StandardScaler().fit(np.asarray(S, dtype=np.float64)))

    def transform(self, S):
        return self.scaler.transform(np.asarray(S, dtype=np.float64))

    def inverse_transform(self, Z):
        return self.scaler.inverse_transform(np.asarray(Z, dtype=np.float64))

    def stats(self):
        return {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist()}
