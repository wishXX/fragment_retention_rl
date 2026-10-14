from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import pandas as pd


@dataclass(frozen=True)
class Fragment:
    name: str
    start: object
    end: object
    times: tuple

    @property
    def observed_hours(self) -> int:
        return len(self.times)


def build_natural_fragments(valid_times, sampling_hours: int = 1):
    valid_times = sorted(valid_times)
    groups = []
    current = []
    prev = None
    step = timedelta(hours=sampling_hours)
    for t in valid_times:
        if prev is None or t - prev == step:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
        prev = t
    if current:
        groups.append(current)

    ranked = sorted(groups, key=lambda g: (-len(g), g[0]))
    return [
        Fragment(name=f"F{i+1}", start=g[0], end=g[-1], times=tuple(g))
        for i, g in enumerate(ranked)
    ]


def order_fragments_primary(fragments):
    longest = fragments[0]
    rest = sorted(fragments[1:], key=lambda f: (-f.observed_hours, f.start))
    return [longest] + rest


def _gap_from_longest_hours(fragment, longest):
    if fragment.end < longest.start:
        return (longest.start - fragment.end).total_seconds() / 3600.0
    if fragment.start > longest.end:
        return (fragment.start - longest.end).total_seconds() / 3600.0
    return 0.0


def order_fragments_temporal_proximity(fragments):
    longest = fragments[0]
    rest = sorted(
        fragments[1:],
        key=lambda f: (_gap_from_longest_hours(f, longest), -f.observed_hours, f.start),
    )
    return [longest] + rest


def fragment_inventory(fragments):
    return pd.DataFrame(
        [
            {
                "Fragment": f.name,
                "Start": f.start,
                "End": f.end,
                "Observed_Hours": f.observed_hours,
            }
            for f in fragments
        ]
    )
