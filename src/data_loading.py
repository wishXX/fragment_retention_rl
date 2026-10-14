from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


def _find_single_raw_file(project_root: Path, raw_glob: str) -> Path:
    supported = {".csv", ".xlsx", ".xlsm", ".xls"}
    files = [
        p for p in project_root.glob(raw_glob)
        if p.is_file() and not p.name.startswith("~$") and p.suffix.lower() in supported
    ]
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one raw data file matching {raw_glob}, found {len(files)}: "
            f"{[p.name for p in files]}"
        )
    return files[0]


def _read_header(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, nrows=0).columns.tolist()
    raise ValueError(f"Unsupported raw data format: {path.suffix}")


def _signal_columns(headers, cfg: dict):
    phrases = cfg["signals"]["phrases"]
    cols = []
    for h in headers:
        low = str(h).lower()
        if phrases["fts"] in low:
            cols.append(h)
        elif phrases["pm"] in low:
            cols.append(h)
        elif phrases["indoor"] in low and "setpoint" not in low:
            cols.append(h)
        elif phrases["outdoor"] in low:
            cols.append(h)
        elif any(p in low for p in phrases["rtp_any"]):
            cols.append(h)
    return list(dict.fromkeys(cols))


def _needed_controllers(cfg: dict):
    sig = cfg["signals"]
    controllers = []
    for key in [
        "action_controllers",
        "rtp_controllers",
        "indoor_controllers",
        "outdoor_controllers",
    ]:
        controllers.extend(sig[key])
    return set(controllers)


def _read_required_table(path: Path, cfg: dict):
    headers = _read_header(path)
    timestamp_col = cfg["data"]["timestamp_column"]
    controller_col = cfg["data"]["controller_column"]
    signal_cols = _signal_columns(headers, cfg)
    usecols = [controller_col, timestamp_col] + signal_cols

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    else:
        df = pd.read_excel(path, usecols=usecols, dtype=object)

    df = df[df[controller_col].isin(_needed_controllers(cfg))].copy()
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        format=cfg["data"]["timestamp_format"],
        errors="coerce",
    )
    df = df[df[timestamp_col].notna()].copy()
    return df, signal_cols


def load_raw_signals(cfg: dict):
    """Load only the telemetry signals required by the formal pipeline.

    Parsing is vectorized for reproducibility and performance. The returned
    dictionary exposes `(controller, variable) -> {time: value}` for the formal pipeline
    contract used by the downstream signal-selection code.
    """
    root = Path(cfg["_project_root"])
    path = _find_single_raw_file(root, cfg["data"]["raw_glob"])
    timestamp_col = cfg["data"]["timestamp_column"]
    controller_col = cfg["data"]["controller_column"]

    df, signal_cols = _read_required_table(path, cfg)
    raw = defaultdict(dict)

    for col in signal_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        if not mask.any():
            continue
        sub = pd.DataFrame({
            "controller": df.loc[mask, controller_col].astype(str),
            "time": df.loc[mask, timestamp_col],
            "value": values.loc[mask].astype(float),
        })
        for controller, group in sub.groupby("controller", sort=False):
            raw[(controller, col)] = dict(zip(group["time"].array.to_pydatetime(), group["value"]))

    all_times = sorted(df[timestamp_col].array.to_pydatetime().tolist())
    # `all_times` may contain duplicates across controllers. The original loader
    # returned unique timestamps, so preserve that contract explicitly.
    all_times = sorted(set(all_times))
    return path, raw, all_times


def _exact_item(raw, controller: str, phrase: str):
    matches = [(cc, var, d) for (cc, var), d in raw.items() if cc == controller and phrase in var.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one signal for {controller} / {phrase}, found {len(matches)}")
    return matches[0]


def select_modeling_signals(raw, cfg: dict):
    sig = cfg["signals"]
    phrases = sig["phrases"]
    indoor_items = [_exact_item(raw, c, phrases["indoor"]) for c in sig["indoor_controllers"]]
    outdoor_items = [_exact_item(raw, c, phrases["outdoor"]) for c in sig["outdoor_controllers"]]
    fts_items = [_exact_item(raw, c, phrases["fts"]) for c in sig["action_controllers"]]
    pm_items = [_exact_item(raw, c, phrases["pm"]) for c in sig["action_controllers"]]

    rtp_items = []
    for c in sig["rtp_controllers"]:
        matches = []
        for (cc, var), d in raw.items():
            if cc == c and any(p in var.lower() for p in phrases["rtp_any"]):
                matches.append((cc, var, d))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one RTP signal for {c}, found {len(matches)}")
        rtp_items.append(matches[0])

    return indoor_items, outdoor_items, fts_items, pm_items, rtp_items


def make_joint_valid_hourly_data(all_times, selected, cfg: dict):
    indoor_items, outdoor_items, fts_items, pm_items, rtp_items = selected

    def available(t) -> bool:
        groups = [indoor_items, outdoor_items, fts_items, pm_items, rtp_items]
        return all(t in d for group in groups for _, _, d in group)

    valid_times = [t for t in all_times if available(t)]
    data = {}
    target_temp = float(cfg["reward"]["target_temp_c"])
    for t in valid_times:
        tin_channels = np.asarray([d[t] for _, _, d in indoor_items], dtype=np.float64)
        tout_channels = np.asarray([d[t] for _, _, d in outdoor_items], dtype=np.float64)
        rtp_channels = np.asarray([d[t] for _, _, d in rtp_items], dtype=np.float64)
        tin_mean = float(np.mean(tin_channels))
        tin_min = float(np.min(tin_channels))
        tin_std = float(np.std(tin_channels, ddof=0))
        comfort_u = float(np.mean(np.maximum(target_temp - tin_channels, 0.0) ** 2))
        tout_mean = float(np.mean(tout_channels))
        tout_std = float(np.std(tout_channels, ddof=0))
        rtp_total = float(np.sum(rtp_channels))
        fts = np.array([d[t] for _, _, d in fts_items], dtype=np.float64)
        pm = np.array([d[t] for _, _, d in pm_items], dtype=np.float64)
        data[t] = {
            "tin_channels": tin_channels, "tin_mean": tin_mean, "tin_min": tin_min,
            "tin_std": tin_std, "comfort_u": comfort_u,
            "tout_channels": tout_channels, "tout_mean": tout_mean, "tout_std": tout_std,
            "rtp_channels": rtp_channels, "rtp_total": rtp_total,
            "tin": tin_mean, "tout": tout_mean, "rtp": rtp_total,
            "fts": fts, "pm": pm,
        }
    return sorted(valid_times), data
