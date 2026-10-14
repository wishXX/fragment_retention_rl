from pathlib import Path

import pytest

from src.config import load_config
from src.pipeline import FormalPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pipeline():
    cfg = load_config(ROOT / "config" / "experiment.yaml")
    raw_glob = cfg["data"]["raw_glob"]
    supported = {".csv", ".xlsx", ".xlsm", ".xls"}
    files = [
        p for p in ROOT.glob(raw_glob)
        if p.is_file()
        and not p.name.startswith("~$")
        and p.suffix.lower() in supported
    ]
    if not files:
        pytest.skip("Proprietary BMS data are not distributed with the public repository.")
    return FormalPipeline(cfg).build()
