from pathlib import Path
import json
import platform
import subprocess
import sys
import numpy
import pandas
import sklearn
import xgboost
import torch
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]


def main():
    defaults = XGBRegressor().get_params(deep=False)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
        "xgbregressor_sklearn_wrapper_defaults": defaults,
    }
    out = ROOT / "outputs" / "qa" / "runtime_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    freeze_out = ROOT / "outputs" / "qa" / "pip_freeze.txt"
    freeze_out.write_text(freeze, encoding="utf-8")

    print(out)
    print(freeze_out)


if __name__ == "__main__":
    main()
