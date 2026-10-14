from pathlib import Path
import importlib.util
import numpy as np


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "04_run_formal_retention.py"
    spec = importlib.util.spec_from_file_location("formal_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(fragment, value):
    return {
        "Evaluator": "ARX",
        "Fragment": fragment,
        "Origin_Time": "2024-01-01 00:00:00",
        "Discounted_Return": value,
        "Mean_Reward": value,
        "Mean_Energy_Component": value,
        "Mean_Underheat_Component": value,
        "Mean_Support_Distance": value,
        "Max_Support_Distance": value,
    }


def test_formal_aggregation_weights_fragments_equally():
    runner = load_runner()
    rows = [row("F1", 0.0), row("F1", 0.0), row("F1", 0.0), row("F2", 10.0)]
    result = runner.aggregate(rows, "C1", "IQL", "official", 1)[0]
    assert np.isclose(np.mean([r["Discounted_Return"] for r in rows]), 2.5)
    assert np.isclose(result["Mean_Discounted_Return"], 5.0)
    assert result["N_Origins"] == 4
    assert result["N_Evaluation_Fragments"] == 2
    assert result["Aggregation_Unit"] == "equal_weight_evaluation_fragment"


def test_formal_aggregation_applies_same_fragment_weighting_to_diagnostics():
    runner = load_runner()
    rows = [row("F1", 2.0), row("F1", 4.0), row("F2", 10.0)]
    result = runner.aggregate(rows, "C1", "IQL", "official", 1)[0]
    expected = np.mean([3.0, 10.0])
    assert np.isclose(result["Mean_Reward"], expected)
    assert np.isclose(result["Mean_Energy_Component"], expected)
    assert np.isclose(result["Mean_Underheat_Component"], expected)
    assert np.isclose(result["Mean_Support_Distance"], expected)
    assert np.isclose(result["Max_Support_Distance"], 10.0)


def test_origin_metadata_preserves_fragment_and_origin_time():
    runner = load_runner()
    rows = [row("F7", 1.25)]
    enriched = runner.attach_origin_metadata(rows, "C3", "CQL", "lr0.0001_w5", 4)[0]
    assert enriched["Condition"] == "C3"
    assert enriched["Algorithm"] == "CQL"
    assert enriched["Seed"] == 4
    assert enriched["Fragment"] == "F7"
    assert "Origin_Time" in enriched
