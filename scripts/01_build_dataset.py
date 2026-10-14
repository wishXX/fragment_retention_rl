from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import FormalPipeline
from src.utils import ensure_dir, save_json


def main():
    cfg = load_config(ROOT / "config" / "experiment.yaml")
    pipe = FormalPipeline(cfg).build()
    pipe.assert_frozen_qa()

    qa_dir = ensure_dir(ROOT / "outputs" / "qa")
    frag_dir = ensure_dir(ROOT / "outputs" / "fragments")

    inv, conditions, _, _ = pipe.qa_tables()
    inv.to_csv(frag_dir / "formal_fragment_inventory.csv", index=False)
    conditions.to_csv(frag_dir / "formal_cumulative_conditions.csv", index=False)
    save_json({
        "joint_valid_hours": len(pipe.valid_times),
        "natural_fragments": len(pipe.fragments),
        "f1_rtp_scale": pipe.rtp_scale,
        "state_dim": next(ds.S.shape[1] for ds in pipe.fragment_datasets.values() if len(ds)),
        "action_dim": 8,
        "f1_transitions": len(pipe.fragment_datasets["F1"]),
        "all_transitions": sum(len(ds) for ds in pipe.fragment_datasets.values()),
    }, qa_dir / "dataset_qa.json")

    print("Formal dataset QA passed")
    print(f"Raw file: {pipe.raw_path.name}")
    print(f"Joint-valid hours: {len(pipe.valid_times)}")
    print(f"Natural fragments: {len(pipe.fragments)}")
    print(f"F1 transitions: {len(pipe.fragment_datasets['F1'])}")
    print(f"All transitions: {sum(len(ds) for ds in pipe.fragment_datasets.values())}")
    print(f"F1 RTP scale: {pipe.rtp_scale:.12f}")


if __name__ == "__main__":
    main()
