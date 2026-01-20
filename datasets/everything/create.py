import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent

EVIDENCE_INFERENCE_SAMPLE_SIZE = 450


def load_dataset(name: str) -> list:
    path = DATASETS_DIR / name / "data.json"
    with open(path, "r") as f:
        return json.load(f)


def main():
    random.seed(42)

    wikipedia = load_dataset("wikipedia")
    qasper_generated = load_dataset("qasper-generated")
    evidence_inference = load_dataset("evidence-inference")

    evidence_inference_sample = random.sample(evidence_inference, EVIDENCE_INFERENCE_SAMPLE_SIZE)

    combined = wikipedia + qasper_generated + evidence_inference_sample

    print(f"Wikipedia: {len(wikipedia)} records")
    print(f"Qasper-generated: {len(qasper_generated)} records")
    print(f"Evidence-inference: {len(evidence_inference_sample)} records (sampled from {len(evidence_inference)})")
    print(f"Total: {len(combined)} records")

    output_path = SCRIPT_DIR / "data.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
