"""Predict churn for eligible visitors at a reference date with a trained model."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from churn.models import load_model_bundle
from churn.service import predict_events
from config import MODELS_DIR, PREDICTIONS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="Retail Rocket events CSV")
    parser.add_argument("--date", required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--model-version", help="Model version directory name (default: LATEST)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    version = args.model_version or (args.models_dir / "LATEST").read_text(encoding="ascii").strip()
    bundle = load_model_bundle(args.models_dir / version / "bundle.joblib")
    reference_date = pd.Timestamp(args.date).normalize()
    predictions = predict_events(pd.read_csv(args.events), reference_date, bundle)

    output = args.output or PREDICTIONS_DIR / f"{reference_date.date()}_{version}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)

    print(
        json.dumps(
            {
                "model_version": version,
                "reference_date": reference_date.date().isoformat(),
                "output": str(output),
                "target_count": len(predictions),
                "predicted_churn_count": int(predictions["predicted_churn"].sum()),
                "risk_level_counts": {
                    level: int(count)
                    for level, count in predictions["risk_level"].value_counts(sort=False).items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
