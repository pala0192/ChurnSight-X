"""Train and version the 12-feature XGBoost churn model."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from churn.daily import activity_matrix, complete_date_range, normalize_events, trim_incomplete_days
from churn.features import build_features
from churn.models import DEFAULT_XGBOOST_PARAMETERS, train_xgboost
from churn.sampling import build_training_samples, training_reference_dates
from config import MODELS_DIR, REFERENCE_STRIDE_DAYS, TARGET_RECALL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="Retail Rocket events CSV")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--stride", type=int, default=REFERENCE_STRIDE_DAYS)
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_XGBOOST_PARAMETERS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=DEFAULT_XGBOOST_PARAMETERS["max_depth"])
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_XGBOOST_PARAMETERS["learning_rate"],
    )
    args = parser.parse_args()

    events = normalize_events(pd.read_csv(args.events))
    complete_dates = complete_date_range(events)
    complete_events = trim_incomplete_days(events)
    matrix = activity_matrix(complete_events, dates=complete_dates)
    reference_dates = training_reference_dates(matrix.columns, stride_days=args.stride)
    labeled_samples = build_training_samples(matrix, reference_dates)
    if labeled_samples.empty:
        raise SystemExit("No eligible training samples were generated from the input events.")

    feature_samples = build_features(complete_events, labeled_samples)
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    version_dir = args.models_dir / version
    result = train_xgboost(
        feature_samples,
        version_dir,
        parameters={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        target_recall=args.target_recall,
    )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    (args.models_dir / "LATEST").write_text(version, encoding="ascii")

    print(
        json.dumps(
            {
                "model_version": version,
                "model_dir": str(version_dir),
                "sample_count": len(feature_samples),
                "threshold": result["metadata"]["threshold_selection"],
                "test_metrics": result["metadata"]["test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
