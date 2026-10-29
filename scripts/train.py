"""Train and version the eleven-feature XGBoost churn model."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from churn.daily import read_events
from churn.dataset import feature_samples_from_events, read_feature_samples
from churn.models import DEFAULT_XGBOOST_PARAMETERS, train_xgboost
from churn.split import DATASET_COLUMN
from config import MODELS_DIR, REFERENCE_STRIDE_DAYS, TARGET_RECALL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        nargs="+",
        default=[],
        help="Event CSV files or directories of CSVs",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        nargs="+",
        default=[],
        help="Labeled feature tables in the target_samples.csv layout",
    )
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
    if not args.events and not args.samples:
        parser.error("give --events, --samples, or both")

    tables = []
    if args.events:
        events_samples = feature_samples_from_events(read_events(args.events), stride_days=args.stride)
        tables.append(events_samples.assign(**{DATASET_COLUMN: "events"}))
    if args.samples:
        tables.append(read_feature_samples(args.samples))
    feature_samples = pd.concat(tables, ignore_index=True)
    if feature_samples.empty:
        raise SystemExit("No eligible training samples were generated from the inputs.")

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
                "split": result["metadata"]["split"],
                "threshold": result["metadata"]["threshold_selection"],
                "test_metrics": result["metadata"]["test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()