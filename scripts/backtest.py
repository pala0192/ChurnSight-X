"""Evaluate the current churn feature set across rolling historical periods."""

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
from churn.features import FEATURE_COLUMNS, build_features
from churn.models import train_xgboost
from churn.sampling import build_training_samples, training_reference_dates
from churn.split import rolling_origin_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=PROJECT_ROOT / "data" / "events.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports")
    parser.add_argument("--scale-pos-weight", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-child-weight", type=float)
    args = parser.parse_args()
    parameters = {
        key: value
        for key, value in {
            "scale_pos_weight": args.scale_pos_weight,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
        }.items()
        if value is not None
    }

    events = normalize_events(pd.read_csv(args.events))
    complete_dates = complete_date_range(events)
    complete_events = trim_incomplete_days(events)
    matrix = activity_matrix(complete_events, dates=complete_dates)
    reference_dates = training_reference_dates(matrix.columns)
    samples = build_training_samples(matrix, reference_dates)
    if samples.empty:
        raise SystemExit("No eligible training samples were generated from the input events.")
    feature_samples = build_features(complete_events, samples)
    folds = rolling_origin_splits(reference_dates)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"rolling_backtest_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_number, date_partitions in enumerate(folds, start=1):
        partitions = {
            name: feature_samples.loc[
                pd.to_datetime(feature_samples["reference_date"]).dt.normalize().isin(dates)
            ].copy()
            for name, dates in date_partitions.items()
        }
        result = train_xgboost(
            feature_samples,
            run_dir / f"fold_{fold_number}",
            parameters=parameters,
            partitions=partitions,
        )
        metadata = result["metadata"]
        fold_results.append(
            {
                "fold": fold_number,
                "train": metadata["split"]["train"],
                "validation": metadata["split"]["validation"],
                "test": metadata["split"]["test"],
                "best_iteration": metadata["best_iteration"],
                "threshold": metadata["threshold_selection"]["threshold"],
                "test_metrics": metadata["test_metrics"],
            }
        )
        print(
            f"Fold {fold_number}/{len(folds)}: "
            f"test {metadata['split']['test']['start']}~{metadata['split']['test']['end']}, "
            f"PR-AUC={metadata['test_metrics']['pr_auc']:.4f}, "
            f"precision={metadata['test_metrics']['precision']:.4f}, "
            f"recall={metadata['test_metrics']['recall']:.4f}, "
            f"F1={metadata['test_metrics']['f1']:.4f}",
            flush=True,
        )

    metric_names = ("pr_auc", "precision", "recall", "f1")
    summary = {
        metric: {
            "mean": float(pd.Series([fold["test_metrics"][metric] for fold in fold_results]).mean()),
            "std": float(pd.Series([fold["test_metrics"][metric] for fold in fold_results]).std(ddof=0)),
        }
        for metric in metric_names
    }
    report = {
        "run_id": run_id,
        "events_file": str(args.events.resolve()),
        "parameter_overrides": parameters,
        "feature_columns": list(FEATURE_COLUMNS),
        "fold_count": len(fold_results),
        "summary": summary,
        "folds": fold_results,
    }
    report_path = run_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("\nAverage test metrics across historical folds:")
    for metric, values in summary.items():
        print(f"  {metric}: {values['mean']:.4f} (std {values['std']:.4f})")
    print(f"Report: {report_path}")
    print("This backtest did not change data/models/LATEST; it does not replace the prediction model.")


if __name__ == "__main__":
    main()
