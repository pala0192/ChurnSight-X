"""Search a small reproducible XGBoost parameter set using rolling validation."""

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from churn.daily import activity_matrix, complete_date_range, normalize_events, trim_incomplete_days
from churn.features import FEATURE_COLUMNS, build_features
from churn.models import DEFAULT_XGBOOST_PARAMETERS, train_xgboost
from churn.sampling import build_training_samples, training_reference_dates
from churn.split import rolling_origin_splits


def _candidate_parameters(trials: int, seed: int) -> list[dict]:
    """Return default plus reproducibly sampled, bounded parameter combinations."""
    space = {
        "max_depth": (2, 3, 4, 5),
        "learning_rate": (0.03, 0.05, 0.1),
        "min_child_weight": (1, 3, 5, 10),
        "subsample": (0.7, 0.85, 1.0),
        "colsample_bytree": (0.7, 0.85, 1.0),
        "scale_pos_weight": (None, 0.3, 0.6, 1.0),
    }
    candidates = [{}]
    rng = random.Random(seed)
    seen = {()}
    while len(candidates) < trials:
        candidate = {key: rng.choice(values) for key, values in space.items()}
        candidate = {key: value for key, value in candidate.items() if value is not None}
        signature = tuple(sorted(candidate.items()))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=PROJECT_ROOT / "data" / "events.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "reports")
    parser.add_argument("--trials", type=int, default=6, help="Total combinations, including defaults")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 2:
        raise SystemExit("--trials must be at least 2 so a candidate can be compared with defaults.")

    print("Loading event data and building features...", flush=True)
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
    candidates = _candidate_parameters(args.trials, args.seed)

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"hyperparameter_search_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for candidate_number, parameters in enumerate(candidates, start=1):
        validation_rows = []
        test_rows = []
        print(
            f"\nCandidate {candidate_number}/{len(candidates)}: "
            f"{parameters or 'current defaults'}",
            flush=True,
        )
        for fold_number, date_partitions in enumerate(folds, start=1):
            partitions = {
                name: feature_samples.loc[
                    pd.to_datetime(feature_samples["reference_date"]).dt.normalize().isin(dates)
                ].copy()
                for name, dates in date_partitions.items()
            }
            result = train_xgboost(
                feature_samples,
                run_dir / f"candidate_{candidate_number}" / f"fold_{fold_number}",
                parameters=parameters,
                partitions=partitions,
            )
            metadata = result["metadata"]
            validation_rows.append(metadata["validation_metrics"])
            test_rows.append(metadata["test_metrics"])
            print(
                f"  fold {fold_number}/{len(folds)} validation "
                f"PR-AUC={metadata['validation_metrics']['pr_auc']:.4f}",
                flush=True,
            )

        validation_mean = {
            metric: float(pd.Series([row[metric] for row in validation_rows]).mean())
            for metric in ("pr_auc", "precision", "recall", "f1")
        }
        test_mean = {
            metric: float(pd.Series([row[metric] for row in test_rows]).mean())
            for metric in ("pr_auc", "precision", "recall", "f1")
        }
        results.append(
            {
                "candidate": candidate_number,
                "parameters": parameters,
                "validation_mean": validation_mean,
                "test_mean": test_mean,
            }
        )
        print(f"  mean validation PR-AUC={validation_mean['pr_auc']:.4f}", flush=True)

    winner = max(results, key=lambda row: row["validation_mean"]["pr_auc"])
    baseline = results[0]
    report = {
        "run_id": run_id,
        "events_file": str(args.events.resolve()),
        "feature_columns": list(FEATURE_COLUMNS),
        "selection_metric": "mean rolling validation PR-AUC",
        "seed": args.seed,
        "candidate_count": len(results),
        "candidate_results": results,
        "selected_candidate": winner["candidate"],
        "selected_parameters": winner["parameters"],
        "baseline_test_mean": baseline["test_mean"],
        "selected_test_mean": winner["test_mean"],
        "default_parameters": DEFAULT_XGBOOST_PARAMETERS,
    }
    report_path = run_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print("\nSelected using validation PR-AUC only:")
    print(json.dumps({"candidate": winner["candidate"], "parameters": winner["parameters"]}, indent=2))
    print("Baseline rolling test mean:", json.dumps(baseline["test_mean"], indent=2))
    print("Selected rolling test mean:", json.dumps(winner["test_mean"], indent=2))
    print(f"Report: {report_path}")
    print("The search did not change data/models/LATEST or configs/xgboost.json.")


if __name__ == "__main__":
    main()
