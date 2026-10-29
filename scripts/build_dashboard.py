"""Write the GitHub Pages dashboard data (docs/data.js) from a model and its predictions."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from churn.dataset import read_feature_samples
from churn.explain import FEATURE_LABELS
from churn.features import FEATURE_COLUMNS
from churn.models import load_model_bundle
from churn.split import split_samples
from config import MODELS_DIR, RISK_HIGH_MIN, RISK_MEDIUM_MIN

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "data.js"


def display_id(visitor_id: object) -> str:
    """Public customer label: the visitor id with its last two characters masked."""
    text = str(visitor_id)
    return text[:-2] + "**" if len(text) > 2 else "**"


def scores(actual: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0,
        "accuracy": float((tp + tn) / len(actual)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def partition_summary(partition: pd.DataFrame, bundle: dict) -> dict:
    actual = partition["churn"].astype(int).to_numpy()
    probabilities = bundle["model"].predict_proba(partition.loc[:, FEATURE_COLUMNS])[:, 1]
    dates = pd.to_datetime(partition["reference_date"])
    summary = {
        "rows": int(len(partition)),
        "churn_rate": float(actual.mean()),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "model": scores(actual, probabilities, float(bundle["threshold"])),
        "all_churn": scores(actual, np.ones(len(actual)), 0.5),
    }
    bands = np.select(
        [probabilities >= RISK_HIGH_MIN, probabilities >= RISK_MEDIUM_MIN],
        ["high", "medium"],
        default="low",
    )
    summary["risk_bands"] = [
        {
            "level": level,
            "count": int((bands == level).sum()),
            "actual_churn_rate": float(actual[bands == level].mean()) if (bands == level).any() else None,
        }
        for level in ("high", "medium", "low")
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path, nargs="+", help="Labeled feature tables used for training")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction CSV written by predict.py")
    parser.add_argument("--model-version", help="Model version directory name (default: LATEST)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    version = args.model_version or (args.models_dir / "LATEST").read_text(encoding="ascii").strip()
    model_dir = args.models_dir / version
    bundle = load_model_bundle(model_dir / "bundle.joblib")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    partitions = split_samples(read_feature_samples(args.samples))

    gains = {feature: values["gain"] for feature, values in meta["feature_importance"].items()}
    total_gain = sum(gains.values()) or 1.0

    predictions = pd.read_csv(args.predictions).sort_values("churn_probability", ascending=False)
    customers = [
        {
            "id": display_id(row.visitor_id),
            "probability": round(float(row.churn_probability), 4),
            "risk": row.risk_level,
            "churn": bool(row.predicted_churn),
            "factor": FEATURE_LABELS[row.top_factor],
            "explanation": row.explanation,
        }
        for row in predictions.itertuples(index=False)
    ]
    factor_counts = predictions["top_factor"].value_counts()

    data = {
        "model_version": version,
        "reference_date": str(predictions["reference_date"].iloc[0])[:10],
        "threshold": float(bundle["threshold"]),
        "risk_high_min": RISK_HIGH_MIN,
        "risk_medium_min": RISK_MEDIUM_MIN,
        "best_iteration": meta["best_iteration"],
        "hyperparameters": {
            key: meta["hyperparameters"][key]
            for key in ("learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda")
        },
        "early_stopping_rounds": meta["early_stopping_rounds"],
        "train": {"rows": meta["split"]["train"]["rows"], "start": meta["split"]["train"]["start"], "end": meta["split"]["train"]["end"]},
        "validation": partition_summary(partitions["validation"], bundle),
        "test": partition_summary(partitions["test"], bundle),
        "feature_importance": [
            {"feature": FEATURE_LABELS[feature], "share": gain / total_gain}
            for feature, gain in sorted(gains.items(), key=lambda item: item[1], reverse=True)
        ],
        "factor_counts": [
            {"feature": FEATURE_LABELS[feature], "count": int(count)} for feature, count in factor_counts.items()
        ],
        "customers": customers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.DASHBOARD = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({len(customers)} customers, model {version})")


if __name__ == "__main__":
    main()
