"""XGBoost training, threshold selection, and portable model artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from churn.features import FEATURE_COLUMNS
from churn.split import split_samples
from config import RISK_HIGH_MIN, RISK_MEDIUM_MIN, TARGET_RECALL


XGBOOST_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "xgboost.json"
XGBOOST_CONFIG = json.loads(XGBOOST_CONFIG_PATH.read_text(encoding="utf-8"))
DEFAULT_XGBOOST_PARAMETERS: dict[str, Any] = XGBOOST_CONFIG["parameters"]
EARLY_STOPPING_ROUNDS = int(XGBOOST_CONFIG["early_stopping_rounds"])


def _choose_threshold(
    actual: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> dict[str, float | bool]:
    candidates = np.unique(np.concatenate(([0.0], probabilities)))
    results = []
    for threshold in candidates:
        predicted = probabilities >= threshold
        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(actual, predicted, zero_division=0)),
                "recall": float(recall_score(actual, predicted, zero_division=0)),
            }
        )

    qualifying = [row for row in results if row["recall"] >= target_recall]
    recall_target_met = bool(qualifying)
    candidates_to_rank = qualifying or results
    selected = max(
        candidates_to_rank,
        key=lambda row: (row["precision"], row["recall"], row["threshold"]),
    )
    return {**selected, "target_recall_met": recall_target_met}


def _metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probabilities >= threshold
    return {
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def _date_summary(samples: pd.DataFrame) -> dict[str, str | int]:
    dates = pd.to_datetime(samples["reference_date"]).dt.normalize()
    return {
        "rows": int(len(samples)),
        "unique_reference_dates": int(dates.nunique()),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def train_xgboost(
    samples: pd.DataFrame,
    output_dir: str | Path,
    parameters: dict[str, Any] | None = None,
    target_recall: float = TARGET_RECALL,
    partitions: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Train a temporal XGBoost model and save binary plus inspectable JSON artifacts."""
    required = {*FEATURE_COLUMNS, "churn", "reference_date"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Training samples are missing columns: {sorted(missing)}")

    partitions = partitions or split_samples(samples)
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Partitions must contain exactly train, validation, and test samples.")
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"The {name} partition has no samples.")

    train_y = partitions["train"]["churn"].astype(int).to_numpy()
    validation_y = partitions["validation"]["churn"].astype(int).to_numpy()
    test_y = partitions["test"]["churn"].astype(int).to_numpy()
    if len(np.unique(train_y)) < 2:
        raise ValueError("The training partition must contain both churn labels.")
    if len(np.unique(validation_y)) < 2 or len(np.unique(test_y)) < 2:
        raise ValueError("Validation and test partitions must each contain both churn labels.")

    positive_count = int(train_y.sum())
    negative_count = int(len(train_y) - positive_count)
    model_parameters = {
        **DEFAULT_XGBOOST_PARAMETERS,
        **(parameters or {}),
        "scale_pos_weight": (parameters or {}).get(
            "scale_pos_weight",
            negative_count / positive_count,
        ),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    model = XGBClassifier(**model_parameters)
    model.fit(
        partitions["train"].loc[:, FEATURE_COLUMNS],
        train_y,
        eval_set=[
            (
                partitions["validation"].loc[:, FEATURE_COLUMNS],
                validation_y,
            )
        ],
        verbose=False,
    )

    validation_probabilities = model.predict_proba(
        partitions["validation"].loc[:, FEATURE_COLUMNS]
    )[:, 1]
    threshold_result = _choose_threshold(validation_y, validation_probabilities, target_recall)
    threshold = float(threshold_result["threshold"])
    test_probabilities = model.predict_proba(partitions["test"].loc[:, FEATURE_COLUMNS])[:, 1]

    booster = model.get_booster()
    importance = {
        feature: {
            "gain": float(booster.get_score(importance_type="gain").get(feature, 0.0)),
            "weight": float(booster.get_score(importance_type="weight").get(feature, 0.0)),
            "cover": float(booster.get_score(importance_type="cover").get(feature, 0.0)),
        }
        for feature in FEATURE_COLUMNS
    }
    importance = dict(
        sorted(importance.items(), key=lambda item: item[1]["gain"], reverse=True)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.json"
    bundle_path = output_path / "bundle.joblib"
    config_path = output_path / "model_config.json"
    meta_path = output_path / "meta.json"
    model.save_model(model_path)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
        },
        bundle_path,
    )

    metadata = {
        "model_type": "xgboost.XGBClassifier",
        "xgboost_version": xgboost.__version__,
        "feature_columns": list(FEATURE_COLUMNS),
        "hyperparameters": model_parameters,
        "best_iteration": int(model.best_iteration),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "threshold_selection": {
            "target_recall": target_recall,
            **threshold_result,
        },
        "split": {
            name: _date_summary(partition)
            for name, partition in partitions.items()
        },
        "validation_metrics": _metrics(
            validation_y,
            validation_probabilities,
            threshold,
        ),
        "test_metrics": _metrics(test_y, test_probabilities, threshold),
        "feature_importance": importance,
        "artifacts": {
            "model_json": model_path.name,
            "bundle": bundle_path.name,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    for json_path, value in ((config_path, metadata), (meta_path, metadata)):
        json_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    return {"model": model, "bundle_path": bundle_path, "metadata": metadata}


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trained model bundle created by train_xgboost."""
    bundle = joblib.load(path)
    if tuple(bundle.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("The model bundle feature columns do not match this application.")
    return bundle


def predict_features(
    feature_rows: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Add churn probability, risk band, and threshold-based prediction to features."""
    if tuple(bundle.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("The model bundle feature columns do not match this application.")

    probabilities = bundle["model"].predict_proba(
        feature_rows.loc[:, FEATURE_COLUMNS]
    )[:, 1]
    threshold = float(bundle["threshold"])
    results = feature_rows.copy()
    results["churn_probability"] = probabilities
    results["risk_level"] = np.select(
        [probabilities >= RISK_HIGH_MIN, probabilities >= RISK_MEDIUM_MIN],
        ["high", "medium"],
        default="low",
    )
    results["predicted_churn"] = probabilities >= threshold
    return results
