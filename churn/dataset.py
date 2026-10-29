"""Labeled feature tables: build them from events and read or write them as CSV."""

from pathlib import Path

import pandas as pd

from churn.daily import activity_matrix, complete_date_range, normalize_events, trim_incomplete_days
from churn.features import FEATURE_COLUMNS, build_features
from churn.sampling import build_training_samples, training_reference_dates
from churn.split import DATASET_COLUMN
from config import REFERENCE_STRIDE_DAYS


SAMPLE_COLUMNS = ("visitor_id", "reference_date", *FEATURE_COLUMNS, "churn")


def feature_samples_from_events(
    events: pd.DataFrame,
    stride_days: int = REFERENCE_STRIDE_DAYS,
) -> pd.DataFrame:
    """Return one labeled feature row per eligible visitor and reference date."""
    events = normalize_events(events)
    complete_dates = complete_date_range(events)
    complete_events = trim_incomplete_days(events)
    matrix = activity_matrix(complete_events, dates=complete_dates)
    reference_dates = training_reference_dates(matrix.columns, stride_days=stride_days)
    labeled_samples = build_training_samples(matrix, reference_dates)
    if labeled_samples.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)

    samples = build_features(complete_events, labeled_samples)
    samples = samples.sort_values(["reference_date", "visitor_id"], kind="stable")
    return samples.loc[:, SAMPLE_COLUMNS].reset_index(drop=True)


def _number_text(value: float) -> str:
    """Shortest round-trip text with whole numbers written without '.0'; NaN becomes empty."""
    if pd.isna(value):
        return ""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def write_feature_samples(samples: pd.DataFrame, path: str | Path) -> None:
    """Write a feature table in the target_samples.csv layout."""
    output = samples.loc[:, SAMPLE_COLUMNS].copy()
    output["reference_date"] = pd.to_datetime(output["reference_date"]).dt.strftime("%Y-%m-%d")
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].map(_number_text)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def read_feature_samples(paths: list[str | Path]) -> pd.DataFrame:
    """Read feature tables in the target_samples.csv layout, tagging rows with their file stem.

    The tag lands in the dataset column so training splits each file on its own dates.
    """
    tables = []
    for path in map(Path, paths):
        table = pd.read_csv(path)
        missing = set(SAMPLE_COLUMNS).difference(table.columns)
        if missing:
            raise ValueError(f"{path} is missing feature-table columns: {sorted(missing)}")
        tables.append(table.loc[:, SAMPLE_COLUMNS].assign(**{DATASET_COLUMN: path.stem}))
    samples = pd.concat(tables, ignore_index=True)
    samples["reference_date"] = pd.to_datetime(samples["reference_date"]).dt.normalize()
    return samples
