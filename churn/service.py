"""Feature-to-prediction workflow for a reference date and event dataset."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from churn.daily import activity_matrix, complete_date_range, normalize_events, trim_incomplete_days
from churn.explain import EXPLANATION_COLUMNS
from churn.features import FEATURE_COLUMNS, build_features
from churn.models import load_model_bundle, predict_features
from churn.sampling import target_mask
from config import OBS_DAYS


def predict_events(
    events: pd.DataFrame,
    reference_date: date | datetime | str | pd.Timestamp,
    model_bundle: dict[str, Any] | str | Path,
) -> pd.DataFrame:
    """Predict eligible visitors from Retail Rocket events at a reference date."""
    reference = pd.Timestamp(reference_date)
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    reference = reference.normalize()
    required_dates = pd.date_range(
        reference - pd.Timedelta(days=OBS_DAYS - 1),
        reference,
    )

    normalized = normalize_events(events)
    available_dates = complete_date_range(normalized)
    missing_dates = required_dates.difference(available_dates)
    if not missing_dates.empty:
        raise ValueError(
            "The observation window is incomplete; missing dates: "
            f"{missing_dates.strftime('%Y-%m-%d').tolist()}"
        )

    complete_events = trim_incomplete_days(normalized)
    observation_events = complete_events.loc[
        complete_events["event_date"].le(reference)
    ]
    matrix = activity_matrix(observation_events, dates=required_dates)
    targets = target_mask(matrix, reference)
    target_ids = targets.index[targets]
    bundle = load_model_bundle(model_bundle) if isinstance(model_bundle, (str, Path)) else model_bundle

    if target_ids.empty:
        return pd.DataFrame(
            columns=[
                "visitor_id",
                "reference_date",
                *FEATURE_COLUMNS,
                "churn_probability",
                "risk_level",
                "predicted_churn",
                *EXPLANATION_COLUMNS,
            ]
        )

    samples = pd.DataFrame(
        {
            "visitor_id": target_ids,
            "reference_date": reference,
        }
    )
    features = build_features(complete_events, samples)
    predictions = predict_features(features, bundle)
    predictions["risk_level"] = pd.Categorical(
        predictions["risk_level"],
        categories=["low", "medium", "high"],
        ordered=True,
    )
    return predictions