"""Build the nine requested 30-day churn prediction features."""

import numpy as np
import pandas as pd

from config import OBS_DAYS
from churn.daily import normalize_events


FEATURE_COLUMNS = (
    "active_days_30d",
    "event_count_30d",
    "view_count_30d",
    "cart_count_30d",
    "transaction_count_30d",
    "unique_items_30d",
    "days_since_last_purchase",
    "active_day_gap_mean",
    "active_day_gap_std",
)
INTEGER_FEATURES = FEATURE_COLUMNS[:7]
FLOAT_FEATURES = FEATURE_COLUMNS[7:]


def _window_features(window: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate one reference date's observation-window events per sample."""
    flags = window.assign(
        is_view=window["event"].eq("view").astype(bool),
        is_cart=window["event"].eq("addtocart").astype(bool),
        is_transaction=window["event"].eq("transaction").astype(bool),
    )
    grouped = flags.groupby("sample_idx")
    features = pd.DataFrame(
        {
            "event_count_30d": grouped.size(),
            "view_count_30d": grouped["is_view"].sum(),
            "cart_count_30d": grouped["is_cart"].sum(),
            "transaction_count_30d": grouped["is_transaction"].sum(),
            "unique_items_30d": grouped["itemid"].nunique(),
        }
    )

    last_purchase = flags.loc[flags["is_transaction"]].groupby("sample_idx")["event_date"].max()
    features["days_since_last_purchase"] = (reference_date - last_purchase).dt.days

    active_dates = flags.loc[:, ["sample_idx", "event_date"]].drop_duplicates()
    active_dates = active_dates.sort_values(["sample_idx", "event_date"])
    gaps = active_dates.groupby("sample_idx")["event_date"].diff().dt.days
    gap_groups = gaps.groupby(active_dates["sample_idx"])
    features["active_days_30d"] = active_dates.groupby("sample_idx").size()
    features["active_day_gap_mean"] = gap_groups.mean()
    features["active_day_gap_std"] = gap_groups.std(ddof=0)
    return features


def build_features(events: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Return sample identifiers and nine features from each observation window."""
    required_sample_columns = {"visitor_id", "reference_date"}
    missing_sample_columns = required_sample_columns.difference(samples.columns)
    if missing_sample_columns:
        raise ValueError(f"Missing sample columns: {sorted(missing_sample_columns)}")

    if "event_date" not in events.columns:
        events = normalize_events(events)

    sample_keys = pd.DataFrame(
        {
            "sample_idx": np.arange(len(samples)),
            "visitorid": samples["visitor_id"].to_numpy(),
            "reference_date": pd.to_datetime(samples["reference_date"]).dt.normalize().to_numpy(),
        }
    )
    sample_events = events.loc[
        events["visitorid"].isin(sample_keys["visitorid"]),
        ["visitorid", "event_date", "event", "itemid"],
    ]

    # One reference date at a time keeps the sample-by-event join small.
    parts = []
    for reference_date, keys in sample_keys.groupby("reference_date", sort=False):
        start_date = reference_date - pd.Timedelta(days=OBS_DAYS - 1)
        in_window = sample_events.loc[sample_events["event_date"].between(start_date, reference_date)]
        window = keys.loc[:, ["sample_idx", "visitorid"]].merge(in_window, on="visitorid")
        if not window.empty:
            parts.append(_window_features(window, reference_date))

    feature_frame = pd.concat(parts) if parts else pd.DataFrame(columns=FEATURE_COLUMNS)
    feature_frame = feature_frame.reindex(index=range(len(samples)), columns=FEATURE_COLUMNS)
    feature_frame["days_since_last_purchase"] = feature_frame["days_since_last_purchase"].fillna(OBS_DAYS)
    feature_frame = feature_frame.fillna(0)
    feature_frame = feature_frame.astype(
        {
            **{column: "int64" for column in INTEGER_FEATURES},
            **{column: "float64" for column in FLOAT_FEATURES},
        }
    )
    feature_frame.index = samples.index
    return pd.concat([samples.copy(), feature_frame], axis=1)
