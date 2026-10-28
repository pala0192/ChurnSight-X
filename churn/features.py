"""Build the sixteen 30-day churn prediction features."""

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
    "purchase_days_30d",
    "unique_viewed_items_30d",
    "unique_cart_items_30d",
    "unique_purchased_items_30d",
    "view_event_ratio",
    "cart_event_ratio",
    "purchase_event_ratio",
)
FLOAT_FEATURES = (
    "active_day_gap_mean",
    "active_day_gap_std",
    "view_event_ratio",
    "cart_event_ratio",
    "purchase_event_ratio",
)
INTEGER_FEATURES = tuple(column for column in FEATURE_COLUMNS if column not in FLOAT_FEATURES)


def _window_features(window: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate one reference date's observation-window events per sample."""
    flags = window.assign(
        is_view=window["event"].eq("view").astype(bool),
        is_cart=window["event"].eq("addtocart").astype(bool),
        is_transaction=window["event"].eq("transaction").astype(bool),
    )
    grouped = flags.groupby("sample_idx")
    views = flags.loc[flags["is_view"]].groupby("sample_idx")
    carts = flags.loc[flags["is_cart"]].groupby("sample_idx")
    purchases = flags.loc[flags["is_transaction"]].groupby("sample_idx")
    features = pd.DataFrame(
        {
            "event_count_30d": grouped.size(),
            "view_count_30d": grouped["is_view"].sum(),
            "cart_count_30d": grouped["is_cart"].sum(),
            "purchase_event_count": grouped["is_transaction"].sum(),
            "unique_items_30d": grouped["itemid"].nunique(),
        }
    )

    # Multi-item orders share one transactionid; count each order once when ids exist.
    if "transactionid" in flags.columns:
        features["transaction_count_30d"] = purchases["transactionid"].nunique()
    else:
        features["transaction_count_30d"] = features["purchase_event_count"]
    features["days_since_last_purchase"] = (reference_date - purchases["event_date"].max()).dt.days
    features["purchase_days_30d"] = purchases["event_date"].nunique()
    features["unique_viewed_items_30d"] = views["itemid"].nunique()
    features["unique_cart_items_30d"] = carts["itemid"].nunique()
    features["unique_purchased_items_30d"] = purchases["itemid"].nunique()
    features["view_event_ratio"] = features["view_count_30d"] / features["event_count_30d"]
    features["cart_event_ratio"] = features["cart_count_30d"] / features["event_count_30d"]
    features["purchase_event_ratio"] = features["purchase_event_count"] / features["event_count_30d"]

    active_dates = flags.loc[:, ["sample_idx", "event_date"]].drop_duplicates()
    active_dates = active_dates.sort_values(["sample_idx", "event_date"])
    gaps = active_dates.groupby("sample_idx")["event_date"].diff().dt.days
    gap_groups = gaps.groupby(active_dates["sample_idx"])
    features["active_days_30d"] = active_dates.groupby("sample_idx").size()
    features["active_day_gap_mean"] = gap_groups.mean()
    # Sample standard deviation; a single gap has none and falls back to 0 below.
    features["active_day_gap_std"] = gap_groups.std(ddof=1)
    return features


def build_features(events: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Return sample identifiers and sixteen features from each observation window."""
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
    event_columns = ["visitorid", "event_date", "event", "itemid"]
    if "transactionid" in events.columns:
        event_columns.append("transactionid")
    sample_events = events.loc[events["visitorid"].isin(sample_keys["visitorid"]), event_columns]

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
