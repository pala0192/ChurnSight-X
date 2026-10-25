"""Build the nine requested 30-day churn prediction features."""

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


def build_features(events: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Return sample identifiers and nine features from each observation window."""
    required_sample_columns = {"visitor_id", "reference_date"}
    missing_sample_columns = required_sample_columns.difference(samples.columns)
    if missing_sample_columns:
        raise ValueError(f"Missing sample columns: {sorted(missing_sample_columns)}")

    if "event_date" not in events.columns:
        events = normalize_events(events)
    visitor_events = {
        visitor_id: group
        for visitor_id, group in events.groupby("visitorid", sort=False)
    }

    rows = []
    for sample in samples.itertuples(index=False):
        visitor_id = getattr(sample, "visitor_id")
        reference_date = pd.Timestamp(getattr(sample, "reference_date")).normalize()
        start_date = reference_date - pd.Timedelta(days=OBS_DAYS - 1)
        customer_events = visitor_events.get(visitor_id)
        if customer_events is None:
            window = events.iloc[0:0]
        else:
            in_window = customer_events["event_date"].between(start_date, reference_date)
            window = customer_events.loc[in_window]

        event_dates = pd.DatetimeIndex(window["event_date"].drop_duplicates()).sort_values()
        gaps = event_dates.to_series().diff().dropna().dt.days
        purchases = window.loc[window["event"].eq("transaction")]
        if purchases.empty:
            days_since_purchase = OBS_DAYS
        else:
            days_since_purchase = int(
                (reference_date - purchases["event_date"].max()).days
            )

        rows.append(
            {
                "active_days_30d": len(event_dates),
                "event_count_30d": len(window),
                "view_count_30d": int(window["event"].eq("view").sum()),
                "cart_count_30d": int(window["event"].eq("addtocart").sum()),
                "transaction_count_30d": int(len(purchases)),
                "unique_items_30d": int(window["itemid"].nunique()),
                "days_since_last_purchase": days_since_purchase,
                "active_day_gap_mean": float(gaps.mean()) if not gaps.empty else 0.0,
                "active_day_gap_std": float(gaps.std(ddof=0)) if not gaps.empty else 0.0,
            }
        )

    feature_frame = pd.DataFrame(rows, index=samples.index, columns=FEATURE_COLUMNS)
    return pd.concat([samples.copy(), feature_frame], axis=1)