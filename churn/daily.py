"""Normalize Retail Rocket events and build daily activity matrices."""

import pandas as pd

from config import MAX_INVALID_ROW_RATIO


EVENT_TYPES = ("view", "addtocart", "transaction")
EVENT_COLUMNS = ("timestamp", "visitorid", "event", "itemid")


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate Retail Rocket columns and normalize millisecond UTC timestamps."""
    missing_columns = set(EVENT_COLUMNS).difference(events.columns)
    if missing_columns:
        raise ValueError(f"Missing required event columns: {sorted(missing_columns)}")
    if events.empty:
        raise ValueError("The event dataset is empty.")

    normalized = events.copy()
    epoch_ms = pd.to_numeric(normalized["timestamp"], errors="coerce")
    normalized["timestamp"] = pd.to_datetime(epoch_ms, unit="ms", utc=True, errors="coerce")
    normalized["event"] = normalized["event"].astype("string").str.strip().str.lower()
    valid = (
        normalized["timestamp"].notna()
        & normalized["visitorid"].notna()
        & normalized["event"].isin(EVENT_TYPES)
    )
    invalid_ratio = 1 - float(valid.mean())
    if invalid_ratio > MAX_INVALID_ROW_RATIO:
        raise ValueError(
            f"Invalid event row ratio {invalid_ratio:.1%} exceeds "
            f"the {MAX_INVALID_ROW_RATIO:.1%} limit."
        )

    normalized = normalized.loc[valid].copy()
    normalized["event_date"] = normalized["timestamp"].dt.floor("D").dt.tz_localize(None)
    return normalized.reset_index(drop=True)


def complete_date_range(events: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the UTC calendar dates excluding the first and last collection days."""
    if "event_date" not in events.columns:
        events = normalize_events(events)
    observed = pd.DatetimeIndex(events["event_date"].dropna().unique()).sort_values()
    if len(observed) < 3:
        return pd.DatetimeIndex([])
    return pd.date_range(observed[0] + pd.Timedelta(days=1), observed[-1] - pd.Timedelta(days=1))


def trim_incomplete_days(events: pd.DataFrame) -> pd.DataFrame:
    """Remove events from the collection's first and last calendar days."""
    if "event_date" not in events.columns:
        events = normalize_events(events)
    dates = complete_date_range(events)
    return events.loc[events["event_date"].isin(dates)].copy()


def activity_matrix(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a visitor-by-date binary matrix from recognized activity events."""
    if "event_date" not in events.columns:
        events = normalize_events(events)
    if dates is None:
        dates = complete_date_range(events)

    complete_events = events.loc[events["event_date"].isin(dates)]
    if complete_events.empty:
        return pd.DataFrame(index=pd.Index([], name="visitorid"), columns=dates, dtype="int8")

    daily_activity = complete_events.groupby(["visitorid", "event_date"]).size().gt(0)
    matrix = daily_activity.unstack("event_date", fill_value=False)
    matrix = matrix.reindex(columns=dates, fill_value=False).astype("int8")
    matrix.index.name = "visitorid"
    return matrix