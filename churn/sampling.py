"""Reference-date eligibility and churn labels for activity matrices."""

from datetime import date, datetime

import pandas as pd

from config import OBS_DAYS, PRED_DAYS, REFERENCE_STRIDE_DAYS


def _date_window(
    mat: pd.DataFrame,
    reference_date: date | datetime | str | pd.Timestamp,
    start_offset: int,
    end_offset: int,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(mat.columns)).normalize()
    if dates.has_duplicates:
        raise ValueError("Activity matrix must have unique calendar dates.")

    reference = pd.Timestamp(reference_date).normalize()
    window_dates = pd.date_range(
        reference + pd.Timedelta(days=start_offset),
        reference + pd.Timedelta(days=end_offset),
    )
    missing = window_dates.difference(dates)
    if not missing.empty:
        raise ValueError(f"Activity matrix is missing dates: {missing.strftime('%Y-%m-%d').tolist()}")

    window = mat.copy()
    window.columns = dates
    return window.loc[:, window_dates]


def target_mask(
    mat: pd.DataFrame,
    reference_date: date | datetime | str | pd.Timestamp,
) -> pd.Series:
    """Return visitors eligible for a churn prediction at the reference date."""
    observation = _date_window(mat, reference_date, -(OBS_DAYS - 1), 0).fillna(0).gt(0)
    history = observation.iloc[:, : OBS_DAYS - PRED_DAYS]
    historical_inactivity = (
        (~history).T.rolling(window=PRED_DAYS).sum().eq(PRED_DAYS).any(axis=0)
    )
    was_active_before_gap = ~historical_inactivity

    active_seven_days_ago = observation.iloc[:, -(PRED_DAYS + 1)]
    inactive_since = ~observation.iloc[:, -PRED_DAYS:].any(axis=1)
    return active_seven_days_ago & inactive_since & was_active_before_gap


def label(
    mat: pd.DataFrame,
    reference_date: date | datetime | str | pd.Timestamp,
) -> pd.Series:
    """Label visitors as churned when they are inactive for all seven future days."""
    prediction_period = _date_window(mat, reference_date, 1, PRED_DAYS).fillna(0)
    return prediction_period.gt(0).sum(axis=1).eq(0).astype("int8").rename("churn")


def training_reference_dates(
    activity_dates: pd.Index,
    stride_days: int = REFERENCE_STRIDE_DAYS,
) -> pd.DatetimeIndex:
    """Return reference dates with complete observation and prediction windows."""
    if stride_days < 1:
        raise ValueError("stride_days must be at least 1.")

    dates = pd.DatetimeIndex(pd.to_datetime(activity_dates)).normalize().sort_values()
    if dates.has_duplicates:
        raise ValueError("Activity dates must be unique.")
    if dates.empty:
        return dates

    expected_dates = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected_dates):
        raise ValueError("Activity dates must be a complete daily range.")

    first_index = OBS_DAYS - 1
    stop_index = len(dates) - PRED_DAYS
    return dates[first_index:stop_index:stride_days]


def build_training_samples(
    mat: pd.DataFrame,
    reference_dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Build one labeled row per eligible visitor and reference date."""
    if reference_dates is None:
        reference_dates = training_reference_dates(mat.columns)

    samples = []
    for reference_date in reference_dates:
        targets = target_mask(mat, reference_date)
        target_ids = targets.index[targets]
        if target_ids.empty:
            continue

        labels = label(mat, reference_date).loc[target_ids]
        samples.append(
            pd.DataFrame(
                {
                    "visitor_id": target_ids,
                    "reference_date": pd.Timestamp(reference_date).normalize(),
                    "churn": labels.to_numpy(),
                }
            )
        )

    if not samples:
        return pd.DataFrame(columns=["visitor_id", "reference_date", "churn"])
    return pd.concat(samples, ignore_index=True)