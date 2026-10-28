"""Chronological sample splits with a label-window gap between partitions."""

import pandas as pd

from config import SPLIT_GAP_DAYS, SPLIT_RATIOS


def split_samples(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split samples by reference date and leave seven unused dates at boundaries."""
    if "reference_date" not in samples.columns:
        raise ValueError("Samples must include a reference_date column.")

    dates = pd.DatetimeIndex(
        pd.to_datetime(samples["reference_date"]).dt.normalize().unique()
    ).sort_values()
    usable_count = len(dates) - 2 * SPLIT_GAP_DAYS
    if usable_count < 10:
        raise ValueError("At least 24 distinct reference dates are required for temporal splitting.")

    train_count = int(usable_count * SPLIT_RATIOS[0])
    validation_count = int(usable_count * SPLIT_RATIOS[1])
    test_count = usable_count - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("Not enough reference dates for all three temporal partitions.")

    train_dates = dates[:train_count]
    validation_start = train_count + SPLIT_GAP_DAYS
    validation_dates = dates[validation_start : validation_start + validation_count]
    test_start = validation_start + validation_count + SPLIT_GAP_DAYS
    test_dates = dates[test_start:]

    reference_dates = pd.to_datetime(samples["reference_date"]).dt.normalize()
    return {
        "train": samples.loc[reference_dates.isin(train_dates)].copy(),
        "validation": samples.loc[reference_dates.isin(validation_dates)].copy(),
        "test": samples.loc[reference_dates.isin(test_dates)].copy(),
    }


def rolling_origin_splits(
    activity_dates: pd.Index,
    n_folds: int = 3,
    validation_days: int = 14,
    test_days: int = 14,
    gap_days: int = SPLIT_GAP_DAYS,
    min_train_days: int = 30,
) -> list[dict[str, pd.DatetimeIndex]]:
    """Create expanding chronological folds with label-window gaps."""
    dates = pd.DatetimeIndex(pd.to_datetime(activity_dates)).normalize().sort_values()
    if dates.has_duplicates:
        raise ValueError("Activity dates must be unique.")
    if min(n_folds, validation_days, test_days, min_train_days) < 1 or gap_days < 0:
        raise ValueError("Fold lengths and count must be positive; gap_days cannot be negative.")

    required_days = min_train_days + gap_days + validation_days + n_folds * test_days
    if len(dates) < required_days:
        raise ValueError(
            f"Rolling validation needs at least {required_days} reference dates; "
            f"the data has {len(dates)}. Add more history or reduce the fold windows."
        )

    first_test_start = len(dates) - n_folds * test_days
    folds = []
    for fold_number in range(n_folds):
        test_start = first_test_start + fold_number * test_days
        test_end = test_start + test_days
        validation_end = test_start - gap_days
        validation_start = validation_end - validation_days
        train_end = validation_start - gap_days
        if train_end < min_train_days:
            raise ValueError("Not enough training dates before the first validation window.")
        folds.append(
            {
                "train": dates[:train_end],
                "validation": dates[validation_start:validation_end],
                "test": dates[test_start:test_end],
            }
        )
    return folds
