"""Chronological sample splits with a label-window gap between partitions."""

import pandas as pd

from config import SPLIT_GAP_DAYS, SPLIT_RATIOS


DATASET_COLUMN = "dataset"
PARTITION_NAMES = ("train", "validation", "test")


def split_samples(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split samples by reference date and leave seven unused dates at boundaries.

    When a dataset column is present, each dataset is split on its own dates and the
    matching partitions are concatenated, so one dataset's dates never crowd out another's.
    """
    if "reference_date" not in samples.columns:
        raise ValueError("Samples must include a reference_date column.")
    if DATASET_COLUMN not in samples.columns:
        return _split_by_date(samples)

    splits = [
        _split_by_date(group, label=str(name))
        for name, group in samples.groupby(DATASET_COLUMN, sort=False)
    ]
    return {
        name: pd.concat([split[name] for split in splits])
        for name in PARTITION_NAMES
    }


def _split_by_date(samples: pd.DataFrame, label: str = "samples") -> dict[str, pd.DataFrame]:
    dates = pd.DatetimeIndex(
        pd.to_datetime(samples["reference_date"]).dt.normalize().unique()
    ).sort_values()
    usable_count = len(dates) - 2 * SPLIT_GAP_DAYS
    if usable_count < 10:
        raise ValueError(
            f"{label}: at least 24 distinct reference dates are required for temporal splitting."
        )

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