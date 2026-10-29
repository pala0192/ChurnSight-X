"""Turn event CSVs into a labeled feature table in the target_samples.csv layout."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from churn.daily import read_events
from churn.dataset import feature_samples_from_events, write_feature_samples
from config import REFERENCE_STRIDE_DAYS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        required=True,
        type=Path,
        nargs="+",
        help="Event CSV files or directories of CSVs",
    )
    parser.add_argument("--output", required=True, type=Path, help="Feature table CSV to write")
    parser.add_argument("--stride", type=int, default=REFERENCE_STRIDE_DAYS)
    args = parser.parse_args()

    samples = feature_samples_from_events(read_events(args.events), stride_days=args.stride)
    write_feature_samples(samples, args.output)
    print(
        f"{len(samples):,} samples, {samples['visitor_id'].nunique():,} visitors, "
        f"churn rate {samples['churn'].mean():.4f} -> {args.output}"
    )


if __name__ == "__main__":
    main()
