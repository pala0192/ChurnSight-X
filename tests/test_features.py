import unittest

import numpy as np
import pandas as pd

from churn.daily import normalize_events
from churn.features import FEATURE_COLUMNS, build_features
from config import OBS_DAYS


def reference_build_features(events: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Original per-sample implementation kept as an oracle for the vectorized one."""
    events = normalize_events(events)
    visitor_events = {
        visitor_id: group
        for visitor_id, group in events.groupby("visitorid", sort=False)
    }

    rows = []
    for sample in samples.itertuples(index=False):
        reference_date = pd.Timestamp(sample.reference_date).normalize()
        start_date = reference_date - pd.Timedelta(days=OBS_DAYS - 1)
        customer_events = visitor_events.get(sample.visitor_id)
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
            days_since_purchase = int((reference_date - purchases["event_date"].max()).days)

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


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_date = pd.Timestamp("2026-06-30")
        events = [
            (-30, 1, "view", "outside-window"),
            (-29, 1, "view", "item-1"),
            (-25, 1, "addtocart", "item-1"),
            (-20, 1, "view", "item-2"),
            (-10, 1, "transaction", "item-3"),
            (-7, 1, "view", "item-4"),
            (0, 1, "view", "item-4"),
            (1, 1, "transaction", "future-item"),
            (0, 2, "view", "item-5"),
        ]
        self.events = pd.DataFrame(
            [
                {
                    "timestamp": int(
                        (self.reference_date + pd.Timedelta(days=offset)).tz_localize("UTC").timestamp()
                        * 1000
                    ),
                    "visitorid": visitor_id,
                    "event": event,
                    "itemid": item_id,
                }
                for offset, visitor_id, event, item_id in events
            ]
        )
        self.samples = pd.DataFrame(
            {
                "visitor_id": [1, 2],
                "reference_date": [self.reference_date, self.reference_date],
            }
        )

    def test_builds_nine_features_from_the_exact_observation_window(self) -> None:
        features = build_features(self.events, self.samples)
        customer = features.iloc[0]

        self.assertEqual(tuple(features.columns[-9:]), FEATURE_COLUMNS)
        self.assertEqual(customer["active_days_30d"], 6)
        self.assertEqual(customer["event_count_30d"], 6)
        self.assertEqual(customer["view_count_30d"], 4)
        self.assertEqual(customer["cart_count_30d"], 1)
        self.assertEqual(customer["transaction_count_30d"], 1)
        self.assertEqual(customer["unique_items_30d"], 4)
        self.assertEqual(customer["days_since_last_purchase"], 10)
        self.assertAlmostEqual(customer["active_day_gap_mean"], 5.8)
        self.assertAlmostEqual(customer["active_day_gap_std"], (6.16) ** 0.5)

    def test_missing_purchase_and_single_active_day_use_documented_defaults(self) -> None:
        features = build_features(self.events, self.samples).iloc[1]

        self.assertEqual(features["days_since_last_purchase"], 30)
        self.assertEqual(features["active_day_gap_mean"], 0.0)
        self.assertEqual(features["active_day_gap_std"], 0.0)

    def test_rejects_more_than_five_percent_invalid_events(self) -> None:
        invalid = self.events.iloc[[0]].copy()
        invalid["event"] = "unknown"
        events = pd.concat([self.events, invalid], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_features(events, self.samples)

    def test_matches_per_sample_reference_on_random_events(self) -> None:
        rng = np.random.default_rng(7)
        start = pd.Timestamp("2026-05-01", tz="UTC")
        event_count = 3000
        events = pd.DataFrame(
            {
                "timestamp": (
                    start + pd.to_timedelta(rng.integers(0, 70 * 86_400_000, event_count), unit="ms")
                ).map(lambda value: int(value.timestamp() * 1000)),
                "visitorid": rng.integers(0, 60, event_count),
                "event": rng.choice(["view", "view", "view", "addtocart", "transaction"], event_count),
                "itemid": rng.integers(0, 200, event_count),
            }
        )
        reference_dates = pd.date_range("2026-06-01", "2026-07-05", freq="4D")
        samples = pd.DataFrame(
            [
                {"visitor_id": visitor_id, "reference_date": reference_date}
                for reference_date in reference_dates
                for visitor_id in rng.choice(70, size=25, replace=False)
            ]
        )
        samples.index = samples.index * 3 + 5

        pd.testing.assert_frame_equal(
            build_features(events, samples),
            reference_build_features(events, samples),
        )


if __name__ == "__main__":
    unittest.main()