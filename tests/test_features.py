import unittest

import pandas as pd

from churn.features import FEATURE_COLUMNS, build_features


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


if __name__ == "__main__":
    unittest.main()