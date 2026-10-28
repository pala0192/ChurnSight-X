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
        views = window.loc[window["event"].eq("view")]
        carts = window.loc[window["event"].eq("addtocart")]
        purchases = window.loc[window["event"].eq("transaction")]
        if purchases.empty:
            days_since_purchase = np.nan
        else:
            days_since_purchase = int((reference_date - purchases["event_date"].max()).days)
        if "transactionid" in purchases.columns:
            transaction_count = int(purchases["transactionid"].nunique())
        else:
            transaction_count = len(purchases)
        event_count = len(window)

        rows.append(
            {
                "active_days_45d": len(event_dates),
                "event_count_45d": event_count,
                "view_count_45d": len(views),
                "cart_count_45d": len(carts),
                "transaction_count_45d": transaction_count,
                "unique_items_45d": int(window["itemid"].nunique()),
                "days_since_last_purchase": days_since_purchase,
                "active_day_gap_mean": float(gaps.mean()) if not gaps.empty else 0.0,
                "active_day_gap_std": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                "purchase_days_45d": int(purchases["event_date"].nunique()),
                "unique_viewed_items_45d": int(views["itemid"].nunique()),
                "unique_cart_items_45d": int(carts["itemid"].nunique()),
                "unique_purchased_items_45d": int(purchases["itemid"].nunique()),
                "view_event_ratio": len(views) / event_count if event_count else 0.0,
                "cart_event_ratio": len(carts) / event_count if event_count else 0.0,
                "purchase_event_ratio": len(purchases) / event_count if event_count else 0.0,
            }
        )

    feature_frame = pd.DataFrame(rows, index=samples.index, columns=FEATURE_COLUMNS)
    return pd.concat([samples.copy(), feature_frame], axis=1)


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_date = pd.Timestamp("2026-06-30")
        events = [
            (-45, 1, "view", "outside-window"),
            (-29, 1, "view", "item-1"),
            (-25, 1, "addtocart", "item-1"),
            (-20, 1, "view", "item-2"),
            (-10, 1, "transaction", "item-3"),
            (-7, 1, "view", "item-4"),
            (0, 1, "view", "item-4"),
            (1, 1, "transaction", "future-item"),
            (0, 2, "view", "item-5"),
            (-5, 3, "view", "item-6"),
            (-2, 3, "view", "item-6"),
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
                "visitor_id": [1, 2, 3],
                "reference_date": [self.reference_date] * 3,
            }
        )

    def test_builds_sixteen_features_from_the_exact_observation_window(self) -> None:
        features = build_features(self.events, self.samples)
        customer = features.iloc[0]

        self.assertEqual(tuple(features.columns[-16:]), FEATURE_COLUMNS)
        self.assertEqual(customer["active_days_45d"], 6)
        self.assertEqual(customer["event_count_45d"], 6)
        self.assertEqual(customer["view_count_45d"], 4)
        self.assertEqual(customer["cart_count_45d"], 1)
        self.assertEqual(customer["transaction_count_45d"], 1)
        self.assertEqual(customer["unique_items_45d"], 4)
        self.assertEqual(customer["days_since_last_purchase"], 10)
        self.assertAlmostEqual(customer["active_day_gap_mean"], 5.8)
        self.assertAlmostEqual(customer["active_day_gap_std"], 7.7 ** 0.5)
        self.assertEqual(customer["purchase_days_45d"], 1)
        self.assertEqual(customer["unique_viewed_items_45d"], 3)
        self.assertEqual(customer["unique_cart_items_45d"], 1)
        self.assertEqual(customer["unique_purchased_items_45d"], 1)
        self.assertAlmostEqual(customer["view_event_ratio"], 4 / 6)
        self.assertAlmostEqual(customer["cart_event_ratio"], 1 / 6)
        self.assertAlmostEqual(customer["purchase_event_ratio"], 1 / 6)

    def test_transaction_count_uses_unique_transaction_ids(self) -> None:
        purchase = self.events.loc[self.events["event"].eq("transaction")].iloc[[0]]
        second_item = purchase.assign(itemid="item-7")
        events = pd.concat([self.events, second_item], ignore_index=True)
        events["transactionid"] = np.where(events["event"].eq("transaction"), 501, np.nan)

        customer = build_features(events, self.samples).iloc[0]

        self.assertEqual(customer["transaction_count_45d"], 1)
        self.assertEqual(customer["unique_purchased_items_45d"], 2)
        self.assertAlmostEqual(customer["purchase_event_ratio"], 2 / 7)

    def test_cart_removal_counts_as_activity_but_not_as_cart_add(self) -> None:
        removal = self.events.loc[self.events["visitorid"].eq(2)].assign(itemid="item-8")
        removal["event"] = "removefromcart"
        removal["timestamp"] -= 3 * 86_400_000
        events = pd.concat([self.events, removal], ignore_index=True)

        customer = build_features(events, self.samples).iloc[1]

        self.assertEqual(customer["active_days_45d"], 2)
        self.assertEqual(customer["event_count_45d"], 2)
        self.assertEqual(customer["cart_count_45d"], 0)
        self.assertEqual(customer["unique_items_45d"], 2)
        self.assertAlmostEqual(customer["view_event_ratio"], 0.5)

    def test_single_activity_gap_has_zero_sample_std(self) -> None:
        customer = build_features(self.events, self.samples).iloc[2]

        self.assertEqual(customer["active_day_gap_mean"], 3.0)
        self.assertEqual(customer["active_day_gap_std"], 0.0)

    def test_missing_purchase_and_single_active_day_use_documented_defaults(self) -> None:
        features = build_features(self.events, self.samples).iloc[1]

        self.assertTrue(pd.isna(features["days_since_last_purchase"]))
        self.assertEqual(features["active_day_gap_mean"], 0.0)
        self.assertEqual(features["active_day_gap_std"], 0.0)
        self.assertEqual(features["view_event_ratio"], 1.0)
        self.assertEqual(features["purchase_days_45d"], 0)

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
        events["transactionid"] = np.where(
            events["event"].eq("transaction"),
            rng.integers(0, 300, event_count),
            np.nan,
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