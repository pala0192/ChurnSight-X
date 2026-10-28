import unittest

import pandas as pd

from churn.daily import activity_matrix, normalize_events
from churn.sampling import build_training_samples, label, target_mask, training_reference_dates


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_date = pd.Timestamp("2026-06-30")
        dates = pd.date_range(
            self.reference_date - pd.Timedelta(days=44),
            self.reference_date + pd.Timedelta(days=7),
        )
        self.activity = pd.DataFrame(0, index=["eligible", "recent", "four_days", "three_days"], columns=dates)
        for offset in (-44, -30, -17, -11, -7):
            self.activity.loc["eligible", self.reference_date + pd.Timedelta(days=offset)] = 1
        self.activity.loc["recent"] = self.activity.loc["eligible"]
        self.activity.loc["recent", self.reference_date - pd.Timedelta(days=3)] = 1
        for offset in (-44, -29, -18, -7):
            self.activity.loc["four_days", self.reference_date + pd.Timedelta(days=offset)] = 1
        for offset in (-30, -15, -7):
            self.activity.loc["three_days", self.reference_date + pd.Timedelta(days=offset)] = 1

    def test_target_requires_all_eligibility_conditions(self) -> None:
        targets = target_mask(self.activity, self.reference_date)

        self.assertTrue(targets["eligible"])
        self.assertFalse(targets["recent"])
        self.assertTrue(targets["four_days"])
        self.assertFalse(targets["three_days"])

    def test_label_is_one_only_when_future_period_is_inactive(self) -> None:
        self.activity.loc["recent", self.reference_date + pd.Timedelta(days=4)] = 1

        labels = label(self.activity, self.reference_date)

        self.assertEqual(labels["eligible"], 1)
        self.assertEqual(labels["recent"], 0)

    def test_target_rejects_missing_observation_date(self) -> None:
        missing_date = self.reference_date - pd.Timedelta(days=10)
        incomplete = self.activity.drop(columns=missing_date)

        with self.assertRaisesRegex(ValueError, "missing dates"):
            target_mask(incomplete, self.reference_date)

    def test_label_rejects_missing_prediction_date(self) -> None:
        missing_date = self.reference_date + pd.Timedelta(days=3)
        incomplete = self.activity.drop(columns=missing_date)

        with self.assertRaisesRegex(ValueError, "missing dates"):
            label(incomplete, self.reference_date)

    def test_training_samples_include_only_targets(self) -> None:
        reference_dates = training_reference_dates(self.activity.columns)

        self.assertEqual(reference_dates.tolist(), [self.reference_date])
        samples = build_training_samples(self.activity, reference_dates)

        self.assertEqual(samples.to_dict("records"), [
            {
                "visitor_id": "eligible",
                "reference_date": self.reference_date,
                "churn": 1,
            },
            {
                "visitor_id": "four_days",
                "reference_date": self.reference_date,
                "churn": 1,
            },
        ])

    def test_cart_removal_marks_a_visitor_active(self) -> None:
        reference = pd.Timestamp("2026-06-30", tz="UTC")
        events = pd.DataFrame(
            {
                "timestamp": [int(reference.timestamp() * 1000)],
                "visitorid": [7],
                "event": ["removefromcart"],
                "itemid": [1],
            }
        )

        matrix = activity_matrix(normalize_events(events), dates=pd.DatetimeIndex([reference.tz_localize(None)]))

        self.assertEqual(matrix.loc[7].tolist(), [1])

    def test_training_reference_dates_require_full_windows(self) -> None:
        references = training_reference_dates(self.activity.columns[:-1])

        self.assertTrue(references.empty)


if __name__ == "__main__":
    unittest.main()