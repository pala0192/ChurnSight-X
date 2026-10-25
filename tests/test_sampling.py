import unittest

import pandas as pd

from churn.sampling import build_training_samples, label, target_mask, training_reference_dates


class SamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_date = pd.Timestamp("2026-06-30")
        dates = pd.date_range(
            self.reference_date - pd.Timedelta(days=29),
            self.reference_date + pd.Timedelta(days=7),
        )
        self.activity = pd.DataFrame(0, index=["eligible", "recent", "old_gap"], columns=dates)
        for offset in (-29, -23, -17, -11, -7):
            self.activity.loc["eligible", self.reference_date + pd.Timedelta(days=offset)] = 1
        self.activity.loc["recent"] = self.activity.loc["eligible"]
        self.activity.loc["recent", self.reference_date - pd.Timedelta(days=3)] = 1
        for offset in (-29, -21, -13, -7):
            self.activity.loc["old_gap", self.reference_date + pd.Timedelta(days=offset)] = 1

    def test_target_requires_all_eligibility_conditions(self) -> None:
        targets = target_mask(self.activity, self.reference_date)

        self.assertTrue(targets["eligible"])
        self.assertFalse(targets["recent"])
        self.assertFalse(targets["old_gap"])

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
            }
        ])

    def test_training_reference_dates_require_full_windows(self) -> None:
        references = training_reference_dates(self.activity.columns[:-1])

        self.assertTrue(references.empty)


if __name__ == "__main__":
    unittest.main()