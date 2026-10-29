import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from churn.dataset import SAMPLE_COLUMNS, read_feature_samples, write_feature_samples
from churn.features import FEATURE_COLUMNS
from churn.split import split_samples


def make_samples() -> pd.DataFrame:
    samples = pd.DataFrame(
        {
            "visitor_id": [454, 5117],
            "reference_date": pd.to_datetime(["2015-06-17", "2015-06-18"]),
            "churn": [1, 0],
        }
    )
    for column in FEATURE_COLUMNS:
        samples[column] = [1, 2]
    samples["days_since_last_purchase"] = [np.nan, 8.0]
    samples["active_day_gap_mean"] = [1.0, 7.2]
    samples["active_day_gap_std"] = [0.5, 9.654014708917737]
    samples["view_event_ratio"] = [1.0, 0.8181818181818182]
    return samples


class DatasetTests(unittest.TestCase):
    def test_written_table_matches_target_samples_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            write_feature_samples(make_samples(), path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[0].split(","), list(SAMPLE_COLUMNS))
        first = dict(zip(SAMPLE_COLUMNS, lines[1].split(",")))
        self.assertEqual(first["reference_date"], "2015-06-17")
        self.assertEqual(first["days_since_last_purchase"], "")
        self.assertEqual(first["active_day_gap_mean"], "1")
        self.assertEqual(first["view_event_ratio"], "1")
        second = dict(zip(SAMPLE_COLUMNS, lines[2].split(",")))
        self.assertEqual(second["active_day_gap_std"], "9.654014708917737")
        self.assertEqual(second["churn"], "0")

    def test_round_trip_preserves_values(self) -> None:
        samples = make_samples()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            write_feature_samples(samples, path)
            loaded = read_feature_samples([path])

        expected = samples.loc[:, SAMPLE_COLUMNS]
        self.assertEqual(loaded["dataset"].unique().tolist(), ["samples"])
        pd.testing.assert_frame_equal(loaded.drop(columns="dataset"), expected, check_dtype=False)

    def test_each_dataset_is_split_on_its_own_dates(self) -> None:
        early = pd.DataFrame(
            {"reference_date": pd.date_range("2015-06-01", periods=30), "dataset": "early"}
        )
        late = pd.DataFrame(
            {"reference_date": pd.date_range("2019-11-01", periods=40), "dataset": "late"}
        )
        combined = pd.concat([early, late], ignore_index=True)

        partitions = split_samples(combined)
        alone = split_samples(late.drop(columns="dataset"))

        for name in ("train", "validation", "test"):
            self.assertEqual(set(partitions[name]["dataset"]), {"early", "late"})
            late_part = partitions[name].loc[partitions[name]["dataset"].eq("late"), "reference_date"]
            self.assertEqual(late_part.tolist(), alone[name]["reference_date"].tolist())

    def test_rejects_tables_missing_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            make_samples().drop(columns="churn").to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "churn"):
                read_feature_samples([path])


if __name__ == "__main__":
    unittest.main()
