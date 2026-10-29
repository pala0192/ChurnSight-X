import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from churn.features import FEATURE_COLUMNS
from churn.models import load_model_bundle, predict_features, train_xgboost
from churn.service import predict_events


def make_labeled_features() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    for reference_date in dates:
        for visitor_number in range(8):
            churn = visitor_number % 2
            engagement = 20 - churn * 14 + visitor_number // 2
            rows.append(
                {
                    "visitor_id": f"{reference_date.date()}-{visitor_number}",
                    "reference_date": reference_date,
                    "churn": churn,
                    "active_days_45d": engagement,
                    "cart_count_45d": engagement // 3,
                    "unique_items_45d": engagement // 2,
                    "days_since_last_purchase": 5 + churn * 20,
                    "active_day_gap_mean": 1.0 + churn * 2,
                    "active_day_gap_std": 0.5 + churn,
                    "unique_viewed_items_45d": engagement // 2,
                    "unique_purchased_items_45d": 1 - churn,
                    "view_event_ratio": 0.6 + churn * 0.3,
                    "cart_event_ratio": 0.1 - churn * 0.05,
                    "purchase_event_ratio": 0.05 * (1 - churn),
                }
            )
    return pd.DataFrame(rows, columns=["visitor_id", "reference_date", "churn", *FEATURE_COLUMNS])


class ModelPipelineTests(unittest.TestCase):
    def test_training_saves_inspectable_json_and_predictable_bundle(self) -> None:
        samples = make_labeled_features()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            result = train_xgboost(
                samples,
                output_dir,
                parameters={"n_estimators": 12, "max_depth": 2, "n_jobs": 1},
            )

            config_path = output_dir / "model_config.json"
            tree_model_path = output_dir / "model.json"
            bundle_path = output_dir / "bundle.joblib"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tree_model = json.loads(tree_model_path.read_text(encoding="utf-8"))
            bundle = load_model_bundle(bundle_path)
            predictions = predict_features(samples.iloc[:5], bundle)
            reference_date = pd.Timestamp("2026-06-30")
            event_offsets = [-45, -44, -40, -30, -20, -11, -8, -7, 1]
            event_rows = []
            for offset in event_offsets:
                event_date = reference_date + pd.Timedelta(days=offset)
                event_rows.append(
                    {
                        "timestamp": int(event_date.tz_localize("UTC").timestamp() * 1000),
                        "visitorid": 99 if offset in (-45, 1) else 1,
                        "event": "view",
                        "itemid": f"item-{offset}",
                    }
                )
            service_predictions = predict_events(
                pd.DataFrame(event_rows),
                reference_date,
                bundle,
            )

            self.assertEqual(config["feature_columns"], list(FEATURE_COLUMNS))
            self.assertEqual(config["hyperparameters"]["max_depth"], 2)
            self.assertIn("feature_importance", config)
            self.assertIn("learner", tree_model)
            self.assertEqual(len(predictions), 5)
            self.assertTrue(predictions["churn_probability"].between(0, 1).all())
            self.assertTrue(predictions["top_factor"].isin(FEATURE_COLUMNS).all())
            self.assertTrue(predictions["explanation"].str.startswith("이탈 위험 ").all())
            self.assertEqual(set(config["feature_medians"]), set(FEATURE_COLUMNS))
            self.assertEqual(service_predictions["visitor_id"].tolist(), [1])
            self.assertTrue(
                service_predictions["churn_probability"].between(0, 1).all()
            )
            self.assertTrue((output_dir / "meta.json").exists())
            self.assertEqual(result["metadata"]["test_metrics"], config["test_metrics"])


if __name__ == "__main__":
    unittest.main()