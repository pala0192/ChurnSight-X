import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from churn.explain import (
    explain_predictions,
    factor_sentence,
    feature_contributions,
    feature_medians,
)
from churn.features import FEATURE_COLUMNS


def make_features(row_count: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        {column: rng.integers(0, 20, row_count).astype(float) for column in FEATURE_COLUMNS}
    )
    features["days_since_last_purchase"] = features["days_since_last_purchase"].where(
        rng.random(row_count) > 0.5
    )
    churn = (features["active_days_45d"] + rng.normal(0, 4, row_count) < 10).astype(int)
    return features, churn.to_numpy()


class ExplainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features, churn = make_features()
        self.model = XGBClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=0)
        self.model.fit(self.features, churn)
        self.probabilities = self.model.predict_proba(self.features)[:, 1]

    def test_contributions_add_up_to_the_model_score(self) -> None:
        contributions = feature_contributions(self.model, self.features)
        margin = self.model.predict(self.features, output_margin=True)
        bias = margin - contributions.sum(axis=1).to_numpy()

        np.testing.assert_allclose(bias, bias[0], atol=1e-5)
        np.testing.assert_allclose(1 / (1 + np.exp(-margin)), self.probabilities, rtol=1e-5)

    def test_names_the_factor_pushing_toward_each_prediction(self) -> None:
        threshold = 0.5
        levels = np.where(self.probabilities >= 0.7, "high", "low")
        explanations = explain_predictions(
            self.model,
            self.features,
            self.probabilities,
            levels,
            threshold,
            feature_medians(self.features),
        )
        contributions = feature_contributions(self.model, self.features)

        for row in range(len(self.features)):
            factor = explanations["top_factor"].iloc[row]
            if self.probabilities[row] >= threshold:
                self.assertEqual(factor, contributions.iloc[row].idxmax())
                self.assertIn("높였습니다", explanations["explanation"].iloc[row])
            else:
                self.assertEqual(factor, contributions.iloc[row].idxmin())
                self.assertIn("낮췄습니다", explanations["explanation"].iloc[row])
        self.assertTrue(explanations["explanation"].str.startswith("이탈 위험 ").all())

    def test_prepared_sentences(self) -> None:
        self.assertEqual(
            factor_sentence("active_days_45d", 4, 5, raises_risk=True),
            "최근 45일 접속일이 4일로 일반 고객(5일)보다 적고 이 점이 이탈 위험을 가장 크게 높였습니다.",
        )
        self.assertEqual(
            factor_sentence("view_event_ratio", 0.25, 0.548, raises_risk=False),
            "전체 행동 중 조회 비율이 25.0%로 일반 고객(54.8%)보다 낮고 이 점이 이탈 위험을 가장 크게 낮췄습니다.",
        )
        self.assertEqual(
            factor_sentence("active_days_45d", 5, 5, raises_risk=True),
            "최근 45일 접속일이 5일로 일반 고객과 같은 수준이며 이 점이 이탈 위험을 가장 크게 높였습니다.",
        )
        self.assertEqual(
            factor_sentence("days_since_last_purchase", np.nan, 13, raises_risk=True),
            "최근 45일 동안 구매 기록이 없고 이 점이 이탈 위험을 가장 크게 높였습니다.",
        )


if __name__ == "__main__":
    unittest.main()
