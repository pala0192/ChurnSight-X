"""Per-customer churn explanations from XGBoost feature contributions (TreeSHAP)."""

from typing import Any

import numpy as np
import pandas as pd
import xgboost

from churn.features import FEATURE_COLUMNS


EXPLANATION_COLUMNS = ("top_factor", "top_factor_contribution", "explanation")
RISK_LEVEL_NAMES = {"high": "높음", "medium": "보통", "low": "낮음"}
FEATURE_LABELS = {
    "active_days_45d": "접속일 수",
    "cart_count_45d": "장바구니 담기 횟수",
    "unique_items_45d": "둘러본 상품 수",
    "days_since_last_purchase": "마지막 구매 후 경과일",
    "active_day_gap_mean": "평균 접속 간격",
    "active_day_gap_std": "접속 간격 불규칙성",
    "unique_viewed_items_45d": "조회한 상품 수",
    "unique_purchased_items_45d": "구매한 상품 수",
    "view_event_ratio": "조회 비율",
    "cart_event_ratio": "장바구니 비율",
    "purchase_event_ratio": "구매 비율",
}

# Prepared wording per feature: (subject with particle, unit, value is lower, value is higher).
# Each phrase is used as "<subject> <value><unit>로 일반 고객(<median><unit>)보다 <lower|higher>, ...".
FACTOR_TEXTS = {
    "active_days_45d": ("최근 45일 접속일이", "일", "적고", "많고"),
    "cart_count_45d": ("최근 45일 장바구니 담기가", "회", "적고", "많고"),
    "unique_items_45d": ("최근 45일 둘러본 상품 수가", "개", "적고", "많고"),
    "days_since_last_purchase": ("마지막 구매 후 지난 기간이", "일", "짧고", "길고"),
    "active_day_gap_mean": ("평균 접속 간격이", "일", "짧고", "길고"),
    "active_day_gap_std": ("접속 간격의 불규칙한 정도(표준편차)가", "일", "작고", "크고"),
    "unique_viewed_items_45d": ("최근 45일 조회한 상품 수가", "개", "적고", "많고"),
    "unique_purchased_items_45d": ("최근 45일 구매한 상품 수가", "개", "적고", "많고"),
    "view_event_ratio": ("전체 행동 중 조회 비율이", "%", "낮고", "높고"),
    "cart_event_ratio": ("전체 행동 중 장바구니 비율이", "%", "낮고", "높고"),
    "purchase_event_ratio": ("전체 행동 중 구매 비율이", "%", "낮고", "높고"),
}
NO_PURCHASE_TEXT = "최근 45일 동안 구매 기록이 없고"
RATIO_FEATURES = {"view_event_ratio", "cart_event_ratio", "purchase_event_ratio"}


def feature_medians(features: pd.DataFrame) -> dict[str, float]:
    """Typical-customer reference values; days_since_last_purchase uses purchasers only."""
    return {column: float(features[column].median()) for column in FEATURE_COLUMNS}


def feature_contributions(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    """Return each feature's log-odds contribution to each row's churn score."""
    best_iteration = getattr(model, "best_iteration", None)
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)
    contributions = model.get_booster().predict(
        xgboost.DMatrix(features.loc[:, FEATURE_COLUMNS]),
        pred_contribs=True,
        iteration_range=iteration_range,
    )
    # The last column is the bias shared by every row.
    return pd.DataFrame(contributions[:, :-1], index=features.index, columns=FEATURE_COLUMNS)


def _format_value(feature: str, value: float) -> str:
    if feature in RATIO_FEATURES:
        return f"{value * 100:.1f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def factor_sentence(feature: str, value: float, median: float, raises_risk: bool) -> str:
    """Prepared sentence for one customer's deciding factor."""
    effect = "이 점이 이탈 위험을 가장 크게 높였습니다." if raises_risk else "이 점이 이탈 위험을 가장 크게 낮췄습니다."
    if feature == "days_since_last_purchase" and pd.isna(value):
        return f"{NO_PURCHASE_TEXT} {effect}"

    subject, unit, lower, higher = FACTOR_TEXTS[feature]
    shown_value, shown_median = _format_value(feature, value), _format_value(feature, median)
    if shown_value == shown_median:
        comparison = "일반 고객과 같은 수준이며"
    else:
        comparison = f"일반 고객({shown_median}{unit})보다 {lower if value < median else higher}"
    return f"{subject} {shown_value}{unit}로 {comparison} {effect}"


def explain_predictions(
    model: Any,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    risk_levels: np.ndarray,
    threshold: float,
    medians: dict[str, float],
) -> pd.DataFrame:
    """Name the factor behind each prediction and describe it with prepared wording.

    Predicted churners get the factor that raised their risk most; others get the
    factor that lowered it most. Without such a factor the largest one is used.
    """
    contributions = feature_contributions(model, features)
    values = contributions.to_numpy()
    predicted_churn = probabilities >= threshold
    toward_prediction = np.where(predicted_churn[:, None], values, -values)
    top_index = np.where(
        toward_prediction.max(axis=1) > 0,
        toward_prediction.argmax(axis=1),
        np.abs(values).argmax(axis=1),
    )

    rows = []
    for row, feature_index, probability, risk_level in zip(
        range(len(features)), top_index, probabilities, risk_levels
    ):
        feature = FEATURE_COLUMNS[feature_index]
        contribution = float(values[row, feature_index])
        sentence = factor_sentence(
            feature,
            features[feature].iloc[row],
            medians[feature],
            raises_risk=contribution > 0,
        )
        rows.append(
            {
                "top_factor": feature,
                "top_factor_contribution": contribution,
                "explanation": (
                    f"이탈 위험 {RISK_LEVEL_NAMES[risk_level]}(이탈 확률 {probability:.0%}). {sentence}"
                ),
            }
        )
    return pd.DataFrame(rows, index=features.index, columns=EXPLANATION_COLUMNS)
