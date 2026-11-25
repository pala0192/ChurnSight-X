"""Shared configuration for churn training and prediction."""

from pathlib import Path


OBS_DAYS = 45
PRED_DAYS = 7
MIN_ACTIVE_DAYS = 4
REFERENCE_STRIDE_DAYS = 1

SPLIT_RATIOS = (0.6, 0.2, 0.2)
SPLIT_GAP_DAYS = PRED_DAYS

TARGET_RECALL = 0.70
RISK_HIGH_MIN = 0.70
RISK_MEDIUM_MIN = 0.30

MAX_INVALID_ROW_RATIO = 0.05

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"