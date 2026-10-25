"""Shared configuration for churn training and prediction."""

from pathlib import Path


OBS_DAYS = 30
PRED_DAYS = 7
REFERENCE_STRIDE_DAYS = 1

BOT_QUANTILE = 0.999
BOT_MIN_EVENTS_PER_DAY = 100

SPLIT_RATIOS = (0.6, 0.2, 0.2)
SPLIT_GAP_DAYS = PRED_DAYS

TARGET_RECALL = 0.70
RISK_HIGH_MIN = 0.60
RISK_MEDIUM_MIN = 0.35

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_INVALID_ROW_RATIO = 0.05

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UPLOADS_DIR = DATA_DIR / "uploads"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"