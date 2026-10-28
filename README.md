# ChurnSight X

Retail Rocket 이벤트 데이터의 고객 이탈을 예측하는 XGBoost 프로젝트입니다. 30일 관찰 기간에서 12개 행동 피처를 계산하고, 다음 7일 활동 여부를 학습 라벨로 사용합니다.

## 데이터 형식

학습 입력은 헤더가 있는 CSV입니다.

| Column | Description |
| --- | --- |
| `timestamp` | UTC Unix timestamp in milliseconds |
| `visitorid` | Visitor identifier |
| `event` | `view`, `addtocart`, or `transaction` |
| `itemid` | Item identifier |

실제 고객 이벤트, 학습된 모델, 업로드 파일은 저장소에 커밋하지 마세요. `.gitignore`가 `data/`, 가상환경, 모델 바이너리를 제외합니다.

## 설치 및 학습

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\train.py --events data\raw\events.csv
```

기본 XGBoost 하이퍼파라미터와 early-stopping 설정은 [`configs/xgboost.json`](configs/xgboost.json)에서 확인할 수 있습니다. 학습 결과는 `data/models/<version>/`에 저장됩니다.

## 예측

```powershell
python scripts\predict.py --events data\raw\events.csv --date 2026-06-30
```

`data/models/LATEST` 모델로 예측해 `data/predictions/<기준일>_<모델버전>.csv`에 저장합니다. `--model-version`으로 다른 모델을, `--output`으로 저장 경로를 지정할 수 있습니다.

Python 코드에서 직접 호출할 수도 있습니다.

```python
import pandas as pd

from churn.models import load_model_bundle
from churn.service import predict_events

events = pd.read_csv("data/raw/events.csv")
bundle = load_model_bundle("data/models/<version>/bundle.joblib")
predictions = predict_events(events, "2026-06-30", bundle)
```

예측 결과에는 대상 고객별 9개 피처, 이탈 확률, 위험 등급, 임계값 기준 이탈 판정이 포함됩니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```

상세 설계는 [`ARCHITECTURE.md`](ARCHITECTURE.md)를 참고하세요.
