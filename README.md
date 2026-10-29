# ChurnSight X

이커머스 이벤트 데이터(화장품 쇼핑몰, 2019-10 ~ 2020-02)의 고객 이탈을 예측하는 XGBoost 프로젝트입니다. 45일 관찰 기간에서 11개 행동 피처를 계산하고, 다음 7일 활동 여부를 학습 라벨로 사용합니다.

예측 대상은 기준일 N에서 N-6~N 7일간 미접속, N-7에 접속, N-44~N 45일 중 접속일이 4일 이상인 고객입니다. 관찰 기간에 구매가 없으면 `days_since_last_purchase`는 결측입니다.

## 데이터 형식

학습 입력은 헤더가 있는 CSV입니다.

| Column | Description |
| --- | --- |
| `timestamp` | UTC Unix timestamp in milliseconds |
| `visitorid` | Visitor identifier |
| `event` | `view`, `addtocart`, `removefromcart`, or `transaction` (`removefromcart` counts as activity and toward the event-ratio denominators) |
| `itemid` | Item identifier |

Extra columns such as `transactionid` are allowed and ignored.

피처: `active_days_45d`, `cart_count_45d`, `unique_items_45d`, `days_since_last_purchase`, `active_day_gap_mean`, `active_day_gap_std`, `unique_viewed_items_45d`, `unique_purchased_items_45d`, `view_event_ratio`, `cart_event_ratio`, `purchase_event_ratio`.

실제 고객 이벤트, 학습된 모델, 업로드 파일은 저장소에 커밋하지 마세요. `.gitignore`가 `data/`, 가상환경, 모델 바이너리를 제외합니다.

## 설치 및 학습

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\train.py --events data\raw\events.csv
```

`--events`에는 CSV 파일 여러 개나 폴더를 줄 수 있습니다. 폴더는 안의 `*.csv`를 모두 합쳐 읽으므로, 피처 표와 같은 폴더에 두지 마세요.

피처 계산이 끝난 표(`visitor_id`, `reference_date`, 11개 피처, `churn`)로도 학습할 수 있습니다. `--events`와 `--samples`를 함께 주면 둘을 합쳐 학습합니다. 입력 파일(데이터셋)마다 기준일 순서로 train/validation/test를 따로 나눈 뒤 같은 구간끼리 합치므로, 기간이 다른 데이터셋을 섞어도 한쪽이 train을 독차지하지 않습니다.

```powershell
python scripts\build_samples.py --events data\2019-Oct.csv data\2019-Nov.csv --output data\samples.csv
python scripts\train.py --samples data\samples.csv
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

예측 결과에는 대상 고객별 11개 피처, 이탈 확률, 위험 등급, 임계값 기준 이탈 판정, 판정 설명이 포함됩니다.

| Column | Description |
| --- | --- |
| `top_factor` | 판정에 가장 크게 기여한 피처. 이탈 판정이면 위험을 가장 크게 높인 피처, 유지 판정이면 가장 크게 낮춘 피처 |
| `top_factor_contribution` | 그 피처의 기여도(TreeSHAP, 로그오즈). 양수는 위험을 높이고 음수는 낮춤 |
| `explanation` | 미리 작성한 문장 틀로 만든 설명. 예: `이탈 위험 높음(이탈 확률 71%). 최근 45일 둘러본 상품 수가 1개로 일반 고객(25개)보다 적고 이 점이 이탈 위험을 가장 크게 높였습니다.` |

"일반 고객" 값은 학습 데이터(train)의 피처별 중앙값으로, 학습할 때 모델 번들에 함께 저장됩니다. 문장 틀은 [`churn/explain.py`](churn/explain.py)의 `FACTOR_TEXTS`에서 고칠 수 있습니다.

## 대시보드 (GitHub Pages)

[`docs/`](docs/)는 예측 결과와 모델 성능을 보여주는 정적 웹페이지입니다. 저장소 Settings → Pages에서 Source를 `Deploy from a branch`, 브랜치를 `main`, 폴더를 `/docs`로 지정하면 공개됩니다.

새 모델이나 기준일로 대시보드를 갱신하려면 예측을 실행한 뒤 데이터 파일을 다시 만들고 커밋합니다.

```powershell
python scripts\predict.py --events data\2019-Oct.csv data\2019-Nov.csv data\2019-Dec.csv data\2020-Jan.csv data\2020-Feb.csv --date 2020-02-28
python scripts\build_dashboard.py --samples data\cosmetics_target_samples.csv --predictions data\predictions\2020-02-28_<version>.csv
```

`docs/data.js`에는 성능 지표, 위험 등급별 실제 이탈률, 판정 요인 분포, 피처 중요도, 고객별 예측(ID 끝 두 자리 마스킹)만 들어가며 원본 이벤트는 포함되지 않습니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```

상세 설계는 [`ARCHITECTURE.md`](ARCHITECTURE.md)를 참고하세요.