"""Interactive entry point for the existing training, prediction and test commands."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    events = ROOT / "data" / "events.csv"
    while True:
        print("\nChurnSight X - 고객 이탈 예측")
        print(f"이벤트 파일: {events}")
        print("1. 학습 (학습 후 테스트 성능도 평가)")
        print("2. 예측 (저장된 최신 모델 사용)")
        print("3. 프로그램 자동 테스트")
        print("4. 이벤트 CSV 경로 변경")
        print("5. 여러 과거 시점에서 성능 평가")
        print("6. 하이퍼파라미터 후보 비교 (약 6회 학습)")
        print("0. 종료")
        choice = input("선택: ").strip()
        if choice == "0":
            return
        if choice == "4":
            candidate = Path(input("CSV 전체 경로: ").strip().strip('"'))
            if candidate.is_file():
                events = candidate
            else:
                print("파일을 찾을 수 없습니다.")
            continue
        if choice == "3":
            args = ["-m", "unittest", "discover", "-s", "tests", "-v"]
        elif choice == "1":
            args = ["scripts/train.py", "--events", str(events)]
            print("학습 중입니다. 데이터 크기에 따라 몇 분 이상 걸릴 수 있습니다.")
        elif choice == "2":
            if not (ROOT / "data/models/LATEST").is_file():
                print("학습된 모델이 없습니다. 먼저 1번 학습을 실행하세요.")
                continue
            try:
                timestamps = pd.to_numeric(pd.read_csv(events, usecols=["timestamp"])["timestamp"], errors="coerce")
                dates = pd.to_datetime(timestamps, unit="ms", utc=True, errors="coerce")
                first = dates.min().normalize() + pd.Timedelta(days=30)
                last = dates.max().normalize() - pd.Timedelta(days=1)
                if first > last or pd.isna(first) or pd.isna(last):
                    print("30일 관찰 기간을 확보할 수 없습니다.")
                    continue
                print(f"기준일 범위 (UTC): {first.date()} ~ {last.date()}")
                selected = input(f"예측 기준일 YYYY-MM-DD [Enter: {last.date()}]: ").strip() or str(last.date())
                args = ["scripts/predict.py", "--events", str(events), "--date", selected]
            except (OSError, ValueError, KeyError) as exc:
                print(f"데이터 확인 실패: {exc}")
                continue
        elif choice == "5":
            args = ["scripts/backtest.py", "--events", str(events)]
            print("과거 구간에서 3회 학습·평가합니다. 결과는 data/reports에 저장됩니다.")
        elif choice == "6":
            args = ["scripts/search_hyperparameters.py", "--events", str(events)]
            print("기본값과 5개 후보를 3개 과거 구간에서 비교합니다. 시간이 걸릴 수 있습니다.")
        else:
            print("0~6 중 선택하세요.")
            continue
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        print("완료되었습니다." if result.returncode == 0 else "실패했습니다. 위 오류 메시지를 확인하세요.")


if __name__ == "__main__":
    main()
