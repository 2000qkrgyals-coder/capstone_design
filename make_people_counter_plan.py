from pathlib import Path
import re
import math
import pandas as pd
import numpy as np


# =========================================================
# 인원수 기반 체크인카운터 운영·인력배치 전처리 스크립트
#
# 입력:
#   data/area_count_time_full_2025-08-31.csv
#   data/area_count_time_full_2025-09-01.csv
#   ...
#   data/area_count_time_full_2025-09-30.csv
#
# 출력:
#   data/people_counter_plan_by_time.csv.gz
#
# 핵심:
# - A~N 단일 알파벳 체크인카운터만 사용
# - IM1, IM2, OUTSIDE, GH 제외
# - data_date가 없으면 파일명에서 날짜 추출
# - time_index는 10초 단위
# - 같은 분 안의 여러 10초 데이터는 평균 인원수로 집계
# - 혼잡 기준: 주의 80명, 혼잡 120명, 매우 혼잡 160명
# - 오픈 권고: 실제 인원수 5명당 창구 1개
# =========================================================


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATTERN = "area_count_time_full_*.csv"
OUTPUT_FILE = DATA_DIR / "people_counter_plan_by_time.csv.gz"

COUNTERS = list("ABCDEFGHIJKLMN")

CONGESTION_CAUTION = 80
CONGESTION_CROWDED = 120
CONGESTION_VERY_CROWDED = 160

OPEN_COUNTER_UNIT_PEOPLE = 5
MAX_OPEN_COUNTERS = 40

SUPPORT_START_PEOPLE = 80
SUPPORT_UNIT_PEOPLE = 40
MAX_SUPPORT_STAFF = 3


def find_input_files():
    files = sorted(DATA_DIR.glob(INPUT_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"입력 CSV 파일을 찾지 못했습니다: {DATA_DIR / INPUT_PATTERN}"
        )

    return files


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 읽기 실패: {path}\n마지막 오류: {last_error}")


def extract_date_from_filename(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)

    if not match:
        raise ValueError(f"파일명에서 날짜를 찾지 못했습니다: {path.name}")

    return match.group(1)


def normalize_area(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def time_index_to_minute_index(value):
    try:
        idx = int(float(value))
    except Exception:
        return np.nan

    if idx < 1:
        return np.nan

    minute_index = (idx - 1) // 6

    if minute_index < 0 or minute_index > 1439:
        return np.nan

    return int(minute_index)


def minute_index_to_hhmm(minute_index: int) -> str:
    hour = int(minute_index) // 60
    minute = int(minute_index) % 60
    return f"{hour:02d}:{minute:02d}"


def get_counter_type(counter: str) -> str:
    counter = str(counter).strip().upper()

    if counter == "A":
        return "프리미엄"

    if counter in ["B", "L"]:
        return "셀프"

    if counter in ["F", "G"]:
        return "혼합"

    return "일반"


def get_congestion_level(num_people: float) -> str:
    if num_people >= CONGESTION_VERY_CROWDED:
        return "매우 혼잡"

    if num_people >= CONGESTION_CROWDED:
        return "혼잡"

    if num_people >= CONGESTION_CAUTION:
        return "주의"

    return "보통"


def get_congestion_score(level: str) -> int:
    score_map = {
        "보통": 0,
        "주의": 1,
        "혼잡": 2,
        "매우 혼잡": 3,
    }

    return score_map.get(level, 0)


def get_recommended_open_counters(num_people: float) -> int:
    if num_people <= 0:
        return 0

    open_count = math.ceil(num_people / OPEN_COUNTER_UNIT_PEOPLE)
    open_count = max(open_count, 1)
    open_count = min(open_count, MAX_OPEN_COUNTERS)

    return int(open_count)


def get_support_staff(num_people: float) -> int:
    if num_people <= SUPPORT_START_PEOPLE:
        return 0

    support = math.ceil((num_people - SUPPORT_START_PEOPLE) / SUPPORT_UNIT_PEOPLE)
    support = max(support, 0)
    support = min(support, MAX_SUPPORT_STAFF)

    return int(support)


def preprocess_one_file(path: Path) -> pd.DataFrame:
    print(f"[LOAD] {path.name}")

    df = read_csv_safely(path)

    required_cols = ["time_index", "area", "num_people"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 누락: {missing}")

    if "data_date" not in df.columns:
        file_date = extract_date_from_filename(path)
        df["data_date"] = file_date
        print(f"  [INFO] data_date 없음 → 파일명 날짜 사용: {file_date}")

    df = df[["data_date", "time_index", "area", "num_people"]].copy()

    df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["counter"] = df["area"].apply(normalize_area)

    df = df[df["counter"].isin(COUNTERS)].copy()

    df["num_people"] = pd.to_numeric(df["num_people"], errors="coerce").fillna(0)
    df["minute_index"] = df["time_index"].apply(time_index_to_minute_index)

    df = df.dropna(subset=["data_date", "minute_index"]).copy()
    df["minute_index"] = df["minute_index"].astype(int)
    df["time_hhmm"] = df["minute_index"].apply(minute_index_to_hhmm)
    df["counter_type"] = df["counter"].apply(get_counter_type)

    grouped = (
        df.groupby(
            ["data_date", "minute_index", "time_hhmm", "counter", "counter_type"],
            as_index=False
        )
        .agg(num_people=("num_people", "mean"))
    )

    grouped["num_people"] = grouped["num_people"].round(2)

    print(
        f"  [OK] {path.name}: "
        f"{grouped['data_date'].min()} / "
        f"{grouped['counter'].nunique()}개 카운터 / "
        f"{len(grouped):,}행"
    )

    return grouped


def make_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("[GRID] 날짜 × 1440분 × A~N 카운터 전체 틀 생성")
    print("=" * 80)

    dates = sorted(df["data_date"].dropna().unique().tolist())
    minutes = list(range(1440))

    grid = pd.MultiIndex.from_product(
        [dates, minutes, COUNTERS],
        names=["data_date", "minute_index", "counter"]
    ).to_frame(index=False)

    grid["time_hhmm"] = grid["minute_index"].apply(minute_index_to_hhmm)
    grid["counter_type"] = grid["counter"].apply(get_counter_type)

    merged = grid.merge(
        df,
        on=["data_date", "minute_index", "time_hhmm", "counter", "counter_type"],
        how="left"
    )

    merged["num_people"] = merged["num_people"].fillna(0).round(2)

    print(f"[OK] 전체 행 수: {len(merged):,}")
    print(f"[OK] 날짜 수: {len(dates):,}")
    print(f"[OK] 카운터: {', '.join(COUNTERS)}")

    return merged


def add_operation_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("[CALC] 혼잡등급·오픈카운터·인력배치 계산")
    print("=" * 80)

    df = df.copy()

    df["혼잡등급"] = df["num_people"].apply(get_congestion_level)
    df["혼잡점수"] = df["혼잡등급"].apply(get_congestion_score)

    df["최대오픈카운터수"] = MAX_OPEN_COUNTERS
    df["권고오픈카운터수"] = df["num_people"].apply(get_recommended_open_counters)

    df["창구운영직원수"] = df["권고오픈카운터수"]
    df["현장지원직원수"] = df["num_people"].apply(get_support_staff)
    df["권고직원수"] = df["창구운영직원수"] + df["현장지원직원수"]

    df["주의기준"] = CONGESTION_CAUTION
    df["혼잡기준"] = CONGESTION_CROWDED
    df["매우혼잡기준"] = CONGESTION_VERY_CROWDED
    df["창구1개당기준인원"] = OPEN_COUNTER_UNIT_PEOPLE
    df["현장지원시작기준"] = SUPPORT_START_PEOPLE
    df["현장지원최대"] = MAX_SUPPORT_STAFF

    df["계산설명"] = (
        "실제 인원수 기반 / "
        "1분 평균 / "
        "주의 80명, 혼잡 120명, 매우 혼잡 160명 / "
        "5명당 창구 1개 / "
        "80명 초과부터 현장지원 추가"
    )

    output_cols = [
        "data_date",
        "minute_index",
        "time_hhmm",
        "counter",
        "counter_type",
        "num_people",
        "혼잡등급",
        "혼잡점수",
        "최대오픈카운터수",
        "권고오픈카운터수",
        "창구운영직원수",
        "현장지원직원수",
        "권고직원수",
        "주의기준",
        "혼잡기준",
        "매우혼잡기준",
        "창구1개당기준인원",
        "현장지원시작기준",
        "현장지원최대",
        "계산설명",
    ]

    df = df[output_cols].sort_values(
        ["data_date", "minute_index", "counter"]
    ).reset_index(drop=True)

    print("[OK] 계산 완료")

    return df


def save_output(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("[SAVE] 앱용 압축 파일 저장")
    print("=" * 80)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"[DELETE] 기존 파일 삭제: {OUTPUT_FILE}")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        compression="gzip"
    )

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print(f"[SAVE] {OUTPUT_FILE}")
    print(f"[SIZE] {size_mb:,.2f} MB")


def print_final_check(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("[CHECK] 최종 확인")
    print("=" * 80)

    print(f"최소 일자: {df['data_date'].min()}")
    print(f"최대 일자: {df['data_date'].max()}")
    print(f"총 행 수: {len(df):,}")
    print(f"카운터 목록: {', '.join(sorted(df['counter'].unique()))}")

    print("\n시각 예시:")
    print(df[["minute_index", "time_hhmm"]].drop_duplicates().head(10).to_string(index=False))

    print("\n혼잡등급 분포:")
    print(df["혼잡등급"].value_counts().to_string())

    print("\n인원수 요약:")
    print(df["num_people"].describe().round(2).to_string())

    print("\n권고오픈카운터수 요약:")
    print(df["권고오픈카운터수"].describe().round(2).to_string())

    if df["data_date"].min() == "2025-08-31":
        print("\n[OK] 2025-08-31 포함")
    else:
        print("\n[WARNING] 2025-08-31이 시작일로 잡히지 않았습니다.")

    if df["data_date"].max() == "2025-09-30":
        print("[OK] 2025-09-30 포함")
    else:
        print("[WARNING] 2025-09-30이 종료일로 잡히지 않았습니다.")


def main():
    print("=" * 80)
    print("인원수 기반 체크인카운터 운영·인력배치 전처리 시작")
    print("=" * 80)

    input_files = find_input_files()

    print(f"[INFO] 입력 파일 수: {len(input_files):,}")

    for path in input_files:
        print(f"  - {path.name}")

    parts = []

    for path in input_files:
        part = preprocess_one_file(path)
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)

    print("\n" + "=" * 80)
    print("[MERGE] 전체 파일 병합 완료")
    print("=" * 80)
    print(f"[INFO] 병합 행 수: {len(combined):,}")
    print(f"[INFO] 최소 일자: {combined['data_date'].min()}")
    print(f"[INFO] 최대 일자: {combined['data_date'].max()}")

    combined = make_full_grid(combined)
    combined = add_operation_columns(combined)

    save_output(combined)
    print_final_check(combined)

    print("\n[DONE] 전처리 완료")
    print("\n다음 명령어로 실행하세요.")
    print(r'cd /d "G:\캡디\2026-06-22 인원수 기반 카운터 배치"')
    print("streamlit cache clear")
    print("streamlit run app.py")


if __name__ == "__main__":
    main()