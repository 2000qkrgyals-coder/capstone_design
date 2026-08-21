from pathlib import Path
from collections import defaultdict
import math
import re
import shutil

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FLIGHT_PATH = BASE_DIR / "8월31일~10월31일_카운터_기종_정리.csv"
PEOPLE_DIR = BASE_DIR / "data(9~10)"

# 기존 하나짜리 대용량 파일 대신 날짜별 파일로 저장
OUTPUT_DIR = BASE_DIR / "operation_daily"
DATES_PATH = BASE_DIR / "operation_dates.csv"

# 기존 파일은 더 이상 앱에서 사용하지 않음
OLD_OUT_PATH = BASE_DIR / "operation_dashboard_data.csv.gz"


COUNTERS = list("ABCDEFGHIJKLMN")
AREAS = COUNTERS + ["IM1", "IM2"]

SELF_COUNTERS = {"B", "F", "G", "L"}
GENERAL_COUNTERS = set(COUNTERS) - {"A"} - SELF_COUNTERS

IM1_COUNTERS = set("ABCDEFG")
IM2_COUNTERS = set("HIJKLMN")


CHECKIN_WINDOW_MINUTES = 120

# 항공편 좌석수를 해당 구역에 머무르는 예상 인원으로 환산하기 위한 체류시간
A_DWELL_MINUTES = 18
GENERAL_DWELL_MINUTES = 22
SELF_DWELL_MINUTES = 14
IM_DWELL_MINUTES = 10


# 운영 기준
A_PEOPLE_PER_COUNTER = 8
GENERAL_PEOPLE_PER_COUNTER = 5
SELF_PEOPLE_PER_DEVICE = 6

# 기존 전처리 기준 유지
IM_PEOPLE_PER_GATE = 40

MAX_COUNTER_UNITS = 40
MAX_IM_GATES = 6
IM_MIN_GATES_WHEN_ACTIVE = 2


def read_csv_any(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]:
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                low_memory=False,
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV 읽기 실패: {path}\n"
        f"마지막 오류: {last_error}"
    )


def pick_col(columns, candidates):
    cols = list(columns)

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        for col in cols:
            if cand.lower() in str(col).lower():
                return col

    return None


def parse_date(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    m = re.search(
        r"20\d{2}[-./]\d{1,2}[-./]\d{1,2}",
        text,
    )

    if m:
        dt = pd.to_datetime(
            m.group(0)
            .replace(".", "-")
            .replace("/", "-"),
            errors="coerce",
        )

        return (
            None
            if pd.isna(dt)
            else dt.strftime("%Y-%m-%d")
        )

    m = re.search(
        r"(20\d{6})",
        text,
    )

    if m:
        dt = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        return (
            None
            if pd.isna(dt)
            else dt.strftime("%Y-%m-%d")
        )

    dt = pd.to_datetime(
        text,
        errors="coerce",
    )

    if (
        pd.isna(dt)
        or dt.year < 2020
    ):
        return None

    return dt.strftime("%Y-%m-%d")


def normalize_time(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    m = re.search(
        r"(\d{1,2}):(\d{2})",
        text,
    )

    if m:
        hour = int(
            m.group(1)
        )

        minute = int(
            m.group(2)
        )

        if (
            0 <= hour <= 23
            and 0 <= minute <= 59
        ):
            return (
                f"{hour:02d}:"
                f"{minute:02d}"
            )

    if re.fullmatch(
        r"\d{3,4}",
        text,
    ):
        text = text.zfill(4)

        hour = int(
            text[:2]
        )

        minute = int(
            text[2:]
        )

        if (
            0 <= hour <= 23
            and 0 <= minute <= 59
        ):
            return (
                f"{hour:02d}:"
                f"{minute:02d}"
            )

    try:
        num = float(text)

        if 0 <= num < 1:
            total_min = int(
                round(
                    num
                    * 24
                    * 60
                )
            )

            total_min = max(
                0,
                min(
                    1439,
                    total_min,
                ),
            )

            return (
                f"{total_min // 60:02d}:"
                f"{total_min % 60:02d}"
            )

    except Exception:
        pass

    dt = pd.to_datetime(
        text,
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    return dt.strftime("%H:%M")


def date_from_filename(path: Path):
    m = re.search(
        r"(20\d{2}-\d{2}-\d{2})",
        path.name,
    )

    return (
        m.group(1)
        if m
        else None
    )


def parse_counters(value):
    if pd.isna(value):
        return []

    text = (
        str(value)
        .upper()
        .strip()
    )

    if text in [
        "",
        "-",
        "NAN",
        "NONE",
    ]:
        return []

    found = re.findall(
        r"[A-N]",
        text,
    )

    result = []

    for counter in found:
        if counter not in result:
            result.append(
                counter
            )

    return result


def ceil_div(value, base):
    value = float(value)

    if value <= 0:
        return 0

    return math.ceil(
        value / base
    )


def minute_to_hhmm(minute):
    minute = int(minute)

    minute = max(
        0,
        min(
            1439,
            minute,
        ),
    )

    return (
        f"{minute // 60:02d}:"
        f"{minute % 60:02d}"
    )


def counter_dwell_minutes(area):
    if area == "A":
        return A_DWELL_MINUTES

    if area in SELF_COUNTERS:
        return SELF_DWELL_MINUTES

    return GENERAL_DWELL_MINUTES


def counter_support_staff(
    area,
    demand,
    open_units,
):
    demand = float(demand)
    open_units = int(open_units)

    if (
        demand <= 0
        or open_units <= 0
    ):
        return 0

    if area == "A":
        return (
            1
            if demand >= 24
            else 0
        )

    if area in SELF_COUNTERS:
        return min(
            ceil_div(
                open_units,
                6,
            ),
            3,
        )

    if demand < 40:
        return 0

    if demand < 80:
        return 1

    if demand < 120:
        return 2

    return 3


def im_line_staff(demand):
    demand = float(demand)

    if demand <= 80:
        return 0

    if demand <= 160:
        return 1

    if demand <= 240:
        return 2

    return 3


def calc_im_gates(demand):
    demand = float(demand)

    if demand <= 0:
        return 0

    gates = ceil_div(
        demand,
        IM_PEOPLE_PER_GATE,
    )

    gates = max(
        IM_MIN_GATES_WHEN_ACTIVE,
        gates,
    )

    gates = min(
        gates,
        MAX_IM_GATES,
    )

    return gates


def calc_operation(
    area,
    demand,
):
    demand = max(
        float(demand),
        0,
    )

    if area == "A":
        area_type = (
            "프리미엄 체크인"
        )

        unit = "창구"

        open_units = min(
            ceil_div(
                demand,
                A_PEOPLE_PER_COUNTER,
            ),
            MAX_COUNTER_UNITS,
        )

        main_staff = open_units

        support_staff = (
            counter_support_staff(
                area,
                demand,
                open_units,
            )
        )

    elif area in SELF_COUNTERS:
        area_type = (
            "셀프 체크인"
        )

        unit = "기기"

        open_units = min(
            ceil_div(
                demand,
                SELF_PEOPLE_PER_DEVICE,
            ),
            MAX_COUNTER_UNITS,
        )

        main_staff = 0

        support_staff = (
            counter_support_staff(
                area,
                demand,
                open_units,
            )
        )

    elif area in GENERAL_COUNTERS:
        area_type = (
            "일반 체크인"
        )

        unit = "창구"

        open_units = min(
            ceil_div(
                demand,
                GENERAL_PEOPLE_PER_COUNTER,
            ),
            MAX_COUNTER_UNITS,
        )

        main_staff = open_units

        support_staff = (
            counter_support_staff(
                area,
                demand,
                open_units,
            )
        )

    elif area in [
        "IM1",
        "IM2",
    ]:
        area_type = (
            "출국장 진입"
        )

        unit = "출입문"

        open_units = calc_im_gates(
            demand
        )

        main_staff = open_units

        support_staff = (
            im_line_staff(
                demand
            )
        )

    else:
        area_type = "기타"
        unit = "개"

        open_units = 0
        main_staff = 0
        support_staff = 0

    return {
        "유형": area_type,
        "단위": unit,
        "필요수": int(
            open_units
        ),
        "기본직원수": int(
            main_staff
        ),
        "지원직원수": int(
            support_staff
        ),
        "총직원수": int(
            main_staff
            + support_staff
        ),
    }


def make_flight_plan():
    if not FLIGHT_PATH.exists():
        raise FileNotFoundError(
            f"항공편 파일 없음: "
            f"{FLIGHT_PATH}"
        )

    df = read_csv_any(
        FLIGHT_PATH
    )

    date_col = pick_col(
        df.columns,
        [
            "일자",
            "날짜",
            "date",
            "data_date",
        ],
    )

    time_col = pick_col(
        df.columns,
        [
            "계획시간",
            "계획 출발시간",
            "계획출발시간",
            "계획",
            "STD",
            "scheduled",
        ],
    )

    airline_col = pick_col(
        df.columns,
        [
            "항공사",
            "airline",
        ],
    )

    seat_col = pick_col(
        df.columns,
        [
            "좌석수",
            "seats",
        ],
    )

    counter_col = pick_col(
        df.columns,
        [
            "체크인카운터",
            "체크인 카운터",
            "counter",
        ],
    )

    missing = []

    for name, col in {
        "일자": date_col,
        "계획시간": time_col,
        "항공사": airline_col,
        "좌석수": seat_col,
        "체크인카운터": counter_col,
    }.items():
        if col is None:
            missing.append(
                name
            )

    if missing:
        raise ValueError(
            f"항공편 파일 필수 컬럼 누락: "
            f"{missing}"
        )

    print(
        "[항공편 컬럼]"
    )

    print(
        f"일자={date_col}, "
        f"계획시간={time_col}, "
        f"항공사={airline_col}, "
        f"좌석수={seat_col}, "
        f"체크인카운터={counter_col}"
    )

    df["_date"] = (
        df[date_col]
        .apply(parse_date)
    )

    df["_time"] = (
        df[time_col]
        .apply(normalize_time)
    )

    df["_dt"] = pd.to_datetime(
        df["_date"]
        + " "
        + df["_time"],
        errors="coerce",
    )

    df["_seats"] = (
        pd.to_numeric(
            df[seat_col],
            errors="coerce",
        )
        .fillna(0)
    )

    df["_airline"] = (
        df[airline_col]
        .astype(str)
    )

    df["_counters"] = (
        df[counter_col]
        .apply(parse_counters)
    )

    df = df.dropna(
        subset=[
            "_dt",
        ]
    )

    df = df[
        df["_seats"] > 0
    ]

    df = df[
        df["_counters"]
        .apply(len)
        > 0
    ]

    if df.empty:
        raise RuntimeError(
            "항공편 데이터 정리 후 "
            "남은 행이 없습니다."
        )

    demand = defaultdict(
        float
    )

    for _, row in df.iterrows():
        planned_dt = (
            row["_dt"]
        )

        seats = float(
            row["_seats"]
        )

        airline = (
            row["_airline"]
        )

        counters = (
            row["_counters"]
        )

        allocation = {}

        if (
            "대한항공"
            in airline
            and "A"
            in counters
        ):
            premium = (
                seats
                * 0.10
            )

            normal_counters = [
                counter
                for counter in counters
                if counter != "A"
            ]

            allocation["A"] = (
                premium
            )

            if normal_counters:
                normal_each = (
                    seats
                    - premium
                ) / len(
                    normal_counters
                )

                for counter in normal_counters:
                    allocation[
                        counter
                    ] = normal_each

        else:
            each = (
                seats
                / len(counters)
            )

            for counter in counters:
                allocation[
                    counter
                ] = each

        start_dt = (
            planned_dt
            - pd.Timedelta(
                minutes=180
            )
        )

        end_dt = (
            planned_dt
            - pd.Timedelta(
                minutes=60
            )
        )

        current = (
            start_dt
            .floor("min")
        )

        while current < end_dt:
            date = (
                current
                .strftime(
                    "%Y-%m-%d"
                )
            )

            minute = (
                current.hour
                * 60
                + current.minute
            )

            im1_value = 0
            im2_value = 0

            for (
                area,
                seats_value,
            ) in allocation.items():

                dwell = (
                    counter_dwell_minutes(
                        area
                    )
                )

                planned_people = (
                    seats_value
                    * dwell
                    / CHECKIN_WINDOW_MINUTES
                )

                demand[
                    (
                        date,
                        minute,
                        area,
                    )
                ] += planned_people

                im_people = (
                    seats_value
                    * IM_DWELL_MINUTES
                    / CHECKIN_WINDOW_MINUTES
                )

                if area in IM1_COUNTERS:
                    im1_value += (
                        im_people
                    )

                elif area in IM2_COUNTERS:
                    im2_value += (
                        im_people
                    )

            if im1_value > 0:
                demand[
                    (
                        date,
                        minute,
                        "IM1",
                    )
                ] += im1_value

            if im2_value > 0:
                demand[
                    (
                        date,
                        minute,
                        "IM2",
                    )
                ] += im2_value

            current += pd.Timedelta(
                minutes=1
            )

    rows = [
        {
            "일자": key[0],
            "분": key[1],
            "구역": key[2],
            "계획수요": round(
                value,
                1,
            ),
        }
        for key, value
        in demand.items()
    ]

    plan = pd.DataFrame(
        rows
    )

    if plan.empty:
        raise RuntimeError(
            "항공편 기반 계획 데이터가 "
            "비어 있습니다."
        )

    plan = (
        plan.groupby(
            [
                "일자",
                "분",
                "구역",
            ],
            as_index=False,
        )[
            "계획수요"
        ]
        .sum()
        .round(
            {
                "계획수요": 1,
            }
        )
    )

    print(
        f"[항공편 계획] "
        f"{len(plan):,}행"
    )

    print(
        f"[항공편 계획 날짜] "
        f"{plan['일자'].min()} "
        f"~ "
        f"{plan['일자'].max()}"
    )

    return plan


def make_people_data():
    files = sorted(
        PEOPLE_DIR.glob(
            "area_count_time_full_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"인원수 파일 없음: "
            f"{PEOPLE_DIR}"
        )

    print(
        f"[인원수 파일 수] "
        f"{len(files):,}"
    )

    frames = []

    for path in files:
        file_date = (
            date_from_filename(
                path
            )
        )

        if file_date is None:
            print(
                f"[SKIP] "
                f"파일명 날짜 인식 실패: "
                f"{path.name}"
            )
            continue

        df = read_csv_any(
            path
        )

        time_col = pick_col(
            df.columns,
            [
                "time_index",
                "minute_index",
                "시각",
                "time",
            ],
        )

        area_col = pick_col(
            df.columns,
            [
                "area",
                "구역",
            ],
        )

        people_col = pick_col(
            df.columns,
            [
                "num_people",
                "인원수",
                "count",
            ],
        )

        if (
            time_col is None
            or area_col is None
            or people_col is None
        ):
            print(
                f"[SKIP] "
                f"필수 컬럼 부족: "
                f"{path.name}"
            )

            print(
                f"       columns="
                f"{list(df.columns)}"
            )

            continue

        df["_date"] = (
            file_date
        )

        df["_area"] = (
            df[area_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df = df[
            df["_area"]
            .isin(AREAS)
        ]

        if df.empty:
            continue

        t = pd.to_numeric(
            df[time_col],
            errors="coerce",
        )

        valid_t = (
            t.dropna()
        )

        if valid_t.empty:
            hhmm = (
                df[time_col]
                .apply(
                    normalize_time
                )
            )

            minute = (
                pd.to_numeric(
                    hhmm.str.slice(
                        0,
                        2,
                    ),
                    errors="coerce",
                )
                * 60
                + pd.to_numeric(
                    hhmm.str.slice(
                        3,
                        5,
                    ),
                    errors="coerce",
                )
            )

        else:
            min_t = (
                valid_t.min()
            )

            max_t = (
                valid_t.max()
            )

            if max_t > 1440:
                minute = (
                    (
                        t - 1
                    )
                    // 6
                ).astype(
                    "Int64"
                )

            elif min_t >= 1:
                minute = (
                    t - 1
                ).astype(
                    "Int64"
                )

            else:
                minute = (
                    t.astype(
                        "Int64"
                    )
                )

        df["_minute"] = (
            minute
        )

        df["_people"] = (
            pd.to_numeric(
                df[people_col],
                errors="coerce",
            )
            .fillna(0)
        )

        part = (
            df[
                [
                    "_date",
                    "_minute",
                    "_area",
                    "_people",
                ]
            ]
            .dropna()
            .rename(
                columns={
                    "_date": "일자",
                    "_minute": "분",
                    "_area": "구역",
                    "_people": "실시간인원수",
                }
            )
        )

        if part.empty:
            continue

        part["분"] = (
            part["분"]
            .astype(int)
        )

        part = part[
            (
                part["분"] >= 0
            )
            & (
                part["분"] <= 1439
            )
        ]

        if part.empty:
            continue

        frames.append(
            part
        )

    if not frames:
        raise RuntimeError(
            "인원수 데이터 변환 결과가 "
            "비어 있습니다. "
            "area 컬럼 값이 A~N, "
            "IM1, IM2 형태인지 "
            "확인해야 합니다."
        )

    people = pd.concat(
        frames,
        ignore_index=True,
    )

    people = (
        people.groupby(
            [
                "일자",
                "분",
                "구역",
            ],
            as_index=False,
        )[
            "실시간인원수"
        ]
        .mean()
        .round(
            {
                "실시간인원수": 1,
            }
        )
    )

    if people.empty:
        raise RuntimeError(
            "인원수 데이터가 "
            "groupby 이후 "
            "0행이 되었습니다."
        )

    print(
        f"[인원수 데이터] "
        f"{len(people):,}행"
    )

    print(
        f"[인원수 날짜] "
        f"{people['일자'].min()} "
        f"~ "
        f"{people['일자'].max()}"
    )

    print(
        f"[인원수 구역] "
        f"{', '.join(sorted(people['구역'].unique()))}"
    )

    return people


def im_status(
    area,
    value,
    gates,
):
    if area not in [
        "IM1",
        "IM2",
    ]:
        return ""

    value = float(value)
    gates = int(gates)

    if value <= 0:
        return (
            "출입문 대기 수요 없음"
        )

    if gates <= 2:
        return (
            "기본 개방 수준"
        )

    if gates <= 4:
        return (
            "부분 증설 권고"
        )

    return (
        "집중 운영 권고"
    )


def make_action(
    area,
    unit,
    diff,
):
    diff = int(diff)

    if diff > 0:
        if area in [
            "IM1",
            "IM2",
        ]:
            return (
                f"출입문 {diff}개 "
                "추가 개방 필요"
            )

        if unit == "기기":
            return (
                f"셀프기기 {diff}대 "
                "추가 운영 필요"
            )

        return (
            f"창구 {diff}개 "
            "추가 운영 필요"
        )

    if diff < 0:
        if area in [
            "IM1",
            "IM2",
        ]:
            return (
                f"출입문 "
                f"{abs(diff)}개 "
                "감축 가능"
            )

        if unit == "기기":
            return (
                f"셀프기기 "
                f"{abs(diff)}대 "
                "감축 가능"
            )

        return (
            f"창구 "
            f"{abs(diff)}개 "
            "감축 가능"
        )

    return "계획 유지"


def add_operation_columns(df):
    rows = []

    for _, row in df.iterrows():
        area = (
            row["구역"]
        )

        plan_calc = (
            calc_operation(
                area,
                row["계획수요"],
            )
        )

        real_calc = (
            calc_operation(
                area,
                row[
                    "실시간인원수"
                ],
            )
        )

        diff_units = (
            real_calc["필요수"]
            - plan_calc["필요수"]
        )

        diff_staff = (
            real_calc["총직원수"]
            - plan_calc["총직원수"]
        )

        if diff_units > 0:
            status = (
                "추가 필요"
            )

        elif diff_units < 0:
            status = (
                "감축 가능"
            )

        else:
            status = (
                "계획 유지"
            )

        rows.append(
            {
                "일자": (
                    row["일자"]
                ),
                "시각": (
                    minute_to_hhmm(
                        row["분"]
                    )
                ),
                "분": int(
                    row["분"]
                ),
                "구역": area,
                "유형": (
                    real_calc["유형"]
                ),
                "단위": (
                    real_calc["단위"]
                ),
                "계획수요": round(
                    float(
                        row["계획수요"]
                    ),
                    1,
                ),
                "실시간인원수": round(
                    float(
                        row[
                            "실시간인원수"
                        ]
                    ),
                    1,
                ),
                "계획오픈수": (
                    plan_calc["필요수"]
                ),
                "실시간필요수": (
                    real_calc["필요수"]
                ),
                "필요수차이": (
                    diff_units
                ),
                "계획기본직원수": (
                    plan_calc[
                        "기본직원수"
                    ]
                ),
                "계획지원직원수": (
                    plan_calc[
                        "지원직원수"
                    ]
                ),
                "계획총직원수": (
                    plan_calc[
                        "총직원수"
                    ]
                ),
                "실시간기본직원수": (
                    real_calc[
                        "기본직원수"
                    ]
                ),
                "실시간지원직원수": (
                    real_calc[
                        "지원직원수"
                    ]
                ),
                "실시간총직원수": (
                    real_calc[
                        "총직원수"
                    ]
                ),
                "직원차이": (
                    diff_staff
                ),
                "상태": status,
                "권고": make_action(
                    area,
                    real_calc["단위"],
                    diff_units,
                ),
                "IM판단": im_status(
                    area,
                    row[
                        "실시간인원수"
                    ],
                    real_calc[
                        "필요수"
                    ],
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def prepare_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_daily_files(out):
    prepare_output_dir()

    manifest_rows = []

    for (
        date_value,
        part,
    ) in out.groupby(
        "일자",
        sort=True,
    ):
        date_value = str(
            date_value
        )

        file_name = (
            f"operation_"
            f"{date_value}"
            f".csv.gz"
        )

        file_path = (
            OUTPUT_DIR
            / file_name
        )

        part = (
            part.sort_values(
                [
                    "분",
                    "구역",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        part.to_csv(
            file_path,
            index=False,
            encoding="utf-8-sig",
            compression="gzip",
        )

        size_mb = (
            file_path
            .stat()
            .st_size
            / (
                1024
                * 1024
            )
        )

        manifest_rows.append(
            {
                "일자": date_value,
                "파일": file_name,
                "행수": len(part),
                "용량MB": round(
                    size_mb,
                    3,
                ),
            }
        )

        print(
            f"[SAVE] "
            f"{date_value} -> "
            f"{file_name} "
            f"({len(part):,}행 / "
            f"{size_mb:.2f} MB)"
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    if manifest.empty:
        raise RuntimeError(
            "날짜별 저장 결과가 없습니다."
        )

    manifest.to_csv(
        DATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return manifest


def main():
    print(
        "=" * 80
    )

    print(
        "T2 운영 데이터 날짜별 분할 생성"
    )

    print(
        "=" * 80
    )

    print(
        "\n[1] "
        "항공편 기반 사전 운영계획 생성"
    )

    plan = make_flight_plan()

    print(
        "\n[2] "
        "인원수 데이터 통합"
    )

    people = make_people_data()

    valid_dates = sorted(
        set(
            plan[
                "일자"
            ].dropna()
        )
        |
        set(
            people[
                "일자"
            ].dropna()
        )
    )

    valid_dates = [
        date_value
        for date_value
        in valid_dates
        if str(
            date_value
        ).startswith(
            "2025-"
        )
    ]

    plan = (
        plan[
            plan["일자"]
            .isin(
                valid_dates
            )
        ]
        .copy()
    )

    people = (
        people[
            people["일자"]
            .isin(
                valid_dates
            )
        ]
        .copy()
    )

    print(
        "\n[3] "
        "계획 데이터와 "
        "실시간 인원수 데이터 결합"
    )

    df = pd.merge(
        plan,
        people,
        on=[
            "일자",
            "분",
            "구역",
        ],
        how="outer",
    )

    del plan
    del people

    df["계획수요"] = (
        df["계획수요"]
        .fillna(0)
    )

    df["실시간인원수"] = (
        df["실시간인원수"]
        .fillna(0)
    )

    df["분"] = pd.to_numeric(
        df["분"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "분",
        ]
    )

    df["분"] = (
        df["분"]
        .astype(int)
    )

    df = df[
        (
            df["분"] >= 0
        )
        &
        (
            df["분"] <= 1439
        )
    ]

    df = (
        df[
            df["구역"]
            .isin(
                AREAS
            )
        ]
        .copy()
    )

    if df.empty:
        raise RuntimeError(
            "최종 결합 데이터가 0행입니다."
        )

    print(
        f"[결합 행 수] "
        f"{len(df):,}"
    )

    print(
        "\n[4] "
        "운영 필요 수와 직원 수 계산"
    )

    out = add_operation_columns(
        df
    )

    del df

    out = (
        out.sort_values(
            [
                "일자",
                "분",
                "구역",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n[5] "
        "날짜별 파일 저장"
    )

    manifest = (
        save_daily_files(
            out
        )
    )

    total_size_mb = (
        sum(
            path.stat().st_size
            for path
            in OUTPUT_DIR.glob(
                "operation_*.csv.gz"
            )
        )
        / (
            1024
            * 1024
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "[완료]"
    )

    print(
        f"날짜 수: "
        f"{len(manifest):,}"
    )

    print(
        f"전체 행 수: "
        f"{len(out):,}"
    )

    print(
        f"날짜 범위: "
        f"{out['일자'].min()} "
        f"~ "
        f"{out['일자'].max()}"
    )

    print(
        f"날짜별 파일 전체 용량: "
        f"{total_size_mb:.2f} MB"
    )

    print(
        f"manifest: "
        f"{DATES_PATH.name}"
    )

    print(
        f"data folder: "
        f"{OUTPUT_DIR.name}"
    )

    if OLD_OUT_PATH.exists():
        print(
            f"[참고] 기존 "
            f"{OLD_OUT_PATH.name} "
            "파일은 이제 앱에서 사용하지 않습니다. "
            "로컬 백업으로 두거나 "
            "Git에서 제외해도 됩니다."
        )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
