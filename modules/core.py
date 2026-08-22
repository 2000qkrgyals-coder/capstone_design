from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re
import numpy as np
import pandas as pd

from modules.simulation import (
    ALL_AREAS,
    AREA_TYPES,
    CHECKIN_AREAS,
    IM_AREAS,
    SELF_AREAS,
    STAFFED_AREAS,
    MAX_UNITS,
    SERVICE_RATE_PER_MIN,
    recommended_units_from_row,
    infer_arrivals_from_observed,
)

DATE_MIN = "2025-09-01"
DATE_MAX = "2025-10-31"


def load_operation_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig", compression="infer")
    df["일자"] = df["일자"].astype(str)
    df["구역"] = df["구역"].astype(str)
    df["시각"] = df["시각"].astype(str)
    df = df[(df["일자"] >= DATE_MIN) & (df["일자"] <= DATE_MAX)].copy()
    df = df[df["구역"].isin(ALL_AREAS)].copy()
    numeric = [
        "분",
        "계획수요",
        "실시간인원수",
        "계획오픈수",
        "실시간필요수",
        "필요수차이",
        "계획기본직원수",
        "계획지원직원수",
        "계획총직원수",
        "실시간기본직원수",
        "실시간지원직원수",
        "실시간총직원수",
        "직원차이",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["분"] = df["분"].astype(int)
    return df.sort_values(["일자", "분", "구역"]).reset_index(drop=True)


def _parse_hhmm_to_min(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not m:
        return None
    h, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= minute <= 59):
        return None
    return h * 60 + minute


def load_flight_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["일자_dt"] = pd.to_datetime(df["일자"].astype(str), format="%Y%m%d", errors="coerce")
    # 9/1 이른 항공편의 전날 체크인 수요를 계산하기 위해 8/31을 유지한다.
    df = df[(df["일자_dt"] >= pd.Timestamp("2025-08-31")) & (df["일자_dt"] <= pd.Timestamp("2025-10-31"))].copy()
    df = df[df["구분"].astype(str).eq("여객")].copy()
    df = df[~df["상태"].astype(str).eq("취소")].copy()
    df["좌석수"] = pd.to_numeric(df["좌석수"], errors="coerce").fillna(0).clip(lower=0)

    # 항공사별 가장 흔한 체크인 구역을 결측치 fallback으로 사용한다.
    mode_map: Dict[str, str] = {}
    for airline, g in df.groupby("항공사"):
        modes = g["체크인카운터"].dropna().astype(str)
        if not modes.empty:
            mode_map[str(airline)] = modes.mode().iloc[0]
    df["체크인카운터_보정"] = [
        (str(v) if not pd.isna(v) and str(v).strip() else mode_map.get(str(a), ""))
        for a, v in zip(df["항공사"], df["체크인카운터"])
    ]

    # 예상시간이 있으면 예상시간, 없으면 계획시간을 사용한다.
    expected = df["예상시간"].apply(_parse_hhmm_to_min)
    planned = df["계획시간"].apply(_parse_hhmm_to_min)
    df["출발기준분"] = expected.where(expected.notna(), planned)
    df = df[df["출발기준분"].notna()].copy()
    df["출발기준분"] = df["출발기준분"].astype(int)
    df["출발기준시각"] = df["일자_dt"] + pd.to_timedelta(df["출발기준분"], unit="m")
    return df.reset_index(drop=True)


def date_options(operation_df: pd.DataFrame) -> List[str]:
    return sorted(operation_df["일자"].dropna().astype(str).unique().tolist())


def minute_to_hhmm(minute: int) -> str:
    minute = int(max(0, min(1439, minute)))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def hhmm_to_minute(text: str) -> int:
    h, m = str(text).split(":")
    return int(h) * 60 + int(m)


def snapshot(operation_df: pd.DataFrame, date: str, minute: int) -> pd.DataFrame:
    rows = operation_df[(operation_df["일자"] == date) & (operation_df["분"] == int(minute))].copy()
    rows = rows.drop_duplicates("구역", keep="last")
    base = pd.DataFrame({"구역": ALL_AREAS})
    rows = base.merge(rows, on="구역", how="left")
    for col in [
        "계획수요",
        "실시간인원수",
        "계획오픈수",
        "실시간필요수",
        "계획총직원수",
        "실시간총직원수",
    ]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0)
    rows["유형"] = rows["유형"].fillna(rows["구역"].map(AREA_TYPES))
    rows["권고필요수"] = rows.apply(recommended_units_from_row, axis=1)
    rows["최대운영수"] = rows["구역"].map(MAX_UNITS).astype(int)
    return rows


def horizon_frame(operation_df: pd.DataFrame, date: str, start_minute: int, horizon_min: int) -> pd.DataFrame:
    end = min(1439, int(start_minute) + int(horizon_min) - 1)
    d = operation_df[
        (operation_df["일자"] == date)
        & (operation_df["분"] >= int(start_minute))
        & (operation_df["분"] <= end)
    ].copy()
    return d.sort_values(["분", "구역"])


def build_baseline_inputs(
    operation_df: pd.DataFrame,
    date: str,
    start_minute: int,
    horizon_min: int,
) -> Dict[str, object]:
    d = horizon_frame(operation_df, date, start_minute, horizon_min)
    if d.empty:
        raise ValueError("선택한 날짜/시간의 운영 데이터가 없습니다.")
    minutes = sorted(d["분"].unique().tolist())
    n = len(minutes)
    idx = pd.Index(minutes, name="분")

    observed: Dict[str, np.ndarray] = {}
    units: Dict[str, np.ndarray] = {}
    arrivals: Dict[str, np.ndarray] = {}
    initial: Dict[str, float] = {}
    baseline_processed: Dict[str, np.ndarray] = {}

    for area in ALL_AREAS:
        g = d[d["구역"] == area].drop_duplicates("분", keep="last").set_index("분").reindex(idx)
        # 전후 값으로 보간한 뒤 남는 결측은 0 처리
        people = pd.to_numeric(g["실시간인원수"], errors="coerce").interpolate(limit_direction="both").fillna(0).to_numpy(float)
        # 권고 수는 행 단위로 계산. 결측행은 0.
        rec: List[int] = []
        for _, row in g.reset_index().iterrows():
            if pd.isna(row.get("구역")):
                row["구역"] = area
            rec.append(recommended_units_from_row(row))
        unit_arr = np.asarray(rec, dtype=float)
        observed[area] = people
        units[area] = unit_arr
        initial[area] = float(people[0]) if len(people) else 0.0
        arrivals[area] = infer_arrivals_from_observed(
            observed_queue=people,
            baseline_units=unit_arr,
            service_rate_per_min=SERVICE_RATE_PER_MIN[area],
        )

    # 기준 체크인 처리량은 결합 시뮬레이션의 하류(IM) 전이를 계산할 때 사용한다.
    from modules.simulation import simulate_single_area
    for area in CHECKIN_AREAS:
        f = simulate_single_area(
            arrivals=arrivals[area],
            units=units[area],
            initial_queue=initial[area],
            area=area,
        )
        baseline_processed[area] = f["processed"].to_numpy()

    im_total = observed["IM1"] + observed["IM2"]
    im_split = np.divide(
        observed["IM1"],
        im_total,
        out=np.full(n, 0.5, dtype=float),
        where=im_total > 1e-9,
    )
    im_split = pd.Series(im_split).rolling(10, min_periods=1, center=True).mean().clip(0.10, 0.90).to_numpy()

    return {
        "minutes": minutes,
        "observed": observed,
        "baseline_units": units,
        "arrivals": arrivals,
        "initial_queue": initial,
        "baseline_checkin_processed": baseline_processed,
        "baseline_im_arrivals": {"IM1": arrivals["IM1"], "IM2": arrivals["IM2"]},
        "im_split": im_split,
    }


def airline_area_weights(counter_text: str) -> Dict[str, float]:
    areas = [a.strip() for a in str(counter_text).split(",") if a.strip() in CHECKIN_AREAS]
    if not areas:
        return {}
    staffed = [a for a in areas if a in STAFFED_AREAS]
    self_areas = [a for a in areas if a in SELF_AREAS]
    weights: Dict[str, float] = {}

    if "A" in staffed:
        # 대한항공 계열: 프리미엄 A는 소수, 셀프 B와 일반 C/D에 나머지를 분배.
        premium_share = 0.08
        self_share = 0.25 if self_areas else 0.0
        regular = [a for a in staffed if a != "A"]
        weights["A"] = premium_share
        if self_areas:
            for a in self_areas:
                weights[a] = self_share / len(self_areas)
        remaining = 1.0 - premium_share - self_share
        if regular:
            for a in regular:
                weights[a] = remaining / len(regular)
        elif staffed:
            weights["A"] += remaining
    else:
        self_share = 0.35 if self_areas else 0.0
        staffed_share = 1.0 - self_share
        if staffed:
            for a in staffed:
                weights[a] = staffed_share / len(staffed)
        if self_areas:
            for a in self_areas:
                weights[a] = self_share / len(self_areas)

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def _checkin_profile_weight(minutes_before_departure: float) -> float:
    # 체크인 180~60분 전을 활성 구간으로 두고 120분 전에 가장 큰 가중치를 준다.
    if minutes_before_departure < 60 or minutes_before_departure > 180:
        return 0.0
    center_distance = abs(minutes_before_departure - 120.0)
    return 0.20 + 0.80 * max(0.0, 1.0 - center_distance / 60.0)


def build_airline_share_multipliers(
    flight_df: pd.DataFrame,
    date: str,
    minutes: Sequence[int],
    selected_airline: Optional[str],
    demand_change_pct: float,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """항공편 좌석 가중치를 이용해 선택 항공사의 각 구역 기여율을 계산한다.

    절대 승객 수를 새로 만드는 대신, 이미 역산된 분당 유입량에 항공사 점유율만큼
    수요 충격을 적용한다. 따라서 원본 운영 데이터의 스케일을 보존한다.
    """
    n = len(minutes)
    multipliers = {a: np.ones(n, dtype=float) for a in ALL_AREAS}
    if not selected_airline or abs(float(demand_change_pct)) < 1e-9:
        return multipliers, pd.DataFrame()

    base_date = pd.Timestamp(date)
    # 전날 밤 출발편은 다음날 이른 시각 체크인에 영향을 주지 않으므로 해당 날짜와 인접 날짜만 충분하다.
    relevant = flight_df[
        (flight_df["출발기준시각"] >= base_date + pd.Timedelta(minutes=min(minutes) + 60))
        & (flight_df["출발기준시각"] <= base_date + pd.Timedelta(minutes=max(minutes) + 180))
    ].copy()
    if relevant.empty:
        return multipliers, pd.DataFrame()

    total_weight = {a: np.zeros(n, dtype=float) for a in CHECKIN_AREAS}
    selected_weight = {a: np.zeros(n, dtype=float) for a in CHECKIN_AREAS}
    detail_rows: List[Dict[str, object]] = []

    for _, row in relevant.iterrows():
        airline = str(row["항공사"])
        seats = float(row["좌석수"])
        weights = airline_area_weights(str(row["체크인카운터_보정"]))
        if not weights or seats <= 0:
            continue
        dep = pd.Timestamp(row["출발기준시각"])
        for i, minute in enumerate(minutes):
            current = base_date + pd.Timedelta(minutes=int(minute))
            before = (dep - current).total_seconds() / 60.0
            profile = _checkin_profile_weight(before)
            if profile <= 0:
                continue
            seat_weight = seats * profile
            for area, area_share in weights.items():
                w = seat_weight * area_share
                total_weight[area][i] += w
                if airline == selected_airline:
                    selected_weight[area][i] += w
        if airline == selected_airline:
            detail_rows.append(
                {
                    "항공사": airline,
                    "편명": row["편명"],
                    "도착지": row["도착지"],
                    "출발기준시각": dep,
                    "좌석수": int(seats),
                    "체크인구역": row["체크인카운터_보정"],
                }
            )

    shock = float(demand_change_pct) / 100.0
    for area in CHECKIN_AREAS:
        share = np.divide(
            selected_weight[area],
            total_weight[area],
            out=np.zeros(n, dtype=float),
            where=total_weight[area] > 1e-9,
        )
        multipliers[area] = np.clip(1.0 + shock * share, 0.05, 3.0)

    details = pd.DataFrame(detail_rows).drop_duplicates() if detail_rows else pd.DataFrame()
    return multipliers, details


def airline_summary(flight_df: pd.DataFrame, date: str, start_minute: int, horizon_min: int) -> pd.DataFrame:
    start = pd.Timestamp(date) + pd.Timedelta(minutes=int(start_minute))
    end = start + pd.Timedelta(minutes=int(horizon_min) + 180)
    d = flight_df[(flight_df["출발기준시각"] >= start + pd.Timedelta(minutes=60)) & (flight_df["출발기준시각"] <= end)].copy()
    if d.empty:
        return pd.DataFrame(columns=["항공사", "항공편수", "좌석수", "대표 체크인구역"])
    rows = []
    for airline, g in d.groupby("항공사"):
        counter = g["체크인카운터_보정"].mode().iloc[0] if not g["체크인카운터_보정"].mode().empty else ""
        rows.append(
            {
                "항공사": airline,
                "항공편수": int(len(g)),
                "좌석수": int(g["좌석수"].sum()),
                "대표 체크인구역": counter,
            }
        )
    return pd.DataFrame(rows).sort_values(["좌석수", "항공편수"], ascending=False).reset_index(drop=True)


def data_quality_report(operation_df: pd.DataFrame, flight_df: pd.DataFrame) -> Dict[str, object]:
    op_sep = operation_df[(operation_df["일자"] >= DATE_MIN) & (operation_df["일자"] <= DATE_MAX)]
    flight_sep = flight_df[(flight_df["일자_dt"] >= pd.Timestamp(DATE_MIN)) & (flight_df["일자_dt"] <= pd.Timestamp(DATE_MAX))]
    duplicates = int(op_sep.duplicated(["일자", "분", "구역"]).sum())
    missing_counter = int(flight_sep["체크인카운터"].isna().sum())
    return {
        "operation_rows": int(len(op_sep)),
        "operation_dates": int(op_sep["일자"].nunique()),
        "operation_duplicates": duplicates,
        "areas": sorted(op_sep["구역"].unique().tolist()),
        "flight_rows": int(len(flight_sep)),
        "airlines": int(flight_sep["항공사"].nunique()),
        "missing_counter_rows": missing_counter,
        "missing_counter_rate": float(missing_counter / len(flight_sep)) if len(flight_sep) else 0.0,
    }
