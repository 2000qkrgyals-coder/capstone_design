from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd

STAFFED_AREAS: List[str] = ["A", "C", "D", "E", "H", "J", "K", "M", "N"]
SELF_AREAS: List[str] = ["B", "F", "G", "L"]
IM_AREAS: List[str] = ["IM1", "IM2"]
CHECKIN_AREAS: List[str] = STAFFED_AREAS + SELF_AREAS
ALL_AREAS: List[str] = CHECKIN_AREAS + IM_AREAS

AREA_TYPES: Dict[str, str] = {
    "A": "프리미엄 체크인",
    "B": "셀프 체크인",
    "C": "일반 체크인",
    "D": "일반 체크인",
    "E": "일반 체크인",
    "F": "셀프 체크인",
    "G": "셀프 체크인",
    "H": "일반 체크인",
    "J": "일반 체크인",
    "K": "일반 체크인",
    "L": "셀프 체크인",
    "M": "일반 체크인",
    "N": "일반 체크인",
    "IM1": "출국장 진입",
    "IM2": "출국장 진입",
}

UNIT_LABELS: Dict[str, str] = {
    **{a: "창구" for a in STAFFED_AREAS},
    **{a: "기기" for a in SELF_AREAS},
    "IM1": "출입문",
    "IM2": "출입문",
}

# 사용자 설명과 원본 운영 데이터의 최대값을 반영한 기본 물리적 상한.
# 일반/프리미엄 체크인 라인은 1~40번 연속 슬롯으로 취급한다.
MAX_UNITS: Dict[str, int] = {
    **{a: 40 for a in STAFFED_AREAS},
    **{a: 40 for a in SELF_AREAS},
    "IM1": 6,
    "IM2": 6,
}

# 시뮬레이션 처리율은 현재 운영 데이터의 자원 산정 규칙과 목표 대기시간을 일관되게
# 연결한 보정값이다. 실제 승객별 서비스 완료 로그가 확보되면 교체해야 한다.
# - 일반: 5명/창구를 10분 목표로 처리 -> 0.5명/분
# - 프리미엄: 계획 데이터가 약 8명/창구 -> 0.8명/분
# - 셀프: 6명/기기를 5분 목표로 처리 -> 1.2명/분
# - IM: 30명/출입문을 3분 목표로 처리 -> 10명/분
SERVICE_RATE_PER_MIN: Dict[str, float] = {
    "A": 0.8,
    "B": 1.2,
    "C": 0.5,
    "D": 0.5,
    "E": 0.5,
    "F": 1.2,
    "G": 1.2,
    "H": 0.5,
    "J": 0.5,
    "K": 0.5,
    "L": 1.2,
    "M": 0.5,
    "N": 0.5,
    "IM1": 10.0,
    "IM2": 10.0,
}

TARGET_WAIT_MIN: Dict[str, float] = {
    **{a: 10.0 for a in STAFFED_AREAS},
    **{a: 5.0 for a in SELF_AREAS},
    "IM1": 3.0,
    "IM2": 3.0,
}

KEEP_RATE: Dict[str, float] = {
    "A": 0.70,
    **{a: 0.50 for a in SELF_AREAS},
    **{a: 0.60 for a in ["C", "D", "E", "H", "J", "K", "M", "N"]},
    "IM1": 0.50,
    "IM2": 0.50,
}


@dataclass(frozen=True)
class ScenarioMetrics:
    avg_wait_min: float
    p90_wait_min: float
    max_wait_min: float
    max_queue: float
    total_processed: float
    sla_violation_rate: float
    congestion_minutes: int
    avg_staff: float
    peak_staff: int
    avg_utilization: float
    peak_utilization: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "avg_wait_min": self.avg_wait_min,
            "p90_wait_min": self.p90_wait_min,
            "max_wait_min": self.max_wait_min,
            "max_queue": self.max_queue,
            "total_processed": self.total_processed,
            "sla_violation_rate": self.sla_violation_rate,
            "congestion_minutes": self.congestion_minutes,
            "avg_staff": self.avg_staff,
            "peak_staff": self.peak_staff,
            "avg_utilization": self.avg_utilization,
            "peak_utilization": self.peak_utilization,
        }


def clamp_units(area: str, units: int) -> int:
    return int(max(0, min(MAX_UNITS[area], int(round(units)))))


def calc_im_gates(people: float) -> int:
    people = max(0.0, float(people))
    if people <= 0:
        return 0
    return int(min(6, max(3, math.ceil(people / 30.0))))


def calc_im_support_staff(gates: int) -> int:
    gates = int(max(0, gates))
    if gates <= 0:
        return 0
    if gates <= 3:
        return 1
    if gates <= 5:
        return 2
    return 3


def staff_from_units(area: str, units: int) -> int:
    """원본 운영 앱의 직원 산정 규칙을 재사용한다."""
    units = clamp_units(area, units)
    if units <= 0:
        return 0
    if area == "A":
        return units + (1 if units >= 3 else 0)
    if area in SELF_AREAS:
        return min(math.ceil(units / 6), 3)
    if area in IM_AREAS:
        return units + calc_im_support_staff(units)
    if units < 8:
        support = 0
    elif units < 16:
        support = 1
    elif units < 24:
        support = 2
    else:
        support = 3
    return units + support


def total_staff(units: Mapping[str, int]) -> int:
    return int(sum(staff_from_units(a, int(units.get(a, 0))) for a in ALL_AREAS))


def recommended_units_from_row(row: pd.Series) -> int:
    """기존 2번 앱의 add_recommendation_columns 로직을 단일 행에 적용한다."""
    area = str(row["구역"])
    plan = int(round(float(row.get("계획오픈수", 0))))
    live = int(round(float(row.get("실시간필요수", 0))))

    if area in IM_AREAS:
        plan = calc_im_gates(float(row.get("계획수요", 0)))
        live = calc_im_gates(float(row.get("실시간인원수", 0)))

    if plan <= 0:
        final = live if live > 0 else 0
    else:
        diff = live - plan
        if diff >= 2:
            final = live
        elif diff <= -2:
            minimum = math.ceil(plan * KEEP_RATE.get(area, 0.60))
            final = max(live, minimum)
        else:
            final = plan
    return clamp_units(area, final)


def minimum_operating_units(area: str, plan_units: int, demand_present: bool = True) -> int:
    plan_units = clamp_units(area, plan_units)
    if area in IM_AREAS:
        return 3 if demand_present else 0
    if plan_units <= 0:
        return 0
    return clamp_units(area, math.ceil(plan_units * KEEP_RATE.get(area, 0.60)))


def infer_arrivals_from_observed(
    observed_queue: np.ndarray,
    baseline_units: np.ndarray,
    service_rate_per_min: float,
) -> np.ndarray:
    """관측 체류 인원과 기준 운영 수로 분당 유입량을 역산한다.

    operation_dashboard_data의 '실시간인원수'는 실제 대기열 자체가 아니라 구역 부하를
    나타내는 값이므로, 결과는 '추정 유입량'으로 취급해야 한다.
    """
    q = np.asarray(observed_queue, dtype=float)
    units = np.asarray(baseline_units, dtype=float)
    n = len(q)
    if n == 0:
        return np.array([], dtype=float)
    arrivals = np.zeros(n, dtype=float)
    for t in range(n):
        q_t = max(0.0, q[t])
        q_next = max(0.0, q[t + 1]) if t + 1 < n else q_t
        cap = max(0.0, units[t] * service_rate_per_min)
        # 현재 부하에서 실제로 처리 가능한 최대량을 기준으로 역산.
        potential_service = min(q_t, cap)
        arrivals[t] = max(0.0, q_next - q_t + potential_service)
    return arrivals


def simulate_single_area(
    arrivals: np.ndarray,
    units: np.ndarray,
    initial_queue: float,
    area: str,
    service_noise: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    arrivals = np.asarray(arrivals, dtype=float)
    units = np.asarray(units, dtype=float)
    n = len(arrivals)
    if len(units) != n:
        raise ValueError("arrivals와 units 길이가 동일해야 합니다.")
    if service_noise is None:
        service_noise = np.ones(n, dtype=float)
    service_noise = np.asarray(service_noise, dtype=float)

    rate = SERVICE_RATE_PER_MIN[area]
    target = TARGET_WAIT_MIN[area]
    queue = np.zeros(n, dtype=float)
    wait = np.zeros(n, dtype=float)
    processed = np.zeros(n, dtype=float)
    utilization = np.zeros(n, dtype=float)

    q = max(0.0, float(initial_queue))
    for t in range(n):
        u = clamp_units(area, int(round(units[t])))
        capacity = max(0.0, u * rate * max(0.05, service_noise[t]))
        incoming = max(0.0, arrivals[t])
        available = q + incoming
        served = min(available, capacity)
        q = max(0.0, available - served)
        queue[t] = q
        processed[t] = served
        if capacity > 1e-9:
            wait[t] = min(240.0, q / capacity)
            utilization[t] = min(2.0, available / capacity)
        else:
            wait[t] = 240.0 if q > 0 else 0.0
            utilization[t] = 2.0 if q > 0 else 0.0

    return pd.DataFrame(
        {
            "queue": queue,
            "wait_min": wait,
            "processed": processed,
            "utilization": utilization,
            "sla_violation": wait > target,
        }
    )


def _as_unit_array(units: Mapping[str, Sequence[float] | float | int], area: str, n: int) -> np.ndarray:
    value = units.get(area, 0)
    if np.isscalar(value):
        return np.full(n, clamp_units(area, int(round(float(value)))), dtype=float)
    arr = np.asarray(value, dtype=float)
    if len(arr) != n:
        raise ValueError(f"{area} 운영 수 배열 길이가 시뮬레이션 길이와 다릅니다.")
    return np.clip(np.rint(arr), 0, MAX_UNITS[area]).astype(float)


def simulate_coupled_system(
    arrivals_by_area: Mapping[str, np.ndarray],
    initial_queue_by_area: Mapping[str, float],
    units_by_area: Mapping[str, Sequence[float] | float | int],
    baseline_checkin_processed: Optional[Mapping[str, np.ndarray]] = None,
    baseline_im_arrivals: Optional[Mapping[str, np.ndarray]] = None,
    im_split: Optional[np.ndarray] = None,
    travel_lag_min: int = 8,
    arrival_noise_by_area: Optional[Mapping[str, np.ndarray]] = None,
    service_noise_by_area: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """체크인 → IM 병목 전이를 포함한 결합 시뮬레이션."""
    first = next(iter(arrivals_by_area.values()))
    n = len(first)
    results: Dict[str, pd.DataFrame] = {}

    # 1) 체크인/셀프체크인
    for area in CHECKIN_AREAS:
        arrivals = np.asarray(arrivals_by_area.get(area, np.zeros(n)), dtype=float).copy()
        if arrival_noise_by_area is not None and area in arrival_noise_by_area:
            arrivals *= np.asarray(arrival_noise_by_area[area], dtype=float)
        units = _as_unit_array(units_by_area, area, n)
        service_noise = None
        if service_noise_by_area is not None and area in service_noise_by_area:
            service_noise = np.asarray(service_noise_by_area[area], dtype=float)
        results[area] = simulate_single_area(
            arrivals=arrivals,
            units=units,
            initial_queue=float(initial_queue_by_area.get(area, 0.0)),
            area=area,
            service_noise=service_noise,
        )

    # 2) 시나리오 체크인 처리량 변화가 IM 유입에 미치는 영향 계산
    extra_processed = np.zeros(n, dtype=float)
    if baseline_checkin_processed is not None:
        for area in CHECKIN_AREAS:
            base = np.asarray(baseline_checkin_processed.get(area, np.zeros(n)), dtype=float)
            extra_processed += results[area]["processed"].to_numpy() - base

    if im_split is None:
        im_split = np.full(n, 0.5, dtype=float)
    im_split = np.clip(np.asarray(im_split, dtype=float), 0.05, 0.95)
    delayed_extra = np.zeros(n, dtype=float)
    lag = max(0, int(travel_lag_min))
    if lag == 0:
        delayed_extra = extra_processed
    elif lag < n:
        delayed_extra[lag:] = extra_processed[:-lag]

    # 3) IM1 / IM2
    for area in IM_AREAS:
        base_arrivals = np.asarray(
            (baseline_im_arrivals or arrivals_by_area).get(area, np.zeros(n)), dtype=float
        ).copy()
        if area == "IM1":
            arrivals = np.maximum(0.0, base_arrivals + delayed_extra * im_split)
        else:
            arrivals = np.maximum(0.0, base_arrivals + delayed_extra * (1.0 - im_split))
        if arrival_noise_by_area is not None and area in arrival_noise_by_area:
            arrivals *= np.asarray(arrival_noise_by_area[area], dtype=float)
        units = _as_unit_array(units_by_area, area, n)
        service_noise = None
        if service_noise_by_area is not None and area in service_noise_by_area:
            service_noise = np.asarray(service_noise_by_area[area], dtype=float)
        results[area] = simulate_single_area(
            arrivals=arrivals,
            units=units,
            initial_queue=float(initial_queue_by_area.get(area, 0.0)),
            area=area,
            service_noise=service_noise,
        )

    # Long form
    long_frames: List[pd.DataFrame] = []
    for area, frame in results.items():
        f = frame.copy()
        f["area"] = area
        f["minute_index"] = np.arange(n)
        f["units"] = _as_unit_array(units_by_area, area, n)
        f["staff"] = f["units"].astype(int).map(lambda u: staff_from_units(area, u))
        long_frames.append(f)
    long_df = pd.concat(long_frames, ignore_index=True)
    return long_df, results


def compute_metrics(long_df: pd.DataFrame) -> ScenarioMetrics:
    if long_df.empty:
        return ScenarioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    waits = pd.to_numeric(long_df["wait_min"], errors="coerce").fillna(0).to_numpy()
    queues = pd.to_numeric(long_df["queue"], errors="coerce").fillna(0).to_numpy()
    processed = pd.to_numeric(long_df["processed"], errors="coerce").fillna(0).to_numpy()
    util = pd.to_numeric(long_df["utilization"], errors="coerce").fillna(0).to_numpy()

    # 대기열이 존재하는 시점에 더 큰 가중치를 부여한다. 모두 0이면 단순 평균.
    weights = np.maximum(queues, 1.0)
    avg_wait = float(np.average(waits, weights=weights))
    p90 = float(np.quantile(waits, 0.90))
    max_wait = float(np.max(waits))
    max_queue = float(np.max(queues))
    sla_rate = float(pd.to_numeric(long_df["sla_violation"], errors="coerce").fillna(False).mean())

    any_violation_by_min = long_df.groupby("minute_index")["sla_violation"].any()
    congestion_minutes = int(any_violation_by_min.sum())

    staff_by_min = long_df.groupby("minute_index")["staff"].sum()
    avg_staff = float(staff_by_min.mean()) if not staff_by_min.empty else 0.0
    peak_staff = int(staff_by_min.max()) if not staff_by_min.empty else 0

    return ScenarioMetrics(
        avg_wait_min=avg_wait,
        p90_wait_min=p90,
        max_wait_min=max_wait,
        max_queue=max_queue,
        total_processed=float(processed.sum()),
        sla_violation_rate=sla_rate,
        congestion_minutes=congestion_minutes,
        avg_staff=avg_staff,
        peak_staff=peak_staff,
        avg_utilization=float(np.mean(util)),
        peak_utilization=float(np.max(util)),
    )



def temporal_comparison(
    reference_long: pd.DataFrame,
    candidate_long: pd.DataFrame,
    tolerance_queue: float = 1e-6,
    tolerance_wait: float = 1e-6,
) -> Tuple[Dict[str, float | int | bool], pd.DataFrame]:
    """Compare two simulations minute-by-minute.

    The strict non-worsening rule uses the strongest quantities observable from the
    current aggregate data: at every minute the candidate must not exceed the
    reference in (1) total estimated queue/load, (2) queue-weighted mean wait, and
    (3) the worst area wait. Equality is allowed because both plans share the same
    initial state, so a literal strict decrease at minute zero is generally impossible.
    """

    def aggregate(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        x = df.copy()
        x["queue"] = pd.to_numeric(x["queue"], errors="coerce").fillna(0.0)
        x["wait_min"] = pd.to_numeric(x["wait_min"], errors="coerce").fillna(0.0)
        x["wait_burden"] = x["queue"] * x["wait_min"]
        g = (
            x.groupby("minute_index", as_index=False)
            .agg(
                total_queue=("queue", "sum"),
                wait_burden=("wait_burden", "sum"),
                max_area_wait=("wait_min", "max"),
            )
            .sort_values("minute_index")
        )
        denom = g["total_queue"].to_numpy(dtype=float)
        burden = g["wait_burden"].to_numpy(dtype=float)
        g["weighted_wait"] = np.divide(burden, denom, out=np.zeros_like(burden), where=denom > 1e-9)
        return g.rename(columns={
            "total_queue": f"{prefix}_queue",
            "wait_burden": f"{prefix}_wait_burden",
            "max_area_wait": f"{prefix}_max_area_wait",
            "weighted_wait": f"{prefix}_weighted_wait",
        })

    ref = aggregate(reference_long, "reference")
    cand = aggregate(candidate_long, "candidate")
    comp = ref.merge(cand, on="minute_index", how="inner").sort_values("minute_index").reset_index(drop=True)

    comp["queue_delta"] = comp["candidate_queue"] - comp["reference_queue"]
    comp["weighted_wait_delta"] = comp["candidate_weighted_wait"] - comp["reference_weighted_wait"]
    comp["max_area_wait_delta"] = comp["candidate_max_area_wait"] - comp["reference_max_area_wait"]
    comp["queue_saved"] = -comp["queue_delta"]
    comp["wait_burden_saved"] = comp["reference_wait_burden"] - comp["candidate_wait_burden"]

    queue_ok = comp["queue_delta"] <= float(tolerance_queue)
    weighted_wait_ok = comp["weighted_wait_delta"] <= float(tolerance_wait)
    max_wait_ok = comp["max_area_wait_delta"] <= float(tolerance_wait)
    comp["nonworsening"] = queue_ok & weighted_wait_ok & max_wait_ok
    comp["queue_state"] = np.where(
        comp["queue_delta"] > float(tolerance_queue),
        "악화",
        np.where(comp["queue_delta"] < -float(tolerance_queue), "개선", "동일"),
    )

    strict_ok = bool(comp["nonworsening"].all()) if not comp.empty else True
    any_improvement = bool(
        ((comp["queue_delta"] < -float(tolerance_queue))
         | (comp["weighted_wait_delta"] < -float(tolerance_wait))
         | (comp["max_area_wait_delta"] < -float(tolerance_wait))).any()
    ) if not comp.empty else False

    worsening_mask = ~comp["nonworsening"]
    queue_worse_mask = comp["queue_delta"] > float(tolerance_queue)
    queue_better_mask = comp["queue_delta"] < -float(tolerance_queue)

    crossing_indices: List[int] = []
    if len(comp) >= 2:
        signs = np.sign(comp["queue_delta"].to_numpy(dtype=float))
        for i in range(1, len(signs)):
            if signs[i] == 0 or signs[i-1] == 0:
                continue
            if signs[i] != signs[i-1]:
                crossing_indices.append(int(comp.iloc[i]["minute_index"]))

    first_better = None
    if queue_better_mask.any():
        first_better = int(comp.loc[queue_better_mask, "minute_index"].iloc[0])

    stats: Dict[str, float | int | bool | list] = {
        "strict_nonworsening": strict_ok,
        "any_improvement": any_improvement,
        "worsening_minutes": int(worsening_mask.sum()),
        "queue_worsening_minutes": int(queue_worse_mask.sum()),
        "queue_improvement_minutes": int(queue_better_mask.sum()),
        "improvement_ratio": float(queue_better_mask.mean()) if len(comp) else 0.0,
        "max_queue_worsening": float(max(0.0, comp["queue_delta"].max())) if len(comp) else 0.0,
        "max_weighted_wait_worsening": float(max(0.0, comp["weighted_wait_delta"].max())) if len(comp) else 0.0,
        "max_area_wait_worsening": float(max(0.0, comp["max_area_wait_delta"].max())) if len(comp) else 0.0,
        "cumulative_queue_saved_person_min": float(comp["queue_saved"].sum()) if len(comp) else 0.0,
        "cumulative_wait_burden_saved": float(comp["wait_burden_saved"].sum()) if len(comp) else 0.0,
        "first_queue_improvement_minute": first_better if first_better is not None else -1,
        "crossing_minute_indices": crossing_indices,
    }
    return stats, comp


def is_temporally_nonworsening(reference_long: pd.DataFrame, candidate_long: pd.DataFrame) -> bool:
    stats, _ = temporal_comparison(reference_long, candidate_long)
    return bool(stats["strict_nonworsening"])

def objective_score(metrics: ScenarioMetrics, objective: str = "balanced") -> float:
    """Composite score for allocation search.

    Peak queue and maximum wait are included explicitly so the optimizer does not
    improve the global average by concentrating passengers into one severe bottleneck.
    """
    objective = str(objective)
    if objective == "대기시간 최소화":
        return (
            metrics.avg_wait_min * 0.30
            + metrics.p90_wait_min * 0.25
            + metrics.max_wait_min * 0.15
            + metrics.max_queue * 0.020
            + metrics.sla_violation_rate * 100 * 0.15
            + metrics.peak_staff * 0.05
            + metrics.congestion_minutes * 0.05
        )
    if objective == "최소 인력 운영":
        return (
            metrics.avg_wait_min * 0.18
            + metrics.p90_wait_min * 0.12
            + metrics.max_wait_min * 0.08
            + metrics.max_queue * 0.010
            + metrics.sla_violation_rate * 100 * 0.12
            + metrics.peak_staff * 0.40
            + metrics.congestion_minutes * 0.05
        )
    # Balanced: explicitly penalize concentrated peak queues.
    return (
        metrics.avg_wait_min * 0.25
        + metrics.p90_wait_min * 0.20
        + metrics.max_wait_min * 0.12
        + metrics.max_queue * 0.020
        + metrics.sla_violation_rate * 100 * 0.15
        + metrics.peak_staff * 0.10
        + metrics.congestion_minutes * 0.05
    )


def monte_carlo_compare(
    arrivals_by_area: Mapping[str, np.ndarray],
    initial_queue_by_area: Mapping[str, float],
    baseline_units: Mapping[str, Sequence[float] | float | int],
    scenario_units: Mapping[str, Sequence[float] | float | int],
    baseline_checkin_processed: Mapping[str, np.ndarray],
    baseline_im_arrivals: Mapping[str, np.ndarray],
    im_split: np.ndarray,
    scenario_arrival_multiplier: Optional[Mapping[str, np.ndarray]] = None,
    travel_lag_min: int = 8,
    iterations: int = 30,
    seed: int = 42,
    arrival_sigma: float = 0.06,
    service_sigma: float = 0.05,
) -> pd.DataFrame:
    """공통 난수를 사용해 기준안과 변경안을 반복 비교한다."""
    n = len(next(iter(arrivals_by_area.values())))
    rows: List[Dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    iterations = max(1, int(iterations))

    for i in range(iterations):
        arrival_noise: Dict[str, np.ndarray] = {}
        service_noise: Dict[str, np.ndarray] = {}
        for area in ALL_AREAS:
            arrival_noise[area] = np.clip(rng.normal(1.0, arrival_sigma, n), 0.75, 1.30)
            service_noise[area] = np.clip(rng.normal(1.0, service_sigma, n), 0.75, 1.25)

        base_long, _ = simulate_coupled_system(
            arrivals_by_area=arrivals_by_area,
            initial_queue_by_area=initial_queue_by_area,
            units_by_area=baseline_units,
            baseline_checkin_processed=baseline_checkin_processed,
            baseline_im_arrivals=baseline_im_arrivals,
            im_split=im_split,
            travel_lag_min=travel_lag_min,
            arrival_noise_by_area=arrival_noise,
            service_noise_by_area=service_noise,
        )

        scenario_arrivals: Dict[str, np.ndarray] = {}
        for area in ALL_AREAS:
            arr = np.asarray(arrivals_by_area.get(area, np.zeros(n)), dtype=float).copy()
            if scenario_arrival_multiplier and area in scenario_arrival_multiplier:
                arr *= np.asarray(scenario_arrival_multiplier[area], dtype=float)
            scenario_arrivals[area] = arr

        scen_long, _ = simulate_coupled_system(
            arrivals_by_area=scenario_arrivals,
            initial_queue_by_area=initial_queue_by_area,
            units_by_area=scenario_units,
            baseline_checkin_processed=baseline_checkin_processed,
            baseline_im_arrivals=baseline_im_arrivals,
            im_split=im_split,
            travel_lag_min=travel_lag_min,
            arrival_noise_by_area=arrival_noise,
            service_noise_by_area=service_noise,
        )

        for label, metrics in [("기준안", compute_metrics(base_long)), ("변경안", compute_metrics(scen_long))]:
            row: Dict[str, float | int | str] = {"iteration": i, "scenario": label}
            row.update(metrics.as_dict())
            rows.append(row)

    return pd.DataFrame(rows)


def optimize_fixed_allocation(
    arrivals_by_area: Mapping[str, np.ndarray],
    initial_queue_by_area: Mapping[str, float],
    starting_units: Mapping[str, int],
    minimum_units: Mapping[str, int],
    staff_budget: int,
    baseline_checkin_processed: Mapping[str, np.ndarray],
    baseline_im_arrivals: Mapping[str, np.ndarray],
    im_split: np.ndarray,
    travel_lag_min: int = 8,
    objective: str = "균형 운영",
    max_steps: int = 24,
    require_temporal_nonworsening: bool = True,
) -> Tuple[Dict[str, int], ScenarioMetrics, pd.DataFrame]:
    """설명 가능한 탐욕형 자원 재배치 최적화.

    - 시작 운영안에서 출발한다.
    - 최소 유지 수량을 지킨다.
    - 가용 직원 수를 넘지 않는다.
    - 1개 추가, 1개 감축, donor→receiver 1개 이동 후보 중 목적함수를 가장 개선하는 조치를 반복한다.

    정수계획의 전역 최적해를 보장하지 않지만, 캡스톤 시연에서 조치 이유를 설명하기 쉽고
    1~3시간 시뮬레이션을 빠르게 반복할 수 있다.
    """
    allocation = {a: clamp_units(a, int(starting_units.get(a, 0))) for a in ALL_AREAS}
    mins = {a: clamp_units(a, int(minimum_units.get(a, 0))) for a in ALL_AREAS}
    staff_budget = max(0, int(staff_budget))

    def run(units_map: Mapping[str, int]) -> Tuple[ScenarioMetrics, pd.DataFrame]:
        long_df, _ = simulate_coupled_system(
            arrivals_by_area=arrivals_by_area,
            initial_queue_by_area=initial_queue_by_area,
            units_by_area=units_map,
            baseline_checkin_processed=baseline_checkin_processed,
            baseline_im_arrivals=baseline_im_arrivals,
            im_split=im_split,
            travel_lag_min=travel_lag_min,
        )
        return compute_metrics(long_df), long_df

    # 예산 초과 시작안은 영향이 가장 작은 감축부터 줄인다.
    while total_staff(allocation) > staff_budget:
        candidates = []
        base_metrics, _ = run(allocation)
        base_score = objective_score(base_metrics, objective)
        for area in ALL_AREAS:
            if allocation[area] <= mins[area]:
                continue
            cand = dict(allocation)
            cand[area] -= 1
            if total_staff(cand) >= total_staff(allocation):
                continue
            m, _ = run(cand)
            penalty = objective_score(m, objective) - base_score
            candidates.append((penalty, area, cand))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        allocation = candidates[0][2]

    current_metrics, current_long = run(allocation)
    # This is the same-demand, same-starting-plan reference used by the strict rule.
    reference_long = current_long.copy()
    reference_metrics = current_metrics
    current_score = objective_score(current_metrics, objective)

    def candidate_allowed(candidate_long: pd.DataFrame) -> bool:
        if not require_temporal_nonworsening:
            return True
        stats, _ = temporal_comparison(reference_long, candidate_long)
        return bool(stats["strict_nonworsening"])

    for _ in range(max(1, int(max_steps))):
        area_stats = (
            current_long.groupby("area")
            .agg(avg_wait=("wait_min", "mean"), max_wait=("wait_min", "max"), avg_util=("utilization", "mean"))
            .reset_index()
        )
        receivers = area_stats.sort_values(["max_wait", "avg_wait"], ascending=False)["area"].head(5).tolist()
        donors = area_stats.sort_values(["avg_util", "avg_wait"], ascending=True)["area"].head(7).tolist()

        best = None  # (score, description, allocation, metrics, long)

        # 1) 추가
        for receiver in receivers:
            if allocation[receiver] >= MAX_UNITS[receiver]:
                continue
            cand = dict(allocation)
            cand[receiver] += 1
            if total_staff(cand) > staff_budget:
                continue
            m, l = run(cand)
            if not candidate_allowed(l):
                continue
            score = objective_score(m, objective)
            if best is None or score < best[0]:
                best = (score, f"{receiver} +1", cand, m, l)

        # 2) 감축 - 최소 인력 목표에서 특히 의미가 있음
        for donor in donors:
            if allocation[donor] <= mins[donor]:
                continue
            cand = dict(allocation)
            cand[donor] -= 1
            m, l = run(cand)
            if not candidate_allowed(l):
                continue
            score = objective_score(m, objective)
            if best is None or score < best[0]:
                best = (score, f"{donor} -1", cand, m, l)

        # 3) 재배치
        for donor in donors:
            if allocation[donor] <= mins[donor]:
                continue
            for receiver in receivers:
                if donor == receiver or allocation[receiver] >= MAX_UNITS[receiver]:
                    continue
                cand = dict(allocation)
                cand[donor] -= 1
                cand[receiver] += 1
                if total_staff(cand) > staff_budget:
                    continue
                m, l = run(cand)
                if not candidate_allowed(l):
                    continue
                score = objective_score(m, objective)
                if best is None or score < best[0]:
                    best = (score, f"{donor}→{receiver}", cand, m, l)

        # 4) Downstream-safe paired additions. A check-in increase can push a
        # temporary bottleneck to IM1/IM2, so evaluate the coordinated move too.
        for receiver in [a for a in receivers if a not in IM_AREAS]:
            if allocation[receiver] >= MAX_UNITS[receiver]:
                continue
            for im_area in IM_AREAS:
                if allocation[im_area] >= MAX_UNITS[im_area]:
                    continue
                cand = dict(allocation)
                cand[receiver] += 1
                cand[im_area] += 1
                if total_staff(cand) > staff_budget:
                    continue
                m, l = run(cand)
                if not candidate_allowed(l):
                    continue
                score = objective_score(m, objective)
                if best is None or score < best[0]:
                    best = (score, f"{receiver}+1 & {im_area}+1", cand, m, l)

        if best is None or best[0] >= current_score - 1e-6:
            break
        current_score, _, allocation, current_metrics, current_long = best

    return allocation, current_metrics, current_long
