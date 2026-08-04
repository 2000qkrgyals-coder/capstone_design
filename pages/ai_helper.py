from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional
import json
import re


def parse_question_constraints(question: str, airlines: Iterable[str]) -> Dict[str, object]:
    """간단한 자연어에서 자주 쓰는 운영 제약을 추출한다.

    LLM 키가 없어도 질문을 시나리오 조건으로 일부 반영할 수 있도록 만든 보조 파서다.
    """
    q = str(question or "")
    result: Dict[str, object] = {}

    # 항공사명은 긴 이름부터 검사해 부분 문자열 충돌을 줄인다.
    for airline in sorted([str(a) for a in airlines], key=len, reverse=True):
        if airline and airline in q:
            result["airline"] = airline
            break

    pct_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", q)
    if pct_matches:
        try:
            result["demand_change_pct"] = float(pct_matches[0])
        except ValueError:
            pass

    # '추가 직원 4명', '직원은 4명뿐', '인력 3명' 등을 포괄.
    staff_patterns = [
        r"(?:추가\s*)?(?:직원|인력)\D{0,8}(\d+)\s*명",
        r"(\d+)\s*명\D{0,8}(?:직원|인력)",
    ]
    for p in staff_patterns:
        m = re.search(p, q)
        if m:
            result["additional_staff"] = int(m.group(1))
            break

    h = re.search(r"(\d+)\s*시간", q)
    if h:
        result["horizon_min"] = min(240, max(30, int(h.group(1)) * 60))
    else:
        m = re.search(r"(\d+)\s*분", q)
        if m:
            result["horizon_min"] = min(240, max(30, int(m.group(1))))

    if any(k in q for k in ["인력 최소", "최소 인력", "인원 최소"]):
        result["objective"] = "최소 인력 운영"
    elif any(k in q for k in ["대기시간 최소", "대기 시간 최소", "가장 빨리", "혼잡 최소"]):
        result["objective"] = "대기시간 최소화"
    elif any(k in q for k in ["균형", "종합"]):
        result["objective"] = "균형 운영"

    return result


def build_structured_context(
    question: str,
    date: str,
    time_text: str,
    horizon_min: int,
    baseline_metrics: Mapping[str, object],
    scenario_metrics: Mapping[str, object],
    units_before: Mapping[str, int],
    units_after: Mapping[str, int],
    bottlenecks: List[Mapping[str, object]],
    airline: Optional[str] = None,
    demand_change_pct: float = 0.0,
    assumptions: Optional[List[str]] = None,
) -> Dict[str, object]:
    changes = []
    for area in sorted(set(units_before) | set(units_after)):
        before = int(units_before.get(area, 0))
        after = int(units_after.get(area, 0))
        if before != after:
            changes.append({"area": area, "before": before, "after": after, "delta": after - before})

    return {
        "question": question,
        "data_time": f"{date} {time_text}",
        "horizon_minutes": int(horizon_min),
        "airline_scenario": {
            "airline": airline,
            "demand_change_pct": float(demand_change_pct),
        },
        "baseline_metrics": dict(baseline_metrics),
        "scenario_metrics": dict(scenario_metrics),
        "resource_changes": changes,
        "top_bottlenecks": bottlenecks,
        "assumptions": assumptions or [],
    }


def deterministic_operation_answer(context: Mapping[str, object]) -> str:
    base = context.get("baseline_metrics", {}) or {}
    scen = context.get("scenario_metrics", {}) or {}
    changes = context.get("resource_changes", []) or []
    bottlenecks = context.get("top_bottlenecks", []) or []
    airline_scenario = context.get("airline_scenario", {}) or {}

    def f(d: Mapping[str, object], key: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, default))
        except Exception:
            return default

    base_wait = f(base, "avg_wait_min")
    scen_wait = f(scen, "avg_wait_min")
    base_p90 = f(base, "p90_wait_min")
    scen_p90 = f(scen, "p90_wait_min")
    base_queue = f(base, "max_queue")
    scen_queue = f(scen, "max_queue")
    base_staff = f(base, "peak_staff")
    scen_staff = f(scen, "peak_staff")

    wait_delta = scen_wait - base_wait
    queue_delta = scen_queue - base_queue
    staff_delta = scen_staff - base_staff
    queue_ratio = (scen_queue / base_queue) if base_queue > 0 else 1.0

    lines: List[str] = []
    lines.append("### AI 운영 분석")
    lines.append(f"기준 시각: {context.get('data_time', '')} / 분석 범위: 향후 {context.get('horizon_minutes', 0)}분")

    airline = airline_scenario.get("airline")
    shock = float(airline_scenario.get("demand_change_pct", 0.0) or 0.0)
    if airline and abs(shock) > 1e-9:
        sign = "+" if shock > 0 else ""
        lines.append(f"항공사 수요 시나리오: {airline} {sign}{shock:.0f}%")

    # Overall verdict considers both average wait and peak queue.
    if wait_delta < -0.2 and queue_ratio <= 1.10:
        verdict = "추천"
        lines.append(f"종합 판정: {verdict} — 평균 대기시간과 피크 대기열이 모두 허용 범위에서 개선됩니다.")
    elif wait_delta < -0.2 and queue_ratio > 1.10:
        verdict = "조건부 추천"
        lines.append(
            f"종합 판정: {verdict} — 평균 대기시간은 줄지만 최대 대기열이 "
            f"{base_queue:.0f}명 → {scen_queue:.0f}명({queue_delta:+.0f}명)으로 증가합니다. "
            "특정 구역에 혼잡이 집중될 가능성이 있어 그대로 적용하기보다 병목 구역 자원을 추가 보정해야 합니다."
        )
    elif wait_delta > 0.2:
        verdict = "비추천"
        lines.append(f"종합 판정: {verdict} — 평균 대기시간이 {base_wait:.1f}분 → {scen_wait:.1f}분으로 증가합니다.")
    else:
        verdict = "효과 제한"
        lines.append("종합 판정: 효과 제한 — 전체 평균 개선 폭이 작아 구역별 변화 확인이 필요합니다.")

    lines.append(
        f"핵심 지표: 평균 대기 {base_wait:.1f}분 → {scen_wait:.1f}분({wait_delta:+.1f}분), "
        f"P90 {base_p90:.1f}분 → {scen_p90:.1f}분, "
        f"최대 대기열 {base_queue:.0f}명 → {scen_queue:.0f}명({queue_delta:+.0f}명)."
    )

    if changes:
        lines.append("\n#### 권장 자원 조정")
        for item in sorted(changes, key=lambda x: abs(int(x.get("delta", 0))), reverse=True)[:8]:
            delta = int(item.get("delta", 0))
            action = "추가" if delta > 0 else "감축"
            lines.append(f"- {item.get('area')}: {item.get('before')} → {item.get('after')} ({abs(delta)}개 {action})")
    else:
        lines.append("\n현재 계산에서는 기준 운영 수를 유지하는 편이 가장 안정적입니다.")

    lines.append(f"\n피크 필요 인력: {base_staff:.0f}명 → {scen_staff:.0f}명({staff_delta:+.0f}명)")

    if bottlenecks:
        lines.append("\n#### 주의할 병목")
        for b in bottlenecks[:3]:
            lines.append(
                f"- {b.get('area')}: 최대 대기 {float(b.get('max_wait_min', 0)):.1f}분, "
                f"최대 대기열 {float(b.get('max_queue', 0)):.0f}명"
            )
        if any(str(b.get("area", "")).startswith("IM") for b in bottlenecks[:3]):
            lines.append("- 체크인 처리량 증가로 IM1/IM2에 병목이 전이되는지 함께 확인해야 합니다.")

    if queue_ratio > 1.10:
        lines.append(
            "\n#### 안전 보정 권고\n"
            "최대 대기열이 기준안보다 10% 이상 증가했으므로 현재 안을 즉시 확정하지 말고, "
            "최대 대기열이 발생한 구역에 1~2개 자원을 추가하거나 인접 저부하 구역에서 재배치한 뒤 다시 시뮬레이션하는 것이 좋습니다."
        )

    lines.append(
        "\n※ 본 수치는 2025년 9월 1분 단위 운영·인원 데이터를 재생한 시뮬레이션 추정값입니다. "
        "실제 승객별 서비스 완료 로그가 아니므로 절대 대기시간보다 기준안 대비 변화량을 중심으로 해석하세요."
    )
    return "\n".join(lines)


def _prompt_from_context(context: Mapping[str, object]) -> str:
    return f"""
당신은 인천공항 제2여객터미널 운영 의사결정 보조 AI입니다.
아래 JSON은 별도의 시뮬레이션 엔진이 계산한 결과입니다. 숫자를 새로 추정하거나 만들어내지 마세요.
반드시 JSON에 있는 값만 근거로 답하세요.

답변 순서:
1. 현재 상황 요약
2. 주요 병목
3. 권장 운영 조치(우선순위 포함)
4. 기준안 대비 예상 효과
5. 필요한 인력/자원 변화
6. 혼잡 전이 또는 부작용 위험
7. 데이터 한계와 가정

실행 불가능한 운영안을 임의로 제시하지 마세요.
답변은 한국어로, 운영자가 바로 읽을 수 있게 간결하고 구체적으로 작성하세요.

시뮬레이션 결과 JSON:
{json.dumps(context, ensure_ascii=False, indent=2, default=str)}
""".strip()


def generate_llm_answer(
    provider: str,
    model: str,
    api_key: str,
    context: Mapping[str, object],
) -> str:
    prompt = _prompt_from_context(context)
    provider = str(provider)
    if provider == "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai 패키지가 설치되어 있지 않습니다.") from exc
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model or "gpt-5",
            input=prompt,
            store=False,
        )
        return str(response.output_text)

    if provider == "Gemini":
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai 패키지가 설치되어 있지 않습니다.") from exc
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or "gemini-3.5-flash",
            contents=prompt,
        )
        return str(response.text)

    return deterministic_operation_answer(context)
