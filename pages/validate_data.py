from pathlib import Path
from core import load_operation_data, load_flight_data, data_quality_report
from simulation import ALL_AREAS

BASE = Path(__file__).resolve().parent
op = load_operation_data(BASE / "data" / "operation_dashboard_sep2025.csv.gz")
fl = load_flight_data(BASE / "data" / "flight_counter_sep2025.csv")
report = data_quality_report(op, fl)

print("=== DATA QA ===")
for k, v in report.items():
    print(f"{k}: {v}")

assert report["operation_dates"] == 30, "9월 30일치 운영 데이터가 아닙니다."
assert report["operation_duplicates"] == 0, "일자/분/구역 중복 행이 있습니다."
assert set(report["areas"]) == set(ALL_AREAS), "예상 구역 목록과 실제 데이터가 다릅니다."
print("\n검수 통과")
