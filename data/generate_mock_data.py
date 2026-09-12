"""
가상 주택 데이터 + 구성원별 이동 시간 데이터 생성 스크립트.

출력 (data/ 하위)
  listings.csv           : 공통 데이터 (전용면적/방/주차/향/역까지 도보 시간) + 부가 정보
  travel_times.csv       : 개별 데이터 (구성원 × 목적지 × 이동수단, 분 단위)
  members.json           : 구성원 정의 + 목적지 좌표/이동수단

실행:
  python data/generate_mock_data.py --n 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import DEFAULT_MEMBERS, ORIENTATION_SCORE, assert_no_price_columns  # noqa: E402
from src.travel import EstimatedTravelProvider, travel_matrix  # noqa: E402

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DISTRICTS = [
    ("강남구",   37.5172, 127.0473, 1.0), ("송파구",   37.5145, 127.1059, 1.0),
    ("마포구",   37.5638, 126.9084, 0.9), ("성동구",   37.5633, 127.0371, 0.9),
    ("노원구",   37.6542, 127.0568, 0.8), ("양천구",   37.5170, 126.8664, 0.8),
    ("영등포구", 37.5264, 126.8963, 0.9), ("은평구",   37.6027, 126.9291, 0.7),
    ("동작구",   37.5124, 126.9393, 0.8), ("광진구",   37.5385, 127.0823, 0.8),
]


def generate_listings(n: int, rng: np.random.Generator, outlier_rate: float) -> pd.DataFrame:
    weights = np.array([d[3] for d in DISTRICTS], dtype=float)
    idx = rng.choice(len(DISTRICTS), size=n, p=weights / weights.sum())

    lat = np.array([DISTRICTS[i][1] for i in idx]) + rng.normal(0, 0.013, n)
    lon = np.array([DISTRICTS[i][2] for i in idx]) + rng.normal(0, 0.016, n)

    area = np.clip(rng.gamma(9.0, 8.0, n) + 26, 29, 175).round(1)
    # 이상치(초대형 펜트하우스)를 의도적으로 섞어 클리핑/RobustScaler 효과를 검증 가능하게 한다
    n_out = max(int(n * outlier_rate), 1)
    out_idx = rng.choice(n, n_out, replace=False)
    area[out_idx] = np.round(rng.uniform(320, 640, n_out), 1)

    rooms = np.clip(np.round(area / 27 + rng.normal(0, 0.45, n)), 1, 7).astype(int)
    built_year = np.clip(np.round(rng.normal(2007, 12, n)), 1980, 2026).astype(int)
    total_floors = np.clip(np.round(rng.normal(19, 6, n)), 5, 45).astype(int)
    floor = np.array([rng.integers(1, tf + 1) for tf in total_floors])

    orient_keys = list(ORIENTATION_SCORE)
    orient_p = np.array([0.34, 0.17, 0.14, 0.13, 0.11, 0.06, 0.05])
    orientation = rng.choice(orient_keys, size=n, p=orient_p / orient_p.sum())

    complex_units = np.clip(np.round(rng.lognormal(6.4, 0.75, n)), 60, 6000).astype(int)
    parking = np.clip(np.round(rng.normal(1.05, 0.32, n) + (built_year - 2007) * 0.012, 2),
                      0.25, 3.0)
    parking[out_idx] = np.round(rng.uniform(3.5, 6.0, n_out), 2)   # 이상치 동반

    # 역까지 도보 시간(분) — 거리가 아니라 시간으로 표현
    transit_walk = np.clip(rng.lognormal(1.95, 0.55, n), 1, 45).round(1)

    green = np.clip(rng.normal(28, 9, n) + np.log1p(complex_units) * 1.9
                    + (built_year - 2007) * 0.12, 5, 62).round(1)
    daylight = np.clip(rng.normal(5.2, 1.1, n)
                       + np.array([ORIENTATION_SCORE[o] for o in orientation]) * 2.2
                       + (floor / total_floors) * 1.1, 1.0, 10.0).round(1)
    noise = np.clip(rng.normal(52, 6.5, n) - green * 0.14
                    + (total_floors - floor) * 0.06, 33, 74).round(1)

    df = pd.DataFrame({
        "listing_id": [f"H{str(i + 1).zfill(4)}" for i in range(n)],
        "name": [f"{DISTRICTS[i][0]} {rng.choice(['그린', '센트럴', '파크', '리버', '하이', '포레'])}"
                 f"{rng.choice(['빌', '타운', '스퀘어', '캐슬', '아파트'])} {rng.integers(1, 15)}단지"
                 for i in idx],
        "gu": [DISTRICTS[i][0] for i in idx],
        "lat": lat.round(6), "lon": lon.round(6),
        # --- 공통 피처 (거리 계산에 사용) ---
        "area_m2": area, "rooms": rooms, "parking_slots": parking,
        "orientation": orientation, "transit_walk_min": transit_walk,
        # --- 부가 정보 (필터/표시/안정성 지수) ---
        "built_year": built_year, "floor": floor, "total_floors": total_floors,
        "green_ratio": green, "complex_units": complex_units,
        "daylight_hours": daylight, "noise_db": noise,
    })
    assert_no_price_columns(df.columns)
    return df


def inject_missing(df: pd.DataFrame, cols, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df.loc[rng.random(len(df)) < rate, c] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--missing-rate", type=float, default=0.04)
    ap.add_argument("--outlier-rate", type=float, default=0.015,
                    help="초대형 이상치 비율 (클리핑/RobustScaler 검증용)")
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    members = DEFAULT_MEMBERS

    listings = generate_listings(args.n, rng, args.outlier_rate)
    travel = travel_matrix(listings, members, EstimatedTravelProvider(seed=args.seed),
                           all_modes=True)
    assert_no_price_columns(travel.columns)

    listings = inject_missing(listings, ["parking_slots", "transit_walk_min",
                                         "green_ratio", "daylight_hours", "noise_db"],
                              args.missing_rate, rng)
    travel = inject_missing(travel, [c for c in travel.columns if c != "listing_id"],
                            args.missing_rate, rng)

    os.makedirs(args.out, exist_ok=True)
    listings.to_csv(os.path.join(args.out, "listings.csv"), index=False)
    travel.to_csv(os.path.join(args.out, "travel_times.csv"), index=False)
    with open(os.path.join(args.out, "members.json"), "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in members], f, ensure_ascii=False, indent=2)

    print(f"✅ listings.csv     : {listings.shape[0]}행 × {listings.shape[1]}열 "
          f"(결측 {int(listings.isna().sum().sum())}, 이상치 {max(int(args.n*args.outlier_rate),1)}건 포함)")
    print(f"✅ travel_times.csv : {travel.shape[0]}행 × {travel.shape[1]}열 "
          f"(결측 {int(travel.isna().sum().sum())}) — 모두 분 단위")
    print(f"✅ members.json     : " + ", ".join(
        f"{m.name}({len(m.destinations)}개 목적지)" for m in members))
    print("ℹ️  가격 컬럼 없음 · 거리(m) 컬럼 없음 — 이동은 전부 시간(분)")


if __name__ == "__main__":
    main()
