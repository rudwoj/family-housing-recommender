"""
가상 주택 물리 데이터 + 구성원별 위치/동선 데이터 생성 스크립트.

출력 (모두 data/ 하위):
  - listings.csv            : 공통 데이터 (주택 물리/환경 특성)  ※ 가격 컬럼 없음
  - individual_features.csv : 개별 데이터 (구성원 × 동선 피처)
  - members.json            : 구성원 정의 + 주요 목적지(직장/학교) 좌표

실행:
  python data/generate_mock_data.py --n 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import (  # noqa: E402
    DEFAULT_MEMBERS, ORIENTATION_SCORE, ROLE_FEATURES, assert_no_price_columns,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# 서울 주요 생활권 중심 (mock) : (이름, lat, lon, 밀도가중치)
DISTRICTS = [
    ("강남구",   37.5172, 127.0473, 1.0),
    ("송파구",   37.5145, 127.1059, 1.0),
    ("마포구",   37.5638, 126.9084, 0.9),
    ("성동구",   37.5633, 127.0371, 0.9),
    ("노원구",   37.6542, 127.0568, 0.8),
    ("양천구",   37.5170, 126.8664, 0.8),
    ("영등포구", 37.5264, 126.8963, 0.9),
    ("은평구",   37.6027, 126.9291, 0.7),
    ("동작구",   37.5124, 126.9393, 0.8),
    ("광진구",   37.5385, 127.0823, 0.8),
]

# 생활 편의 앵커 (마트/병원/공원/도서관 군집 중심)
AMENITY_HUBS = [
    (37.5172, 127.0473), (37.5638, 126.9084), (37.6542, 127.0568),
    (37.5264, 126.8963), (37.5385, 127.0823), (37.5145, 127.1059),
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_hub_m(lat, lon, hubs, rng, jitter=0.55):
    """가장 가까운 허브까지의 보행 접근 거리(m). 도로 우회율 + 잡음 반영."""
    d = np.min([haversine_km(lat, lon, h[0], h[1]) for h in hubs], axis=0)
    detour = 1.25 + rng.normal(0, 0.08, size=np.shape(d))
    return np.clip(d * 1000 * detour * rng.lognormal(0, jitter, size=np.shape(d)) * 0.35, 60, 4500)


# --------------------------------------------------------------------------- #
# 1) 공통 데이터 : 주택 물리/환경 특성
# --------------------------------------------------------------------------- #
def generate_listings(n: int, rng: np.random.Generator) -> pd.DataFrame:
    weights = np.array([d[3] for d in DISTRICTS], dtype=float)
    idx = rng.choice(len(DISTRICTS), size=n, p=weights / weights.sum())

    gu = [DISTRICTS[i][0] for i in idx]
    lat = np.array([DISTRICTS[i][1] for i in idx]) + rng.normal(0, 0.013, n)
    lon = np.array([DISTRICTS[i][2] for i in idx]) + rng.normal(0, 0.016, n)

    # 면적 -> 방/욕실 수는 면적에 종속 (현실적 상관관계)
    area = np.clip(rng.gamma(shape=9.0, scale=8.0, size=n) + 26, 29, 175).round(1)
    rooms = np.clip(np.round(area / 27 + rng.normal(0, 0.45, n)), 1, 5).astype(int)
    baths = np.clip(np.round(rooms / 2.2 + rng.normal(0, 0.3, n)), 1, 3).astype(int)

    built_year = np.clip(np.round(rng.normal(2007, 12, n)), 1980, 2026).astype(int)
    total_floors = np.clip(np.round(rng.normal(19, 6, n)), 5, 45).astype(int)
    floor = np.array([rng.integers(1, tf + 1) for tf in total_floors])

    orient_keys = list(ORIENTATION_SCORE)
    orient_p = np.array([0.34, 0.17, 0.14, 0.13, 0.11, 0.06, 0.05])
    orientation = rng.choice(orient_keys, size=n, p=orient_p / orient_p.sum())

    complex_units = np.clip(np.round(rng.lognormal(6.4, 0.75, n)), 60, 6000).astype(int)
    # 대단지일수록 녹지/주차 여건이 좋고, 신축일수록 주차가 넉넉한 구조로 생성
    green = np.clip(rng.normal(28, 9, n) + np.log1p(complex_units) * 1.9
                    + (built_year - 2007) * 0.12, 5, 62).round(1)
    parking = np.clip(rng.normal(1.05, 0.32, n) + (built_year - 2007) * 0.012, 0.25, 2.6).round(2)
    daylight = np.clip(rng.normal(5.2, 1.1, n)
                       + np.array([ORIENTATION_SCORE[o] for o in orientation]) * 2.2
                       + (floor / total_floors) * 1.1, 1.0, 10.0).round(1)
    noise = np.clip(rng.normal(52, 6.5, n) - green * 0.14 + (total_floors - floor) * 0.06, 33, 74).round(1)

    df = pd.DataFrame({
        "listing_id": [f"H{str(i + 1).zfill(4)}" for i in range(n)],
        "name": [f"{DISTRICTS[i][0]} {rng.choice(['그린', '센트럴', '파크', '리버', '하이', '포레'])}"
                 f"{rng.choice(['빌', '타운', '스퀘어', '캐슬', '아파트'])} {rng.integers(1, 15)}단지"
                 for i in idx],
        "gu": gu,
        "lat": lat.round(6),
        "lon": lon.round(6),
        "area_m2": area,
        "rooms": rooms,
        "bathrooms": baths,
        "orientation": orientation,
        "built_year": built_year,
        "floor": floor,
        "total_floors": total_floors,
        "green_ratio": green,
        "parking_per_unit": parking,
        "complex_units": complex_units,
        "daylight_hours": daylight,
        "noise_db": noise,
    })
    assert_no_price_columns(df.columns)
    return df


# --------------------------------------------------------------------------- #
# 2) 개별 데이터 : 구성원별 생활 동선
# --------------------------------------------------------------------------- #
def generate_individual(listings: pd.DataFrame, members, rng: np.random.Generator) -> pd.DataFrame:
    out = pd.DataFrame({"listing_id": listings["listing_id"]})
    lat, lon = listings["lat"].to_numpy(), listings["lon"].to_numpy()
    n = len(listings)

    for m in members:
        keys = {f.key for f in ROLE_FEATURES[m.role]}
        d_km = haversine_km(lat, lon, m.anchor_lat, m.anchor_lon)

        if "commute_min" in keys:
            # 도어투도어 = 고정 오버헤드 + 거리 비례 + 개인 편차
            out[m.column("commute_min")] = np.clip(
                12 + d_km * 3.05 + rng.normal(0, 6.5, n), 8, 120).round(0)
        if "transfer_count" in keys:
            out[m.column("transfer_count")] = np.clip(
                np.round(d_km / 7.5 + rng.normal(0, 0.55, n)), 0, 4).astype(int)
        if "station_walk_m" in keys:
            out[m.column("station_walk_m")] = np.clip(
                rng.lognormal(6.0, 0.55, n), 90, 2200).round(0)
        if "mart_dist_m" in keys:
            out[m.column("mart_dist_m")] = nearest_hub_m(lat, lon, AMENITY_HUBS, rng, 0.5).round(0)
        if "hospital_dist_m" in keys:
            out[m.column("hospital_dist_m")] = nearest_hub_m(lat, lon, AMENITY_HUBS, rng, 0.6).round(0)
        if "school_walk_m" in keys:
            out[m.column("school_walk_m")] = np.clip(
                rng.lognormal(6.35, 0.5, n), 120, 3000).round(0)
        if "academy_dist_m" in keys:
            out[m.column("academy_dist_m")] = np.clip(
                d_km * 1000 * 1.2 * rng.lognormal(0, 0.18, n), 200, 26000).round(0)
        if "park_dist_m" in keys:
            out[m.column("park_dist_m")] = nearest_hub_m(lat, lon, AMENITY_HUBS, rng, 0.55).round(0)
        if "library_dist_m" in keys:
            out[m.column("library_dist_m")] = nearest_hub_m(lat, lon, AMENITY_HUBS, rng, 0.7).round(0)

    assert_no_price_columns(out.columns)
    return out


# --------------------------------------------------------------------------- #
# 3) 결측 주입 (KNNImputer 검증용)
# --------------------------------------------------------------------------- #
def inject_missing(df: pd.DataFrame, cols, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, c] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="생성할 매물 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--missing-rate", type=float, default=0.04,
                    help="결측 주입 비율 (KNNImputer 보정 검증용)")
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    members = DEFAULT_MEMBERS

    listings = generate_listings(args.n, rng)
    individual = generate_individual(listings, members, rng)

    listings = inject_missing(
        listings, ["green_ratio", "parking_per_unit", "daylight_hours", "noise_db"],
        args.missing_rate, rng)
    individual = inject_missing(
        individual, [c for c in individual.columns if c != "listing_id"],
        args.missing_rate, rng)

    os.makedirs(args.out, exist_ok=True)
    listings.to_csv(os.path.join(args.out, "listings.csv"), index=False)
    individual.to_csv(os.path.join(args.out, "individual_features.csv"), index=False)
    with open(os.path.join(args.out, "members.json"), "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in members], f, ensure_ascii=False, indent=2)

    miss_c = listings.isna().sum().sum()
    miss_i = individual.isna().sum().sum()
    print(f"✅ listings.csv            : {listings.shape[0]} rows × {listings.shape[1]} cols (결측 {miss_c})")
    print(f"✅ individual_features.csv : {individual.shape[0]} rows × {individual.shape[1]} cols (결측 {miss_i})")
    print(f"✅ members.json            : {[m.name for m in members]}")
    print("ℹ️  가격 관련 컬럼 없음 (assert_no_price_columns 통과)")


if __name__ == "__main__":
    main()
