"""
실제 API로 data/listings.csv, data/travel_times.csv, data/members.json 를 생성한다.

데이터 소스 (전부 scripts/check_api_access.py 로 실키 검증 완료):
  - 공공데이터포털 국토교통부_아파트 전월세 실거래가 (RTMSDataSvcAptRent)
    -> 단지명/전용면적/층/건축년도/주소 (실측). 가격 필드는 절대 사용하지 않는다.
  - 카카오 로컬 API -> 지오코딩, 최근접 지하철역 거리(역세권 도보시간) (실측)

방 개수/주차 가능 대수/향/총층수/단지 규모/녹지비율/일조시간/소음도는 어느 API에도
없어 data/generate_mock_data.py 와 동일한 통계적 추정 공식으로 채운다(재현성을 위해
시드 고정 rng 사용). 최종 컬럼만 봐서는 실측/추정 구분이 안 되므로 data/real_data_report.json
에 기록해 둔다.

travel_times.csv 는 (아직 사용자가 입력하지 않은) 기본 데모 구성원 템플릿 기준 스냅샷이며,
실제 앱은 사용자가 입력한 목적지로 매 세션 실시간 계산한다(src/travel.py travel_matrix).

실행:
    python scripts/fetch_real_dataset.py [--out data] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_sources import kakao, public_data  # noqa: E402
from src.data_sources.config import SEOUL_GU_NAMES, load_settings  # noqa: E402
from src.schema import ORIENTATION_SCORE, assert_no_price_columns, make_default_members  # noqa: E402
from src.travel import EstimatedTravelProvider, travel_matrix  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEED = 42

#: 서울 25개 자치구. 앱의 자치구 선택 UI 와 같은 표를 봐야 코드-이름 매핑이 어긋나지 않는다.
GU_NAMES = SEOUL_GU_NAMES

CITY = "서울특별시"
WALK_M_PER_MIN = 67.0  # 도보 평균 속도(분속) — mock 의 transit_walk_min 분포와 맞춘 근사치

#: 카카오 지오코딩·지하철역 조회는 매물 1건당 2회씩 발생해 순차로 돌면 압도적 병목이다.
#: 호출 간격(sleep) 대신 워커 수로 동시성을 제한한다.
MAX_WORKERS = 12

#: 시군구당 받아오는 실거래 원시 행 수. 한 달치 거래가 가장 많은 구(강서구)가 1,300건 수준이라
#: 넉넉히 잡아 한 번에 다 받는다(2000행 응답도 0.5초 수준).
RAW_ROWS_PER_SIGUNGU = 2000


def _walk_minutes_estimate(rng: np.random.Generator, n: int) -> np.ndarray:
    """지하철 검색이 실패했을 때만 쓰는 최후 폴백 (mock과 동일 분포)."""
    return np.clip(rng.lognormal(1.95, 0.55, n), 1, 45).round(1)


class Stats:
    def __init__(self):
        self.real = 0
        self.estimated_fallback = 0
        self.skipped_no_geocode = 0
        self.skipped_bad_row = 0

    def as_dict(self) -> dict:
        return vars(self)

def fetch_raw_rent_rows(settings, stats: Stats, codes: list[str] | None = None) -> list[dict]:
    """전월세 실거래가 API에서 시군구별로 원시 행을 모으고 가격 필드를 제거한다.

    codes 로 조회할 자치구를 좁힐 수 있다(없으면 설정값 전체).
    구마다 settings.max_listings_per_gu 건까지 담는다 — 거래가 적은 구는 있는 만큼만
    담기고, 많은 구는 상한까지 채운다. 여러 구를 고르면 그만큼 합계가 늘어난다.
    """
    seen = set()

    def fetch_one(code: str):
        return public_data.fetch_apt_rent(code, settings.target_deal_ym,
                                          settings.data_go_kr_service_key,
                                          endpoint=settings.apt_rent_endpoint,
                                          num_of_rows=RAW_ROWS_PER_SIGUNGU)

    codes = list(codes if codes is not None else settings.target_sigungu_codes)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(len(codes), 1))) as pool:
        responses = list(pool.map(fetch_one, codes))

    # 시군구 순서대로 훑어야 listing_id 부여가 재현 가능하다.
    cap = settings.max_listings_per_gu
    raw_rows: list[dict] = []
    for code, resp in zip(codes, responses):
        gu = GU_NAMES.get(code, code)
        if not resp.ok:
            print(f"⚠️  [{gu}] 실거래 조회 실패: {resp.error}")
            continue

        taken = 0
        for item in resp.items:
            if taken >= cap:
                break
            try:
                area = float(item.get("excluUseAr", ""))
                floor = int(str(item.get("floor", "")).strip())
                built_year = int(str(item.get("buildYear", "")).strip())
            except (TypeError, ValueError):
                stats.skipped_bad_row += 1
                continue
            if area <= 0 or floor <= 0 or built_year <= 0:
                stats.skipped_bad_row += 1
                continue

            name = item.get("aptNm", "").strip() or "이름없음"
            umd = item.get("umdNm", "").strip()
            jibun = item.get("jibun", "").strip()
            roadnm = item.get("roadnm", "").strip()

            dedup_key = (name, umd, jibun, floor, area)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            road_address = f"{CITY} {gu} {roadnm}".strip() if roadnm else None
            jibun_address = f"{CITY} {gu} {umd} {jibun}".strip() if umd else None

            raw_rows.append({
                "name": name, "gu": gu,
                "area_m2": area, "floor": floor, "built_year": built_year,
                "road_address": road_address, "jibun_address": jibun_address,
            })
            taken += 1

        print(f"✅ [{gu}] {taken}건 수집 (전체 {len(resp.items)}건 중, 중복/불량 제외)")

    return raw_rows


def geocode_rows(raw_rows: list[dict], settings, stats: Stats) -> list[dict]:
    """주소 -> 좌표. 도로명주소 실패 시 지번주소로 재시도, 캐시로 중복 호출을 줄인다."""
    addresses = {addr for row in raw_rows
                 for addr in (row["road_address"], row["jibun_address"]) if addr}

    def resolve(addr: str):
        # 네트워크 일시 오류(DNS/타임아웃)까지 삼켜야 한다 — 수백 건을 병렬로 던지면
        # 한두 건은 실패하는데, 여기서 새면 갱신 전체가 죽는다.
        try:
            return kakao.geocode_address(addr, settings.kakao_rest_api_key)
        except Exception as e:                          # noqa: BLE001
            print(f"   지오코딩 오류 ({addr}): {str(e)[:100]}")
            return None

    ordered = sorted(addresses)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        cache = dict(zip(ordered, pool.map(resolve, ordered)))

    geocoded = []
    for row in raw_rows:
        result = next((cache.get(addr) for addr in (row["road_address"], row["jibun_address"])
                       if addr and cache.get(addr) is not None), None)
        if result is None:
            stats.skipped_no_geocode += 1
            continue
        geocoded.append({**row, "lat": result.lat, "lon": result.lon})

    return geocoded

def fetch_transit_walk_minutes(geocoded: list[dict], settings, rng: np.random.Generator,
                               stats: Stats) -> list[float]:
    """카카오 로컬 API로 가장 가까운 지하철역까지의 거리를 조회해 도보 시간(분)으로 환산한다.
    검색 실패시에만 mock과 동일한 분포로 추정치를 채운다."""
    def nearest(row: dict):
        # 실패하면 아래에서 추정치로 메운다. 네트워크 오류까지 잡아야 갱신이 안 죽는다.
        try:
            return kakao.search_category_nearest(row["lat"], row["lon"],
                                                 kakao.CATEGORY_CODES["subway"],
                                                 settings.kakao_rest_api_key)
        except Exception:                               # noqa: BLE001
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        places = list(pool.map(nearest, geocoded))

    # rng 는 순서대로 뽑아야 시드가 같을 때 결과가 재현된다 — 폴백 채우기는 순차로.
    out = []
    for place in places:
        if place is not None and place.distance_m is not None:
            out.append(round(np.clip(place.distance_m / WALK_M_PER_MIN, 1, 60), 1))
            stats.real += 1
        else:
            out.append(float(_walk_minutes_estimate(rng, 1)[0]))
            stats.estimated_fallback += 1
    return out


def derive_synthetic_common_fields(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """실측 불가 필드(방/주차/향/총층수/단지규모/녹지/일조/소음) — mock과 동일 공식."""
    n = len(df)
    df = df.copy()

    area = df["area_m2"].to_numpy()
    floor = df["floor"].to_numpy()
    built_year = df["built_year"].to_numpy()

    df["rooms"] = np.clip(np.round(area / 27 + rng.normal(0, 0.45, n)), 1, 7).astype(int)

    df["parking_slots"] = np.clip(
        np.round(rng.normal(1.05, 0.32, n) + (built_year - 2007) * 0.012, 2), 0.25, 3.0)

    orient_keys = list(ORIENTATION_SCORE)
    orient_p = np.array([0.34, 0.17, 0.14, 0.13, 0.11, 0.06, 0.05])
    df["orientation"] = rng.choice(orient_keys, size=n, p=orient_p / orient_p.sum())

    sampled_total_floors = np.clip(np.round(rng.normal(19, 6, n)), 5, 45).astype(int)
    df["total_floors"] = np.maximum(floor, sampled_total_floors)

    complex_units = np.clip(np.round(rng.lognormal(6.4, 0.75, n)), 60, 6000).astype(int)
    df["complex_units"] = complex_units
    df["green_ratio"] = np.clip(rng.normal(28, 9, n) + np.log1p(complex_units) * 1.9
                                 + (built_year - 2007) * 0.12, 5, 62).round(1)
    df["daylight_hours"] = np.clip(
        rng.normal(5.2, 1.1, n)
        + np.array([ORIENTATION_SCORE[o] for o in df["orientation"]]) * 2.2
        + (floor / df["total_floors"].to_numpy()) * 1.1, 1.0, 10.0).round(1)
    df["noise_db"] = np.clip(rng.normal(52, 6.5, n) - df["green_ratio"].to_numpy() * 0.14
                              + (df["total_floors"].to_numpy() - floor) * 0.06, 33, 74).round(1)
    return df


def build_listings(geocoded: list[dict], transit_walk_min: list[float],
                   rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(geocoded)
    df.insert(0, "listing_id", [f"H{str(i + 1).zfill(4)}" for i in range(len(df))])
    df["transit_walk_min"] = transit_walk_min
    df = derive_synthetic_common_fields(df, rng)
    df = df[["listing_id", "name", "gu", "lat", "lon", "area_m2", "rooms", "parking_slots",
             "orientation", "transit_walk_min", "built_year", "floor", "total_floors",
             "green_ratio", "complex_units", "daylight_hours", "noise_db"]]
    assert_no_price_columns(df.columns)
    return df


def build_live_listings(settings, codes: list[str] | None = None) -> tuple[pd.DataFrame, Stats]:
    """공공데이터와 카카오 API에서 추천 후보를 메모리에 생성한다.

    codes 로 조회할 자치구를 좁힐 수 있다(없으면 설정값 전체).
    CLI 배치와 Streamlit 앱이 같은 수집 규칙을 사용하며, 이 함수는 파일을 쓰지
    않아 Community Cloud의 읽기 전용 배포 환경에서도 호출할 수 있다.
    """
    if not settings.has_data_go_kr_key or not settings.has_kakao_key:
        raise ValueError("DATA_GO_KR_SERVICE_KEY와 KAKAO_REST_API_KEY가 모두 필요합니다.")

    stats = Stats()
    rng = np.random.default_rng(SEED)
    raw_rows = fetch_raw_rent_rows(settings, stats, codes)
    geocoded = geocode_rows(raw_rows, settings, stats)
    if not geocoded:
        raise ValueError("공공데이터에서 좌표가 확인된 후보를 가져오지 못했습니다.")

    transit_walk_min = fetch_transit_walk_minutes(geocoded, settings, rng, stats)
    return build_listings(geocoded, transit_walk_min, rng), stats

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA_DIR)
    ap.add_argument("--dry-run", action="store_true", help="CSV 저장 없이 통계만 출력")
    args = ap.parse_args()

    settings = load_settings()
    if not settings.has_data_go_kr_key or not settings.has_kakao_key:
        print("❌ .env 에 DATA_GO_KR_SERVICE_KEY / KAKAO_REST_API_KEY 가 필요합니다.")
        sys.exit(1)

    stats = Stats()
    rng = np.random.default_rng(SEED)

    print("=== 1) 공공데이터포털: 전월세 실거래가 수집 ===")
    raw_rows = fetch_raw_rent_rows(settings, stats)
    print(f"총 {len(raw_rows)}건 원시 수집 (불량 {stats.skipped_bad_row}건 제외)")

    print("\n=== 2) 카카오 지오코딩 ===")
    geocoded = geocode_rows(raw_rows, settings, stats)
    print(f"지오코딩 성공 {len(geocoded)}건 (실패 {stats.skipped_no_geocode}건)")

    if not geocoded:
        print("❌ 지오코딩된 매물이 하나도 없습니다. 종료합니다.")
        sys.exit(1)

    print("\n=== 3) 카카오 최근접 지하철역 조회 (역세권 도보시간) ===")
    transit_walk_min = fetch_transit_walk_minutes(geocoded, settings, rng, stats)

    print("\n=== 4) listings.csv 구성 (실측 + 추정 공식) ===")
    listings = build_listings(geocoded, transit_walk_min, rng)
    print(listings[["listing_id", "name", "gu", "area_m2", "floor", "built_year",
                    "transit_walk_min"]].to_string(index=False))

    print("\n=== 5) travel_times.csv / members.json (기본 데모 구성원 스냅샷) ===")
    members = make_default_members()
    travel = travel_matrix(listings, members, EstimatedTravelProvider(seed=SEED), all_modes=True)
    assert_no_price_columns(travel.columns)

    print(f"\n=== 완료 === 지하철역 실측 {stats.real}건, 검색 실패로 추정 대체 {stats.estimated_fallback}건")

    if args.dry_run:
        print("\n--dry-run 이라 파일을 저장하지 않았습니다.")
        return

    os.makedirs(args.out, exist_ok=True)
    listings.to_csv(os.path.join(args.out, "listings.csv"), index=False)
    travel.to_csv(os.path.join(args.out, "travel_times.csv"), index=False)
    with open(os.path.join(args.out, "members.json"), "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in members], f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "real_data_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "국토교통부_아파트 전월세 실거래가 + 카카오 로컬 API",
            "deal_ym": settings.target_deal_ym,
            "sigungu_codes": settings.target_sigungu_codes,
            "n_listings": len(listings),
            "transit_walk_stats": stats.as_dict(),
            "real_fields": ["name", "gu", "lat", "lon", "area_m2", "floor", "built_year",
                            "transit_walk_min (실측 우선, 검색 실패시 추정)"],
            "estimated_fields": ["rooms", "parking_slots", "orientation", "total_floors",
                                 "green_ratio", "complex_units", "daylight_hours", "noise_db"],
            "note": "travel_times.csv 는 기본 데모 구성원 템플릿 스냅샷이며, 앱은 사용자가 "
                    "입력한 목적지로 매 세션 실시간 계산한다(경로 API 사용 토글 시 "
                    "src/travel.py ApiTravelProvider, 지도 실경로는 src/data_sources/kakao.py).",
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장 완료: {args.out}/listings.csv ({len(listings)}행), "
          f"{args.out}/travel_times.csv, {args.out}/members.json, {args.out}/real_data_report.json")


if __name__ == "__main__":
    main()
