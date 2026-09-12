"""
.env 에 채운 API 키로 실제 어떤 데이터를 받아올 수 있는지 확인하는 진단 스크립트.

실제 수집 파이프라인(scripts/fetch_real_dataset.py, 아직 미구현)을 만들기 전에
반드시 먼저 실행해서:
  1) 공공데이터포털 서비스키가 유효한지, 국토교통부_아파트 전월세 실거래가 API가
     실제로 어떤 필드를 반환하는지 (data.go.kr 는 API 버전이 종종 바뀐다)
  2) 카카오 REST API 키로 지오코딩 / 카테고리·키워드 장소검색이 되는지
  3) 카카오모빌리티 자동차 길찾기(실측 이동시간)가 이 키로 동작하는지
  4) 실험적인 대중교통 경로 API가 이 키로 동작하는지
를 눈으로 확인한다.

주의: 전월세 실거래가 API는 가격(보증금/월세) 필드를 포함한다. 이 프로젝트는
가격을 절대 쓰지 않으므로(src/schema.py assert_no_price_columns), 화면 출력에서
가격 필드는 별도로 표시해 "이건 쓰면 안 되는 값"임을 명확히 한다.

실행:
    python scripts/check_api_access.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_sources import kakao, public_data  # noqa: E402
from src.data_sources.config import load_settings  # noqa: E402

LINE = "-" * 70


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def check_public_data(settings) -> str | None:
    section("① 공공데이터포털 — 아파트 전월세 실거래가")
    if not settings.has_data_go_kr_key:
        print("⚠️  DATA_GO_KR_SERVICE_KEY 가 .env 에 비어 있습니다. 이 단계는 건너뜁니다.")
        return None

    sample_code = settings.target_sigungu_codes[0]
    print(f"엔드포인트: {settings.apt_rent_endpoint}")
    print(f"LAWD_CD={sample_code}, DEAL_YMD={settings.target_deal_ym} 로 실거래 조회 중...")
    resp = public_data.fetch_apt_rent(sample_code, settings.target_deal_ym,
                                       settings.data_go_kr_service_key,
                                       endpoint=settings.apt_rent_endpoint, num_of_rows=5)
    if not resp.ok:
        print(f"❌ 조회 실패: {resp.error}")
        print("   원본 응답(앞 500자):", resp.raw_text[:500])
        if (resp.error or "").startswith("12:") or "폐기됨" in (resp.error or ""):
            print("   -> 이 엔드포인트 자체가 존재하지 않는다는 뜻입니다 (키 문제 아님).")
        print("   -> data.go.kr 로그인 > 마이페이지 > 활용신청 현황 > 해당 API 클릭 시 나오는")
        print("      '요청 URL' 예제를 그대로 복사해 .env 의 DATA_GO_KR_APT_RENT_ENDPOINT 에 넣고 재시도하세요.")
        return None

    print(f"✅ 조회 성공 — {len(resp.items)}건 (요청 5건 기준, {settings.target_deal_ym} 기준)")
    if not resp.items:
        print("   해당 달에 거래가 없을 수 있습니다. TARGET_DEAL_YM 을 다른 달로 바꿔 재시도해보세요.")
        return None

    print("\n첫 번째 item 의 전체 필드 (가격 필드는 [가격-사용금지] 로 표시):")
    first = resp.items[0]
    address = None
    for k, v in first.items():
        tag = " [가격-사용금지]" if k in public_data.KNOWN_PRICE_FIELDS else ""
        print(f"   {k:20s} = {v}{tag}")
        if any(t in k for t in ("주소", "지번", "addr", "Jibun", "umd", "Umd")):
            address = address or str(v)

    return address


def check_kakao(settings, sample_address: str | None) -> None:
    section("② 카카오 로컬 API — 지오코딩 / 장소검색")
    if not settings.has_kakao_key:
        print("⚠️  KAKAO_REST_API_KEY 가 .env 에 비어 있습니다. 이 단계는 건너뜁니다.")
        return

    address = sample_address or "서울특별시 강남구 테헤란로 152"
    print(f"주소 '{address}' 지오코딩 중...")
    try:
        geo = kakao.geocode_address(address, settings.kakao_rest_api_key)
    except kakao.KakaoError as e:
        print(f"❌ 지오코딩 실패: {e}")
        return

    if geo is None:
        print("⚠️  검색 결과 없음. 다른 주소로 재시도 필요.")
        return
    print(f"✅ 지오코딩 성공 — lat={geo.lat}, lon={geo.lon} ({geo.address_name})")

    print("\n반경 5km 내 카테고리별 최근접 장소:")
    for label, code in kakao.CATEGORY_CODES.items():
        try:
            place = kakao.search_category_nearest(geo.lat, geo.lon, code, settings.kakao_rest_api_key)
        except kakao.KakaoError as e:
            print(f"   {label:10s} ❌ {e}")
            continue
        if place is None:
            print(f"   {label:10s} 결과 없음")
        else:
            print(f"   {label:10s} {place.place_name} — {place.distance_m:.0f}m")
        kakao.sleep_between_calls(settings.request_interval_sec)

    print("\n반경 5km 내 키워드 검색(카테고리 코드 없는 것들):")
    for label, kw in kakao.KEYWORD_ONLY.items():
        try:
            place = kakao.search_keyword_nearest(geo.lat, geo.lon, kw, settings.kakao_rest_api_key)
        except kakao.KakaoError as e:
            print(f"   {label:10s} ❌ {e}")
            continue
        if place is None:
            print(f"   {label:10s} 결과 없음")
        else:
            print(f"   {label:10s} {place.place_name} — {place.distance_m:.0f}m")
        kakao.sleep_between_calls(settings.request_interval_sec)

    section("③ 카카오모빌리티 — 자동차 길찾기 (실제 이동시간/거리)")
    dest_lat, dest_lon = 37.4979, 127.0276  # 강남 anchor 샘플
    try:
        route = kakao.car_directions(geo.lat, geo.lon, dest_lat, dest_lon, settings.kakao_rest_api_key)
    except kakao.KakaoError as e:
        print(f"❌ 자동차 길찾기 실패: {e}")
        print("   -> 카카오 디벨로퍼스 콘솔에서 이 앱에 '카카오모빌리티' 상품이 활성화되어 있는지 확인하세요.")
        print("      (로컬 API 와 REST 키는 같지만 상품 활성화가 별도인 경우가 있음)")
    else:
        if route is None:
            print("⚠️  경로를 찾지 못했습니다 (좌표/도로망 확인 필요).")
        else:
            print(f"✅ 자동차 길찾기 성공 — {route.duration_sec/60:.0f}분 · {route.distance_m/1000:.1f}km")
            print("   -> commute_min 후보로 이 값을 실측치로 쓸 수 있습니다 (대중교통이 아닌 자가용 기준).")

    section("④ 카카오맵 — 대중교통 경로 조회 (2026-07-21 신규 오픈)")
    try:
        transit = kakao.transit_route(geo.lat, geo.lon, dest_lat, dest_lon, settings.kakao_rest_api_key)
    except kakao.KakaoError as e:
        print(f"❌ 대중교통 경로 조회 실패: {e}")
        print("   -> 카카오 디벨로퍼스 콘솔 [앱] > [제품 설정] > [카카오맵] 에서")
        print("      '사용 설정'을 켰는지 확인하세요 (별도 심사 없이 바로 켤 수 있음).")
    else:
        if transit is None:
            print("⚠️  경로를 찾지 못했습니다 (좌표 확인 필요).")
        else:
            fare_txt = f"{transit.fare_won:.0f}원" if transit.fare_won is not None else "요금 정보 없음"
            print(f"✅ 대중교통 경로 조회 성공 — {transit.total_time_sec/60:.0f}분 · "
                  f"환승 {transit.transfers}회 · {transit.route_type} · {fare_txt}")
            print("   -> commute_min 을 대중교통 실측치로 채울 수 있습니다.")


def main() -> None:
    settings = load_settings()
    print("설정 로드 완료:")
    print(f"  대상 시군구코드: {settings.target_sigungu_codes}")
    print(f"  대상 계약년월: {settings.target_deal_ym}")
    print(f"  공공데이터포털 키 설정됨: {settings.has_data_go_kr_key}")
    print(f"  카카오 키 설정됨: {settings.has_kakao_key}")

    if not settings.has_data_go_kr_key and not settings.has_kakao_key:
        print("\n.env 에 키가 하나도 없습니다. .env.example 을 참고해 .env 를 채운 뒤 다시 실행하세요.")
        return

    sample_address = check_public_data(settings)
    check_kakao(settings, sample_address)

    section("완료")
    print("위 출력을 근거로 실제 필드명이 확인되면 scripts/fetch_real_dataset.py 파이프라인을 완성합니다.")


if __name__ == "__main__":
    main()
