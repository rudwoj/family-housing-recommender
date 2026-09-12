"""
카카오 로컬 API (developers.kakao.com/docs/latest/ko/local) +
카카오모빌리티 길찾기 API (developers.kakaomobility.com) 클라이언트.

제공 확인된 기능:
  - 주소->좌표 변환, 키워드/카테고리 장소 검색(거리순 정렬 시 거리 포함) — 카카오 로컬 API
  - 자동차 길찾기(실제 소요시간/거리) — 카카오모빌리티 Directions API. 문서상으로는
    로컬 API와 동일한 "REST API 키"(Authorization: KakaoAK ...)를 쓰지만, 카카오
    디벨로퍼스 콘솔에서 해당 앱에 "카카오모빌리티" 상품을 별도로 활성화해야 호출이
    되는 경우가 있어 — 실제 동작 여부는 check_api_access.py 로 확인한다.

  - 대중교통 경로 조회 — 2026-07-21 카카오맵 신규 API 4종으로 공식 오픈됨
    (dapi.kakao.com/v2/routing/publictraffic). 디벨로퍼스 콘솔의 [앱] > [제품 설정] >
    [카카오맵] 에서 사용 설정을 켜야 호출된다. 실제 호출로 정식 엔드포인트임을 확인함
    (더미 키로도 로컬 API와 동일한 AccessDeniedError 형식 응답).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

BASE = "https://dapi.kakao.com/v2/local"
ADDRESS_SEARCH_URL = f"{BASE}/search/address.json"
KEYWORD_SEARCH_URL = f"{BASE}/search/keyword.json"
CATEGORY_SEARCH_URL = f"{BASE}/search/category.json"

# 카카오모빌리티 자동차 길찾기 (공식 문서 확인됨: developers.kakaomobility.com/docs/navi-api/directions)
CAR_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

# 카카오맵 대중교통 경로 조회 (공식, 2026-07-21 오픈: [앱]>[제품 설정]>[카카오맵] 사용 설정 필요)
TRANSIT_ROUTE_URL = "https://dapi.kakao.com/v2/routing/publictraffic"

DEFAULT_TIMEOUT = 8

# 카테고리 그룹 코드 (공식 문서 기준)
CATEGORY_CODES = {
    "mart": "MT1",       # 대형마트
    "hospital": "HP8",   # 병원
    "school": "SC4",     # 학교
    "academy": "AC5",    # 학원
    "subway": "SW8",     # 지하철역
    "pharmacy": "PM9",   # 약국
}

# 카테고리 코드가 없어 키워드 검색으로 대체해야 하는 것들
KEYWORD_ONLY = {
    "park": "공원",
    "library": "도서관",
}


class KakaoError(RuntimeError):
    """카카오 API 호출/응답 오류."""


@dataclass
class GeocodeResult:
    lat: float
    lon: float
    address_name: str
    road_address_name: str | None = None


@dataclass
class PlaceResult:
    place_name: str
    distance_m: float | None
    lat: float
    lon: float


def _headers(api_key: str) -> dict:
    return {"Authorization": f"KakaoAK {api_key}"}


def _get(url: str, api_key: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.get(url, headers=_headers(api_key), params=params, timeout=timeout)
    if resp.status_code != 200:
        raise KakaoError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def geocode_address(query: str, api_key: str) -> GeocodeResult | None:
    """주소 문자열 -> 좌표. 매칭 결과가 없으면 None."""
    data = _get(ADDRESS_SEARCH_URL, api_key, {"query": query, "size": 1})
    docs = data.get("documents", [])
    if not docs:
        return None
    d = docs[0]
    road = d.get("road_address")
    return GeocodeResult(
        lat=float(d["y"]), lon=float(d["x"]),
        address_name=d.get("address_name", query),
        road_address_name=road.get("address_name") if road else None,
    )


def _nearest_from_documents(docs: list[dict]) -> PlaceResult | None:
    if not docs:
        return None
    d = docs[0]
    dist = d.get("distance")
    return PlaceResult(
        place_name=d.get("place_name", ""),
        distance_m=float(dist) if dist not in (None, "") else None,
        lat=float(d["y"]), lon=float(d["x"]),
    )


def search_category_nearest(lat: float, lon: float, category_group_code: str, api_key: str,
                             radius: int = 5000) -> PlaceResult | None:
    """좌표 주변에서 가장 가까운 카테고리(MT1/HP8/SC4/AC5/SW8/PM9) 장소."""
    data = _get(CATEGORY_SEARCH_URL, api_key, {
        "category_group_code": category_group_code,
        "x": lon, "y": lat, "radius": radius, "sort": "distance", "size": 1,
    })
    return _nearest_from_documents(data.get("documents", []))


def search_keyword_nearest(lat: float, lon: float, keyword: str, api_key: str,
                            radius: int = 5000) -> PlaceResult | None:
    """카테고리 코드가 없는 장소(공원/도서관 등)를 키워드로 검색."""
    data = _get(KEYWORD_SEARCH_URL, api_key, {
        "query": keyword, "x": lon, "y": lat, "radius": radius, "sort": "distance", "size": 1,
    })
    return _nearest_from_documents(data.get("documents", []))


def _vertexes_to_latlon(vertexes: list[float]) -> list[tuple[float, float]]:
    """[x1, y1, x2, y2, ...] (경도, 위도 평탄화) -> [(위도, 경도), ...]."""
    return [(vertexes[i + 1], vertexes[i]) for i in range(0, len(vertexes) - 1, 2)]


def _extract_route_path(route: dict) -> list[tuple[float, float]]:
    """
    routes[0] 하나에서 실제 이동 경로 좌표열을 뽑아낸다. 두 API가 서로 다른 스키마를
    쓰는 걸 실제 응답으로 확인함:
      - 자동차 길찾기(Directions API): sections[].roads[].vertexes = [x1,y1,x2,y2,...] (평탄화)
      - 대중교통(publictraffic API): steps[].path.points = [[x, y], [x, y], ...] (도보/버스/지하철
        구간 전부 포함, 좌표 순서대로 이어붙이면 전체 동선이 된다)
    둘 다 없으면 빈 리스트를 반환해 호출부가 직선으로 폴백하게 한다.
    """
    points: list[tuple[float, float]] = []

    for section in route.get("sections", []) or []:
        for road in section.get("roads", []) or []:
            verts = road.get("vertexes") or []
            if verts:
                points.extend(_vertexes_to_latlon(verts))

    for step in route.get("steps", []) or []:
        for p in (step.get("path") or {}).get("points") or []:
            if len(p) >= 2:
                points.append((float(p[1]), float(p[0])))

    return points


@dataclass
class CarRoute:
    duration_sec: float
    distance_m: float
    path: list[tuple[float, float]] = field(default_factory=list)


def car_directions(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
                    api_key: str, priority: str = "RECOMMEND") -> CarRoute | None:
    """
    카카오모빌리티 자동차 길찾기. 좌표는 "경도,위도" 순서로 보낸다.
    앱에 카카오모빌리티 상품이 활성화되어 있지 않으면 403/401 로 KakaoError 가 발생한다.
    """
    params = {
        "origin": f"{origin_lon},{origin_lat}",
        "destination": f"{dest_lon},{dest_lat}",
        "priority": priority,
    }
    data = _get(CAR_DIRECTIONS_URL, api_key, params)
    routes = data.get("routes", [])
    if not routes:
        return None
    summary = routes[0].get("summary", {})
    duration = summary.get("duration")
    distance = summary.get("distance")
    if duration is None:
        return None
    return CarRoute(duration_sec=float(duration), distance_m=float(distance or 0),
                     path=_extract_route_path(routes[0]))


@dataclass
class TransitRoute:
    total_time_sec: float
    transfers: int
    route_type: str
    fare_won: float | None
    path: list[tuple[float, float]] = field(default_factory=list)


def transit_route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float,
                   api_key: str) -> TransitRoute | None:
    """
    카카오맵 대중교통 경로 조회. 디벨로퍼스 콘솔에서 [앱]>[제품 설정]>[카카오맵]
    사용 설정이 켜져 있어야 한다 (꺼져 있으면 KakaoError).
    """
    params = {
        "start_x": origin_lon, "start_y": origin_lat,
        "end_x": dest_lon, "end_y": dest_lat,
    }
    data = _get(TRANSIT_ROUTE_URL, api_key, params)
    status = data.get("status")
    if status and status != "OK":
        return None
    routes = data.get("routes", [])
    if not routes:
        return None
    props = routes[0].get("properties", {})
    total_time = props.get("totalTime")
    if total_time is None:
        return None
    fare = props.get("fare")
    return TransitRoute(
        total_time_sec=float(total_time),
        transfers=int(props.get("transfers", 0) or 0),
        route_type=props.get("type", ""),
        fare_won=float(fare["value"]) if isinstance(fare, dict) and fare.get("value") is not None else None,
        path=_extract_route_path(routes[0]),
    )


def sleep_between_calls(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
