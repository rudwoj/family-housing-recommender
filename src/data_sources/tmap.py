"""
TMap 보행자 경로안내 클라이언트 (SK open API).

도보는 카카오·네이버 모두 전용 길찾기를 제공하지 않는다. 국내에서 실제
인도/횡단보도를 따라가는 보행자 경로를 주는 건 사실상 TMap 뿐이라 여기만 붙인다.

키 발급
  https://openapi.sk.com > 로그인 > 대시보드 > 프로젝트(앱) 생성
  > PRODUCTS > 교통/위치 > TMAP > API 사용 요금 > 무료체험 Free > 사용하기
  앱에 상품이 붙어 있지 않으면 앱키가 있어도 403 INVALID_API_KEY 가 난다.
  .env 에 TMAP_APP_KEY=..., 배포본은 Secrets 최상위 키로 TMAP_APP_KEY = "..."

주의
  - Free 요금제 보행자 경로안내 한도는 1,000건/일이다.
  - 서비스 지역이 서울·수도권·6대광역시·제주 등으로 제한된다.
  - startName/endName 은 URL 인코딩해서 보내야 한다.
  - 실패하면 호출부가 거리 기반 추정으로 폴백해야 한다.

응답(GeoJSON FeatureCollection)
  features[].properties.totalTime      총 소요 시간(초)  ← Point 피처에 실려온다
  features[].properties.totalDistance  총 거리(m)
  features[].geometry.coordinates      LineString 이면 [[경도, 위도], ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote

import requests

PEDESTRIAN_URL = "https://apis.openapi.sk.com/tmap/routes/pedestrian"
DEFAULT_TIMEOUT = 10


class TmapError(RuntimeError):
    """TMap API 호출/응답 오류."""


@dataclass
class WalkRoute:
    total_time_sec: float
    distance_m: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)


def pedestrian_route(origin_lat: float, origin_lon: float,
                     dest_lat: float, dest_lon: float,
                     app_key: str, timeout: int = DEFAULT_TIMEOUT) -> WalkRoute | None:
    """보행자 경로 1건. 경로가 없으면 None, 호출 실패는 TmapError."""
    body = {
        "startX": str(origin_lon), "startY": str(origin_lat),
        "endX": str(dest_lon), "endY": str(dest_lat),
        "startName": quote("출발"), "endName": quote("도착"),
        "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO",
        "searchOption": "0",
    }
    resp = requests.post(
        PEDESTRIAN_URL,
        params={"version": "1", "format": "json"},
        headers={"Content-Type": "application/json", "appKey": app_key},
        json=body, timeout=timeout)

    if resp.status_code != 200:
        raise TmapError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except ValueError as e:
        raise TmapError(f"JSON 파싱 실패: {resp.text[:200]}") from e

    if "error" in data:
        err = data["error"] or {}
        raise TmapError(f"{err.get('code', '?')}: {err.get('message', '')}"[:200])

    features = data.get("features") or []
    if not features:
        return None

    total_time = None
    total_distance = 0.0
    path: list[tuple[float, float]] = []

    for feat in features:
        props = feat.get("properties") or {}
        if total_time is None and props.get("totalTime") is not None:
            total_time = float(props["totalTime"])
            total_distance = float(props.get("totalDistance") or 0.0)

        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        for c in geom.get("coordinates") or []:
            if len(c) >= 2:
                try:
                    path.append((float(c[1]), float(c[0])))
                except (TypeError, ValueError):
                    continue

    if total_time is None:
        return None

    return WalkRoute(total_time_sec=total_time, distance_m=total_distance, path=path)
