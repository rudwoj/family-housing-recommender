"""
ODsay 대중교통 길찾기 클라이언트 (lab.odsay.com).

카카오맵 대중교통 경로 조회는 응답이 크고 느려 배포 환경에서 타임아웃에 걸리는
일이 잦아, ODSAY_API_KEY 가 있으면 대중교통은 ODsay 를 먼저 쓰고 실패하면
카카오로 넘어간다. 키가 없으면 기존대로 카카오만 쓴다.

키 발급: https://lab.odsay.com > 회원가입 > API 신청 > 대중교통 길찾기
  - 발급된 키가 URL 인코딩된 형태(`%2B` 등)로 복사되는 경우가 있다. 그대로
    requests 에 넘기면 이중 인코딩되어 인증 오류가 나므로 여기서 먼저 디코딩한다.

응답 구조 (searchPubTransPathT)
  result.path[0].info.totalTime        총 소요 시간(분)
  result.path[0].info.payment          요금(원)
  result.path[0].subPath[].trafficType 1=지하철 2=버스 3=도보
  result.path[0].subPath[].passStopList.stations[] 정류장 좌표(x=경도, y=위도)
오류일 때는 HTTP 200 인 채로 {"error": {"code": ..., "msg": ...}} 가 온다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import unquote

import requests

SEARCH_PATH_URL = "https://api.odsay.com/v1/api/searchPubTransPathT"
DEFAULT_TIMEOUT = 10


class OdsayError(RuntimeError):
    """ODsay API 호출/응답 오류."""


@dataclass
class TransitRoute:
    total_time_sec: float
    transfers: int
    fare_won: float | None = None
    path: list[tuple[float, float]] = field(default_factory=list)


def normalize_key(raw: str) -> str:
    """URL 인코딩된 채로 복사해 온 키를 한 번 디코딩한다."""
    key = (raw or "").strip()
    return unquote(key) if "%" in key else key


def _extract_path(path_obj: dict,
                  origin: tuple[float, float],
                  dest: tuple[float, float]) -> list[tuple[float, float]]:
    """subPath 들을 이어 붙여 (위도, 경도) 좌표열을 만든다.

    도보 구간은 좌표가 없으므로 앞뒤 정류장이 직선으로 이어진다. 출발/도착
    좌표를 양 끝에 붙여 실제 매물·목적지까지 선이 닿게 한다.
    """
    points: list[tuple[float, float]] = [origin]

    for sub in path_obj.get("subPath") or []:
        stations = ((sub.get("passStopList") or {}).get("stations")) or []
        for st in stations:
            try:
                points.append((float(st["y"]), float(st["x"])))
            except (KeyError, TypeError, ValueError):
                continue
        if not stations:
            # 정류장 목록이 없는 구간은 시작/끝 좌표라도 쓴다.
            for xk, yk in (("startX", "startY"), ("endX", "endY")):
                if sub.get(xk) is not None and sub.get(yk) is not None:
                    try:
                        points.append((float(sub[yk]), float(sub[xk])))
                    except (TypeError, ValueError):
                        continue

    points.append(dest)

    # 연속 중복 제거
    deduped: list[tuple[float, float]] = []
    for p in points:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return deduped


def transit_route(origin_lat: float, origin_lon: float,
                  dest_lat: float, dest_lon: float,
                  api_key: str, timeout: int = DEFAULT_TIMEOUT,
                  referer: str = "") -> TransitRoute | None:
    """대중교통 최적 경로 1건. 경로가 없으면 None, 호출 실패는 OdsayError.

    referer: ODsay 마이페이지에 "URI" 플랫폼으로 등록한 서비스 도메인. 이 키가
    서버(IP) 플랫폼이 아니라 URI 플랫폼으로만 등록돼 있으면, ODsay 는 요청의
    Referer 헤더가 등록된 도메인과 일치하는지로 인증한다 — 안 보내면 서버 사이드
    호출(requests) 은 Referer 가 비어 있어 500 ApiKeyAuthFailed 로 거부된다.
    """
    params = {
        "apiKey": normalize_key(api_key),
        "SX": origin_lon, "SY": origin_lat,
        "EX": dest_lon, "EY": dest_lat,
        "SearchPathType": 0,
    }
    headers = {"Referer": referer} if referer else None
    resp = requests.get(SEARCH_PATH_URL, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise OdsayError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
    except ValueError as e:
        raise OdsayError(f"JSON 파싱 실패: {resp.text[:200]}") from e

    if "error" in data:
        err = data["error"]
        if isinstance(err, list):
            err = err[0] if err else {}
        raise OdsayError(f"{err.get('code', '?')}: {err.get('msg', err.get('message', ''))}"[:200])

    paths = (data.get("result") or {}).get("path") or []
    if not paths:
        return None

    best = paths[0]
    info = best.get("info") or {}
    total_min = info.get("totalTime")
    if total_min is None:
        return None

    fare = info.get("payment")
    transfers = int(info.get("busTransitCount", 0) or 0) + \
        int(info.get("subwayTransitCount", 0) or 0)

    return TransitRoute(
        total_time_sec=float(total_min) * 60.0,
        transfers=max(transfers - 1, 0),
        fare_won=float(fare) if fare is not None else None,
        path=_extract_path(best, (origin_lat, origin_lon), (dest_lat, dest_lon)),
    )
