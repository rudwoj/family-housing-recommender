"""
목적지까지의 이동 시간 계산 (차 / 대중교통 / 도보).

구성원이 입력한 목적지 좌표와 각 매물 좌표로부터 이동 시간을 만든다.
Provider 를 갈아끼우는 구조라, 실제 경로 API 키가 있으면 그대로 대체된다.

  EstimatedTravelProvider : 거리 기반 추정 (기본값, 오프라인 동작)
  ApiTravelProvider       : 외부 경로 API 연동 지점 (키 필요)

※ ApiTravelProvider 는 키가 없어 이 환경에서 실행 검증하지 못했다.
   키를 넣고 쓰기 전에 반드시 한 번 호출해 응답 형식을 확인할 것.
   실패 시에는 예외를 삼키고 추정치로 자동 폴백한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import TRAVEL_MODES, Destination, Member

EARTH_R = 6371.0
DETOUR = 1.32          # 직선거리 → 실제 경로 보정 계수


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
@dataclass
class ModeProfile:
    """이동수단별 속도(km/h)와 고정 오버헤드(분)."""
    speed_kmh: float
    overhead_min: float
    per_km_extra: float = 0.0      # 거리 비례 추가 지연 (환승/신호 등)


MODE_PROFILES: dict[str, ModeProfile] = {
    # 도심 평균 주행 + 주차/진출입 오버헤드
    "drive":   ModeProfile(speed_kmh=27.0, overhead_min=6.0, per_km_extra=0.10),
    # 도보 접근 + 대기 + 환승을 오버헤드로 흡수
    "transit": ModeProfile(speed_kmh=22.0, overhead_min=13.0, per_km_extra=0.45),
    "walk":    ModeProfile(speed_kmh=4.4,  overhead_min=1.0),
}


class TravelProvider:
    name = "base"

    def minutes(self, lat, lon, dest: Destination, mode: str) -> np.ndarray:
        raise NotImplementedError


class EstimatedTravelProvider(TravelProvider):
    """직선거리 × 우회계수 기반 추정. 외부 의존성이 없어 항상 동작한다."""

    name = "거리 기반 추정"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def minutes(self, lat, lon, dest: Destination, mode: str) -> np.ndarray:
        p = MODE_PROFILES[mode]
        d_km = haversine_km(np.asarray(lat), np.asarray(lon), dest.lat, dest.lon) * DETOUR
        minutes = p.overhead_min + d_km / p.speed_kmh * 60 + d_km * p.per_km_extra
        if self.rng is not None:      # 목업 데이터에 현실적인 산포를 준다
            minutes = minutes * self.rng.normal(1.0, 0.07, size=np.shape(minutes))
        return np.clip(np.round(minutes, 1), 1.0, 600.0)


class ApiTravelProvider(TravelProvider):
    """
    외부 경로 API 연동 지점.

    환경변수로 키를 주면 사용한다.
      KAKAO_REST_API_KEY : 자동차 경로 (Kakao Mobility)
      ODSAY_API_KEY      : 대중교통 경로 (ODsay)
    키가 없거나 호출이 실패하면 추정치로 폴백한다.
    매물 수 × 목적지 수만큼 호출이 발생하므로 캐시를 반드시 함께 쓸 것.
    """

    name = "경로 API"

    def __init__(self, fallback: TravelProvider | None = None, timeout: float = 3.0):
        self.fallback = fallback or EstimatedTravelProvider()
        self.timeout = timeout
        self.kakao_key = os.getenv("KAKAO_REST_API_KEY")
        self.odsay_key = os.getenv("ODSAY_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.kakao_key or self.odsay_key)

    def minutes(self, lat, lon, dest: Destination, mode: str) -> np.ndarray:
        if not self.available or mode == "walk":
            return self.fallback.minutes(lat, lon, dest, mode)
        try:
            return self._call_api(lat, lon, dest, mode)
        except Exception:
            return self.fallback.minutes(lat, lon, dest, mode)

    def _call_api(self, lat, lon, dest: Destination, mode: str) -> np.ndarray:
        import requests  # 선택 의존성

        out = []
        for la, lo in zip(np.asarray(lat), np.asarray(lon)):
            if mode == "drive" and self.kakao_key:
                r = requests.get(
                    "https://apis-navi.kakaomobility.com/v1/directions",
                    params={"origin": f"{lo},{la}", "destination": f"{dest.lon},{dest.lat}"},
                    headers={"Authorization": f"KakaoAK {self.kakao_key}"},
                    timeout=self.timeout)
                sec = r.json()["routes"][0]["summary"]["duration"]
            elif mode == "transit" and self.odsay_key:
                r = requests.get(
                    "https://api.odsay.com/v1/api/searchPubTransPathT",
                    params={"apiKey": self.odsay_key, "SX": lo, "SY": la,
                            "EX": dest.lon, "EY": dest.lat},
                    timeout=self.timeout)
                sec = r.json()["result"]["path"][0]["info"]["totalTime"] * 60
            else:
                raise RuntimeError("no key for mode")
            out.append(sec / 60.0)
        return np.round(np.array(out), 1)


# --------------------------------------------------------------------------- #
# 이동 시간 행렬
# --------------------------------------------------------------------------- #
def travel_matrix(listings: pd.DataFrame, members: list[Member],
                  provider: TravelProvider | None = None,
                  all_modes: bool = False) -> pd.DataFrame:
    """
    매물 × (구성원·목적지·이동수단) 이동 시간(분) 테이블.

    all_modes=True 면 차/대중교통/도보를 모두 계산한다(데이터 스냅샷용).
    기본값은 각 목적지에 선택된 이동수단만 계산한다(앱 실시간 계산용).
    """
    provider = provider or EstimatedTravelProvider()
    lat, lon = listings["lat"].to_numpy(), listings["lon"].to_numpy()
    out = pd.DataFrame({"listing_id": listings["listing_id"].to_numpy()})

    for m in members:
        for d in m.destinations:
            modes = list(TRAVEL_MODES) if all_modes else [d.mode]
            for mode in modes:
                out[d.column(m.key, mode)] = provider.minutes(lat, lon, d, mode)
    return out
