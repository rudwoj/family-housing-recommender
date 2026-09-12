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
from concurrent.futures import ThreadPoolExecutor
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


def _secret(name: str) -> str:
    """환경변수 → .env → Streamlit Secrets 순으로 키를 찾는다.

    여기만 os.getenv 를 보면, Secrets 로만 키를 넣은 배포본에서 지도 선은 실제
    경로인데 이동시간만 조용히 추정치로 떨어진다. 설정 로더를 import 하지 않고
    자립적으로 두는 이유는, 배포본의 config.py 가 뒤처져 있어도 깨지지 않게
    하기 위함이다(실제로 ImportError 로 앱이 죽은 적이 있다).
    """
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:
        from dotenv import load_dotenv
        load_dotenv()
        value = os.environ.get(name, "")
        if value:
            return value.strip()
    except Exception:
        pass
    try:
        import streamlit as st
        secret = st.secrets.get(name, "")
        return str(secret).strip() if secret is not None else ""
    except Exception:
        return ""


class ApiTravelProvider(TravelProvider):
    """
    실제 경로 API 연동.

      KAKAO_REST_API_KEY : 자동차(카카오모빌리티) + 대중교통(카카오맵 경로 조회)
      ODSAY_API_KEY      : 대중교통 (있으면 이쪽을 먼저 쓴다 — 카카오 대중교통은
                           응답이 커서 배포 환경에서 타임아웃이 잦다)
      TMAP_APP_KEY       : 도보 (SK open API. 국내에서 보행자 경로를 주는 곳이
                           사실상 여기뿐이라, 없으면 도보는 거리 기반 추정)

    도보는 TMAP_APP_KEY 가 있을 때만 실측하고, 없으면 거리 기반 추정을 쓴다.
    호출이 실패하면 해당 매물만 추정치로 메운다(전체를 버리지 않는다).
    매물 수 × 목적지 수만큼 호출이 발생하므로 캐시를 반드시 함께 쓸 것.
    """

    name = "경로 API"

    #: 대중교통까지 켜면 호출 수가 배로 늘어 순차 호출은 너무 느리다.
    MAX_WORKERS = 6

    def __init__(self, fallback: TravelProvider | None = None, timeout: float = 15.0):
        self.fallback = fallback or EstimatedTravelProvider()
        self.timeout = timeout
        self.kakao_key = _secret("KAKAO_REST_API_KEY")
        self.odsay_key = _secret("ODSAY_API_KEY")
        self.tmap_key = _secret("TMAP_APP_KEY")
        #: 모드별 마지막 실패 사유 — 화면에서 "왜 추정치인가" 를 보여주기 위함
        self.last_error: dict[str, str] = {}

    @property
    def available(self) -> bool:
        return bool(self.kakao_key or self.odsay_key or self.tmap_key)

    def supports(self, mode: str) -> bool:
        """실제 API 로 계산 가능한 모드인지."""
        if mode == "drive":
            return bool(self.kakao_key)
        if mode == "transit":
            return bool(self.kakao_key or self.odsay_key)
        # 도보는 카카오/ODsay 모두 전용 길찾기가 없다. TMap 이 있을 때만 실측한다.
        return bool(self.tmap_key)

    def minutes(self, lat, lon, dest: Destination, mode: str) -> np.ndarray:
        est = self.fallback.minutes(lat, lon, dest, mode)
        if not self.supports(mode):
            if mode == "walk":
                self.last_error.setdefault(
                    mode, "TMAP_APP_KEY 가 없어 거리 기반 추정 "
                         "(카카오·ODsay 는 도보 길찾기를 제공하지 않는다)")
            return est

        lats, lons = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        out = np.array(est, dtype=float, copy=True)

        def one(i: int) -> tuple[int, float | None]:
            try:
                return i, self._one_minute(float(lats[i]), float(lons[i]), dest, mode)
            except Exception as e:                      # noqa: BLE001 - 사유를 보여주려 넓게 잡는다
                self.last_error.setdefault(mode, str(e)[:160])
                return i, None

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            for i, val in pool.map(one, range(len(lats))):
                if val is not None:
                    out[i] = val

        return np.clip(np.round(out, 1), 1.0, 600.0)

    def _one_minute(self, la: float, lo: float, dest: Destination, mode: str) -> float | None:
        from .data_sources import kakao as kakao_api

        if mode == "walk":
            from .data_sources import tmap as tmap_api
            route = tmap_api.pedestrian_route(la, lo, dest.lat, dest.lon, self.tmap_key,
                                              timeout=int(self.timeout))
            return route.total_time_sec / 60.0 if route else None

        if mode == "drive":
            route = kakao_api.car_directions(la, lo, dest.lat, dest.lon, self.kakao_key,
                                             timeout=int(self.timeout))
            return route.duration_sec / 60.0 if route else None

        # 대중교통: ODsay 를 먼저 쓴다. 카카오 대중교통은 경로 수십 개의 좌표까지
        # 실려와 응답이 크고, 배포 환경의 느린 네트워크에서 타임아웃이 잦았다.
        if self.odsay_key:
            from .data_sources import odsay as odsay_api
            try:
                route = odsay_api.transit_route(la, lo, dest.lat, dest.lon,
                                                self.odsay_key, timeout=int(self.timeout))
                if route is not None:
                    return route.total_time_sec / 60.0
            except Exception as e:                      # noqa: BLE001
                self.last_error.setdefault("transit", f"ODsay: {str(e)[:120]}")
                if not self.kakao_key:
                    raise

        if self.kakao_key:
            route = kakao_api.transit_route(la, lo, dest.lat, dest.lon, self.kakao_key,
                                            timeout=int(self.timeout))
            if route is not None:
                return route.total_time_sec / 60.0

        return None


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
