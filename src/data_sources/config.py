"""
.env 로부터 외부 API 연동 설정을 읽어온다.

`DATA_GO_KR_SERVICE_KEY` 는 공공데이터포털이 Encoding/Decoding 두 형태로
발급하기 때문에, 어느 쪽을 넣어도 requests 가 정확히 한 번만 percent-encoding
하도록 여기서 미리 디코딩해 둔다(이미 인코딩된 키를 그대로 넘기면 이중 인코딩되어
서비스키 인증 오류(SERVICE_KEY_IS_NOT_REGISTERED_ERROR 등)가 발생하기 쉽다).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import unquote

from dotenv import load_dotenv

from .public_data import DEFAULT_APT_RENT_ENDPOINT

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _normalize_data_go_kr_key(raw: str) -> str:
    key = raw.strip()
    if "%" in key:
        key = unquote(key)
    return key


def _previous_year_month() -> str:
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}{m:02d}"


DEFAULT_SIGUNGU_CODES = [
    "11680", "11710", "11440", "11200", "11350",
    "11470", "11560", "11380", "11590", "11215",
]


@dataclass
class Settings:
    data_go_kr_service_key: str
    kakao_rest_api_key: str
    kakao_js_key: str = ""
    apt_rent_endpoint: str = DEFAULT_APT_RENT_ENDPOINT
    target_sigungu_codes: list[str] = field(default_factory=lambda: list(DEFAULT_SIGUNGU_CODES))
    target_deal_ym: str = field(default_factory=_previous_year_month)
    max_complexes_per_sigungu: int = 0
    request_interval_sec: float = 0.2

    @property
    def has_data_go_kr_key(self) -> bool:
        return bool(self.data_go_kr_service_key)

    @property
    def has_kakao_key(self) -> bool:
        return bool(self.kakao_rest_api_key)

    @property
    def has_kakao_js_key(self) -> bool:
        return bool(self.kakao_js_key)


def load_settings() -> Settings:
    _ensure_env_loaded()

    raw_key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    kakao_key = os.environ.get("KAKAO_REST_API_KEY", "").strip()
    kakao_js_key = os.environ.get("KAKAO_JS_KEY", "").strip()

    codes_raw = os.environ.get("TARGET_SIGUNGU_CODES", "").strip()
    codes = [c.strip() for c in codes_raw.split(",") if c.strip()] or list(DEFAULT_SIGUNGU_CODES)

    max_n = os.environ.get("MAX_COMPLEXES_PER_SIGUNGU", "0").strip()
    interval = os.environ.get("REQUEST_INTERVAL_SEC", "0.2").strip()

    rent_endpoint = os.environ.get("DATA_GO_KR_APT_RENT_ENDPOINT", "").strip() or DEFAULT_APT_RENT_ENDPOINT
    deal_ym = os.environ.get("TARGET_DEAL_YM", "").strip() or _previous_year_month()

    return Settings(
        data_go_kr_service_key=_normalize_data_go_kr_key(raw_key) if raw_key else "",
        kakao_rest_api_key=kakao_key,
        kakao_js_key=kakao_js_key,
        apt_rent_endpoint=rent_endpoint,
        target_sigungu_codes=codes,
        target_deal_ym=deal_ym,
        max_complexes_per_sigungu=int(max_n) if max_n else 0,
        request_interval_sec=float(interval) if interval else 0.2,
    )
