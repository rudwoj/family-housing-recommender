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


def _read_secret(name: str) -> str:
    """환경변수와 Streamlit Cloud Secrets에서 설정값을 안전하게 읽는다.

    로컬 실행과 데이터 수집 스크립트는 ``.env``/환경변수를 사용한다. 반면
    Streamlit Community Cloud에서 Settings > Secrets로 추가한 값은 일반적인
    ``os.environ``이 아니라 ``st.secrets``에 있으므로, 배포 환경에서는 여기서
    명시적으로 읽어야 한다.
    """
    value = os.environ.get(name, "")
    if value:
        return value.strip()

    try:
        import streamlit as st

        secret = st.secrets.get(name, "")
        return str(secret).strip() if secret is not None else ""
    except Exception:
        # Streamlit 외부에서 실행하거나 Secrets 파일이 아직 없을 수 있다.
        return ""


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
    kakao_javascript_key: str = ""
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
        """지도 렌더링용 JavaScript 키. REST API 키와 별개로 발급된다."""
        return bool(self.kakao_javascript_key)


def load_settings() -> Settings:
    _ensure_env_loaded()

    raw_key = _read_secret("DATA_GO_KR_SERVICE_KEY")
    kakao_key = _read_secret("KAKAO_REST_API_KEY")
    kakao_js_key = _read_secret("KAKAO_JAVASCRIPT_KEY")

    codes_raw = _read_secret("TARGET_SIGUNGU_CODES")
    codes = [c.strip() for c in codes_raw.split(",") if c.strip()] or list(DEFAULT_SIGUNGU_CODES)

    max_n = _read_secret("MAX_COMPLEXES_PER_SIGUNGU") or "0"
    interval = _read_secret("REQUEST_INTERVAL_SEC") or "0.2"

    rent_endpoint = _read_secret("DATA_GO_KR_APT_RENT_ENDPOINT") or DEFAULT_APT_RENT_ENDPOINT
    deal_ym = _read_secret("TARGET_DEAL_YM") or _previous_year_month()

    return Settings(
        data_go_kr_service_key=_normalize_data_go_kr_key(raw_key) if raw_key else "",
        kakao_rest_api_key=kakao_key,
        kakao_javascript_key=kakao_js_key,
        apt_rent_endpoint=rent_endpoint,
        target_sigungu_codes=codes,
        target_deal_ym=deal_ym,
        max_complexes_per_sigungu=int(max_n) if max_n else 0,
        request_interval_sec=float(interval) if interval else 0.2,
    )
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
    kakao_javascript_key: str = ""
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
        """지도 렌더링용 JavaScript 키. REST API 키와 별개로 발급된다."""
        return bool(self.kakao_javascript_key)


def load_settings() -> Settings:
    _ensure_env_loaded()

    raw_key = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")
    kakao_key = os.environ.get("KAKAO_REST_API_KEY", "").strip()
    kakao_js_key = os.environ.get("KAKAO_JAVASCRIPT_KEY", "").strip()

    codes_raw = os.environ.get("TARGET_SIGUNGU_CODES", "").strip()
    codes = [c.strip() for c in codes_raw.split(",") if c.strip()] or list(DEFAULT_SIGUNGU_CODES)

    max_n = os.environ.get("MAX_COMPLEXES_PER_SIGUNGU", "0").strip()
    interval = os.environ.get("REQUEST_INTERVAL_SEC", "0.2").strip()

    rent_endpoint = os.environ.get("DATA_GO_KR_APT_RENT_ENDPOINT", "").strip() or DEFAULT_APT_RENT_ENDPOINT
    deal_ym = os.environ.get("TARGET_DEAL_YM", "").strip() or _previous_year_month()

    return Settings(
        data_go_kr_service_key=_normalize_data_go_kr_key(raw_key) if raw_key else "",
        kakao_rest_api_key=kakao_key,
        kakao_javascript_key=kakao_js_key,
        apt_rent_endpoint=rent_endpoint,
        target_sigungu_codes=codes,
        target_deal_ym=deal_ym,
        max_complexes_per_sigungu=int(max_n) if max_n else 0,
        request_interval_sec=float(interval) if interval else 0.2,
    )
