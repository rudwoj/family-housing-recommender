"""
공공데이터포털(data.go.kr) · 국토교통부_아파트 전월세 실거래가 자료 클라이언트.

이 API는 "단지"가 아니라 실제 개별 전월세 계약 건 단위 데이터라서, 이 프로젝트의
"매물" 단위와 정확히 맞는다 — 계약 1건 = 매물 1건으로 다루면 된다. 대신 이 API는
본질적으로 가격(보증금/월세) 데이터셋이므로, 이 프로젝트에서는 면적/층/건축년도/
주소 등 물리적 속성만 뽑아 쓰고 금액 필드는 절대 사용하지 않는다
(src/schema.py 의 assert_no_price_columns() 로 이중 검증한다).

엔드포인트는 실제 호출로 존재를 확인했다(2026-09, dummy key 로
SERVICE_KEY_IS_NOT_REGISTERED_ERROR 응답 확인 — 즉 서비스는 살아있고 키만 필요한 상태).
다만 응답 포맷(JSON/XML), 필드명은 실제 키로 scripts/check_api_access.py 를 돌려 확정한다.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

DEFAULT_APT_RENT_ENDPOINT = "http://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent"

DEFAULT_TIMEOUT = 10

# 응답에 섞여 올 수 있는 가격 관련 필드(참고용) — 실제 배제는
# src.schema.assert_no_price_columns() 로 강제한다.
KNOWN_PRICE_FIELDS = {
    "deposit", "monthlyRent", "preDeposit", "preMonthlyRent",
    "보증금액", "월세금액", "종전계약보증금", "종전계약월세",
}


class PublicDataError(RuntimeError):
    """공공데이터포털 API 호출/응답 오류."""


# data.go.kr 는 API 마다 성공 코드 자릿수가 다르다 (00 / 0 / 000 / 0000 등 전부 관찰됨).
_SUCCESS_CODES = {"00", "0", "000", "0000"}
_SUCCESS_MSGS = {"ok", "normal_code", "normal service.", "success"}


def _is_success(code: str | None, msg: str | None) -> bool:
    if code is None and msg is None:
        return True
    if code in _SUCCESS_CODES:
        return True
    return (msg or "").strip().lower() in _SUCCESS_MSGS


@dataclass
class RawApiResponse:
    ok: bool
    items: list[dict]
    raw_text: str
    error: str | None = None


def _xml_items_to_dicts(items_el: ET.Element) -> list[dict]:
    out = []
    for item_el in items_el.findall("item"):
        out.append({child.tag: (child.text or "") for child in item_el})
    return out


def _parse_xml(text: str) -> RawApiResponse | None:
    """data.go.kr 는 오류를 XML(OpenAPI_ServiceResponse)로, 경우에 따라 정상 응답도
    XML(response/header/body/items/item)로 반환한다. XML이 아니면 None을 반환해
    호출부가 JSON 파싱을 시도하게 한다."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    if root.tag == "OpenAPI_ServiceResponse" or root.find(".//cmmMsgHeader") is not None:
        code = root.findtext(".//returnReasonCode")
        msg = root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
        return RawApiResponse(ok=False, items=[], raw_text=text, error=f"{code}: {msg}")

    if root.tag == "response":
        result_code = root.findtext(".//header/resultCode")
        result_msg = root.findtext(".//header/resultMsg")
        if not _is_success(result_code, result_msg):
            return RawApiResponse(ok=False, items=[], raw_text=text, error=f"{result_code}: {result_msg}")
        items_el = root.find(".//body/items")
        items = _xml_items_to_dicts(items_el) if items_el is not None else []
        return RawApiResponse(ok=True, items=items, raw_text=text)

    return RawApiResponse(ok=False, items=[], raw_text=text,
                           error=f"알 수 없는 XML 최상위 태그: {root.tag}")


def _get(endpoint: str, params: dict, timeout: int = DEFAULT_TIMEOUT) -> RawApiResponse:
    try:
        resp = requests.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as e:
        return RawApiResponse(ok=False, items=[], raw_text="", error=f"네트워크 오류: {e}")

    text = resp.text

    xml_result = _parse_xml(text)
    if xml_result is not None:
        return xml_result

    try:
        data = resp.json()
    except ValueError:
        return RawApiResponse(ok=False, items=[], raw_text=text,
                               error=f"JSON/XML 모두 파싱 실패 (status={resp.status_code})")

    # 게이트웨이 오류가 JSON 형식으로 올 때는 최상위 키가 'response' 가 아니라
    # 'OpenAPI_ServiceResponse' 다.
    if "OpenAPI_ServiceResponse" in data:
        header = data["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
        code = header.get("returnReasonCode")
        msg = header.get("returnAuthMsg") or header.get("errMsg")
        return RawApiResponse(ok=False, items=[], raw_text=text, error=f"{code}: {msg}")

    if "response" not in data:
        return RawApiResponse(ok=False, items=[], raw_text=text,
                               error=f"예상치 못한 응답 형식 (status={resp.status_code}, 최상위 키={list(data.keys())})")

    header = data.get("response", {}).get("header", {})
    result_code = header.get("resultCode")
    result_msg = header.get("resultMsg")
    if not _is_success(result_code, result_msg):
        return RawApiResponse(ok=False, items=[], raw_text=text, error=f"{result_code}: {result_msg}")

    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})
    if isinstance(items, dict):
        item = items.get("item", [])
        items = item if isinstance(item, list) else ([item] if item else [])
    elif not isinstance(items, list):
        items = []

    return RawApiResponse(ok=True, items=items, raw_text=text)


def fetch_apt_rent(lawd_cd: str, deal_ymd: str, service_key: str, *,
                    endpoint: str = DEFAULT_APT_RENT_ENDPOINT,
                    num_of_rows: int = 100, page_no: int = 1) -> RawApiResponse:
    """
    법정동코드 5자리(LAWD_CD) + 계약년월 6자리(DEAL_YMD, 예: '202608') 기준
    아파트 전월세 실거래 목록. item 필드 예: 아파트명/주소, 전용면적, 층, 건축년도,
    계약년월일, 보증금/월세(가격 — 사용 금지).
    """
    params = {
        "serviceKey": service_key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "type": "json",
    }
    return _get(endpoint, params)


def raise_if_error(resp: RawApiResponse, context: str) -> None:
    if not resp.ok:
        raise PublicDataError(f"{context} 실패: {resp.error}")
