"""
피처 스키마 정의 모듈.

데이터 계층
  1) 공통 데이터 (Housing Physical Features) — 가족이 공유하는 주택 자체의 조건
       전용면적 / 방 개수 / 주차 가능 대수 / 향 / 버스·지하철역까지 도보 시간
  2) 개별 데이터 (Individual Mobility Features) — 구성원이 직접 입력한 목적지까지의
       차·대중교통·도보 이동 시간 (선택 항목, 이동수단별로 켜고 끌 수 있음)
  3) 부가 정보 (Auxiliary) — 거리 계산에는 쓰지 않고 하드 제약·표시·안정성 지수에만
       사용하는 항목 (준공년도, 층, 녹지 비율, 단지 규모, 일조, 소음)

가격(매매가/전세가/월세/관리비)은 스키마에 존재하지 않으며
`assert_no_price_columns()` 가 유입을 차단한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CURRENT_YEAR = 2026

Direction = Literal["higher", "lower"]
Curve = Literal["linear", "concave", "convex"]


# --------------------------------------------------------------------------- #
# 가격 배제 가드
# --------------------------------------------------------------------------- #
PRICE_TOKENS = (
    "price", "cost", "rent", "deposit", "fee", "won", "krw", "maintenance", "budget",
    "매매", "전세", "월세", "보증금", "관리비", "분양", "호가", "가격", "시세", "예산",
)


def assert_no_price_columns(columns) -> None:
    bad = [c for c in columns if any(tok in str(c).lower() for tok in PRICE_TOKENS)]
    if bad:
        raise ValueError(f"가격 관련 컬럼은 사용할 수 없습니다: {bad}")


# --------------------------------------------------------------------------- #
# 피처 정의
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    unit: str
    direction: Direction        # higher = 클수록 좋음, lower = 작을수록 좋음
    curve: Curve = "linear"     # 효용 함수 형태 (비선형 선호 반영)
    gamma: float = 1.0          # 곡률. concave/convex 에서만 의미
    default_weight: float = 1.0
    hard_filter: Literal["min", "max", None] = None
    short: str = ""

    @property
    def axis(self) -> str:
        return self.short or self.label


# --- 공통 피처 (거리 계산에 사용) ------------------------------------------- #
#  curve 해설
#   concave : 체감 효용 감소. 면적 60→70m² 의 만족 증가폭 > 130→140m²
#   convex  : 체감 불만 가속. 도보 5→10분 보다 20→25분 이 훨씬 괴롭다
COMMON_FEATURES: list[Feature] = [
    Feature("area_m2",          "전용면적",            "m²", "higher", "concave", 1.9, 1.0, "min", "면적"),
    Feature("rooms",            "방 개수",             "개", "higher", "concave", 1.5, 1.0, "min", "방"),
    Feature("parking_slots",    "주차 가능 대수",      "대", "higher", "concave", 2.0, 0.7, "min", "주차"),
    Feature("orientation_score","향(일조 선호도)",     "점", "higher", "linear",  1.0, 0.8, None,  "향"),
    Feature("transit_walk_min", "역까지 도보 시간",    "분", "lower",  "convex",  1.9, 0.9, "max", "역세권"),
]
COMMON_KEYS = [f.key for f in COMMON_FEATURES]


# --- 부가 정보 (거리 계산 제외 · 필터/표시/안정성 지수 전용) ----------------- #
@dataclass(frozen=True)
class AuxColumn:
    key: str
    label: str
    unit: str
    direction: Direction
    hard_filter: Literal["min", "max", None] = None
    relax_priority: int = 0     # 백오프에서 먼저 완화되는 순서 (클수록 먼저 풀린다)


AUX_COLUMNS: list[AuxColumn] = [
    AuxColumn("building_age",   "준공 경과년수", "년",   "lower",  "max", 3),
    AuxColumn("green_ratio",    "녹지 비율",     "%",    "higher", "min", 2),
    AuxColumn("complex_units",  "단지 규모",     "세대", "higher", None,  0),
    AuxColumn("daylight_hours", "일조 시간",     "h",    "higher", None,  0),
    AuxColumn("noise_db",       "주간 소음도",   "dB",   "lower",  "max", 2),
]
AUX_BY_KEY = {a.key: a for a in AUX_COLUMNS}


# --------------------------------------------------------------------------- #
# 이동수단 · 목적지 · 구성원
# --------------------------------------------------------------------------- #
TRAVEL_MODES: dict[str, str] = {"drive": "차", "transit": "대중교통", "walk": "도보"}

# 이동수단별 효용 곡률. 도보는 길어질수록 특히 가파르게 괴롭다.
MODE_GAMMA: dict[str, float] = {"drive": 1.7, "transit": 1.8, "walk": 2.2}


@dataclass
class Destination:
    """구성원이 직접 입력하는 목적지 (직장/학교/학원 등)."""
    key: str
    label: str
    lat: float
    lon: float
    mode: str = "transit"        # 선택된 이동수단 (drive/transit/walk)
    weight: float = 1.0
    enabled: bool = True

    def column(self, member_key: str, mode: str | None = None) -> str:
        return f"{member_key}__{self.key}__{mode or self.mode}_min"

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "lat": self.lat, "lon": self.lon,
                "mode": self.mode, "weight": self.weight, "enabled": self.enabled}


@dataclass
class Member:
    key: str
    name: str
    color: str
    destinations: list[Destination] = field(default_factory=list)
    priority: float = 1.0        # 가족 내 발언권 P_i
    min_satisfaction: float = 0.0  # 하한선(0~100). 이 밑으로 떨어지는 매물은 배제
    active: bool = True

    @property
    def enabled_destinations(self) -> list[Destination]:
        return [d for d in self.destinations if d.enabled]

    def columns(self) -> list[str]:
        return [d.column(self.key) for d in self.enabled_destinations]

    def feature_of(self, column: str) -> tuple[Destination, str] | None:
        for d in self.destinations:
            for mode in TRAVEL_MODES:
                if column == d.column(self.key, mode):
                    return d, mode
        return None

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "color": self.color,
                "priority": self.priority, "min_satisfaction": self.min_satisfaction,
                "destinations": [d.to_dict() for d in self.destinations]}

    @classmethod
    def from_dict(cls, d: dict) -> "Member":
        dests = [Destination(**x) for x in d.get("destinations", [])]
        return cls(key=d["key"], name=d["name"], color=d["color"], destinations=dests,
                   priority=d.get("priority", 1.0),
                   min_satisfaction=d.get("min_satisfaction", 0.0))


MEMBER_COLORS = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]
DEFAULT_MEMBER_COUNT = 4
MAX_MEMBERS = 6

# 구성원 i 의 기본 목적지 템플릿 (이름·좌표·이동수단 모두 앱에서 수정 가능)
DESTINATION_PRESETS = [
    ("직장", 37.5216, 126.9243, "transit"),
    ("직장", 37.4979, 127.0276, "drive"),
    ("학교", 37.5050, 127.0450, "walk"),
    ("학원", 37.4995, 127.0630, "transit"),
    ("주 활동지", 37.5638, 126.9084, "transit"),
    ("주 활동지", 37.5385, 127.0823, "drive"),
]
SECONDARY_PRESET = ("보조 목적지", 37.5100, 127.0200, "drive")


def make_default_members(n: int = DEFAULT_MEMBER_COUNT) -> list[Member]:
    """
    구성원 기본 구성. 이름은 '구성원 1..n' 이며 앱에서 사용자가 직접 바꾼다.
    key(m1, m2, ...)는 컬럼 식별자라 이름을 바꿔도 고정된다.
    """
    members = []
    for i in range(max(1, min(n, MAX_MEMBERS))):
        label, lat, lon, mode = DESTINATION_PRESETS[i % len(DESTINATION_PRESETS)]
        s_label, s_lat, s_lon, s_mode = SECONDARY_PRESET
        members.append(Member(
            key=f"m{i + 1}", name=f"구성원 {i + 1}", color=MEMBER_COLORS[i % len(MEMBER_COLORS)],
            destinations=[
                Destination("dest1", label, lat, lon, mode, 1.0, True),
                Destination("dest2", s_label, s_lat, s_lon, s_mode, 0.5, False),
            ]))
    return members


DEFAULT_MEMBERS: list[Member] = make_default_members()


ORIENTATION_SCORE: dict[str, float] = {
    "남": 1.00, "남동": 0.88, "남서": 0.82, "동": 0.68,
    "서": 0.52, "북동": 0.36, "북": 0.20,
}

FEATURE_BY_KEY: dict[str, Feature] = {f.key: f for f in COMMON_FEATURES}

DERIVED_LABELS: dict[str, str] = {
    "travel_mean_min": "평균 이동시간", "travel_std_min": "이동시간 편차",
    "travel_max_min": "최대 이동시간", "inequality_index": "불평등도 지수",
    "equity_index": "형평성 지수", "stability_index": "주거 안정성",
    "fit_score": "종합 적합도", "sat_min": "최약자 만족도", "sat_gap": "만족도 격차",
    "distance": "이상점 거리",
}


def travel_feature(member: Member, dest: Destination, mode: str) -> Feature:
    """목적지×이동수단 조합에 대한 동적 피처 정의."""
    return Feature(
        key=dest.column(member.key, mode),
        label=f"{dest.label} ({TRAVEL_MODES[mode]})",
        unit="분", direction="lower", curve="convex",
        gamma=MODE_GAMMA.get(mode, 1.8),
        default_weight=dest.weight, hard_filter="max",
        short=f"{dest.label}·{TRAVEL_MODES[mode]}",
    )
