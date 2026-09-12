"""
피처 스키마 정의 모듈.

이 프로젝트의 모든 데이터는 두 계층으로 엄격히 분리된다.

1) 공통 데이터 (Housing Physical & Environmental Features)
   - 가족 전체가 공유하는 주택 자체의 공간/환경 특성 (면적, 방 수, 향, 준공년도 ...)
2) 개별 데이터 (Individual Location & Mobility Features)
   - 구성원 개인의 생활 동선 특성 (통근 시간, 환승 횟수, 학교 도보 거리 ...)

가격(매매가/전세가/월세/관리비 등)은 스키마 자체에 존재하지 않는다.
데이터 로딩 시 `assert_no_price_columns()` 로 한 번 더 강제한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CURRENT_YEAR = 2026

Direction = Literal["higher", "lower"]


# --------------------------------------------------------------------------- #
# 가격 배제 가드
# --------------------------------------------------------------------------- #
PRICE_TOKENS = (
    "price", "cost", "rent", "deposit", "fee", "won", "krw", "maintenance",
    "매매", "전세", "월세", "보증금", "관리비", "분양", "호가", "가격", "시세",
)


def assert_no_price_columns(columns) -> None:
    """가격 계열 컬럼이 유입되면 즉시 실패시킨다 (요구사항: 가격 완전 배제)."""
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
    direction: Direction          # higher = 클수록 좋음, lower = 작을수록 좋음
    default_weight: float = 1.0
    hard_filter: Literal["min", "max", None] = None   # 하드 제약 가능 여부
    minute_factor: float | None = None                # 이동 부담(분) 환산 계수
    short: str = ""                                   # 레이더 차트용 축약 라벨

    @property
    def axis(self) -> str:
        return self.short or self.label


# --- 공통(가족 공유) 피처 ---------------------------------------------------- #
COMMON_FEATURES: list[Feature] = [
    Feature("area_m2",              "전용면적",        "m²",   "higher", 1.0, "min", short="면적"),
    Feature("rooms",                "방 개수",         "개",   "higher", 1.0, "min", short="방"),
    Feature("bathrooms",            "욕실 개수",       "개",   "higher", 0.6, "min", short="욕실"),
    Feature("orientation_score",    "향(일조 선호도)", "점",   "higher", 0.8, None,  short="향"),
    Feature("building_age",         "준공 경과년수",   "년",   "lower",  0.8, "max", short="연식"),
    Feature("floor_comfort",        "층 쾌적도",       "점",   "higher", 0.4, None,  short="층"),
    Feature("green_ratio",          "녹지 비율",       "%",    "higher", 0.7, "min", short="녹지"),
    Feature("parking_per_unit",     "세대당 주차대수", "대",   "higher", 0.7, "min", short="주차"),
    Feature("complex_units",        "단지 규모",       "세대", "higher", 0.4, None,  short="단지"),
    Feature("daylight_hours",       "일조 시간",       "h",    "higher", 0.6, None,  short="일조"),
    Feature("noise_db",             "주간 소음도",     "dB",   "lower",  0.6, "max", short="소음"),
]

COMMON_KEYS = [f.key for f in COMMON_FEATURES]


# --- 역할별 개별(동선) 피처 라이브러리 ---------------------------------------- #
#   minute_factor: 이동 부담 지수를 만들기 위한 '분' 환산 계수
#     - 시간(분) 피처        -> 1.0
#     - 도보 거리(m) 피처    -> 0.0125 (도보 80 m/분 기준)
#     - 환승 횟수            -> 4.0  (환승 1회 ≈ 4분 체감 패널티)
WALK_MIN_PER_M = 0.0125      # 도보 80 m/분
TRANSIT_MIN_PER_M = 0.004    # 버스/자가용 등 근거리 이동 250 m/분

ROLE_FEATURES: dict[str, list[Feature]] = {
    "worker": [
        Feature("commute_min",    "직장 통근 시간",     "분",  "lower", 1.0, "max", 1.0,             "통근"),
        Feature("transfer_count", "대중교통 환승 횟수", "회",  "lower", 0.6, "max", 4.0,             "환승"),
        Feature("station_walk_m", "역까지 도보 거리",   "m",   "lower", 0.5, None,  WALK_MIN_PER_M,  "역세권"),
        Feature("mart_dist_m",    "마트 거리",          "m",   "lower", 0.4, None,  None,            "마트"),
        Feature("hospital_dist_m","병원 거리",          "m",   "lower", 0.4, None,  None,            "병원"),
    ],
    "student": [
        Feature("school_walk_m",  "학교 도보 거리",     "m",   "lower", 1.0, "max", WALK_MIN_PER_M,  "학교"),
        Feature("academy_dist_m", "학원가 거리",        "m",   "lower", 0.8, "max", TRANSIT_MIN_PER_M, "학원"),
        Feature("park_dist_m",    "공원 접근성",        "m",   "lower", 0.5, None,  None,            "공원"),
        Feature("library_dist_m", "도서관 접근성",      "m",   "lower", 0.5, None,  None,            "도서관"),
    ],
}


# --------------------------------------------------------------------------- #
# 가족 구성원
# --------------------------------------------------------------------------- #
@dataclass
class Member:
    key: str                       # 컬럼 prefix (예: dad -> dad__commute_min)
    name: str                      # 표시명
    role: str                      # ROLE_FEATURES 의 키
    color: str                     # 지도/차트 색상
    anchor_label: str = ""         # 주요 목적지 명칭 (직장/학교)
    anchor_lat: float | None = None
    anchor_lon: float | None = None
    priority: float = 1.0          # 가족 내 발언권(구성원 가중치 P_i)
    active: bool = True
    feature_weights: dict[str, float] = field(default_factory=dict)

    @property
    def features(self) -> list[Feature]:
        return ROLE_FEATURES[self.role]

    def column(self, feature_key: str) -> str:
        return f"{self.key}__{feature_key}"

    @property
    def columns(self) -> list[str]:
        return [self.column(f.key) for f in self.features]

    def default_weights(self) -> dict[str, float]:
        return {f.key: f.default_weight for f in self.features}

    def resolved_weights(self) -> dict[str, float]:
        w = self.default_weights()
        w.update({k: v for k, v in self.feature_weights.items() if k in w})
        return w

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "role": self.role, "color": self.color,
            "anchor_label": self.anchor_label, "anchor_lat": self.anchor_lat,
            "anchor_lon": self.anchor_lon, "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Member":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


DEFAULT_MEMBERS: list[Member] = [
    Member("dad",   "아빠",   "worker",  "#2563eb", "여의도 직장", 37.5216, 126.9243),
    Member("mom",   "엄마",   "worker",  "#db2777", "강남 직장",   37.4979, 127.0276),
    Member("child", "자녀",   "student", "#059669", "대치 학원가", 37.4995, 127.0630),
]


ORIENTATION_SCORE: dict[str, float] = {
    "남": 1.00, "남동": 0.88, "남서": 0.82, "동": 0.68,
    "서": 0.52, "북동": 0.36, "북": 0.20,
}

FEATURE_BY_KEY: dict[str, Feature] = {f.key: f for f in COMMON_FEATURES}
for _fs in ROLE_FEATURES.values():
    for _f in _fs:
        FEATURE_BY_KEY.setdefault(_f.key, _f)


# 파생 지표 라벨 (차트/표 표기용)
DERIVED_LABELS: dict[str, str] = {
    "travel_mean_min": "평균 이동시간",
    "travel_std_min": "이동시간 편차",
    "travel_max_min": "최대 이동시간",
    "inequality_index": "불평등도 지수",
    "equity_index": "형평성 지수",
    "stability_index": "주거 안정성",
    "fit_score": "종합 적합도",
    "sat_min": "최약자 만족도",
    "sat_gap": "만족도 격차",
    "distance": "이상점 거리",
}
