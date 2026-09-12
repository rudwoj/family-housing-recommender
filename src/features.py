"""
파생 변수 생성 (Feature Engineering).

단순 물리 거리를 넘어서 가족 단위 의사결정에 필요한 지표를 정량화한다.
  - 구성원별 이동 부담(분) 환산
  - 가족 평균 이동시간 / 이동시간 표준편차(구성원 간 불평등도)
  - 형평성 지수(Equity Index)
  - 주거 환경 종합 안정성 지수(Housing Stability Index)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COMMON_FEATURES, CURRENT_YEAR, ORIENTATION_SCORE, Member

IDEAL_FLOOR_RATIO = 0.62   # 너무 낮지도 높지도 않은 층을 선호한다고 가정
FLOOR_TOLERANCE = 0.35


def derive_common_features(listings: pd.DataFrame) -> pd.DataFrame:
    """원천 공통 데이터로부터 모델이 쓰는 공통 피처를 파생시킨다."""
    df = listings.copy()

    df["building_age"] = (CURRENT_YEAR - df["built_year"]).clip(lower=0)
    df["orientation_score"] = df["orientation"].map(ORIENTATION_SCORE)

    ratio = (df["floor"] / df["total_floors"]).clip(0, 1)
    df["floor_comfort"] = np.exp(-(((ratio - IDEAL_FLOOR_RATIO) / FLOOR_TOLERANCE) ** 2)).round(4)

    for f in COMMON_FEATURES:
        if f.key not in df.columns:
            raise KeyError(f"공통 피처 '{f.key}' 가 데이터에 없습니다.")
    return df


def member_travel_burden(df: pd.DataFrame, member: Member) -> pd.Series:
    """구성원 1인의 체감 이동 부담(분). minute_factor 가 정의된 피처만 합산한다."""
    parts, total_w = [], 0.0
    for f in member.features:
        if f.minute_factor is None:
            continue
        col = member.column(f.key)
        if col not in df.columns:
            continue
        parts.append(df[col].astype(float) * f.minute_factor)
        total_w += 1.0
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1)


def add_derived_metrics(df: pd.DataFrame, members: list[Member]) -> pd.DataFrame:
    """가족 단위 파생 지표를 추가한다 (거리 계산과 무관한 사후 해석 지표)."""
    out = df.copy()
    burden_cols = []
    for m in members:
        col = f"burden_min__{m.key}"
        out[col] = member_travel_burden(out, m).round(1)
        burden_cols.append(col)

    B = out[burden_cols]
    out["travel_mean_min"] = B.mean(axis=1).round(1)
    out["travel_std_min"] = B.std(axis=1, ddof=0).round(1)
    out["travel_max_min"] = B.max(axis=1).round(1)
    # 불평등도 지수: 변동계수(CV) 기반. 0 에 가까울수록 구성원 간 부담이 균등.
    out["inequality_index"] = (out["travel_std_min"] / out["travel_mean_min"].replace(0, np.nan)).round(3)
    out["equity_index"] = (1 - out["inequality_index"].clip(0, 1)).round(3)

    out["stability_index"] = housing_stability_index(out).round(3)
    return out


def housing_stability_index(df: pd.DataFrame) -> pd.Series:
    """
    주거 환경 종합 안정성 지수 (0~1).
    녹지·주차·일조·단지규모·연식·소음을 min-max 정규화 후 가중 합성한다.
    """
    spec = {
        "green_ratio": (0.22, "higher"),
        "parking_per_unit": (0.20, "higher"),
        "daylight_hours": (0.16, "higher"),
        "complex_units": (0.14, "higher"),
        "building_age": (0.16, "lower"),
        "noise_db": (0.12, "lower"),
    }
    score = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for col, (w, direction) in spec.items():
        s = df[col].astype(float)
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or hi == lo:
            continue
        norm = (s - lo) / (hi - lo)
        if direction == "lower":
            norm = 1 - norm
        score = score.add(norm * w, fill_value=0.0)
        wsum = wsum.add(norm.notna().astype(float) * w, fill_value=0.0)
    return (score / wsum.replace(0, np.nan)).clip(0, 1)


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """
    최대화 목적함수 행렬(n × m)에 대한 파레토 비지배 해 마스크.
    어떤 구성원도 손해보지 않으면서 누군가는 이득인 대안이 없는 점 = 파레토 최적.
    """
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    X = np.nan_to_num(objectives, nan=-np.inf)
    for i in range(n):
        if not mask[i]:
            continue
        dominates = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dominates.any():
            mask[i] = False
    return mask
