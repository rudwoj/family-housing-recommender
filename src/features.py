"""
파생 변수 생성 (Feature Engineering).

모든 이동 지표는 '분' 단위 시간으로만 다룬다 (거리 m 는 사용하지 않는다).
  - 구성원별 이동 부담(분): 활성 목적지들의 가중 평균 이동 시간
  - 가족 평균 이동시간 / 표준편차(구성원 간 불평등도)
  - 형평성 지수 / 주거 환경 종합 안정성 지수
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import AUX_COLUMNS, CURRENT_YEAR, ORIENTATION_SCORE, Member

STABILITY_SPEC = {
    "green_ratio": (0.26, "higher"),
    "building_age": (0.22, "lower"),
    "complex_units": (0.16, "higher"),
    "daylight_hours": (0.18, "higher"),
    "noise_db": (0.18, "lower"),
}


def derive_common_features(listings: pd.DataFrame) -> pd.DataFrame:
    """원천 데이터로부터 모델이 쓰는 공통 피처/부가 컬럼을 파생시킨다."""
    df = listings.copy()
    df["building_age"] = (CURRENT_YEAR - df["built_year"]).clip(lower=0)
    df["orientation_score"] = df["orientation"].map(ORIENTATION_SCORE)
    return df


def member_travel_burden(df: pd.DataFrame, member: Member) -> pd.Series:
    """구성원 1인의 체감 이동 부담(분) = 활성 목적지 이동시간의 가중 평균."""
    parts, weights = [], []
    for d in member.enabled_destinations:
        col = d.column(member.key)
        if col in df.columns:
            parts.append(df[col].astype(float) * d.weight)
            weights.append(d.weight)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1) / max(sum(weights), 1e-12)


def add_derived_metrics(df: pd.DataFrame, members: list[Member], engine=None) -> pd.DataFrame:
    out = df.copy()
    burden_cols = []
    for m in members:
        col = f"burden_min__{m.key}"
        out[col] = member_travel_burden(out, m).round(1)
        burden_cols.append(col)

    if burden_cols:
        B = out[burden_cols]
        out["travel_mean_min"] = B.mean(axis=1).round(1)
        out["travel_std_min"] = B.std(axis=1, ddof=0).round(1)
        out["travel_max_min"] = B.max(axis=1).round(1)
        out["inequality_index"] = (out["travel_std_min"] /
                                   out["travel_mean_min"].replace(0, np.nan)).round(3)
        out["equity_index"] = (1 - out["inequality_index"].clip(0, 1)).round(3)
    else:
        for c in ["travel_mean_min", "travel_std_min", "travel_max_min",
                  "inequality_index", "equity_index"]:
            out[c] = np.nan

    out["stability_index"] = housing_stability_index(out).round(3)
    return out


def housing_stability_index(df: pd.DataFrame) -> pd.Series:
    """녹지·연식·단지규모·일조·소음을 정규화 후 가중 합성한 주거 안정성 (0~1)."""
    score = pd.Series(0.0, index=df.index)
    wsum = pd.Series(0.0, index=df.index)
    for col, (w, direction) in STABILITY_SPEC.items():
        if col not in df.columns:
            continue
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
    """최대화 목적함수 행렬(n × m)의 파레토 비지배 해 마스크."""
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    X = np.nan_to_num(objectives, nan=-np.inf)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask
