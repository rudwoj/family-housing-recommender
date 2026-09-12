"""
전처리: 이상치 클리핑 → 스케일링 → 결측 보정 → 비선형 효용 변환.

설계 요지
  ① 클리핑(Winsorizing)
     상·하위 q% 를 잘라 초대형 평수/극단 이동시간 같은 이상치가 MinMax 스케일을
     독점해 일반 매물을 0~0.1 구간에 뭉개는 문제를 막는다.
  ② 스케일러
     minmax  : 클리핑된 범위를 [0,1] 로 선형 매핑
     robust  : 중앙값/IQR 기준. ±2 IQR 밖은 포화시켜 꼬리만 압축한다
     standard: 평균/표준편차 기준. ±3σ 포화
     어느 쪽이든 결과는 [0,1] 의 '상대 위치'로 통일된다.
  ③ 비선형 효용 변환
     사람의 만족도는 선형이 아니다.
       concave : 면적·방 수처럼 체감 효용이 감소하는 재화
       convex  : 이동시간처럼 길어질수록 불만이 가속되는 비용
     변환 후 값은 모두 '만족도(0~1, 1이 최선)'이므로 이상점은 전 차원 1 벡터가 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .schema import Feature

EPS = 1e-12
SATURATION = {"robust": 2.0, "standard": 3.0}   # IQR / σ 배수


def winsorize(df: pd.DataFrame, q: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """상·하위 q 분위로 클리핑. (클리핑된 df, 구간별 잘린 개수 리포트) 반환."""
    if q <= 0:
        return df, pd.DataFrame(columns=["피처", "하한", "상한", "잘린 값"])
    lo, hi = df.quantile(q), df.quantile(1 - q)
    clipped = df.clip(lower=lo, upper=hi, axis=1)
    n_clipped = ((df < lo) | (df > hi)).sum()
    report = pd.DataFrame({
        "피처": df.columns, "하한": lo.round(2).values, "상한": hi.round(2).values,
        "잘린 값": n_clipped.values,
    })
    return clipped, report[report["잘린 값"] > 0].reset_index(drop=True)


def make_scaler(kind: str):
    return {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler}[kind]()


def to_unit_interval(values: np.ndarray, kind: str) -> np.ndarray:
    """스케일러 출력을 [0,1] 상대 위치로 통일한다."""
    if kind == "minmax":
        return np.clip(values, 0.0, 1.0)
    s = SATURATION[kind]
    return (np.clip(values, -s, s) + s) / (2 * s)


def utility(position: np.ndarray, feature: Feature) -> np.ndarray:
    """
    [0,1] 상대 위치 → [0,1] 만족도. 1 이 항상 최선.

      higher + concave : u = t^(1/γ)        (체감 효용 감소)
      lower  + convex  : u = 1 - t^γ        (불만 가속)
      linear           : u = t  또는 1 - t
    """
    t = np.clip(position, 0.0, 1.0)
    if feature.direction == "lower":
        if feature.curve == "convex":
            return 1.0 - np.power(t, max(feature.gamma, EPS))
        if feature.curve == "concave":
            return 1.0 - np.power(t, 1.0 / max(feature.gamma, EPS))
        return 1.0 - t
    if feature.curve == "concave":
        return np.power(t, 1.0 / max(feature.gamma, EPS))
    if feature.curve == "convex":
        return np.power(t, max(feature.gamma, EPS))
    return t


class UtilityPipeline:
    """클리핑 → 스케일 → 보정 → 효용 변환을 한 번에 수행하고 진단을 남긴다."""

    # 분포가 이보다 좁게 눌리면 이상치에 스케일을 빼앗긴 것으로 본다
    COMPRESSION_THRESHOLD = 0.25

    def __init__(self, scaler: str = "robust", clip_q: float = 0.01, n_neighbors: int = 5):
        self.scaler_kind = scaler
        self.clip_q = clip_q
        self.n_neighbors = n_neighbors
        self.report: dict = {}

    def fit_transform(self, raw: pd.DataFrame, features: dict[str, Feature]) -> pd.DataFrame:
        cols = list(raw.columns)
        clipped, clip_report = winsorize(raw.astype(float), self.clip_q)

        scaler = make_scaler(self.scaler_kind)
        scaled = scaler.fit_transform(clipped)          # NaN 은 통과
        n_missing = int(np.isnan(scaled).sum())

        k = max(1, min(self.n_neighbors, len(clipped) - 1))
        filled = KNNImputer(n_neighbors=k, weights="distance").fit_transform(scaled)
        if filled.shape[1] != len(cols):                # 전부 결측인 열은 상위에서 제거됨
            raise ValueError("보정 단계에서 피처가 소실되었습니다.")

        position = to_unit_interval(filled, self.scaler_kind)
        util = np.column_stack([utility(position[:, i], features[c]) for i, c in enumerate(cols)])

        pos_df = pd.DataFrame(position, index=raw.index, columns=cols)
        spread = (pos_df.quantile(0.95) - pos_df.quantile(0.05)).round(3)
        compressed = spread[spread < self.COMPRESSION_THRESHOLD].index.tolist()

        self.scaler = scaler
        self.report = {
            "clip_report": clip_report, "missing_cells": n_missing, "impute_k": k,
            "clip_q": self.clip_q, "scaler": self.scaler_kind,
            "position_spread": spread,
            "compressed_features": compressed,
        }
        self.position = pos_df
        return pd.DataFrame(util, index=raw.index, columns=cols)

    def inverse_raw(self, position_df: pd.DataFrame) -> pd.DataFrame:
        """[0,1] 위치 → 원 단위 (클리핑된 범위 기준)."""
        kind = self.scaler_kind
        if kind == "minmax":
            z = position_df.values
        else:
            s = SATURATION[kind]
            z = position_df.values * (2 * s) - s
        return pd.DataFrame(self.scaler.inverse_transform(z),
                            index=position_df.index, columns=position_df.columns)
