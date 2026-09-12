"""
공통/개별 피처 분리 + 다인(多人) 가중 K-NN 주거지 추천 엔진.

파이프라인
  ① 하드 제약 필터링   : 방 수/면적 등 타협 불가 조건으로 후보 축소 (K-NN 이전)
  ② 스케일링           : MinMax 또는 Standard 로 단위 왜곡 제거
  ③ KNNImputer         : 누락된 공간/환경/동선 정보 보정
  ④ 가중치 결합        : W_c(공통) ⊕ P_i·W_i(구성원별) → 정규화된 피처 가중치 ω
  ⑤ 가족 이상점 벡터   : 방향성(higher/lower) + 만족 임계값(satisficing) 기반
  ⑥ 가중 유클리드 K-NN : Z = X·√ω 변환 후 sklearn NearestNeighbors 로 Top-K 탐색
  ⑦ 해석               : 구성원별 만족도 / 파레토 최적 / 기여도 분해 / 군집 분석

가격 데이터는 파이프라인 어디에도 존재하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .features import add_derived_metrics, pareto_mask
from .schema import COMMON_FEATURES, COMMON_KEYS, Member, assert_no_price_columns

EPS = 1e-12

# 컬럼 라벨링에 쓰는 구성원 key -> 표시명 레지스트리 (엔진 생성 시 갱신)
MEMBER_NAMES: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# 설정 객체
# --------------------------------------------------------------------------- #
@dataclass
class HardConstraints:
    """K-NN 이전에 적용되는 절대 조건."""
    common_min: dict[str, float] = field(default_factory=dict)   # {피처: 최솟값}
    common_max: dict[str, float] = field(default_factory=dict)   # {피처: 최댓값}
    allowed_orientations: list[str] | None = None
    allowed_gu: list[str] | None = None
    member_max: dict[str, float] = field(default_factory=dict)   # {"dad__commute_min": 50}


@dataclass
class WeightConfig:
    """
    가중치 설정.
      alpha            : 공통(주거 자체) vs 개별(생활 동선) 배분. 1.0 이면 공통만.
      common_weights   : W_c  {공통 피처: 가중치}
      member_priority  : P_i  {구성원 key: 발언권}
      member_weights   : W_i  {구성원 key: {피처: 가중치}}
      satisficing      : 만족 임계값 {컬럼명: 원 단위 값}. 이 값보다 좋으면 동일 취급.
    """
    alpha: float = 0.5
    common_weights: dict[str, float] = field(default_factory=dict)
    member_priority: dict[str, float] = field(default_factory=dict)
    member_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    satisficing: dict[str, float] = field(default_factory=dict)


@dataclass
class Recommendation:
    top: pd.DataFrame              # Top-K 추천 (원 단위 + 점수)
    scored: pd.DataFrame           # 전체 후보 점수 (파레토/트레이드오프용)
    weights: pd.Series             # 최종 피처 가중치 ω (합 = 1)
    ideal_raw: pd.Series           # 가족 이상점 벡터 (원 단위)
    ideal_scaled: pd.Series        # 가족 이상점 벡터 (스케일 공간)
    scaled: pd.DataFrame           # 스케일/보정 완료 피처 행렬
    contributions: pd.DataFrame    # 후보별 피처 거리 기여도(비율, 행 합=1)
    clusters: pd.DataFrame         # Top-K 매물 -> 군집 라벨
    cluster_profile: pd.DataFrame  # 군집별 특성 프로파일
    diagnostics: dict


# --------------------------------------------------------------------------- #
# 추천 엔진
# --------------------------------------------------------------------------- #
class FamilyHousingRecommender:
    def __init__(
        self,
        listings: pd.DataFrame,
        individual: pd.DataFrame,
        members: list[Member],
        scaler: str = "minmax",
        n_impute_neighbors: int = 5,
    ):
        assert_no_price_columns(listings.columns)
        assert_no_price_columns(individual.columns)

        self.members = members
        MEMBER_NAMES.update({m.key: m.name for m in members})
        self.scaler_kind = scaler
        self.n_impute_neighbors = n_impute_neighbors

        df = listings.merge(individual, on="listing_id", how="inner")
        self.data = df.set_index("listing_id")

    # ---------------- 공통/개별 피처 분리 ---------------- #
    @property
    def active_members(self) -> list[Member]:
        return [m for m in self.members if m.active]

    def common_columns(self) -> list[str]:
        return [k for k in COMMON_KEYS if k in self.data.columns]

    def individual_columns(self) -> list[str]:
        cols = []
        for m in self.active_members:
            cols += [c for c in m.columns if c in self.data.columns]
        return cols

    def feature_columns(self) -> list[str]:
        return self.common_columns() + self.individual_columns()

    def column_owner(self, col: str) -> str:
        return "공통" if col in COMMON_KEYS else col.split("__", 1)[0]

    # ---------------- ① 하드 제약 필터링 ---------------- #
    def hard_filter(self, hc: HardConstraints) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.data
        mask = pd.Series(True, index=df.index)
        report = []

        def apply(name, cond, kept_before):
            nonlocal mask
            mask &= cond
            report.append({"조건": name, "탈락": int(kept_before - mask.sum()), "잔여": int(mask.sum())})

        for key, v in hc.common_min.items():
            if key in df.columns and v is not None:
                apply(f"{label_of(key)} ≥ {v}", df[key].fillna(np.inf) >= v, mask.sum())
        for key, v in hc.common_max.items():
            if key in df.columns and v is not None:
                apply(f"{label_of(key)} ≤ {v}", df[key].fillna(-np.inf) <= v, mask.sum())
        if hc.allowed_orientations:
            apply(f"향 ∈ {hc.allowed_orientations}", df["orientation"].isin(hc.allowed_orientations), mask.sum())
        if hc.allowed_gu:
            apply(f"지역 ∈ {len(hc.allowed_gu)}곳", df["gu"].isin(hc.allowed_gu), mask.sum())
        for col, v in hc.member_max.items():
            if col in df.columns and v is not None:
                # 결측은 보수적으로 통과시키고 이후 KNNImputer 가 보정한다.
                apply(f"{label_of(col)} ≤ {v}", (df[col] <= v) | df[col].isna(), mask.sum())

        return df[mask], pd.DataFrame(report)

    # ---------------- ②③ 스케일링 + 결측 보정 ---------------- #
    def _preprocess(self, candidates: pd.DataFrame, cols: list[str]):
        raw = candidates[cols].astype(float)

        # 후보 전체가 결측인 피처는 보정 자체가 불가능하므로 제외한다.
        dropped = [c for c in cols if raw[c].isna().all()]
        cols = [c for c in cols if c not in dropped]
        if not cols:
            raise ValueError("후보 매물에 유효한 피처 값이 없습니다. 하드 제약을 완화하세요.")
        raw = raw[cols]

        scaler = MinMaxScaler() if self.scaler_kind == "minmax" else StandardScaler()
        scaled = scaler.fit_transform(raw)              # NaN 은 그대로 통과
        n_missing = int(np.isnan(scaled).sum())

        k = max(1, min(self.n_impute_neighbors, len(raw) - 1))
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        filled = imputer.fit_transform(scaled)

        X = pd.DataFrame(filled, index=raw.index, columns=cols)
        return X, scaler, cols, {"missing_cells": n_missing, "impute_k": k,
                                 "dropped_features": dropped}

    # ---------------- ④ 가중치 결합 ---------------- #
    def build_weights(self, cfg: WeightConfig, cols: list[str]) -> pd.Series:
        w = pd.Series(0.0, index=cols, dtype=float)

        common_cols = [c for c in cols if c in COMMON_KEYS]
        ind_cols = [c for c in cols if c not in COMMON_KEYS]
        alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
        if not ind_cols:
            alpha = 1.0
        if not common_cols:
            alpha = 0.0

        # 공통 파트: alpha × 정규화된 W_c
        if common_cols:
            wc = pd.Series({c: max(cfg.common_weights.get(c, 1.0), 0.0) for c in common_cols})
            if wc.sum() > EPS:
                w.loc[common_cols] = alpha * wc / wc.sum()

        # 개별 파트: (1-alpha) × 정규화된 P_i × 정규화된 W_i
        if ind_cols:
            prio = {m.key: max(cfg.member_priority.get(m.key, m.priority), 0.0)
                    for m in self.active_members}
            p_sum = sum(prio.values())
            if p_sum > EPS:
                for m in self.active_members:
                    mcols = [c for c in ind_cols if c.startswith(f"{m.key}__")]
                    if not mcols:
                        continue
                    wi = cfg.member_weights.get(m.key) or m.resolved_weights()
                    vec = pd.Series({c: max(wi.get(c.split("__", 1)[1], 1.0), 0.0) for c in mcols})
                    if vec.sum() <= EPS:
                        continue
                    w.loc[mcols] = (1 - alpha) * (prio[m.key] / p_sum) * (vec / vec.sum())

        total = w.sum()
        return w / total if total > EPS else pd.Series(1.0 / len(cols), index=cols)

    # ---------------- ⑤ 가족 이상점 벡터 ---------------- #
    def _ideal_vector(self, X: pd.DataFrame, cols: list[str]) -> pd.Series:
        from .schema import FEATURE_BY_KEY
        ideal = {}
        for c in cols:
            key = c if c in COMMON_KEYS else c.split("__", 1)[1]
            f = FEATURE_BY_KEY[key]
            ideal[c] = X[c].max() if f.direction == "higher" else X[c].min()
        return pd.Series(ideal)

    def _scale_value(self, scaler, cols: list[str], col: str, raw_v: float) -> float:
        """원 단위 값 하나를 스케일 공간으로 변환 (스케일러 파라미터 직접 사용)."""
        j = cols.index(col)
        if self.scaler_kind == "minmax":
            return float(raw_v * scaler.scale_[j] + scaler.min_[j])
        return float((raw_v - scaler.mean_[j]) / scaler.scale_[j])

    def _apply_satisficing(self, X: pd.DataFrame, ideal: pd.Series, cfg: WeightConfig,
                           scaler, cols: list[str]) -> pd.DataFrame:
        """
        만족 임계값(satisficing): '이 정도면 충분' 지점을 넘어서는 초과 성능은
        거리 계산에서 상쇄한다. (통근 20분과 10분을 굳이 구분하지 않겠다는 선언)
        """
        if not cfg.satisficing:
            return X
        X = X.copy()
        for col, raw_v in cfg.satisficing.items():
            if col not in cols or raw_v is None:
                continue
            t = self._scale_value(scaler, cols, col, float(raw_v))
            if ideal[col] <= t:      # 작을수록 좋은 피처
                X[col] = X[col].clip(lower=t)
            else:                    # 클수록 좋은 피처
                X[col] = X[col].clip(upper=t)
        return X

    # ---------------- ⑥⑦ 추천 ---------------- #
    def recommend(self, cfg: WeightConfig, hc: HardConstraints | None = None,
                  k: int = 10, n_clusters: int = 3) -> Recommendation:
        hc = hc or HardConstraints()
        cols = self.feature_columns()
        if not cols:
            raise ValueError("활성화된 피처가 없습니다. 구성원 또는 공통 피처를 선택하세요.")

        candidates, filter_report = self.hard_filter(hc)
        if candidates.empty:
            raise ValueError("하드 제약을 만족하는 매물이 없습니다. 조건을 완화하세요.")

        X, scaler, cols, prep_info = self._preprocess(candidates, cols)

        # 보정된 값을 원 단위로 되돌려 결측 칸을 채운다(파생 지표·표시용).
        raw_filled = pd.DataFrame(scaler.inverse_transform(X.values), index=X.index, columns=cols)
        candidates = candidates.copy()
        for c in cols:
            candidates[c] = candidates[c].astype(float).fillna(raw_filled[c].round(2))

        ideal = self._ideal_vector(X, cols)
        Xe = self._apply_satisficing(X, ideal, cfg, scaler, cols)

        w = self.build_weights(cfg, cols)
        sqrt_w = np.sqrt(w.values)

        # 가중 유클리드 = 좌표에 √ω 를 곱한 뒤의 표준 유클리드
        Z = Xe.values * sqrt_w
        z_ideal = (ideal.values * sqrt_w).reshape(1, -1)

        n_neighbors = int(min(max(k, 1), len(Z)))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto").fit(Z)
        knn_dist, knn_idx = nn.kneighbors(z_ideal)
        knn_dist, knn_idx = knn_dist[0], knn_idx[0]

        diff = Xe.values - ideal.values                      # (n, p)
        sq = (diff ** 2) * w.values                          # 가중 제곱 편차
        dist_all = np.sqrt(sq.sum(axis=1))

        span = (Xe.max() - Xe.min()).clip(lower=EPS).values
        d_max = float(np.sqrt(((span ** 2) * w.values).sum())) or 1.0
        score_all = 1 - np.clip(dist_all / d_max, 0, 1)

        scored = candidates.copy()
        scored["distance"] = dist_all.round(5)
        scored["fit_score"] = (score_all * 100).round(1)

        # --- 구성원별 만족도 & 공통 만족도 --- #
        group_cols = {"공통": [c for c in cols if c in COMMON_KEYS]}
        for m in self.active_members:
            group_cols[m.key] = [c for c in cols if c.startswith(f"{m.key}__")]

        sat_cols = {}
        for gkey, gcols in group_cols.items():
            if not gcols:
                continue
            sub_w = w[gcols]
            sub_w = sub_w / sub_w.sum() if sub_w.sum() > EPS else pd.Series(1 / len(gcols), index=gcols)
            gdiff = Xe[gcols].values - ideal[gcols].values
            gdist = np.sqrt(((gdiff ** 2) * sub_w.values).sum(axis=1))
            gspan = (Xe[gcols].max() - Xe[gcols].min()).clip(lower=EPS).values
            gmax = float(np.sqrt(((gspan ** 2) * sub_w.values).sum())) or 1.0
            name = f"sat__{gkey}"
            scored[name] = ((1 - np.clip(gdist / gmax, 0, 1)) * 100).round(1)
            sat_cols[gkey] = name

        member_sat_cols = [sat_cols[m.key] for m in self.active_members if m.key in sat_cols]
        if member_sat_cols:
            S = scored[member_sat_cols]
            scored["sat_min"] = S.min(axis=1).round(1)             # 최약자 만족도(maximin)
            scored["sat_gap"] = (S.max(axis=1) - S.min(axis=1)).round(1)   # 이해충돌 지수
            obj = S.values.astype(float)
            if "sat__공통" in scored.columns:
                obj = np.column_stack([obj, scored["sat__공통"].values])
            scored["pareto"] = pareto_mask(obj)
        else:
            scored["sat_min"] = scored["fit_score"]
            scored["sat_gap"] = 0.0
            scored["pareto"] = True

        scored = add_derived_metrics(scored, self.active_members)
        scored = scored.sort_values("distance")

        # --- Top-K (sklearn K-NN 결과) --- #
        top_ids = Xe.index[knn_idx]
        top = scored.loc[top_ids].copy()
        top.insert(0, "rank", range(1, len(top) + 1))
        top["knn_distance"] = knn_dist.round(5)

        # --- 기여도 분해: 왜 이 매물인가 / 무엇이 발목을 잡는가 --- #
        # (정렬 기준을 바꿔도 조회 가능하도록 전체 후보에 대해 계산한다)
        contrib = pd.DataFrame(sq, index=Xe.index, columns=cols)
        contrib = contrib.div(contrib.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(4)
        contrib = contrib.reindex(scored.index)

        clusters, cluster_profile = self._cluster(Z, Xe, top_ids, scored, n_clusters)
        top = top.join(clusters["cluster"], how="left")

        diagnostics = {
            "total_listings": len(self.data),
            "candidates": len(candidates),
            "filter_report": filter_report,
            "feature_count": len(cols),
            "common_features": len(group_cols["공통"]),
            "individual_features": len(cols) - len(group_cols["공통"]),
            "d_max": d_max,
            **prep_info,
        }
        ideal_raw = pd.Series(scaler.inverse_transform(ideal.values.reshape(1, -1))[0], index=cols)

        return Recommendation(
            top=top, scored=scored, weights=w, ideal_raw=ideal_raw, ideal_scaled=ideal,
            scaled=Xe, contributions=contrib, clusters=clusters,
            cluster_profile=cluster_profile, diagnostics=diagnostics,
        )

    # ---------------- 군집 분석 ---------------- #
    def _cluster(self, Z: np.ndarray, X: pd.DataFrame, top_ids, scored: pd.DataFrame,
                 n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        idx_pos = [X.index.get_loc(i) for i in top_ids]
        Zt = Z[idx_pos]
        n_clusters = int(min(max(n_clusters, 1), len(Zt)))
        if n_clusters < 2:
            lab = pd.DataFrame({"cluster": np.zeros(len(Zt), dtype=int)}, index=top_ids)
            return lab, pd.DataFrame()

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Zt)
        lab = pd.DataFrame({"cluster": km.labels_}, index=top_ids)

        profile_cols = [c for c in ["area_m2", "rooms", "building_age", "green_ratio",
                                    "parking_per_unit", "travel_mean_min", "travel_std_min",
                                    "stability_index", "fit_score"] if c in scored.columns]
        prof = scored.loc[top_ids, profile_cols].join(lab).groupby("cluster").mean().round(2)
        prof["매물수"] = lab.groupby("cluster").size()
        prof["대표지역"] = scored.loc[top_ids].join(lab).groupby("cluster")["gu"] \
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "-")
        return lab, prof


def label_of(col: str) -> str:
    """컬럼명 -> 사람이 읽는 라벨."""
    from .schema import DERIVED_LABELS, FEATURE_BY_KEY
    if "__" in col:
        owner, key = col.split("__", 1)
        f = FEATURE_BY_KEY.get(key)
        return f"{MEMBER_NAMES.get(owner, owner)}·{f.axis if f else key}"
    if col in FEATURE_BY_KEY:
        return FEATURE_BY_KEY[col].label
    return DERIVED_LABELS.get(col, col)


__all__ = [
    "FamilyHousingRecommender", "WeightConfig", "HardConstraints",
    "Recommendation", "label_of", "COMMON_FEATURES",
]
