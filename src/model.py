"""
다인(多人) 가중 K-NN 주거지 추천 엔진.

파이프라인
  ① 하드 제약 + 백오프  : 후보가 min_candidates 미만이면 덜 중요한 제약부터
                          단계적으로 완화해 후보군을 되살린다 (0건 조회 방지)
  ② 전처리              : 클리핑 → 스케일링(minmax/robust) → KNNImputer
  ③ 비선형 효용 변환    : 체감 효용 감소(concave) / 불만 가속(convex) 반영
                          → 모든 피처가 '만족도 0~1' 이 되어 이상점은 1 벡터
  ④ 가중 유클리드 K-NN  : Z = U·√ω 변환 후 sklearn NearestNeighbors 로 Top-K
  ⑤ 이해충돌 해소       : 구성원별 만족도 하한선, 파레토 최적, 개인 Top-N 교집합

가격 데이터는 파이프라인 어디에도 존재하지 않는다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .features import add_derived_metrics, pareto_mask
from .preprocess import UtilityPipeline
from .schema import (
    AUX_BY_KEY, COMMON_FEATURES, COMMON_KEYS, DERIVED_LABELS, FEATURE_BY_KEY,
    Feature, Member, assert_no_price_columns, travel_feature,
)

EPS = 1e-12
MEMBER_NAMES: dict[str, str] = {}
FEATURE_REGISTRY: dict[str, Feature] = {}     # 동적 이동시간 피처 라벨링용


# --------------------------------------------------------------------------- #
# 제약 조건
# --------------------------------------------------------------------------- #
@dataclass
class Constraint:
    """하드 제약 1건. priority 가 클수록 백오프에서 먼저 완화된다."""
    column: str
    op: Literal["min", "max"]
    value: float
    priority: int = 1
    locked: bool = False          # True 면 절대 완화하지 않는다
    label: str = ""

    def text(self) -> str:
        sign = "≥" if self.op == "min" else "≤"
        return f"{self.label or self.column} {sign} {self.value:g}"

    def relaxed(self, step: float) -> "Constraint":
        c = deepcopy(self)
        c.value = self.value - step if self.op == "min" else self.value + step
        return c


@dataclass
class CategoryConstraint:
    column: str
    allowed: list[str]
    priority: int = 1
    locked: bool = False
    label: str = ""


@dataclass
class HardConstraints:
    numeric: list[Constraint] = field(default_factory=list)
    category: list[CategoryConstraint] = field(default_factory=list)
    min_candidates: int = 5       # 이보다 적으면 백오프 작동
    max_relax_steps: int = 6      # 제약 1건당 완화 시도 횟수


# --------------------------------------------------------------------------- #
# 가중치 / 결과
# --------------------------------------------------------------------------- #
@dataclass
class WeightConfig:
    alpha: float = 0.5                                   # 공통 vs 개별 배분
    common_weights: dict[str, float] = field(default_factory=dict)
    member_priority: dict[str, float] = field(default_factory=dict)
    dest_weights: dict[str, float] = field(default_factory=dict)   # {컬럼: 가중치}
    satisficing: dict[str, float] = field(default_factory=dict)    # {컬럼: 충분 기준(분)}


@dataclass
class Recommendation:
    top: pd.DataFrame
    scored: pd.DataFrame
    weights: pd.Series
    utility: pd.DataFrame          # 만족도 공간 (0~1, 1이 최선)
    ideal_raw: pd.Series           # 이상점의 원 단위 값
    contributions: pd.DataFrame
    clusters: pd.DataFrame
    cluster_profile: pd.DataFrame
    member_tops: dict[str, list[str]]      # 구성원별 독립 Top-N
    intersection: list[str]                # 교집합
    diagnostics: dict


# --------------------------------------------------------------------------- #
# 엔진
# --------------------------------------------------------------------------- #
class FamilyHousingRecommender:
    def __init__(self, listings: pd.DataFrame, travel: pd.DataFrame, members: list[Member],
                 scaler: str = "robust", clip_q: float = 0.01, n_impute_neighbors: int = 5):
        assert_no_price_columns(listings.columns)
        assert_no_price_columns(travel.columns)

        self.members = members
        MEMBER_NAMES.update({m.key: m.name for m in members})
        self.scaler_kind = scaler
        self.clip_q = clip_q
        self.n_impute_neighbors = n_impute_neighbors

        self.data = listings.merge(travel, on="listing_id", how="inner").set_index("listing_id")
        self._register_features()

    def _register_features(self) -> None:
        # 인스턴스별로 들고 있는다. FEATURE_REGISTRY 는 label_of() 용 전역이라
        # 한 프로세스의 여러 Streamlit 세션이 함께 쓰는데, 예전처럼 clear() 하면
        # 다른 세션이 계산 중인 컬럼이 사라져 KeyError 로 터진다.
        features: dict[str, Feature] = {f.key: f for f in COMMON_FEATURES}
        for m in self.members:
            for d in m.destinations:
                for mode in ("drive", "transit", "walk"):
                    col = d.column(m.key, mode)
                    if col in self.data.columns:
                        features[col] = travel_feature(m, d, mode)
        self.features = features
        FEATURE_REGISTRY.update(features)

    # ---------------- 피처 구성 ---------------- #
    @property
    def active_members(self) -> list[Member]:
        return [m for m in self.members if m.active]

    def common_columns(self) -> list[str]:
        return [k for k in COMMON_KEYS if k in self.data.columns]

    def member_columns(self, member: Member) -> list[str]:
        return [c for c in member.columns() if c in self.data.columns]

    def individual_columns(self) -> list[str]:
        return [c for m in self.active_members for c in self.member_columns(m)]

    def feature_columns(self) -> list[str]:
        return self.common_columns() + self.individual_columns()

    # ---------------- ① 하드 제약 + 백오프 ---------------- #
    def _apply(self, df: pd.DataFrame, hc: HardConstraints,
               numeric: list[Constraint]) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for c in numeric:
            if c.column not in df.columns:
                continue
            col = df[c.column]
            ok = col >= c.value if c.op == "min" else col <= c.value
            mask &= ok | col.isna()          # 결측은 통과시키고 뒤에서 보정
        for cc in hc.category:
            if cc.column in df.columns and cc.allowed:
                mask &= df[cc.column].isin(cc.allowed)
        return mask

    def filter_with_backoff(self, hc: HardConstraints):
        """
        제약을 적용하고, 후보가 min_candidates 미만이면 단계적으로 완화한다.

        규칙
          - locked 제약은 절대 건드리지 않는다
          - priority 가 높은(=덜 중요한) 제약부터 푼다
          - 단, 실제로 후보를 막고 있는(binding) 제약만 완화한다.
            걸림돌이 아닌 제약을 푸는 헛수고를 피한다
          - 완화값은 데이터의 실제 범위를 넘지 않는다. 한계에 닿으면 즉시 완전 해제
        """
        df = self.data
        numeric = deepcopy(hc.numeric)
        relaxations: list[dict] = []

        mask = self._apply(df, hc, numeric)
        report = [{"단계": "원래 조건", "내용": f"{len(numeric)}개 제약",
                   "잔여 후보": int(mask.sum())}]
        steps: dict[str, int] = {}

        def bound(c: Constraint) -> float:
            col = df[c.column]
            return float(col.min() if c.op == "min" else col.max())

        def is_binding(c: Constraint) -> bool:
            """이 제약을 없애면 후보가 늘어나는가."""
            others = [x for x in numeric if x is not c]
            return int(self._apply(df, hc, others).sum()) > int(mask.sum())

        while mask.sum() < hc.min_candidates:
            relaxable = sorted([c for c in numeric if not c.locked],
                               key=lambda c: (-c.priority, c.column))
            target = next((c for c in relaxable if is_binding(c)), None)
            if target is None:
                break                                  # 더 풀 수 있는 걸림돌이 없다

            span = float(df[target.column].max() - df[target.column].min())
            step = max(span * 0.10, EPS)
            n = steps.get(target.column, 0)
            before = target.text()
            relaxed = target.relaxed(step)

            # 데이터 범위를 넘어서거나 시도 횟수를 소진하면 완전 해제
            limit = bound(target)
            exceeds = relaxed.value < limit if target.op == "min" else relaxed.value > limit
            if exceeds or n >= hc.max_relax_steps:
                numeric = [c for c in numeric if c is not target]
                relaxations.append({"제약": before, "조치": "완전 해제"})
            else:
                numeric[numeric.index(target)] = relaxed
                steps[target.column] = n + 1
                relaxations.append({"제약": before, "조치": f"→ {relaxed.text()}"})

            mask = self._apply(df, hc, numeric)
            report.append({"단계": f"완화 {len(relaxations)}",
                           "내용": f"{relaxations[-1]['제약']} {relaxations[-1]['조치']}",
                           "잔여 후보": int(mask.sum())})

        detail = [{"조건": c.text(), "통과": int(self._apply(df, hc, [c]).sum()),
                   "탈락": int(len(df) - self._apply(df, hc, [c]).sum())} for c in numeric]
        return df[mask], pd.DataFrame(detail), pd.DataFrame(relaxations), pd.DataFrame(report)

    # ---------------- ④ 가중치 ---------------- #
    def build_weights(self, cfg: WeightConfig, cols: list[str]) -> pd.Series:
        w = pd.Series(0.0, index=cols, dtype=float)
        common_cols = [c for c in cols if c in COMMON_KEYS]
        ind_cols = [c for c in cols if c not in COMMON_KEYS]

        alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
        if not ind_cols:
            alpha = 1.0
        if not common_cols:
            alpha = 0.0

        if common_cols:
            wc = pd.Series({c: max(cfg.common_weights.get(c, 1.0), 0.0) for c in common_cols})
            if wc.sum() > EPS:
                w.loc[common_cols] = alpha * wc / wc.sum()

        if ind_cols:
            prio = {m.key: max(cfg.member_priority.get(m.key, m.priority), 0.0)
                    for m in self.active_members}
            total = sum(prio.values())
            if total > EPS:
                for m in self.active_members:
                    mcols = [c for c in ind_cols if c.startswith(f"{m.key}__")]
                    if not mcols:
                        continue
                    vec = pd.Series({c: max(cfg.dest_weights.get(c, 1.0), 0.0) for c in mcols})
                    if vec.sum() <= EPS:
                        continue
                    w.loc[mcols] = (1 - alpha) * (prio[m.key] / total) * (vec / vec.sum())

        s = w.sum()
        return w / s if s > EPS else pd.Series(1.0 / len(cols), index=cols)

    # ---------------- 만족 임계값 ---------------- #
    @staticmethod
    def _apply_satisficing(util: pd.DataFrame, raw: pd.DataFrame,
                           cfg: WeightConfig) -> pd.DataFrame:
        """'이 정도면 충분' 기준(분)을 넘어서는 초과 성능은 만점 처리한다."""
        if not cfg.satisficing:
            return util
        util = util.copy()
        for col, threshold in cfg.satisficing.items():
            if col not in util.columns or threshold is None:
                continue
            good = raw[col] <= float(threshold)
            util.loc[good, col] = 1.0
        return util

    # ---------------- 추천 ---------------- #
    def recommend(self, cfg: WeightConfig, hc: HardConstraints | None = None,
                  k: int = 10, n_clusters: int = 3, member_top_n: int = 3) -> Recommendation:
        hc = hc or HardConstraints()
        cols = self.feature_columns()
        if not cols:
            raise ValueError("활성화된 피처가 없습니다. 구성원 또는 목적지를 선택하세요.")

        candidates, filter_detail, relaxations, backoff_log = self.filter_with_backoff(hc)
        if candidates.empty:
            raise ValueError("모든 제약을 완화해도 후보가 없습니다. 조건을 근본적으로 재검토하세요.")

        raw = candidates[cols].astype(float)
        dropped = [c for c in cols if raw[c].isna().all()]
        cols = [c for c in cols if c not in dropped]
        if not cols:
            raise ValueError("후보 매물에 유효한 피처 값이 없습니다.")
        raw = raw[cols]

        pipe = UtilityPipeline(self.scaler_kind, self.clip_q, self.n_impute_neighbors)
        util = pipe.fit_transform(raw, {c: self.features[c] for c in cols})

        raw_filled = pipe.inverse_raw(pipe.position)
        candidates = candidates.copy()
        for c in cols:
            candidates[c] = candidates[c].astype(float).fillna(raw_filled[c].round(1))

        util = self._apply_satisficing(util, candidates[cols], cfg)

        w = self.build_weights(cfg, cols)
        sqrt_w = np.sqrt(w.values)

        # 이상점 = 모든 차원에서 만족도 1
        gap = 1.0 - util.values
        sq = (gap ** 2) * w.values
        dist_all = np.sqrt(sq.sum(axis=1))          # Σω=1, 편차∈[0,1] → d∈[0,1]

        n_neighbors = int(min(max(k, 1), len(util)))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(util.values * sqrt_w)
        knn_dist, knn_idx = nn.kneighbors(np.ones((1, len(cols))) * sqrt_w)

        scored = candidates.copy()
        scored["distance"] = dist_all.round(5)
        scored["fit_score"] = ((1 - dist_all) * 100).round(1)

        # --- 구성원별 / 공통 만족도 --- #
        groups = {"공통": [c for c in cols if c in COMMON_KEYS]}
        for m in self.active_members:
            groups[m.key] = [c for c in cols if c.startswith(f"{m.key}__")]

        for gkey, gcols in groups.items():
            if not gcols:
                continue
            sub = w[gcols]
            sub = sub / sub.sum() if sub.sum() > EPS else pd.Series(1 / len(gcols), index=gcols)
            gd = np.sqrt((((1 - util[gcols].values) ** 2) * sub.values).sum(axis=1))
            scored[f"sat__{gkey}"] = ((1 - gd) * 100).round(1)

        member_cols = [f"sat__{m.key}" for m in self.active_members if f"sat__{m.key}" in scored]
        if member_cols:
            S = scored[member_cols]
            scored["sat_min"] = S.min(axis=1).round(1)
            scored["sat_gap"] = (S.max(axis=1) - S.min(axis=1)).round(1)
            obj = S.values.astype(float)
            if "sat__공통" in scored.columns:
                obj = np.column_stack([obj, scored["sat__공통"].values])
            scored["pareto"] = pareto_mask(obj)
        else:
            scored["sat_min"] = scored["fit_score"]
            scored["sat_gap"] = 0.0
            scored["pareto"] = True

        scored = add_derived_metrics(scored, self.active_members, self)

        # --- ⑤ 구성원별 하한선 (Averaging Effect 방지) --- #
        floor_mask = pd.Series(True, index=scored.index)
        floors = []
        for m in self.active_members:
            col = f"sat__{m.key}"
            if col in scored.columns and m.min_satisfaction > 0:
                ok = scored[col] >= m.min_satisfaction
                floors.append({"구성원": m.name, "하한선": m.min_satisfaction,
                               "통과": int(ok.sum())})
                floor_mask &= ok
        scored["passes_floor"] = floor_mask

        eligible = scored[floor_mask]
        floor_relaxed = False
        if len(eligible) < 1:
            eligible = scored                       # 하한선이 전부 배제하면 경고 후 무시
            floor_relaxed = True

        eligible = eligible.sort_values("distance")
        top = eligible.head(k).copy()
        top.insert(0, "rank", range(1, len(top) + 1))

        # --- 구성원별 독립 Top-N 과 교집합 --- #
        member_tops: dict[str, list[str]] = {}
        for m in self.active_members:
            col = f"sat__{m.key}"
            if col in scored.columns:
                member_tops[m.key] = scored.sort_values(col, ascending=False) \
                                           .head(member_top_n).index.tolist()
        sets = [set(v) for v in member_tops.values()]
        intersection = sorted(set.intersection(*sets)) if sets else []

        contrib = pd.DataFrame(sq, index=util.index, columns=cols)
        contrib = contrib.div(contrib.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(4)

        clusters, profile = self._cluster(util.values * sqrt_w, util, top.index, scored, n_clusters)
        top = top.join(clusters["cluster"], how="left")

        ideal_raw = self._ideal_raw(pipe, cols)
        diagnostics = {
            "total_listings": len(self.data), "candidates": len(candidates),
            "feature_count": len(cols), "common_features": len(groups["공통"]),
            "individual_features": len(cols) - len(groups["공통"]),
            "dropped_features": dropped, "filter_detail": filter_detail,
            "relaxations": relaxations, "backoff_log": backoff_log,
            "floors": pd.DataFrame(floors), "floor_relaxed": floor_relaxed,
            "eligible": int(floor_mask.sum()), "knn_distance": float(knn_dist[0][0]),
            **pipe.report,
        }
        return Recommendation(top=top, scored=scored, weights=w, utility=util,
                              ideal_raw=ideal_raw, contributions=contrib, clusters=clusters,
                              cluster_profile=profile, member_tops=member_tops,
                              intersection=intersection, diagnostics=diagnostics)

    def _ideal_raw(self, pipe: UtilityPipeline, cols: list[str]) -> pd.Series:
        """만족도 1 에 해당하는 원 단위 값 (방향성에 따라 관측 최댓값/최솟값)."""
        ideal_pos = pd.DataFrame([[1.0 if self.features[c].direction == "higher" else 0.0
                                   for c in cols]], columns=cols)
        return pipe.inverse_raw(ideal_pos).iloc[0].round(1)

    def _cluster(self, Z, util, top_ids, scored, n_clusters):
        pos = [util.index.get_loc(i) for i in top_ids]
        Zt = Z[pos]
        n_clusters = int(min(max(n_clusters, 1), len(Zt)))
        if n_clusters < 2:
            return pd.DataFrame({"cluster": np.zeros(len(Zt), dtype=int)}, index=top_ids), pd.DataFrame()

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(Zt)
        lab = pd.DataFrame({"cluster": km.labels_}, index=top_ids)
        pcols = [c for c in ["area_m2", "rooms", "parking_slots", "transit_walk_min",
                             "travel_mean_min", "travel_std_min", "stability_index", "fit_score"]
                 if c in scored.columns]
        prof = scored.loc[top_ids, pcols].join(lab).groupby("cluster").mean().round(2)
        prof["매물수"] = lab.groupby("cluster").size()
        prof["대표지역"] = scored.loc[top_ids].join(lab).groupby("cluster")["gu"] \
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "-")
        return lab, prof


def label_of(col: str) -> str:
    if col in FEATURE_REGISTRY:
        f = FEATURE_REGISTRY[col]
        if "__" in col:
            return f"{MEMBER_NAMES.get(col.split('__', 1)[0], '')}·{f.axis}"
        return f.label
    if col in FEATURE_BY_KEY:
        return FEATURE_BY_KEY[col].label
    if col in AUX_BY_KEY:
        return AUX_BY_KEY[col].label
    return DERIVED_LABELS.get(col, col)


__all__ = ["FamilyHousingRecommender", "WeightConfig", "HardConstraints", "Constraint",
           "CategoryConstraint", "Recommendation", "label_of", "FEATURE_REGISTRY"]
