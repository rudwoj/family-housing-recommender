"""엔진 검증: python tests/test_model.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import derive_common_features, pareto_mask
from src.model import (Constraint, FamilyHousingRecommender, HardConstraints, WeightConfig)
from src.preprocess import UtilityPipeline, utility
from src.schema import (COMMON_FEATURES, FEATURE_BY_KEY, Member,
                        assert_no_price_columns, make_default_members)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load():
    listings = derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))
    travel = pd.read_csv(os.path.join(DATA, "travel_times.csv"))
    members = make_default_members()          # 기본 4명 (이름은 앱에서 사용자가 입력)
    return listings, travel, members


def base_constraints(**kw):
    return HardConstraints(numeric=[
        Constraint("rooms", "min", 3, priority=0, locked=True, label="방 개수"),
        Constraint("area_m2", "min", 60, priority=0, label="전용면적"),
        Constraint("parking_slots", "min", 0.5, priority=2, label="주차 가능 대수"),
        Constraint("building_age", "max", 30, priority=3, label="준공 경과년수"),
    ], **kw)


def test_price_and_distance_guard():
    for bad in (["area_m2", "매매가"], ["rooms", "price_krw"]):
        try:
            assert_no_price_columns(bad)
            raise AssertionError(f"가격 가드 실패: {bad}")
        except ValueError:
            pass
    listings, travel, _ = load()
    dist_cols = [c for c in list(listings.columns) + list(travel.columns) if c.endswith("_m")]
    assert not dist_cols, f"거리(m) 컬럼이 남아있음: {dist_cols}"
    print("✅ 가격/거리 컬럼 없음 — 이동은 전부 분 단위")


def test_outlier_robustness():
    """① 이상치가 있어도 일반 매물의 변별력이 유지되는가"""
    listings, _, _ = load()
    x = listings[["area_m2"]].dropna()
    feats = {"area_m2": FEATURE_BY_KEY["area_m2"]}
    normal = x["area_m2"] < 200

    widths = {}
    for kind, q in [("minmax", 0.0), ("minmax", 0.02), ("robust", 0.01)]:
        p = UtilityPipeline(scaler=kind, clip_q=q)
        p.fit_transform(x, feats)
        pos = p.position["area_m2"][normal.values]
        widths[f"{kind}/clip{q}"] = round(float(pos.quantile(.95) - pos.quantile(.05)), 3)

    assert widths["minmax/clip0.02"] > widths["minmax/clip0.0"] * 2, widths
    assert widths["robust/clip0.01"] > widths["minmax/clip0.0"] * 2, widths
    print(f"✅ 이상치 대응: 일반매물 분포 폭 {widths}")


def test_nonlinear_utility():
    """④ 비선형 효용: 통근 50→60분 손실 > 10→20분 손실"""
    f = FEATURE_BY_KEY["transit_walk_min"]          # lower + convex
    lo = utility(np.array([0.1]), f)[0] - utility(np.array([0.2]), f)[0]
    hi = utility(np.array([0.8]), f)[0] - utility(np.array([0.9]), f)[0]
    assert hi > lo * 2, (lo, hi)

    a = FEATURE_BY_KEY["area_m2"]                   # higher + concave
    gain_low = utility(np.array([0.2]), a)[0] - utility(np.array([0.1]), a)[0]
    gain_high = utility(np.array([0.9]), a)[0] - utility(np.array([0.8]), a)[0]
    assert gain_low > gain_high, (gain_low, gain_high)
    print(f"✅ 비선형 효용: 시간 후반 손실 {hi:.3f} vs 초반 {lo:.3f} / "
          f"면적 체감효용 {gain_low:.3f} → {gain_high:.3f}")


def test_backoff():
    """③ 과도한 제약 → 0건 대신 단계적 완화로 후보 복원"""
    listings, travel, members = load()
    eng = FamilyHousingRecommender(listings, travel, members)

    harsh = HardConstraints(numeric=[
        Constraint("rooms", "min", 6, priority=0, locked=True, label="방 개수"),
        Constraint("area_m2", "min", 160, priority=1, label="전용면적"),
        Constraint("parking_slots", "min", 2.5, priority=2, label="주차"),
        Constraint("building_age", "max", 3, priority=3, label="준공 경과년수"),
        Constraint("transit_walk_min", "max", 3, priority=2, label="역까지 도보"),
    ], min_candidates=5)

    cand, _, relax, log = eng.filter_with_backoff(harsh)
    assert len(cand) >= 5, f"백오프 실패: {len(cand)}건"
    assert len(relax) > 0, "완화 기록이 없음"
    # 잠긴 제약은 절대 완화되지 않아야 한다
    assert cand["rooms"].min() >= 6, "locked 제약이 완화됨"
    print(f"✅ 백오프: 0건 → {len(cand)}건 복원 (완화 {len(relax)}단계, 방 개수 잠금 유지)")
    print(log.to_string(index=False))

    rec = eng.recommend(WeightConfig(), harsh, k=5)
    assert len(rec.top) >= 1
    return rec


def test_conflict_resolution():
    """② 상충 선호 → 하한선 + 파레토 + 개인 Top-N 교집합"""
    listings, travel, members = load()
    for m in members:
        m.min_satisfaction = 55.0
    eng = FamilyHousingRecommender(listings, travel, members)

    cfg = WeightConfig(alpha=0.35,
                       member_priority={m.key: 1.0 for m in members})
    rec = eng.recommend(cfg, base_constraints(), k=10, member_top_n=5)

    for m in eng.active_members:
        col = f"sat__{m.key}"
        assert rec.top[col].min() >= 55.0 - 1e-6, f"{m.name} 하한선 위반"
    assert rec.scored["pareto"].sum() >= 1
    assert set(rec.member_tops) == {m.key for m in eng.active_members}
    print(f"✅ 하한선 55점 적용: 전체 {len(rec.scored)}건 → 통과 {rec.diagnostics['eligible']}건")
    print(f"   구성원별 Top5 교집합: {rec.intersection or '없음(트레이드오프 존재)'}")
    return rec


def test_pipeline():
    listings, travel, members = load()
    eng = FamilyHousingRecommender(listings, travel, members)
    cfg = WeightConfig(alpha=0.45, satisficing={"m1__dest1__transit_min": 30.0})
    rec = eng.recommend(cfg, base_constraints(), k=10, n_clusters=3)

    assert abs(rec.weights.sum() - 1.0) < 1e-9
    assert rec.utility.isna().sum().sum() == 0
    assert (rec.utility.values >= -1e-9).all() and (rec.utility.values <= 1 + 1e-9).all()
    assert rec.top["distance"].is_monotonic_increasing
    assert abs(rec.contributions.sum(axis=1) - 1).max() < 1e-3
    assert rec.top["travel_mean_min"].notna().all()
    assert list(pareto_mask(np.array([[1., 1.], [2., 2.], [3., 0.]]))) == [False, True, True]

    d = rec.diagnostics
    print(f"✅ 파이프라인: 후보 {d['candidates']}건 / 피처 {d['feature_count']}개"
          f" (공통 {d['common_features']} + 개별 {d['individual_features']})")
    print(f"   클리핑 {d['clip_q']:.0%}: {len(d['clip_report'])}개 피처에서 이상치 절단, "
          f"KNNImputer 보정 {d['missing_cells']}셀")
    cols = (["name", "gu", "rank", "fit_score"]
            + [f"sat__{m.key}" for m in eng.active_members]
            + ["sat_gap", "travel_mean_min", "cluster"])
    print(rec.top[cols].head(5).to_string())
    return rec


if __name__ == "__main__":
    test_price_and_distance_guard()
    test_outlier_robustness()
    test_nonlinear_utility()
    test_backoff()
    test_conflict_resolution()
    test_pipeline()
    print("\n✅ 모든 테스트 통과")
