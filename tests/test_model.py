"""엔진 스모크 테스트: python tests/test_model.py 또는 pytest tests/"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import add_derived_metrics, derive_common_features, pareto_mask
from src.model import FamilyHousingRecommender, HardConstraints, WeightConfig
from src.schema import DEFAULT_MEMBERS, Member, assert_no_price_columns

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load():
    listings = derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))
    individual = pd.read_csv(os.path.join(DATA, "individual_features.csv"))
    members = [Member.from_dict(d) for d in json.load(open(os.path.join(DATA, "members.json"), encoding="utf-8"))]
    return listings, individual, members


def test_price_guard():
    try:
        assert_no_price_columns(["area_m2", "매매가"])
    except ValueError:
        return True
    raise AssertionError("가격 컬럼 가드가 동작하지 않음")


def test_pipeline():
    listings, individual, members = load()
    eng = FamilyHousingRecommender(listings, individual, members)

    hc = HardConstraints(common_min={"rooms": 3, "area_m2": 60}, common_max={"building_age": 30},
                         member_max={"dad__commute_min": 60})
    cfg = WeightConfig(
        alpha=0.45,
        common_weights={"area_m2": 1.5, "green_ratio": 1.0, "building_age": 0.8},
        member_priority={"dad": 1.0, "mom": 1.2, "child": 1.4},
        satisficing={"dad__commute_min": 30.0},
    )
    rec = eng.recommend(cfg, hc, k=10, n_clusters=3)

    assert abs(rec.weights.sum() - 1.0) < 1e-9, "가중치 합이 1이 아님"
    assert rec.scaled.isna().sum().sum() == 0, "KNNImputer 이후에도 결측 존재"
    assert len(rec.top) == 10
    assert rec.top["distance"].is_monotonic_increasing, "Top-K 정렬 오류"
    assert rec.top["fit_score"].iloc[0] >= rec.top["fit_score"].iloc[-1]
    assert abs(rec.contributions.sum(axis=1) - 1).max() < 1e-3, "기여도 합 != 1"
    assert rec.scored["pareto"].sum() >= 1
    assert rec.top["travel_mean_min"].notna().all(), "파생 이동지표 결측"

    # 하드 제약 실제 적용 여부
    assert rec.scored["rooms"].min() >= 3 and rec.scored["area_m2"].min() >= 60

    # 가중치 변화 → 추천 결과 변화 (개인화가 실제로 작동하는지)
    cfg_child = WeightConfig(alpha=0.1, member_priority={"dad": 0.1, "mom": 0.1, "child": 5.0})
    rec2 = eng.recommend(cfg_child, hc, k=10)
    assert set(rec.top.index) != set(rec2.top.index), "가중치를 바꿔도 결과가 동일함"
    assert rec2.scored["sat__child"].loc[rec2.top.index[0]] >= rec.scored["sat__child"].loc[rec.top.index[0]] - 1e-6

    # 파레토 정의 검증
    m = pareto_mask(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.0]]))
    assert list(m) == [False, True, True], f"파레토 마스크 오류: {m}"

    print(f"후보 {rec.diagnostics['candidates']}건 / 피처 {rec.diagnostics['feature_count']}개"
          f" (공통 {rec.diagnostics['common_features']} + 개별 {rec.diagnostics['individual_features']})")
    print(f"보정된 결측 셀: {rec.diagnostics['missing_cells']} (k={rec.diagnostics['impute_k']})")
    print(rec.top[["name", "gu", "rank", "fit_score", "sat__dad", "sat__mom", "sat__child",
                   "sat_gap", "travel_mean_min", "equity_index", "cluster"]].head(5).to_string())
    print("\n[군집 프로파일]\n" + rec.cluster_profile.to_string())
    top_contrib = rec.contributions.iloc[0].sort_values(ascending=False).head(4)
    print("\n[1위 매물 거리 기여도 Top4]\n" + top_contrib.to_string())
    return True


if __name__ == "__main__":
    test_price_guard()
    test_pipeline()
    print("\n✅ 모든 테스트 통과")
