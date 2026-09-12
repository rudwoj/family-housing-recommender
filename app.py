"""
가족 맞춤 주거지 추천 (Multi-Person Weighted K-NN)

실행:
    streamlit run app.py

가격 정보를 전혀 사용하지 않고, 주택의 공간·환경 조건(공통 데이터)과
구성원 각자의 생활 동선(개별 데이터)만으로 최적 주거지를 추천한다.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from src import viz
from src.features import derive_common_features
from src.model import FamilyHousingRecommender, HardConstraints, WeightConfig, label_of
from src.schema import COMMON_FEATURES, FEATURE_BY_KEY, Member, ORIENTATION_SCORE

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

st.set_page_config(page_title="가족 맞춤 주거지 추천", page_icon="🏡", layout="wide")

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False


# --------------------------------------------------------------------------- #
# 데이터 로딩
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    listings = derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))
    individual = pd.read_csv(os.path.join(DATA, "individual_features.csv"))
    with open(os.path.join(DATA, "members.json"), encoding="utf-8") as f:
        members = [Member.from_dict(d) for d in json.load(f)]
    return listings, individual, members


if not os.path.exists(os.path.join(DATA, "listings.csv")):
    st.error("데이터가 없습니다. 먼저 실행하세요:  `python data/generate_mock_data.py`")
    st.stop()

listings, individual, members = load_data()

PRESETS = {
    "⚖️ 균형": dict(alpha=0.5, prio={"dad": 1.0, "mom": 1.0, "child": 1.0}),
    "👶 자녀 중심": dict(alpha=0.35, prio={"dad": 0.6, "mom": 0.8, "child": 2.2}),
    "🚇 통근 우선": dict(alpha=0.25, prio={"dad": 1.6, "mom": 1.6, "child": 0.6}),
    "🏠 집 자체 우선": dict(alpha=0.85, prio={"dad": 1.0, "mom": 1.0, "child": 1.0}),
}


def ensure_defaults():
    """
    위젯 기본값 보증.
    st.rerun() 직후에는 아직 렌더링되지 않은 위젯의 상태가 정리될 수 있으므로
    매 실행마다 setdefault 로 복구한다(값이 이미 있으면 사용자 설정을 보존).
    """
    d = {"alpha": 0.5}
    for m in members:
        d[f"active__{m.key}"] = True
        d[f"prio__{m.key}"] = float(m.priority)
        for f in m.features:
            d[f"w__{m.key}__{f.key}"] = f.default_weight
    for f in COMMON_FEATURES:
        d[f"wc__{f.key}"] = f.default_weight
    for k_, v_ in d.items():
        st.session_state.setdefault(k_, v_)


ensure_defaults()

# --------------------------------------------------------------------------- #
# 사이드바
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.title("🏡 가족 주거 조건")
sb.caption("가격은 일절 반영하지 않습니다. 공간·환경·동선만으로 계산합니다.")

sb.markdown("**빠른 프리셋**")
pcols = sb.columns(2)
for i, (pname, pval) in enumerate(PRESETS.items()):
    if pcols[i % 2].button(pname, use_container_width=True):
        st.session_state.alpha = pval["alpha"]
        for k, v in pval["prio"].items():
            if f"prio__{k}" in st.session_state:
                st.session_state[f"prio__{k}"] = v
        st.rerun()

sb.divider()

# --- ① 하드 제약 --- #
sb.subheader("① 하드 제약 (타협 불가)")
hc_min, hc_max = {}, {}
hc_min["rooms"] = sb.slider("최소 방 개수", 1, 5, 3)
hc_min["area_m2"] = sb.slider("최소 전용면적 (m²)", 29, 150, 60, step=1)
hc_max["building_age"] = sb.slider("최대 준공 경과년수 (년)", 0, 46, 30)
hc_min["parking_per_unit"] = sb.slider("최소 세대당 주차대수", 0.0, 2.5, 0.5, step=0.05)
allowed_or = sb.multiselect("허용 향", list(ORIENTATION_SCORE), default=list(ORIENTATION_SCORE))
allowed_gu = sb.multiselect("지역 한정 (미선택 = 전체)", sorted(listings["gu"].unique()), default=[])

# --- ② 공통 vs 개별 균형 --- #
sb.divider()
sb.subheader("② 무엇을 더 중요하게?")
alpha = sb.slider("🏠 집 자체 ←→ 🚶 생활 동선", 0.0, 1.0, key="alpha", step=0.05,
                  help="1.0 에 가까울수록 주택의 물리·환경 조건, 0.0 에 가까울수록 구성원 동선을 우선합니다.")
sb.caption(f"공통 피처 비중 **{alpha:.0%}** · 개별 동선 비중 **{1 - alpha:.0%}**")

# --- ③ 구성원별 설정 --- #
sb.divider()
sb.subheader("③ 구성원별 가중치")
member_priority, member_weights, member_max, satisficing = {}, {}, {}, {}
use_satisficing = sb.toggle("만족 임계값 사용", value=True,
                            help="'이 정도면 충분' 기준을 넘어서는 초과 성능은 점수 차이로 보지 않습니다.")

for m in members:
    with sb.expander(f"{m.name} · {m.anchor_label}", expanded=(m.key == "child")):
        m.active = st.checkbox("추천에 반영", key=f"active__{m.key}")
        member_priority[m.key] = st.slider("가족 내 발언권 $P_i$", 0.0, 3.0,
                                           key=f"prio__{m.key}", step=0.1)
        w = {}
        for f in m.features:
            w[f.key] = st.slider(f"{f.label} ({f.unit})", 0.0, 2.0,
                                 key=f"w__{m.key}__{f.key}", step=0.1)
            if f.hard_filter == "max":
                lo, hi = float(individual[m.column(f.key)].min()), float(individual[m.column(f.key)].max())
                v = st.number_input(f"⛔ {f.label} 상한", value=float(round(hi)), min_value=lo,
                                    max_value=float(round(hi)), step=(1.0 if f.unit == "분" else 50.0),
                                    key=f"hard__{m.key}__{f.key}")
                if v < hi:
                    member_max[m.column(f.key)] = v
            if use_satisficing and f.minute_factor is not None and f.unit == "분":
                s = st.number_input(f"✅ {f.label} 충분 기준", value=25.0, min_value=0.0,
                                    step=5.0, key=f"sat__{m.key}__{f.key}")
                satisficing[m.column(f.key)] = s
        member_weights[m.key] = w

# --- ④ 공통 피처 가중치 --- #
sb.divider()
sb.subheader("④ 공통(주거) 피처 가중치")
common_weights = {}
with sb.expander("공통 피처 상세 조정", expanded=False):
    for f in COMMON_FEATURES:
        common_weights[f.key] = st.slider(f"{f.label} ({f.unit})", 0.0, 2.0,
                                          key=f"wc__{f.key}", step=0.1)

sb.divider()
k = sb.slider("추천 매물 수 (K)", 3, 30, 10)
n_clusters = sb.slider("군집 수", 2, 5, 3)
scaler_kind = sb.radio("스케일링", ["minmax", "standard"], horizontal=True)
sort_mode = sb.radio("정렬 기준", ["종합 적합도", "최약자 우선 (Maximin)", "형평성 우선"], index=0,
                     help="구성원 간 이해충돌을 다루는 세 가지 사회적 선택 규칙입니다.")

# --------------------------------------------------------------------------- #
# 추천 실행
# --------------------------------------------------------------------------- #
engine = FamilyHousingRecommender(listings, individual, members, scaler=scaler_kind)
hc = HardConstraints(
    common_min=hc_min, common_max=hc_max,
    allowed_orientations=allowed_or if len(allowed_or) < len(ORIENTATION_SCORE) else None,
    allowed_gu=allowed_gu or None, member_max=member_max)
cfg = WeightConfig(alpha=alpha, common_weights=common_weights, member_priority=member_priority,
                   member_weights=member_weights, satisficing=satisficing if use_satisficing else {})

active = [m for m in members if m.active]
if not active and alpha < 1.0:
    st.warning("활성 구성원이 없어 공통(주거) 피처만으로 계산합니다.")

try:
    rec = engine.recommend(cfg, hc, k=k, n_clusters=n_clusters)
except ValueError as e:
    st.error(f"⚠️ {e}")
    st.stop()

if sort_mode != "종합 적합도":
    key = "sat_min" if sort_mode.startswith("최약자") else "sat_gap"
    asc = key == "sat_gap"
    order = rec.scored.sort_values(key, ascending=asc).head(k).index
    rec.top = rec.scored.loc[order].copy()
    rec.top.insert(0, "rank", range(1, len(order) + 1))
    rec.top = rec.top.join(rec.clusters["cluster"], how="left")

top = rec.top

# --------------------------------------------------------------------------- #
# 헤더 지표
# --------------------------------------------------------------------------- #
st.title("🏡 가족 맞춤 주거지 추천")
st.caption("Multi-Person Weighted K-NN · 가격 배제 · 공통(주거 물리/환경) + 개별(구성원 동선) 결합")

best = top.iloc[0]
c = st.columns(6)
c[0].metric("후보 매물", f"{rec.diagnostics['candidates']}건",
            f"전체 {rec.diagnostics['total_listings']}건 중")
c[1].metric("1위 종합 적합도", f"{best['fit_score']:.1f}점")
c[2].metric("최약자 만족도", f"{best['sat_min']:.1f}점",
            help="가장 불만족한 구성원의 점수 (Maximin 관점)")
c[3].metric("이해충돌 지수", f"{best['sat_gap']:.1f}p",
            help="구성원 간 만족도 최대-최소 격차. 낮을수록 합의가 쉽습니다.")
c[4].metric("평균 이동부담", f"{best['travel_mean_min']:.0f}분",
            f"편차 ±{best['travel_std_min']:.0f}분", delta_color="off")
c[5].metric("주거 안정성 지수", f"{best['stability_index']:.2f}",
            help="녹지·주차·일조·단지규모·연식·소음 종합 (0~1)")

tabs = st.tabs(["🥇 추천 결과", "⚖️ 이해충돌·파레토", "🗺️ 동선 지도", "🔎 군집 분석", "🧪 모델 진단"])

# --------------------------------------------------------------------------- #
# 탭 1: 추천 결과
# --------------------------------------------------------------------------- #
with tabs[0]:
    show = {"rank": "순위", "name": "매물", "gu": "지역", "fit_score": "종합",
            "sat__공통": "주거", **{f"sat__{m.key}": m.name for m in active},
            "sat_min": "최약자", "sat_gap": "격차", "area_m2": "면적(m²)", "rooms": "방",
            "orientation": "향", "building_age": "연식", "green_ratio": "녹지(%)",
            "travel_mean_min": "평균이동(분)", "travel_std_min": "편차(분)",
            "equity_index": "형평성", "stability_index": "안정성",
            "pareto": "파레토", "cluster": "군집"}
    cols = [c_ for c_ in show if c_ in top.columns]
    table = top[cols].rename(columns=show)

    st.dataframe(
        table, use_container_width=True, height=430,
        column_config={
            "종합": st.column_config.ProgressColumn("종합", min_value=0, max_value=100, format="%.1f"),
            "최약자": st.column_config.ProgressColumn("최약자", min_value=0, max_value=100, format="%.1f"),
            "파레토": st.column_config.CheckboxColumn("파레토"),
        })

    st.divider()
    left, right = st.columns([1, 1])
    labels = {i: f"{int(r['rank'])}위 · {r['name']} ({r['fit_score']:.1f}점)" for i, r in top.iterrows()}
    sel = left.selectbox("매물 선택", list(top.index), format_func=lambda i: labels[i])
    compare = right.multiselect("비교 대상 (최대 3)", [i for i in top.index if i != sel],
                                default=[i for i in top.index if i != sel][:1], max_selections=3,
                                format_func=lambda i: labels[i])

    st.plotly_chart(viz.member_radar(rec, [sel] + compare, active),
                    use_container_width=True, key="radar_members")

    cc = st.columns([1, 1])
    cc[0].plotly_chart(viz.contribution_bar(rec, sel), use_container_width=True, key="contrib")
    common_cols = [f.key for f in COMMON_FEATURES if f.key in rec.scaled.columns]
    cc[1].plotly_chart(viz.feature_radar(rec, [sel] + compare, common_cols,
                                         "공통(주거) 피처 프로파일 — 정규화 0~1"),
                       use_container_width=True, key="radar_common")

    r = top.loc[sel]
    st.info(
        f"**{r['name']}** · {r['gu']} — 전용 {r['area_m2']:.0f}m² / 방 {int(r['rooms'])} / "
        f"{r['orientation']}향 / 준공 {int(r['building_age'])}년차 / 녹지 {r['green_ratio']:.0f}% · "
        + " · ".join(f"{m.name} 이동부담 {r[f'burden_min__{m.key}']:.0f}분" for m in active)
        + f" → 격차 {r['sat_gap']:.1f}p"
        + ("  ⭐ **파레토 최적** (누구도 손해보지 않고 개선할 대안이 없음)" if r["pareto"] else ""))

# --------------------------------------------------------------------------- #
# 탭 2: 이해충돌 / 파레토
# --------------------------------------------------------------------------- #
with tabs[1]:
    if len(active) < 2:
        st.info("구성원을 2명 이상 활성화하면 트레이드오프 분석이 가능합니다.")
    else:
        opts = {m.key: m.name for m in active}
        cA, cB = st.columns(2)
        x_key = cA.selectbox("X축 구성원", list(opts), format_func=lambda k_: opts[k_], index=0)
        y_key = cB.selectbox("Y축 구성원", list(opts), format_func=lambda k_: opts[k_],
                             index=min(1, len(opts) - 1))
        st.plotly_chart(
            viz.tradeoff_scatter(rec, x_key, y_key, opts[x_key], opts[y_key], list(top.index)),
            use_container_width=True, key="tradeoff")

        pareto = rec.scored[rec.scored["pareto"]].sort_values("sat_min", ascending=False)
        st.markdown(f"#### 파레토 최적 대안 {len(pareto)}건")
        st.caption("한 구성원의 만족도를 높이려면 반드시 다른 구성원이 손해를 봐야 하는 '균형점' 집합입니다.")
        pcols = ["name", "gu", "fit_score"] + [f"sat__{m.key}" for m in active] + \
                ["sat_min", "sat_gap", "travel_mean_min", "equity_index"]
        st.dataframe(pareto[[c_ for c_ in pcols if c_ in pareto.columns]].head(20).rename(
            columns={"name": "매물", "gu": "지역", "fit_score": "종합",
                     **{f"sat__{m.key}": m.name for m in active},
                     "sat_min": "최약자", "sat_gap": "격차",
                     "travel_mean_min": "평균이동(분)", "equity_index": "형평성"}),
            use_container_width=True, height=320)

        st.markdown("#### 규칙별 최선의 선택 비교")
        rules = {
            "공리주의 (가중 합 최대)": rec.scored.sort_values("distance").index[0],
            "롤스적 최약자 우선 (Maximin)": rec.scored.sort_values("sat_min", ascending=False).index[0],
            "형평성 우선 (격차 최소)": rec.scored.sort_values("sat_gap").index[0],
        }
        rc = st.columns(len(rules))
        for (rule, lid), col in zip(rules.items(), rc):
            row = rec.scored.loc[lid]
            col.markdown(f"**{rule}**")
            col.write(f"🏠 {row['name']} ({row['gu']})")
            col.write(f"종합 {row['fit_score']:.1f} · 최약자 {row['sat_min']:.1f} · 격차 {row['sat_gap']:.1f}p")
        st.plotly_chart(viz.member_radar(rec, list(dict.fromkeys(rules.values())), active),
                        use_container_width=True, key="radar_rules")

# --------------------------------------------------------------------------- #
# 탭 3: 지도
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.caption("추천 매물 위치와, 선택 매물에서 각 구성원의 주요 목적지(직장/학원가)까지의 동선입니다.")
    map_sel = st.selectbox("동선을 볼 매물", list(top.index),
                           format_func=lambda i: labels[i], key="map_sel")
    kind, obj = viz.build_map(top, active, map_sel)
    if kind == "folium" and HAS_ST_FOLIUM:
        st_folium(obj, height=560, use_container_width=True)
    elif kind == "folium":
        st.components.v1.html(obj._repr_html_(), height=560)
    else:
        st.plotly_chart(obj, use_container_width=True, key="map_plotly")
        st.caption("※ `folium`, `streamlit-folium` 설치 시 folium 지도로 자동 전환됩니다.")

    mrow = top.loc[map_sel]
    mc = st.columns(len(active) or 1)
    for col, m in zip(mc, active):
        col.metric(f"{m.name} 이동부담", f"{mrow[f'burden_min__{m.key}']:.0f}분",
                   f"만족도 {mrow[f'sat__{m.key}']:.1f}점", delta_color="off")

# --------------------------------------------------------------------------- #
# 탭 4: 군집
# --------------------------------------------------------------------------- #
with tabs[3]:
    if rec.cluster_profile.empty:
        st.info("군집을 나눌 만큼 추천 매물이 많지 않습니다. K 를 늘려보세요.")
    else:
        st.plotly_chart(viz.cluster_profile_fig(rec.cluster_profile), use_container_width=True, key="clusters")
        st.markdown("#### 군집별 프로파일 (평균)")
        st.dataframe(rec.cluster_profile.rename(columns={
            "area_m2": "면적(m²)", "rooms": "방", "building_age": "연식", "green_ratio": "녹지(%)",
            "parking_per_unit": "주차", "travel_mean_min": "평균이동(분)",
            "travel_std_min": "이동편차(분)", "stability_index": "안정성", "fit_score": "종합"}),
            use_container_width=True)
        for cid in sorted(rec.clusters["cluster"].unique()):
            ids = rec.clusters.index[rec.clusters["cluster"] == cid]
            with st.expander(f"군집 {cid} — {len(ids)}건"):
                st.dataframe(top.loc[[i for i in ids if i in top.index],
                                     ["name", "gu", "fit_score", "area_m2", "green_ratio",
                                      "travel_mean_min", "sat_gap"]].rename(
                    columns={"name": "매물", "gu": "지역", "fit_score": "종합", "area_m2": "면적(m²)",
                             "green_ratio": "녹지(%)", "travel_mean_min": "평균이동(분)", "sat_gap": "격차"}),
                    use_container_width=True)

# --------------------------------------------------------------------------- #
# 탭 5: 모델 진단
# --------------------------------------------------------------------------- #
with tabs[4]:
    d = rec.diagnostics
    m1 = st.columns(4)
    m1[0].metric("사용 피처 수", d["feature_count"], f"공통 {d['common_features']} / 개별 {d['individual_features']}")
    m1[1].metric("KNNImputer 보정 셀", d["missing_cells"], f"k = {d['impute_k']}")
    m1[2].metric("이론적 최대 거리", f"{d['d_max']:.3f}")
    m1[3].metric("스케일링", scaler_kind)

    if d.get("dropped_features"):
        st.warning("후보 전체가 결측이라 제외된 피처: " +
                   ", ".join(label_of(c) for c in d["dropped_features"]))

    st.plotly_chart(viz.weight_bar(rec), use_container_width=True, key="weights")

    st.markdown("#### 하드 제약 필터링 단계")
    st.dataframe(d["filter_report"], use_container_width=True, hide_index=True)

    st.markdown("#### 가족 이상점 벡터 (Ideal Point)")
    ideal_df = pd.DataFrame({
        "피처": [label_of(c) for c in rec.ideal_raw.index],
        "이상값(원 단위)": rec.ideal_raw.round(1).values,
        "단위": [FEATURE_BY_KEY[c if "__" not in c else c.split("__", 1)[1]].unit
                 for c in rec.ideal_raw.index],
        "방향": [FEATURE_BY_KEY[c if "__" not in c else c.split("__", 1)[1]].direction
                 for c in rec.ideal_raw.index],
        "가중치 ω(%)": (rec.weights.values * 100).round(2),
    })
    st.dataframe(ideal_df, use_container_width=True, hide_index=True, height=380)

    st.markdown("#### 가격 배제 검증")
    st.success("데이터 로딩 시 `assert_no_price_columns()` 통과 — 매매/전세/월세/관리비 컬럼 없음")
    with st.expander("원천 데이터 컬럼 보기"):
        st.write("**공통(주거):**", list(listings.columns))
        st.write("**개별(동선):**", list(individual.columns))
