"""
가족 맞춤 주거지 추천 (Multi-Person Weighted K-NN)

실행: streamlit run app.py

가격을 배제하고, 주택의 공간 조건(공통)과 구성원이 입력한 목적지까지의
이동 시간(개별)만으로 추천한다. 모든 이동 지표는 분 단위 시간이다.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src import viz
from src.features import derive_common_features
from src.model import (CategoryConstraint, Constraint, FamilyHousingRecommender,
                       HardConstraints, WeightConfig, label_of)
from src.schema import (AUX_COLUMNS, COMMON_FEATURES, DEFAULT_MEMBER_COUNT, MAX_MEMBERS,
                        ORIENTATION_SCORE, TRAVEL_MODES, Destination, Member,
                        make_default_members)
from src.travel import ApiTravelProvider, EstimatedTravelProvider, travel_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

st.set_page_config(page_title="가족 맞춤 주거지 추천", page_icon="🏡", layout="wide")

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False

# 목적지 좌표 입력을 돕는 프리셋 (직접 입력도 가능)
PLACE_PRESETS = {
    "여의도": (37.5216, 126.9243), "강남역": (37.4979, 127.0276),
    "광화문": (37.5720, 126.9769), "판교": (37.3947, 127.1112),
    "구로디지털단지": (37.4853, 126.9015), "성수": (37.5446, 127.0559),
    "대치 학원가": (37.4995, 127.0630), "잠실": (37.5133, 127.1000),
    "상암DMC": (37.5796, 126.8896), "용산": (37.5298, 126.9648),
}


@st.cache_data(show_spinner=False)
def load_listings() -> pd.DataFrame:
    return derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))


@st.cache_data(show_spinner="이동 시간 계산 중...")
def compute_travel(signature: tuple, use_api: bool) -> pd.DataFrame:
    """목적지 구성이 바뀔 때만 이동 시간을 다시 계산한다."""
    listings = load_listings()
    members = [Member(key=mk, name=mk, color="#000",
                      destinations=[Destination(key=dk, label=dk, lat=la, lon=lo, mode=mo)])
               for mk, dk, la, lo, mo in signature]
    provider = ApiTravelProvider() if use_api else EstimatedTravelProvider()
    frames = [travel_matrix(listings, [m], provider) for m in members]
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="listing_id")
    return out


if not os.path.exists(os.path.join(DATA, "listings.csv")):
    st.error("데이터가 없습니다. 먼저 실행하세요:  `python data/generate_mock_data.py`")
    st.stop()

listings = load_listings()

# --------------------------------------------------------------------------- #
# 사이드바 ① 구성원 · 목적지
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.title("🏡 가족 주거 조건")
sb.caption("가격은 반영하지 않습니다. 공간 조건과 이동 시간(분)만으로 계산합니다.")

use_api = sb.toggle("경로 API 사용", value=False,
                    help="KAKAO_REST_API_KEY / ODSAY_API_KEY 환경변수가 있을 때만 동작하며, "
                         "실패 시 거리 기반 추정으로 자동 폴백합니다.")
if use_api and not ApiTravelProvider().available:
    sb.warning("API 키가 없어 거리 기반 추정으로 동작합니다.")

sb.subheader("① 구성원과 목적지")
n_members = sb.number_input("구성원 수", min_value=1, max_value=MAX_MEMBERS,
                            value=DEFAULT_MEMBER_COUNT, step=1, key="n_members",
                            help="구성원 이름은 각 항목에서 직접 입력할 수 있습니다.")
defaults = make_default_members(int(n_members))

members: list[Member] = []
for base in defaults:
    mk = base.key
    display = st.session_state.get(f"mname__{mk}", base.name) or base.name
    with sb.expander(display, expanded=(mk == "m1")):
        name = st.text_input("이름", value=base.name, key=f"mname__{mk}",
                             placeholder="예: 아빠, 엄마, 첫째")
        name = name.strip() or base.name
        active = st.checkbox("추천에 반영", value=True, key=f"active__{mk}")
        priority = st.slider("가족 내 발언권 $P_i$", 0.0, 3.0, value=1.0, step=0.1,
                             key=f"prio__{mk}")
        floor = st.slider("만족도 하한선 (이 밑이면 후보에서 제외)", 0, 90, value=0, step=5,
                          key=f"floor__{mk}",
                          help="상충하는 선호가 '어중간한 타협안'으로 수렴하는 것을 막습니다.")

        dests: list[Destination] = []
        for base_d in base.destinations:
            dk = base_d.key
            st.markdown("---")
            enabled = st.checkbox(f"목적지 사용 ({base_d.label})", value=base_d.enabled,
                                  key=f"den__{mk}__{dk}")
            label = st.text_input("목적지 이름", value=base_d.label, key=f"dlabel__{mk}__{dk}")
            label = label.strip() or base_d.label
            c1, c2 = st.columns([1, 1])
            mode = c1.selectbox("이동수단", list(TRAVEL_MODES),
                                index=list(TRAVEL_MODES).index(base_d.mode),
                                format_func=lambda k: TRAVEL_MODES[k],
                                key=f"dmode__{mk}__{dk}")
            weight = c2.slider("중요도", 0.0, 2.0, value=float(base_d.weight), step=0.1,
                               key=f"dw__{mk}__{dk}")
            preset = st.selectbox("위치", ["(좌표 직접 입력)"] + list(PLACE_PRESETS), index=0,
                                  key=f"dplace__{mk}__{dk}")
            if preset == "(좌표 직접 입력)":
                c3, c4 = st.columns(2)
                lat = c3.number_input("위도", value=float(base_d.lat), format="%.4f",
                                      key=f"dlat__{mk}__{dk}")
                lon = c4.number_input("경도", value=float(base_d.lon), format="%.4f",
                                      key=f"dlon__{mk}__{dk}")
            else:
                lat, lon = PLACE_PRESETS[preset]
                st.caption(f"{preset} ({lat:.4f}, {lon:.4f})")
            cap = st.number_input("⛔ 이동시간 상한 (분, 0=제한 없음)", min_value=0, value=0, step=5,
                                  key=f"dcap__{mk}__{dk}")
            st.session_state[f"cap__{mk}__{dk}"] = cap
            dests.append(Destination(dk, label, lat, lon, mode, weight, enabled))

        m = Member(mk, name, base.color, dests, priority, float(floor))
        m.active = active
        members.append(m)

signature = tuple(sorted((m.key, d.key, round(d.lat, 5), round(d.lon, 5), d.mode)
                         for m in members for d in m.enabled_destinations))
if not signature:
    st.error("활성화된 목적지가 없습니다. 구성원의 목적지를 하나 이상 켜주세요.")
    st.stop()
travel = compute_travel(signature, use_api)

# --------------------------------------------------------------------------- #
# 사이드바 ② 하드 제약 (백오프)
# --------------------------------------------------------------------------- #
sb.divider()
sb.subheader("② 하드 제약")
min_candidates = sb.slider("최소 확보 후보 수", 1, 30, 5, key="min_candidates",
                           help="이보다 적으면 덜 중요한 제약부터 자동 완화(백오프)합니다.")

constraints: list[Constraint] = []
c_rooms = sb.slider("최소 방 개수", 1, 7, 3, key="c_rooms")
lock_rooms = sb.checkbox("🔒 방 개수 절대 사수", value=True)
constraints.append(Constraint("rooms", "min", c_rooms, priority=0,
                              locked=lock_rooms, label="방 개수"))

c_area = sb.slider("최소 전용면적 (m²)", 29, 200, 60, key="c_area")
lock_area = sb.checkbox("🔒 면적 절대 사수", value=False)
constraints.append(Constraint("area_m2", "min", c_area, priority=1,
                              locked=lock_area, label="전용면적"))

c_park = sb.slider("최소 주차 가능 대수", 0.0, 3.0, 0.5, step=0.05, key="c_park")
constraints.append(Constraint("parking_slots", "min", c_park, priority=2, label="주차 가능 대수"))

c_walk = sb.slider("역까지 도보 시간 상한 (분)", 1, 45, 20, key="c_walk")
constraints.append(Constraint("transit_walk_min", "max", c_walk, priority=2, label="역까지 도보"))

with sb.expander("부가 조건 (가장 먼저 완화됨)", expanded=False):
    c_age = st.slider("최대 준공 경과년수 (년)", 0, 46, 30)
    constraints.append(Constraint("building_age", "max", c_age, priority=3, label="준공 경과년수"))
    c_noise = st.slider("최대 주간 소음도 (dB)", 33, 74, 74)
    if c_noise < 74:
        constraints.append(Constraint("noise_db", "max", c_noise, priority=3, label="소음도"))

for m in members:
    for d in m.enabled_destinations:
        cap = st.session_state.get(f"cap__{m.key}__{d.key}", 0)
        if cap and cap > 0:
            constraints.append(Constraint(d.column(m.key), "max", float(cap), priority=1,
                                          label=f"{m.name}·{d.label}"))

allowed_or = sb.multiselect("허용 향", list(ORIENTATION_SCORE), default=list(ORIENTATION_SCORE))
allowed_gu = sb.multiselect("지역 한정 (미선택 = 전체)", sorted(listings["gu"].unique()), default=[])
categories = []
if len(allowed_or) < len(ORIENTATION_SCORE):
    categories.append(CategoryConstraint("orientation", allowed_or, priority=2, label="향"))
if allowed_gu:
    categories.append(CategoryConstraint("gu", allowed_gu, priority=1, label="지역"))

# --------------------------------------------------------------------------- #
# 사이드바 ③ 가중치 · 전처리
# --------------------------------------------------------------------------- #
sb.divider()
sb.subheader("③ 가중치")
alpha = sb.slider("🏠 집 자체 ←→ 🚶 이동 시간", 0.0, 1.0, value=0.5, step=0.05, key="alpha")
sb.caption(f"공통 {alpha:.0%} · 개별 동선 {1 - alpha:.0%}")

common_weights = {}
with sb.expander("공통 피처 가중치", expanded=False):
    for f in COMMON_FEATURES:
        common_weights[f.key] = st.slider(f"{f.label} ({f.unit})", 0.0, 2.0,
                                          value=float(f.default_weight), step=0.1,
                                          key=f"wc__{f.key}")

use_satisficing = sb.toggle("만족 임계값 사용", value=True,
                            help="'이 정도면 충분' 기준보다 빠르면 모두 만점 처리합니다.")
satisficing = {}
if use_satisficing:
    with sb.expander("충분 기준 (분)", expanded=False):
        for m in members:
            for d in m.enabled_destinations:
                v = st.number_input(f"{m.name}·{d.label}", min_value=0, value=25, step=5,
                                    key=f"sat__{m.key}__{d.key}")
                satisficing[d.column(m.key)] = float(v)

sb.divider()
sb.subheader("④ 전처리 · 출력")
scaler_kind = sb.selectbox("스케일러", ["robust", "minmax", "standard"], index=0,
                           format_func=lambda k: {"robust": "RobustScaler (이상치에 강함, 권장)",
                                                  "minmax": "MinMax", "standard": "Standard"}[k],
                           key="scaler_kind")
clip_q = sb.slider("이상치 클리핑 (상·하위 %)", 0.0, 5.0, 1.0, step=0.5, key="clip_q") / 100
k = sb.slider("추천 매물 수 (K)", 3, 30, 10, key="k")
member_top_n = sb.slider("구성원별 개인 Top-N (교집합용)", 1, 15, 5, key="member_top_n")
n_clusters = sb.slider("군집 수", 2, 5, 3, key="n_clusters")
sort_mode = sb.radio("정렬 기준", ["종합 적합도", "최약자 우선 (Maximin)", "형평성 우선"], index=0,
                     key="sort_mode")

# --------------------------------------------------------------------------- #
# 추천 실행
# --------------------------------------------------------------------------- #
dest_weights = {d.column(m.key): d.weight for m in members for d in m.enabled_destinations}
cfg = WeightConfig(alpha=alpha, common_weights=common_weights,
                   member_priority={m.key: m.priority for m in members},
                   dest_weights=dest_weights,
                   satisficing=satisficing if use_satisficing else {})
hc = HardConstraints(numeric=constraints, category=categories, min_candidates=min_candidates)

engine = FamilyHousingRecommender(listings, travel, members,
                                  scaler=scaler_kind, clip_q=clip_q)
if not engine.active_members and alpha < 1.0:
    st.warning("활성 구성원이 없어 공통 피처만으로 계산합니다.")

try:
    rec = engine.recommend(cfg, hc, k=k, n_clusters=n_clusters, member_top_n=member_top_n)
except ValueError as e:
    st.error(f"⚠️ {e}")
    st.stop()

if sort_mode != "종합 적합도":
    key = "sat_min" if sort_mode.startswith("최약자") else "sat_gap"
    pool = rec.scored[rec.scored["passes_floor"]] if rec.scored["passes_floor"].any() else rec.scored
    order = pool.sort_values(key, ascending=(key == "sat_gap")).head(k).index
    rec.top = pool.loc[order].copy()
    rec.top.insert(0, "rank", range(1, len(order) + 1))
    rec.top = rec.top.join(rec.clusters["cluster"], how="left")

top = rec.top
d = rec.diagnostics
active = engine.active_members

# --------------------------------------------------------------------------- #
# 헤더
# --------------------------------------------------------------------------- #
st.title("🏡 가족 맞춤 주거지 추천")
st.caption("Multi-Person Weighted K-NN · 가격 배제 · 이동은 모두 시간(분) · 비선형 효용 + 파레토")

if len(d["relaxations"]):
    st.warning(f"⚙️ 하드 제약이 과도해 후보가 {min_candidates}건 미만이었습니다. "
               f"덜 중요한 제약부터 **{len(d['relaxations'])}단계 자동 완화**해 "
               f"{d['candidates']}건을 확보했습니다. (모델 진단 탭에서 내역 확인)")
if d["floor_relaxed"]:
    st.warning("⚠️ 구성원 하한선을 모두 만족하는 매물이 없어 하한선을 일시 해제했습니다. "
               "하한선을 낮추거나 발언권을 조정해 보세요.")
if d["compressed_features"]:
    st.info("ℹ️ 이상치로 분포가 눌린 피처: " +
            ", ".join(label_of(c) for c in d["compressed_features"]) +
            " → 클리핑 비율을 올리거나 RobustScaler 를 쓰세요.")

best = top.iloc[0]
c = st.columns(6)
c[0].metric("후보 매물", f"{d['candidates']}건", f"전체 {d['total_listings']}건 중")
c[1].metric("1위 종합 적합도", f"{best['fit_score']:.1f}점")
c[2].metric("최약자 만족도", f"{best['sat_min']:.1f}점")
c[3].metric("이해충돌 지수", f"{best['sat_gap']:.1f}p")
c[4].metric("평균 이동시간", f"{best['travel_mean_min']:.0f}분",
            f"편차 ±{best['travel_std_min']:.0f}분", delta_color="off")
c[5].metric("주거 안정성", f"{best['stability_index']:.2f}")

tabs = st.tabs(["🥇 추천 결과", "⚖️ 이해충돌·파레토", "🗺️ 동선 지도", "🔎 군집", "🧪 모델 진단"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    show = {"rank": "순위", "name": "매물", "gu": "지역", "fit_score": "종합",
            "sat__공통": "주거", **{f"sat__{m.key}": m.name for m in active},
            "sat_min": "최약자", "sat_gap": "격차", "area_m2": "면적(m²)", "rooms": "방",
            "parking_slots": "주차", "orientation": "향", "transit_walk_min": "역도보(분)",
            **{f"burden_min__{m.key}": f"{m.name} 이동(분)" for m in active},
            "travel_mean_min": "평균이동(분)", "travel_std_min": "편차(분)",
            "equity_index": "형평성", "stability_index": "안정성",
            "pareto": "파레토", "cluster": "군집"}
    cols = [c_ for c_ in show if c_ in top.columns]
    st.dataframe(top[cols].rename(columns=show), use_container_width=True, height=430,
                 column_config={
                     "종합": st.column_config.ProgressColumn("종합", min_value=0, max_value=100, format="%.1f"),
                     "최약자": st.column_config.ProgressColumn("최약자", min_value=0, max_value=100, format="%.1f"),
                     "파레토": st.column_config.CheckboxColumn("파레토")})

    st.divider()
    left, right = st.columns([1, 1])
    labels = {i: f"{int(r['rank'])}위 · {r['name']} ({r['fit_score']:.1f}점)" for i, r in top.iterrows()}
    sel = left.selectbox("매물 선택", list(top.index), format_func=lambda i: labels[i])
    compare = right.multiselect("비교 대상 (최대 3)", [i for i in top.index if i != sel],
                                default=[i for i in top.index if i != sel][:1],
                                max_selections=3, format_func=lambda i: labels[i])

    st.plotly_chart(viz.member_radar(rec, [sel] + compare, active),
                    use_container_width=True, key="radar_members")
    cc = st.columns(2)
    cc[0].plotly_chart(viz.contribution_bar(rec, sel), use_container_width=True, key="contrib")
    cc[1].plotly_chart(viz.feature_radar(rec, [sel] + compare, list(rec.utility.columns),
                                         "만족도 프로파일 (1 = 이상점)"),
                       use_container_width=True, key="radar_features")

    r = top.loc[sel]
    parts = [f"{m.name} {r[f'burden_min__{m.key}']:.0f}분" for m in active
             if f"burden_min__{m.key}" in r and pd.notna(r[f"burden_min__{m.key}"])]
    st.info(f"**{r['name']}** · {r['gu']} — 전용 {r['area_m2']:.0f}m² / 방 {int(r['rooms'])} / "
            f"주차 {r['parking_slots']:.2f}대 / {r['orientation']}향 / 역도보 {r['transit_walk_min']:.0f}분"
            + ("  ·  " + " · ".join(parts) if parts else "")
            + f"  →  격차 {r['sat_gap']:.1f}p"
            + ("  ⭐ **파레토 최적**" if r["pareto"] else ""))

# --------------------------------------------------------------------------- #
with tabs[1]:
    if len(active) < 2:
        st.info("구성원을 2명 이상 활성화하면 트레이드오프 분석이 가능합니다.")
    else:
        st.markdown("#### 구성원별 독립 Top-N 과 교집합")
        st.caption("각자 자기 기준으로만 뽑은 상위 매물입니다. 교집합이 있으면 그게 가장 안전한 합의안입니다.")
        mc = st.columns(len(active))
        for col, m in zip(mc, active):
            col.markdown(f"**{m.name}**")
            for lid in rec.member_tops.get(m.key, []):
                row = rec.scored.loc[lid]
                mark = "⭐ " if lid in rec.intersection else ""
                col.write(f"{mark}{row['name']} · {row[f'sat__{m.key}']:.0f}점")
        if rec.intersection:
            st.success("🤝 모두의 Top-N 에 든 매물: " +
                       ", ".join(rec.scored.loc[i, "name"] for i in rec.intersection))
        else:
            st.warning("교집합이 없습니다 — 선호가 실제로 상충합니다. "
                       "아래 파레토 프론티어에서 균형점을 고르거나, 하한선을 걸어 타협 범위를 좁히세요.")

        st.divider()
        opts = {m.key: m.name for m in active}
        cA, cB = st.columns(2)
        x_key = cA.selectbox("X축 구성원", list(opts), format_func=lambda k_: opts[k_], index=0)
        y_key = cB.selectbox("Y축 구성원", list(opts), format_func=lambda k_: opts[k_],
                             index=min(1, len(opts) - 1))
        st.plotly_chart(viz.tradeoff_scatter(rec, x_key, y_key, opts[x_key], opts[y_key],
                                             list(top.index)),
                        use_container_width=True, key="tradeoff")

        pareto = rec.scored[rec.scored["pareto"]].sort_values("sat_min", ascending=False)
        st.markdown(f"#### 파레토 최적 대안 {len(pareto)}건")
        pcols = ["name", "gu", "fit_score"] + [f"sat__{m.key}" for m in active] + \
                ["sat_min", "sat_gap", "travel_mean_min", "passes_floor"]
        st.dataframe(pareto[[c_ for c_ in pcols if c_ in pareto.columns]].head(20).rename(
            columns={"name": "매물", "gu": "지역", "fit_score": "종합",
                     **{f"sat__{m.key}": m.name for m in active},
                     "sat_min": "최약자", "sat_gap": "격차",
                     "travel_mean_min": "평균이동(분)", "passes_floor": "하한선 통과"}),
            use_container_width=True, height=320)

        st.markdown("#### 규칙별 최선의 선택")
        pool = rec.scored[rec.scored["passes_floor"]] if rec.scored["passes_floor"].any() else rec.scored
        rules = {"공리주의 (가중 합)": pool.sort_values("distance").index[0],
                 "롤스적 최약자 우선": pool.sort_values("sat_min", ascending=False).index[0],
                 "형평성 우선 (격차 최소)": pool.sort_values("sat_gap").index[0]}
        rc = st.columns(len(rules))
        for (rule, lid), col in zip(rules.items(), rc):
            row = rec.scored.loc[lid]
            col.markdown(f"**{rule}**")
            col.write(f"🏠 {row['name']} ({row['gu']})")
            col.write(f"종합 {row['fit_score']:.1f} · 최약자 {row['sat_min']:.1f} · 격차 {row['sat_gap']:.1f}p")
        st.plotly_chart(viz.member_radar(rec, list(dict.fromkeys(rules.values())), active),
                        use_container_width=True, key="radar_rules")

# --------------------------------------------------------------------------- #
with tabs[2]:
    map_sel = st.selectbox("동선을 볼 매물", list(top.index),
                           format_func=lambda i: labels[i], key="map_sel")
    kind, obj = viz.build_map(top, active, map_sel)
    if kind == "folium" and HAS_ST_FOLIUM:
        st_folium(obj, height=560, use_container_width=True)
    elif kind == "folium":
        st.components.v1.html(obj._repr_html_(), height=560)
    else:
        st.plotly_chart(obj, use_container_width=True, key="map_plotly")

    mrow = top.loc[map_sel]
    for m in active:
        cols_ = st.columns(max(len(m.enabled_destinations), 1))
        for col, dst in zip(cols_, m.enabled_destinations):
            val = mrow.get(dst.column(m.key))
            col.metric(f"{m.name} → {dst.label}",
                       f"{val:.0f}분" if pd.notna(val) else "-", TRAVEL_MODES[dst.mode],
                       delta_color="off")

# --------------------------------------------------------------------------- #
with tabs[3]:
    if rec.cluster_profile.empty:
        st.info("군집을 나눌 만큼 추천 매물이 많지 않습니다.")
    else:
        st.plotly_chart(viz.cluster_profile_fig(rec.cluster_profile),
                        use_container_width=True, key="clusters")
        st.dataframe(rec.cluster_profile.rename(columns={
            "area_m2": "면적(m²)", "rooms": "방", "parking_slots": "주차",
            "transit_walk_min": "역도보(분)", "travel_mean_min": "평균이동(분)",
            "travel_std_min": "이동편차(분)", "stability_index": "안정성", "fit_score": "종합"}),
            use_container_width=True)

# --------------------------------------------------------------------------- #
with tabs[4]:
    m1 = st.columns(4)
    m1[0].metric("사용 피처", d["feature_count"],
                 f"공통 {d['common_features']} / 개별 {d['individual_features']}")
    m1[1].metric("KNNImputer 보정", f"{d['missing_cells']}셀", f"k = {d['impute_k']}")
    m1[2].metric("스케일러", d["scaler"], f"클리핑 {d['clip_q']:.1%}")
    m1[3].metric("하한선 통과", f"{d['eligible']}건", f"전체 후보 {d['candidates']}건")

    st.markdown("#### ① 이상치 클리핑")
    if len(d["clip_report"]):
        st.dataframe(d["clip_report"].assign(피처=lambda x: x["피처"].map(label_of)),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("클리핑된 값이 없습니다.")
    st.caption("정규화 후 분포 폭(p5~p95) — 좁을수록 이상치에 스케일을 빼앗긴 상태입니다.")
    st.dataframe(pd.DataFrame({"피처": [label_of(c) for c in d["position_spread"].index],
                               "분포 폭": d["position_spread"].values}),
                 use_container_width=True, hide_index=True, height=240)

    st.markdown("#### ② 비선형 효용 변환")
    st.plotly_chart(viz.utility_curve_fig(), use_container_width=True, key="utility_curve")

    st.markdown("#### ③ 하드 제약 · 백오프")
    st.dataframe(d["filter_detail"], use_container_width=True, hide_index=True)
    if len(d["relaxations"]):
        st.markdown("**자동 완화 내역**")
        st.dataframe(d["backoff_log"], use_container_width=True, hide_index=True)
    else:
        st.caption("완화 없이 조건을 만족했습니다.")

    if len(d["floors"]):
        st.markdown("#### ④ 구성원 하한선")
        st.dataframe(d["floors"], use_container_width=True, hide_index=True)

    st.markdown("#### ⑤ 가중치와 이상점")
    st.plotly_chart(viz.weight_bar(rec), use_container_width=True, key="weights")
    st.dataframe(pd.DataFrame({
        "피처": [label_of(c) for c in rec.ideal_raw.index],
        "이상값(원 단위)": rec.ideal_raw.values,
        "가중치 ω(%)": (rec.weights.values * 100).round(2)}),
        use_container_width=True, hide_index=True)

    st.success("가격 컬럼 없음 · 거리(m) 컬럼 없음 — 이동 지표는 전부 분 단위 시간입니다.")
    with st.expander("원천 데이터 컬럼"):
        st.write("**공통/부가:**", list(listings.columns))
        st.write("**개별(이동 시간, 분):**", list(travel.columns))
