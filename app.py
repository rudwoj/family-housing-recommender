"""
가족 맞춤 주거지 추천 (Multi-Person Weighted K-NN)

실행: streamlit run app.py

화면 흐름
  ① 설정 화면 — 구성원·목적지·제약·가중치를 큰 화면에서 입력
  ② [확인하기] → ③ 결과 화면 — 추천 목록과 동선 지도

설정값은 위젯 상태가 아니라 st.session_state["config"] 에 스냅샷으로 저장한다.
화면이 전환되면 설정 위젯이 렌더링되지 않아 위젯 상태가 정리될 수 있기 때문이다.

가격은 반영하지 않으며, 이동 지표는 전부 분 단위 시간이다.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src import viz
from src.features import derive_common_features
from src.model import (CategoryConstraint, Constraint, FamilyHousingRecommender,
                       HardConstraints, WeightConfig, label_of)
from src.schema import (COMMON_FEATURES, DEFAULT_MEMBER_COUNT, MAX_MEMBERS,
                        ORIENTATION_SCORE, TRAVEL_MODES, Destination, Member,
                        make_default_members)
from src.travel import ApiTravelProvider, EstimatedTravelProvider, travel_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

st.set_page_config(page_title="가족 맞춤 주거지 추천", page_icon="🏡", layout="wide",
                   initial_sidebar_state="collapsed")

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False

PLACE_PRESETS = {
    "여의도": (37.5216, 126.9243), "강남역": (37.4979, 127.0276),
    "광화문": (37.5720, 126.9769), "판교": (37.3947, 127.1112),
    "구로디지털단지": (37.4853, 126.9015), "성수": (37.5446, 127.0559),
    "대치 학원가": (37.4995, 127.0630), "잠실": (37.5133, 127.1000),
    "상암DMC": (37.5796, 126.8896), "용산": (37.5298, 126.9648),
}
SORT_MODES = ["종합 적합도", "최약자 우선 (Maximin)", "형평성 우선"]


@st.cache_data(show_spinner=False)
def load_listings() -> pd.DataFrame:
    return derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))


@st.cache_data(show_spinner="이동 시간 계산 중...")
def compute_travel(signature: tuple, use_api: bool) -> pd.DataFrame:
    listings = load_listings()
    members = [Member(key=mk, name=mk, color="#000",
                      destinations=[Destination(key=dk, label=dk, lat=la, lon=lo, mode=mo)])
               for mk, dk, la, lo, mo in signature]
    provider = ApiTravelProvider() if use_api else EstimatedTravelProvider()
    out = travel_matrix(listings, [members[0]], provider)
    for m in members[1:]:
        out = out.merge(travel_matrix(listings, [m], provider), on="listing_id")
    return out


if not os.path.exists(os.path.join(DATA, "listings.csv")):
    st.error("데이터가 없습니다. 먼저 실행하세요:  `python data/generate_mock_data.py`")
    st.stop()

listings = load_listings()
st.session_state.setdefault("stage", "setup")


# =========================================================================== #
# ① 설정 화면
# =========================================================================== #
def destination_form(mk: str, base_d: Destination,
                     prev_cap: int = 0) -> tuple[Destination, int]:
    """목적지 1건의 입력 위젯. (목적지, 이동시간 상한) 반환."""
    dk = base_d.key
    enabled = st.checkbox("이 목적지 사용", value=base_d.enabled, key=f"den__{mk}__{dk}")
    label = st.text_input("목적지 이름", value=base_d.label, key=f"dlabel__{mk}__{dk}").strip()
    label = label or base_d.label

    c1, c2 = st.columns(2)
    mode = c1.selectbox("이동수단", list(TRAVEL_MODES), key=f"dmode__{mk}__{dk}",
                        index=list(TRAVEL_MODES).index(base_d.mode),
                        format_func=lambda k: TRAVEL_MODES[k])
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
        st.caption(f"📍 {preset} ({lat:.4f}, {lon:.4f})")

    cap = st.number_input("⛔ 이동시간 상한 (분, 0 = 제한 없음)", min_value=0,
                          value=int(prev_cap), step=5, key=f"dcap__{mk}__{dk}")
    return Destination(dk, label, lat, lon, mode, weight, enabled), int(cap)


def render_setup() -> None:
    """
    설정 화면. 이전에 [확인하기] 로 저장한 값이 있으면 그 값으로 복원한다.
    (결과 화면을 거치는 동안 설정 위젯의 상태가 정리될 수 있어, 위젯 상태가 아니라
     저장된 스냅샷을 기본값으로 쓴다.)
    """
    saved = st.session_state.get("config", {})
    raw = saved.get("raw", {})

    def D(key, fallback):
        return raw.get(key, fallback)

    st.title("🏡 가족 맞춤 주거지 추천")
    st.caption("가격은 반영하지 않습니다. 주택의 공간 조건과 구성원별 이동 시간(분)만으로 계산합니다.")
    st.markdown("#### 아래 조건을 입력하고 맨 아래 **확인하기**를 누르세요.")
    st.divider()

    # ---------------- ① 구성원과 목적지 ---------------- #
    st.subheader("① 구성원과 목적지")
    head = st.columns([1, 3])
    n_members = head[0].number_input("구성원 수", min_value=1, max_value=MAX_MEMBERS,
                                     value=int(D("n_members", DEFAULT_MEMBER_COUNT)),
                                     step=1, key="n_members")
    use_api = head[1].toggle(
        "경로 API 사용", value=bool(D("use_api", False)), key="use_api",
        help="KAKAO_REST_API_KEY / ODSAY_API_KEY 환경변수가 있을 때만 동작하며, "
             "실패하면 거리 기반 추정으로 자동 폴백합니다.")
    if use_api and not ApiTravelProvider().available:
        st.warning("API 키가 없어 거리 기반 추정으로 동작합니다.")

    # 저장된 구성원이 있으면 그것을 기본값으로 (없거나 인원이 늘면 템플릿으로 채운다)
    template = make_default_members(int(n_members))
    prev = {m.key: m for m in saved.get("members", [])}
    defaults = [prev.get(t.key, t) for t in template]
    prev_caps = D("caps", {})

    members: list[Member] = []
    caps: dict[str, int] = {}

    grid = st.columns(2)
    for i, base in enumerate(defaults):
        mk = base.key
        with grid[i % 2].container(border=True):
            name = st.text_input(f"구성원 {i + 1} 이름", value=base.name, key=f"mname__{mk}",
                                 placeholder="예: 아빠, 엄마, 첫째").strip() or base.name
            cc = st.columns([1, 1])
            active = cc[0].checkbox("추천에 반영", value=bool(base.active), key=f"active__{mk}")
            priority = cc[1].slider("발언권", 0.0, 3.0, value=float(base.priority), step=0.1,
                                    key=f"prio__{mk}")
            floor = st.slider("만족도 하한선 (이 밑이면 후보에서 제외)", 0, 90,
                              value=int(base.min_satisfaction), step=5, key=f"floor__{mk}",
                              help="상충하는 선호가 '어중간한 타협안'으로 수렴하는 것을 막습니다.")

            dests: list[Destination] = []
            for base_d in base.destinations:
                with st.expander(f"목적지 · {base_d.label}",
                                 expanded=(base_d.key == "dest1")):
                    dest, cap = destination_form(mk, base_d,
                                                 prev_caps.get(base_d.column(mk), 0))
                    dests.append(dest)
                    if cap > 0:
                        caps[dest.column(mk)] = cap

            m = Member(mk, name, base.color, dests, priority, float(floor))
            m.active = active
            members.append(m)

    # ---------------- ② 하드 제약 ---------------- #
    st.divider()
    st.subheader("② 하드 제약 — 타협할 수 없는 조건")
    st.caption("조건이 과해 후보가 부족하면, 덜 중요한 조건부터 자동으로 완화합니다(잠금한 조건 제외).")

    r1 = st.columns(4)
    c_rooms = r1[0].slider("최소 방 개수", 1, 7, int(D("c_rooms", 3)), key="c_rooms")
    lock_rooms = r1[0].checkbox("🔒 절대 사수", value=bool(D("lock_rooms", True)), key="lock_rooms")
    c_area = r1[1].slider("최소 전용면적 (m²)", 29, 200, int(D("c_area", 60)), key="c_area")
    lock_area = r1[1].checkbox("🔒 절대 사수", value=bool(D("lock_area", False)), key="lock_area")
    c_park = r1[2].slider("최소 주차 가능 대수", 0.0, 3.0, float(D("c_park", 0.5)), step=0.05, key="c_park")
    c_walk = r1[3].slider("역까지 도보 시간 상한 (분)", 1, 45, int(D("c_walk", 20)), key="c_walk")

    r2 = st.columns(4)
    c_age = r2[0].slider("최대 준공 경과년수 (년)", 0, 46, int(D("c_age", 30)), key="c_age")
    c_noise = r2[1].slider("최대 주간 소음도 (dB)", 33, 74, int(D("c_noise", 74)), key="c_noise")
    min_candidates = r2[2].slider("최소 확보 후보 수", 1, 30, int(D("min_candidates", 5)), key="min_candidates")
    allowed_gu = r2[3].multiselect("지역 한정 (미선택 = 전체)", sorted(listings["gu"].unique()),
                                   default=D("allowed_gu", []), key="allowed_gu")
    allowed_or = st.multiselect("허용 향", list(ORIENTATION_SCORE),
                                default=D("allowed_or", list(ORIENTATION_SCORE)), key="allowed_or")

    constraints = [
        Constraint("rooms", "min", c_rooms, priority=0, locked=lock_rooms, label="방 개수"),
        Constraint("area_m2", "min", c_area, priority=1, locked=lock_area, label="전용면적"),
        Constraint("parking_slots", "min", c_park, priority=2, label="주차 가능 대수"),
        Constraint("transit_walk_min", "max", c_walk, priority=2, label="역까지 도보"),
        Constraint("building_age", "max", c_age, priority=3, label="준공 경과년수"),
    ]
    if c_noise < 74:
        constraints.append(Constraint("noise_db", "max", c_noise, priority=3, label="소음도"))
    # 라벨은 구성원·목적지 이름으로 직접 만든다
    # (엔진이 아직 없어 label_of 사전이 비어 있으므로 내부 컬럼명이 노출된다)
    cap_labels = {d.column(m.key): f"{m.name}·{d.label}"
                  for m in members for d in m.destinations}
    for col, cap in caps.items():
        constraints.append(Constraint(col, "max", float(cap), priority=1,
                                      label=cap_labels.get(col, col)))

    categories = []
    if len(allowed_or) < len(ORIENTATION_SCORE):
        categories.append(CategoryConstraint("orientation", allowed_or, priority=2, label="향"))
    if allowed_gu:
        categories.append(CategoryConstraint("gu", allowed_gu, priority=1, label="지역"))

    # ---------------- ③ 가중치 ---------------- #
    st.divider()
    st.subheader("③ 무엇을 더 중요하게 볼까요")
    alpha = st.slider("🏠 집 자체 ←→ 🚶 이동 시간", 0.0, 1.0, value=float(D("alpha", 0.5)),
                      step=0.05, key="alpha")
    st.caption(f"공통(주택 조건) {alpha:.0%} · 개별(이동 시간) {1 - alpha:.0%}")

    wcols = st.columns(len(COMMON_FEATURES))
    common_weights = {}
    for col, f in zip(wcols, COMMON_FEATURES):
        common_weights[f.key] = col.slider(
            f.axis, 0.0, 2.0, step=0.1, key=f"wc__{f.key}",
            value=float(saved.get("common_weights", {}).get(f.key, f.default_weight)))

    use_satisficing = st.toggle("만족 임계값 사용", value=bool(D("use_satisficing", True)),
                                key="use_satisficing",
                                help="'이 정도면 충분' 기준보다 빠르면 모두 만점 처리합니다.")
    satisficing = {}
    if use_satisficing:
        enabled = [(m, d) for m in members for d in m.enabled_destinations]
        if enabled:
            scols = st.columns(min(len(enabled), 4))
            for j, (m, dst) in enumerate(enabled):
                v = scols[j % len(scols)].number_input(
                    f"{m.name}·{dst.label} (분)", min_value=0, step=5,
                    value=int(saved.get("satisficing", {}).get(dst.column(m.key), 25)),
                    key=f"sat__{m.key}__{dst.key}")
                satisficing[dst.column(m.key)] = float(v)

    # ---------------- ④ 계산 옵션 ---------------- #
    st.divider()
    with st.expander("④ 계산 옵션 (기본값 권장)", expanded=False):
        o = st.columns(4)
        scaler_kind = o[0].selectbox(
            "스케일러", ["robust", "minmax", "standard"], key="scaler_kind",
            index=["robust", "minmax", "standard"].index(D("scaler", "robust")),
            format_func=lambda k: {"robust": "RobustScaler (이상치에 강함)",
                                   "minmax": "MinMax", "standard": "Standard"}[k])
        clip_q = o[1].slider("이상치 클리핑 (상·하위 %)", 0.0, 5.0, float(D("clip_q", 0.01) * 100),
                             step=0.5, key="clip_q") / 100
        k = o[2].slider("추천 매물 수 (K)", 3, 30, int(D("k", 10)), key="k")
        sort_mode = o[3].radio("정렬 기준", SORT_MODES, key="sort_mode",
                               index=SORT_MODES.index(D("sort_mode", SORT_MODES[0])))

    # ---------------- 확인하기 ---------------- #
    st.divider()
    if not any(d for m in members for d in m.enabled_destinations):
        st.error("활성화된 목적지가 없습니다. 구성원마다 목적지를 하나 이상 켜주세요.")
        return
    if st.button("✅ 확인하기", type="primary", use_container_width=True):
        st.session_state.config = {
            "members": members, "constraints": constraints, "categories": categories,
            "min_candidates": min_candidates, "alpha": alpha,
            "common_weights": common_weights,
            "satisficing": satisficing if use_satisficing else {},
            "scaler": scaler_kind, "clip_q": clip_q, "k": k, "sort_mode": sort_mode,
            "use_api": use_api,
            # 설정 화면으로 되돌아왔을 때 그대로 복원하기 위한 원본 값
            "raw": {"n_members": int(n_members), "use_api": use_api, "c_rooms": c_rooms,
                    "lock_rooms": lock_rooms, "c_area": c_area, "lock_area": lock_area,
                    "c_park": c_park, "c_walk": c_walk, "c_age": c_age, "c_noise": c_noise,
                    "min_candidates": min_candidates, "allowed_gu": allowed_gu,
                    "allowed_or": allowed_or, "alpha": alpha,
                    "use_satisficing": use_satisficing, "scaler": scaler_kind,
                    "clip_q": clip_q, "k": k, "sort_mode": sort_mode, "caps": caps},
        }
        st.session_state.stage = "result"
        st.rerun()


# =========================================================================== #
# ③ 결과 화면
# =========================================================================== #
def render_result(conf: dict) -> None:
    members: list[Member] = conf["members"]
    signature = tuple(sorted((m.key, d.key, round(d.lat, 5), round(d.lon, 5), d.mode)
                             for m in members for d in m.enabled_destinations))
    travel = compute_travel(signature, conf["use_api"])

    engine = FamilyHousingRecommender(listings, travel, members,
                                      scaler=conf["scaler"], clip_q=conf["clip_q"])
    cfg = WeightConfig(
        alpha=conf["alpha"], common_weights=conf["common_weights"],
        member_priority={m.key: m.priority for m in members},
        dest_weights={d.column(m.key): d.weight for m in members
                      for d in m.enabled_destinations},
        satisficing=conf["satisficing"])
    hc = HardConstraints(numeric=conf["constraints"], category=conf["categories"],
                         min_candidates=conf["min_candidates"])

    bar = st.columns([1, 4])
    if bar[0].button("⬅ 설정 수정", use_container_width=True):
        st.session_state.stage = "setup"
        st.rerun()

    if not engine.active_members and conf["alpha"] < 1.0:
        st.warning("활성 구성원이 없어 공통(주택 조건) 피처만으로 계산합니다.")

    try:
        rec = engine.recommend(cfg, hc, k=conf["k"])
    except ValueError as e:
        st.error(f"⚠️ {e}")
        st.stop()

    if conf["sort_mode"] != SORT_MODES[0]:
        key = "sat_min" if conf["sort_mode"].startswith("최약자") else "sat_gap"
        pool = rec.scored[rec.scored["passes_floor"]] if rec.scored["passes_floor"].any() \
            else rec.scored
        order = pool.sort_values(key, ascending=(key == "sat_gap")).head(conf["k"]).index
        rec.top = pool.loc[order].copy()
        rec.top.insert(0, "rank", range(1, len(order) + 1))

    top, d, active = rec.top, rec.diagnostics, engine.active_members

    st.title("🏡 추천 결과")
    st.caption("Multi-Person Weighted K-NN · 가격 배제 · 이동은 모두 시간(분)")

    if len(d["relaxations"]):
        st.warning(f"⚙️ 하드 제약이 과도해 후보가 {conf['min_candidates']}건 미만이었습니다. "
                   f"덜 중요한 제약부터 **{len(d['relaxations'])}단계 자동 완화**해 "
                   f"{d['candidates']}건을 확보했습니다.")
        with st.expander("완화 내역 보기"):
            st.dataframe(d["backoff_log"], use_container_width=True, hide_index=True)
    if d["floor_relaxed"]:
        st.warning("⚠️ 하한선을 모두 만족하는 매물이 없어 하한선을 일시 해제했습니다.")
    if d["compressed_features"]:
        st.info("ℹ️ 이상치로 분포가 눌린 피처: " +
                ", ".join(label_of(c) for c in d["compressed_features"]) +
                " → 클리핑 비율을 올리거나 RobustScaler 를 쓰세요.")

    best = top.iloc[0]
    c = st.columns(3)
    c[0].metric("후보 매물", f"{d['candidates']}건", f"전체 {d['total_listings']}건 중")
    c[1].metric("1위 종합 적합도", f"{best['fit_score']:.1f}점")
    c[2].metric("최약자 만족도", f"{best['sat_min']:.1f}점")

    labels = {i: f"{int(r['rank'])}위 · {r['name']} ({r['fit_score']:.1f}점)"
              for i, r in top.iterrows()}
    tabs = st.tabs(["🥇 추천 결과", "🗺️ 동선 지도"])

    with tabs[0]:
        main_cols = {"rank": "순위", "name": "매물", "gu": "지역", "fit_score": "종합",
                     "sat__공통": "주거", **{f"sat__{m.key}": m.name for m in active},
                     "sat_min": "최약자", "sat_gap": "격차", "area_m2": "면적(m²)",
                     "rooms": "방", "parking_slots": "주차", "orientation": "향"}
        detail_cols = {"name": "매물", "transit_walk_min": "역도보(분)",
                       **{f"burden_min__{m.key}": f"{m.name} 이동(분)" for m in active},
                       "travel_mean_min": "평균이동(분)", "travel_std_min": "편차(분)",
                       "equity_index": "형평성", "green_ratio": "녹지(%)",
                       "building_age": "연식(년)", "daylight_hours": "일조(h)",
                       "noise_db": "소음(dB)"}

        st.dataframe(top[[c_ for c_ in main_cols if c_ in top.columns]].rename(columns=main_cols),
                     use_container_width=True, height=430,
                     column_config={
                         "종합": st.column_config.ProgressColumn("종합", min_value=0,
                                                               max_value=100, format="%.1f"),
                         "최약자": st.column_config.ProgressColumn("최약자", min_value=0,
                                                                max_value=100, format="%.1f")})
        with st.expander("상세보기 — 이동 시간과 주거 환경", expanded=False):
            st.dataframe(
                top[[c_ for c_ in detail_cols if c_ in top.columns]].rename(columns=detail_cols),
                use_container_width=True, height=430)

        st.divider()
        left, right = st.columns(2)
        sel = left.selectbox("매물 선택", list(top.index), format_func=lambda i: labels[i])
        compare = right.multiselect("비교 대상 (최대 3)", [i for i in top.index if i != sel],
                                    default=[i for i in top.index if i != sel][:1],
                                    max_selections=3, format_func=lambda i: labels[i])

        st.plotly_chart(viz.member_radar(rec, [sel] + compare, active),
                        use_container_width=True, key="radar_members")
        st.plotly_chart(viz.feature_radar(rec, [sel] + compare, list(rec.utility.columns),
                                          "만족도 프로파일 (1 = 이상점)"),
                        use_container_width=True, key="radar_features")

        r = top.loc[sel]
        parts = [f"{m.name} {r[f'burden_min__{m.key}']:.0f}분" for m in active
                 if f"burden_min__{m.key}" in r and pd.notna(r[f"burden_min__{m.key}"])]
        st.info(f"**{r['name']}** · {r['gu']} — 전용 {r['area_m2']:.0f}m² / "
                f"방 {int(r['rooms'])} / 주차 {r['parking_slots']:.2f}대 / "
                f"{r['orientation']}향 / 역도보 {r['transit_walk_min']:.0f}분"
                + ("  ·  " + " · ".join(parts) if parts else ""))

    with tabs[1]:
        # 후보가 바뀌면 이전 선택이 목록에 없을 수 있으므로 key 를 두지 않는다.
        map_sel = st.selectbox("동선을 볼 매물", list(top.index),
                               format_func=lambda i: labels[i])
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
                           f"{val:.0f}분" if pd.notna(val) else "-",
                           TRAVEL_MODES[dst.mode], delta_color="off")


# =========================================================================== #
if st.session_state.stage == "result" and "config" in st.session_state:
    render_result(st.session_state.config)
else:
    st.session_state.stage = "setup"
    render_setup()
