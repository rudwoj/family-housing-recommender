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

from src import viz, viz_kakao
from src.data_sources import kakao, odsay, tmap
from src.data_sources.config import load_settings
from src.features import derive_common_features
from src.model import (CategoryConstraint, Constraint, FamilyHousingRecommender,
                       HardConstraints, WeightConfig, label_of)
from src.schema import (COMMON_FEATURES, DEFAULT_MEMBER_COUNT, MAX_MEMBERS,
                        ORIENTATION_SCORE, TRAVEL_MODES, Destination, Member,
                        make_default_members)
from src.travel import (ApiTravelProvider, EstimatedTravelProvider, travel_matrix,
                        _secret as read_api_key)
from scripts.fetch_real_dataset import build_live_listings

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

st.set_page_config(page_title="가족 맞춤 주거지 추천", page_icon="🏡", layout="wide",
                   initial_sidebar_state="collapsed")

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


@st.cache_data(show_spinner="공공데이터에서 최신 주거 후보를 불러오는 중...", ttl=21600)
def load_latest_public_listings() -> tuple[pd.DataFrame, dict]:
    """파일 저장 없이 최신 공공데이터 후보를 생성한다."""
    fresh, stats = build_live_listings(load_settings())
    return derive_common_features(fresh), stats.as_dict()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_real_route(lat: float, lon: float, dest_lat: float, dest_lon: float,
                     mode: str, api_key: str, odsay_key: str = "",
                     tmap_key: str = "") -> dict:
    """
    카카오 API로 실제 이동 경로 좌표를 가져온다.

    반환: {"path": [...], "status": ..., "reason": ...}
      real      실제 경로를 받았다
      walk      도보 경로를 못 받아 직선으로 그린다 (TMap 키 없음/거부/실패)
      no_route  API는 응답했지만 경로가 없다
      error     호출 실패 (권한/설정 문제일 가능성이 높다)
    왜 직선으로 나오는지 화면에서 확인할 수 있도록 이유를 삼키지 않고 돌려준다.
    """
    if not api_key and not odsay_key and not tmap_key:
        return {"path": [], "status": "no_key", "reason": "경로 API 키 없음"}

    if mode == "walk":
        # 도보는 카카오·ODsay 모두 길찾기가 없어 TMap 으로만 실측할 수 있다.
        if not tmap_key:
            return {"path": [], "status": "walk",
                    "reason": "TMAP_APP_KEY 가 없어 직선으로 표시합니다"}
        try:
            route = tmap.pedestrian_route(lat, lon, dest_lat, dest_lon, tmap_key, timeout=15)
            if route is not None and len(route.path) >= 2:
                return {"path": route.path, "status": "real", "reason": "TMap",
                        "minutes": route.total_time_sec / 60.0}
            return {"path": [], "status": "walk", "reason": "TMap 경로 결과 없음 — 직선 표시"}
        except Exception as e:                      # noqa: BLE001
            # 출발-도착이 너무 멀면 TMap 이 경로를 거부한다. 타임아웃 같은 네트워크
            # 오류까지 함께 잡아야 한다 — 여기서 새면 앱 전체가 죽는다.
            return {"path": [], "status": "walk", "reason": f"TMap: {str(e)[:120]}"}
    if mode == "transit" and odsay_key:
        # 대중교통은 ODsay 를 먼저 쓴다 (카카오 대중교통은 응답이 커서 느리다).
        try:
            route = odsay.transit_route(lat, lon, dest_lat, dest_lon, odsay_key, timeout=15)
            if route is not None and len(route.path) >= 2:
                return {"path": route.path, "status": "real", "reason": "ODsay",
                        "minutes": route.total_time_sec / 60.0}
        except Exception as e:                      # noqa: BLE001 - 타임아웃 포함
            if not api_key:
                return {"path": [], "status": "error", "reason": f"ODsay: {str(e)[:140]}"}

    if not api_key:
        return {"path": [], "status": "no_key", "reason": "카카오 키 없음"}
    try:
        if mode == "drive":
            route = kakao.car_directions(lat, lon, dest_lat, dest_lon, api_key, timeout=15)
        else:
            route = kakao.transit_route(lat, lon, dest_lat, dest_lon, api_key, timeout=15)
        if route is not None and len(route.path) >= 2:
            secs = getattr(route, "duration_sec", None) or getattr(route, "total_time_sec", None)
            return {"path": route.path, "status": "real", "reason": "",
                    "minutes": (secs / 60.0) if secs else None}
        return {"path": [], "status": "no_route", "reason": "경로 결과 없음"}
    except Exception as e:                          # noqa: BLE001 - 타임아웃 포함
        return {"path": [], "status": "error", "reason": str(e)[:160]}


@st.cache_data(show_spinner="이동 시간 계산 중...")
def compute_travel(signature: tuple, use_api: bool,
                   listings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """이동 시간표와 함께, 어떤 이동수단이 왜 추정치로 떨어졌는지도 돌려준다."""
    members = [Member(key=mk, name=mk, color="#000",
                      destinations=[Destination(key=dk, label=dk, lat=la, lon=lo, mode=mo)])
               for mk, dk, la, lo, mo in signature]
    provider = ApiTravelProvider() if use_api else EstimatedTravelProvider()
    out = travel_matrix(listings, [members[0]], provider)
    for m in members[1:]:
        out = out.merge(travel_matrix(listings, [m], provider), on="listing_id")
    return out, dict(getattr(provider, "last_error", {}))


if not os.path.exists(os.path.join(DATA, "listings.csv")):
    st.error("데이터가 없습니다. 먼저 실행하세요:  `python data/generate_mock_data.py`")
    st.stop()

st.session_state.setdefault("stage", "setup")
st.session_state.setdefault("listing_source", "bundled")
st.session_state.setdefault("listing_source_error", "")

listings = load_listings()
public_listing_stats: dict | None = None
if st.session_state.listing_source == "public":
    try:
        listings, public_listing_stats = load_latest_public_listings()
        st.session_state.listing_source_error = ""
    except Exception as e:
        st.session_state.listing_source = "bundled"
        st.session_state.listing_source_error = str(e)


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
    settings = load_settings()
    can_load_public = settings.has_data_go_kr_key and settings.has_kakao_key
    source_col, action_col = st.columns([3, 2])
    if st.session_state.listing_source == "public" and public_listing_stats:
        source_col.success(
            f"최신 공공데이터 후보 {len(listings)}건 사용 중 · 지하철역 실측 {public_listing_stats['real']}건")
    else:
        source_col.info("기본 내장 후보 데이터를 사용 중입니다.")
    if action_col.button("최신 공공데이터 후보 불러오기", disabled=not can_load_public,
                         use_container_width=True):
        load_latest_public_listings.clear()
        st.session_state.listing_source = "public"
        st.rerun()
    if not can_load_public:
        st.caption("공공데이터 최신 후보는 Secrets에 `DATA_GO_KR_SERVICE_KEY`와 `KAKAO_REST_API_KEY`를 모두 설정하면 사용할 수 있습니다.")
    elif st.session_state.listing_source_error:
        st.warning(f"공공데이터 후보 갱신에 실패해 기본 데이터를 사용합니다: {st.session_state.listing_source_error}")
    st.markdown("#### 아래 조건을 입력하고 맨 아래 **확인하기**를 누르세요.")
    st.divider()

    # ---------------- ① 구성원과 목적지 ---------------- #
    st.subheader("① 구성원과 목적지")
    n_members = st.number_input("구성원 수", min_value=1, max_value=MAX_MEMBERS,
                                value=int(D("n_members", DEFAULT_MEMBER_COUNT)),
                                step=1, key="n_members")
    # 경로 API는 항상 사용한다 — 키가 없거나 호출이 실패하면 거리 기반 추정으로 자동 폴백한다.
    use_api = True
    if not ApiTravelProvider().available:
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
    travel, travel_fallbacks = compute_travel(signature, conf["use_api"], listings)

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

    try:
        rec = engine.recommend(cfg, hc, k=conf["k"])
    except ValueError as e:
        # 지도(전체 화면)를 못 그리는 경로이므로 여기서는 일반 화면으로 안내한다.
        if st.button("⬅ 설정 수정"):
            st.session_state.stage = "setup"
            st.rerun()
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

    # ---------------- 전체 화면 지도 (카카오맵 스타일) ---------------- #
    # 헤더를 숨기고 지도 iframe 을 뷰포트 전체에 고정한다. 진단·상세 정보는
    # 지도 왼쪽 아래 [상세 정보 보기] 버튼 → 다이얼로그로 제공한다.
    st.markdown("""
        <style>
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        section[data-testid="stMain"] { overflow: hidden; }
        iframe[title$="kakao_map"] { position: fixed; inset: 0; z-index: 10;
                                     width: 100vw !important;
                                     height: 100vh !important; border: 0; }
        /* 리런 중 stale 요소가 0.33 으로 어두워지는 것을 완화 — 지도가 거의
           그대로 보이고, 로딩 표시는 지도 컴포넌트가 중앙에 직접 띄운다. */
        [data-stale="true"], .stale-element { opacity: 0.9 !important; }
        </style>""", unsafe_allow_html=True)

    # 선택 상태: 컴포넌트가 돌려준 클릭 값 → session_state → 기본 1위
    map_sel = st.session_state.get("map_sel")
    if map_sel not in top.index:
        map_sel = top.index[0]
    r_sel = top.loc[map_sel]

    routes: dict[tuple[str, str], dict] = {}
    settings = load_settings()
    odsay_key = read_api_key("ODSAY_API_KEY")
    tmap_key = read_api_key("TMAP_APP_KEY")
    # 지도 동선은 "선택한 매물 1건" 에 대해서만 조회하므로 호출이 몇 건뿐이다.
    # 전체 매물 × 목적지를 도는 이동시간 계산(경로 API 토글)과 달리 항상
    # 실제 경로를 가져온다 — 토글이 꺼져 있다고 대중교통까지 직선으로 그리면
    # 지나는 역을 하나도 안 보여주게 된다.
    if settings.has_kakao_key or odsay_key or tmap_key:
        with st.spinner("실제 동선 조회 중..."):
            for m in active:
                for dst in m.enabled_destinations:
                    routes[(m.key, dst.key)] = fetch_real_route(
                        float(r_sel["lat"]), float(r_sel["lon"]),
                        float(dst.lat), float(dst.lon), dst.mode,
                        settings.kakao_rest_api_key, odsay_key, tmap_key)

    # 카카오 JS 키가 있으면 카카오맵, 없으면 Leaflet(OSM) — 오버레이 UI 는 동일.
    js_key = settings.kakao_javascript_key if settings.has_kakao_js_key else ""
    act = viz_kakao.render_map_view(top, active, map_sel, routes, js_key,
                                    height=640, key="map_view")
    # 컴포넌트 값은 세션 동안 유지되므로, nonce(n)가 바뀐 새 이벤트만 처리한다.
    if isinstance(act, dict) and act.get("n") != st.session_state.get("map_view_n"):
        st.session_state["map_view_n"] = act.get("n")
        kind, aid = act.get("action"), act.get("id")
        if kind == "back":
            st.session_state.stage = "setup"
            st.rerun()
        elif kind == "select" and aid in top.index and aid != map_sel:
            st.session_state["map_sel"] = aid
            st.rerun()
        elif kind == "detail":
            detail_dialog(rec, top, active, d, conf, map_sel, js_key)


# =========================================================================== #
# 상세 분석 다이얼로그 — 지도 왼쪽 아래 [상세 정보 보기] 로 연다
# =========================================================================== #
@st.dialog("📊 상세 분석", width="large")
def detail_dialog(rec, top, active, d, conf, map_sel: str, js_key: str) -> None:
    # 진단 메시지 (전체 화면 지도에서는 자리가 없어 여기서 보여준다)
    if not active and conf["alpha"] < 1.0:
        st.warning("활성 구성원이 없어 공통(주택 조건) 피처만으로 계산했습니다.")
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
    st.caption(f"후보 {d['candidates']}건 / 전체 {d['total_listings']}건 · "
               f"1위 적합도 {best['fit_score']:.1f}점 · 최약자 {best['sat_min']:.1f}점")

    # 선택 매물 요약 한 줄
    r_sel = top.loc[map_sel]
    parts = [f"{m.name} {r_sel[f'burden_min__{m.key}']:.0f}분" for m in active
             if f"burden_min__{m.key}" in r_sel and pd.notna(r_sel[f"burden_min__{m.key}"])]
    st.info(f"**{r_sel['name']}** · {r_sel['gu']} — 전용 {r_sel['area_m2']:.0f}m² / "
            f"방 {int(r_sel['rooms'])} / 주차 {r_sel['parking_slots']:.2f}대 / "
            f"{r_sel['orientation']}향 / 역도보 {r_sel['transit_walk_min']:.0f}분"
            + ("  ·  " + " · ".join(parts) if parts else ""))

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
                 use_container_width=True, height=380,
                 column_config={
                     "종합": st.column_config.ProgressColumn("종합", min_value=0,
                                                           max_value=100, format="%.1f"),
                     "최약자": st.column_config.ProgressColumn("최약자", min_value=0,
                                                            max_value=100, format="%.1f")})
    with st.expander("이동 시간과 주거 환경", expanded=False):
        st.dataframe(
            top[[c_ for c_ in detail_cols if c_ in top.columns]].rename(columns=detail_cols),
            use_container_width=True, height=380)

    st.divider()
    # 지도에서 선택한 매물을 기준으로 비교한다.
    labels = {i: f"{int(r['rank'])}위 · {r['name']} ({r['fit_score']:.1f}점)"
              for i, r in top.iterrows()}
    compare = st.multiselect("비교 대상 (최대 3)",
                             [i for i in top.index if i != map_sel],
                             default=[i for i in top.index if i != map_sel][:1],
                             max_selections=3, format_func=lambda i: labels[i])
    st.plotly_chart(viz.member_radar(rec, [map_sel] + compare, active),
                    use_container_width=True, key="radar_members")
    st.plotly_chart(viz.feature_radar(rec, [map_sel] + compare, list(rec.utility.columns),
                                      "만족도 프로파일 (1 = 이상점)"),
                    use_container_width=True, key="radar_features")

    if not js_key:
        st.caption("ℹ️ `KAKAO_JAVASCRIPT_KEY` 를 설정하면 카카오맵으로 표시됩니다.")


# =========================================================================== #
if st.session_state.stage == "result" and "config" in st.session_state:
    render_result(st.session_state.config)
else:
    st.session_state.stage = "setup"
    render_setup()
