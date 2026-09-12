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
from src.data_sources import kakao, tmap
from src.data_sources.config import SEOUL_GU_NAMES, load_settings
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


try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False

SORT_MODES = ["종합 적합도", "최약자 우선 (Maximin)", "형평성 우선"]


@st.cache_data(show_spinner=False, ttl=600)
def search_places(query: str, api_key: str) -> list[dict]:
    """카카오 키워드 장소 검색. 같은 검색어는 캐시로 재호출을 막는다."""
    return kakao.search_keyword_places(query, api_key, size=7)


def run_place_search(query: str) -> tuple[list[dict], str]:
    """장소 검색을 실행하고 (결과, 오류 메시지)를 돌려준다."""
    api_key = read_api_key("KAKAO_REST_API_KEY")
    if not api_key:
        return [], ("카카오 API 키가 설정되지 않았습니다. 환경변수 또는 Secrets에 "
                    "`KAKAO_REST_API_KEY`를 추가해 주세요.")
    query = query.strip()
    if len(query) < 2:
        return [], "검색어를 2글자 이상 입력해 주세요."
    try:
        results = search_places(query, api_key)
    except kakao.KakaoError as e:
        msg = str(e)
        if "HTTP 401" in msg or "HTTP 403" in msg:
            return [], "카카오 API 키가 유효하지 않거나 권한이 없습니다. 키 설정을 확인해 주세요."
        return [], f"장소 검색에 실패했습니다: {msg[:120]}"
    except Exception as e:                              # noqa: BLE001 - 네트워크 오류 포함
        return [], f"네트워크 오류로 장소를 검색하지 못했습니다: {str(e)[:120]}"
    if not results:
        return [], f"'{query}' 검색 결과가 없습니다. 다른 이름으로 검색해 보세요."
    return results, ""


@st.cache_data(show_spinner=False)
def load_listings() -> pd.DataFrame:
    return derive_common_features(pd.read_csv(os.path.join(DATA, "listings.csv")))


@st.cache_data(show_spinner=False, ttl=21600)
def load_one_gu_listings(gu_code: str) -> tuple[pd.DataFrame, dict]:
    """자치구 1곳의 최신 공공데이터 후보. 구 단위로 캐시한다.

    자치구 조합 전체를 캐시 키로 쓰면 구를 하나 추가할 때마다 이미 받아둔 구까지
    전부 다시 수집하게 된다. 구 단위로 끊어두면 새로 고른 구만 받아오면 된다.
    """
    fresh, stats = build_live_listings(load_settings(), [gu_code])
    return fresh, stats.as_dict()


def load_latest_public_listings(gu_codes: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """선택한 자치구들의 후보를 합친다. 파일은 쓰지 않는다."""
    frames, merged = [], {}
    for code in gu_codes:
        with st.spinner(f"{SEOUL_GU_NAMES.get(code, code)} 공공데이터를 불러오는 중..."):
            df, stats = load_one_gu_listings(code)
        frames.append(df)
        for k, v in stats.items():
            merged[k] = merged.get(k, 0) + v

    combined = pd.concat(frames, ignore_index=True)
    # 구별로 따로 만든 id 가 겹치므로 합친 뒤 다시 부여한다.
    combined["listing_id"] = [f"H{str(i + 1).zfill(4)}" for i in range(len(combined))]
    return derive_common_features(combined), merged


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_real_route(lat: float, lon: float, dest_lat: float, dest_lon: float,
                     mode: str, api_key: str, tmap_key: str = "") -> dict:
    """
    카카오 API로 실제 이동 경로 좌표를 가져온다.

    반환: {"path": [...], "status": ..., "reason": ...}
      real      실제 경로를 받았다
      walk      도보 경로를 못 받아 직선으로 그린다 (TMap 키 없음/거부/실패)
      no_route  API는 응답했지만 경로가 없다
      error     호출 실패 (권한/설정 문제일 가능성이 높다)
    왜 직선으로 나오는지 화면에서 확인할 수 있도록 이유를 삼키지 않고 돌려준다.
    """
    if not api_key and not tmap_key:
        return {"path": [], "status": "no_key", "reason": "경로 API 키 없음"}

    if mode == "walk":
        # 도보는 카카오에 길찾기가 없어 TMap 으로만 실측할 수 있다.
        if not tmap_key:
            return {"path": [], "status": "walk",
                    "reason": "TMAP_APP_KEY 가 없어 직선으로 표시합니다"}
        try:
            route = tmap.pedestrian_route(lat, lon, dest_lat, dest_lon, tmap_key)
            if route is not None and len(route.path) >= 2:
                return {"path": route.path, "status": "real", "reason": "TMap",
                        "minutes": route.total_time_sec / 60.0}
            return {"path": [], "status": "walk", "reason": "TMap 경로 결과 없음 — 직선 표시"}
        except Exception as e:                      # noqa: BLE001
            # 출발-도착이 너무 멀면 TMap 이 경로를 거부한다. 타임아웃 같은 네트워크
            # 오류까지 함께 잡아야 한다 — 여기서 새면 앱 전체가 죽는다.
            return {"path": [], "status": "walk", "reason": f"TMap: {str(e)[:120]}"}
    if not api_key:
        return {"path": [], "status": "no_key", "reason": "카카오 키 없음"}
    try:
        if mode == "drive":
            route = kakao.car_directions(lat, lon, dest_lat, dest_lon, api_key)
        else:
            route = kakao.transit_route(lat, lon, dest_lat, dest_lon, api_key)
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
# 위젯 key(selected_gu_widget)가 아니라 별도 키에 담아 둔다. 결과 화면으로 넘어가면
# 설정 화면의 위젯이 렌더링되지 않아 위젯 상태가 정리되는데, 그때 선택이 비어버리면
# 엉뚱하게 내장 목업 데이터로 폴백된다.
st.session_state.setdefault("gu_choice", ["강남구", "송파구", "마포구"])

listings = load_listings()
public_listing_stats: dict | None = None
if st.session_state.listing_source == "public":
    codes = tuple(c for c, name in SEOUL_GU_NAMES.items()
                  if name in st.session_state.gu_choice)
    if not codes:
        st.session_state.listing_source = "bundled"
    else:
        try:
            listings, public_listing_stats = load_latest_public_listings(codes)
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

    c3, c4 = st.columns(2)
    lat = c3.number_input("위도", value=float(base_d.lat), format="%.4f",
                          key=f"dlat__{mk}__{dk}")
    lon = c4.number_input("경도", value=float(base_d.lon), format="%.4f",
                          key=f"dlon__{mk}__{dk}")

    cap = st.number_input("⛔ 이동시간 상한 (분, 0 = 제한 없음)", min_value=0,
                          value=int(prev_cap), step=5, key=f"dcap__{mk}__{dk}")
    return Destination(dk, label, lat, lon, mode, weight, enabled), int(cap)


def render_setup_legacy() -> None:
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
        load_one_gu_listings.clear()
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
    # 경로 API는 항상 사용한다. 키가 없거나 호출이 실패하면 추정치로 폴백한다.
    use_api = True
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
    c_walk = r1[3].slider("역까지 도보 시간 상한 (분)", 1, 45, int(D("c_walk", 10)), key="c_walk")

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
# MOVA-style 4-step setup wizard
# =========================================================================== #
def render_setup() -> None:
    """Collect the same model inputs in a calm, reference-matched wizard."""
    saved = st.session_state.get("config", {})
    raw = saved.get("raw", {})
    draft = st.session_state.setdefault("setup_draft", {})
    step = int(st.session_state.setdefault("setup_step", 1))

    def val(key, fallback):
        return draft.get(key, raw.get(key, fallback))

    if "members" not in draft:
        draft["members"] = saved.get("members") or make_default_members(
            int(raw.get("n_members", 2)))
    if "places" not in draft:
        draft["places"] = dict(raw.get("places", {}))
    members: list[Member] = draft["members"]
    member_places: dict = draft["places"]
    settings = load_settings()
    kakao_ready = settings.has_kakao_key
    tmap_ready = bool(read_api_key("TMAP_APP_KEY"))
    public_data_ready = settings.has_data_go_kr_key and kakao_ready
    member_index = min(int(st.session_state.get("member_index", 0)), len(members) - 1)
    st.session_state.member_index = member_index

    intro = {
        1: ("함께 찾는 새로운 동네", "누구와 함께\n이동하나요?",
            "함께 이동할 구성원과 선호 조건을 알려주세요.\n모두가 만족할 장소를 더 정확하게 추천해 드려요."),
        2: ("우리 가족의 기준", "어떤 집을\n찾고 있나요?",
            "절대 포기할 수 없는 주거 조건을 선택해 주세요.\n후보가 적으면 중요도에 따라 안전하게 조정해요."),
        3: ("모두의 선호 반영", "무엇을 더\n중요하게 볼까요?",
            "집의 조건과 이동 시간 사이의 비중을 조절해 주세요.\n구성원별 중요도도 함께 반영할 수 있어요."),
        4: ("마지막 확인", "추천 방식을\n선택해 주세요",
            "계산 방식과 결과 개수를 확인하면 준비가 끝나요.\n입력 정보는 추천을 위해서만 안전하게 사용됩니다."),
    }
    badge, title, description = intro[step]
    progress = (33, 55, 78, 100)[step - 1]

    st.markdown(f"""
    <style>
      #MainMenu, header[data-testid="stHeader"], footer {{display:none!important}}
      .stApp {{background:#f7f9fc;color:#111827}}
      .block-container {{max-width:1540px;padding:0 42px 145px!important}}
      .mova-header {{height:112px;margin:0 -42px 72px;padding:0 42px;
        border-bottom:1px solid #26302d;display:flex;align-items:center;justify-content:space-between}}
      .mova-brand {{display:flex;align-items:center;gap:13px;font-size:22px;font-weight:850;color:#eef2ef}}
      .mova-logo {{width:46px;height:46px;border-radius:16px;background:#69e6ae;color:#123d32;
        display:grid;place-items:center;font-size:25px}}
      .mova-status {{display:flex;align-items:center;gap:21px;color:#99a29f;font-size:18px;font-weight:650}}
      .mova-status b {{color:#63dfa9;font-size:16px}} .track {{width:180px;height:7px;background:#293330;border-radius:99px;overflow:hidden}}
      .track i {{display:block;width:{progress}%;height:100%;background:#66e7b0;border-radius:99px}}
      .intro .badge {{display:inline-block;background:#edf4ff;color:#2563eb;border-radius:999px;padding:9px 16px;font-weight:800;font-size:16px}}
      .intro h1 {{white-space:pre-line;font-size:48px;line-height:1.16;letter-spacing:-2.8px;margin:28px 0;color:#0f172a}}
      .intro p {{white-space:pre-line;color:#728197;font-size:19px;line-height:1.75;letter-spacing:-.5px;margin:0 0 42px}}
      .member-label {{color:#65748a;font-weight:800;font-size:14px;margin-bottom:12px}}
      .member-tabs-pin {{display:none}}
      div[data-testid="stElementContainer"]:has(.member-tabs-pin) {{display:none!important}}
      div[data-testid="stElementContainer"]:has(.member-tabs-pin) + div[data-testid="stHorizontalBlock"] {{gap:6px;background:#f1f3f6;border-radius:18px;padding:6px}}
      div[data-testid="stElementContainer"]:has(.member-tabs-pin) + div[data-testid="stHorizontalBlock"] .stButton button {{height:52px;border-radius:13px;background:#fff}}
      div[data-testid="stElementContainer"]:has(.member-tabs-pin) + div[data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {{background:#2867e8;color:#fff}}
      .privacy {{border:1px solid #dbe4ef;border-radius:18px;padding:20px 22px;color:#718198;font-size:16px;line-height:1.55;margin-top:34px}}
      .data-source {{margin-top:14px;padding:15px 18px;border-radius:16px;background:#eef8f4;color:#45665c;font-size:14px;line-height:1.5}}
      .data-source b {{display:block;color:#163f34;font-size:15px;margin-bottom:2px}}
      .api-grid {{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px}}
      .api-chip {{padding:7px 11px;border-radius:999px;background:#eef8f4;color:#25765e;font-size:13px;font-weight:750}}
      .api-chip.off {{background:#f3f4f6;color:#87918e}}
      div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{background:#fff;border:1px solid #dbe4ef!important;border-radius:28px!important;
        box-shadow:0 24px 55px rgba(30,53,74,.10);padding:22px 30px 26px}}
      .card-head {{display:flex;align-items:center;gap:16px;margin:2px 0 28px}}
      .step-dot {{width:50px;height:50px;border-radius:50%;background:#2867e8;color:#fff;display:grid;place-items:center;font-size:19px;font-weight:800;flex:none}}
      .card-head h2 {{margin:0;color:#101827;font-size:28px;letter-spacing:-1px}} .card-head p {{margin:3px 0 0;color:#8190a5;font-size:16px}}
      .required {{margin-left:auto;border:1px solid #b8d6ff;color:#2870ee;background:#f4f8ff;border-radius:999px;padding:8px 14px;font-weight:800}}
      div[data-testid="stTextInputRootElement"], div[data-baseweb="base-input"] {{height:64px!important;min-height:64px!important}}
      .stTextInput input, .stNumberInput input {{height:64px!important;min-height:64px!important;border:1px solid #d9e2ed;border-radius:17px;color:#152033;font-size:17px;padding:0 20px;background:#fff}}
      div[data-baseweb="select"]>div {{min-height:64px;border-color:#d9e2ed!important;border-radius:17px!important;font-size:17px;background:#fff}}
      div[role="radiogroup"] {{gap:13px}} div[role="radiogroup"] label {{border:1px solid #d9e2ed;border-radius:17px;padding:17px 22px;min-height:62px;flex:1;justify-content:center}}
      div[role="radiogroup"] label:has(input:checked) {{background:#2867e8;color:white;border-color:#2867e8}}
      div[role="radiogroup"] label:has(input:checked) p {{color:white!important}}
      .stButton button {{height:57px;border-radius:16px;font-weight:800;font-size:17px;border-color:#d6e0ec}}
      .stButton button[kind="primary"] {{background:#2867e8;border-color:#2867e8}}
      .counter {{color:#728197;font-size:17px;padding-top:16px}}
      .footer-pin {{display:none}}
      div[data-testid="stElementContainer"]:has(.footer-pin) {{display:none!important}}
      div[data-testid="stElementContainer"]:has(.footer-pin) + div[data-testid="stHorizontalBlock"] {{position:fixed;z-index:999;left:0;right:0;bottom:0;background:#f7f9fc;
        border-top:1px solid #26302d;padding:25px max(42px,calc((100vw - 1540px)/2 + 42px));box-shadow:0 -8px 24px rgba(30,53,74,.03)}}
      div[data-testid="stElementContainer"]:has(.footer-pin) + div[data-testid="stHorizontalBlock"] .stButton button[kind="primary"] {{background:#69e6ae;border-color:#69e6ae;color:#102d25}}
      @media(max-height:950px) and (min-width:801px) {{
        .block-container {{padding-bottom:105px!important}}
        .mova-header {{height:88px;margin-bottom:38px}}
        .intro h1 {{font-size:42px;margin:22px 0}}
        .intro p {{font-size:17px;line-height:1.65;margin-bottom:28px}}
        .privacy {{margin-top:20px;padding:16px 18px;font-size:14px}}
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{padding:16px 24px 18px}}
        .card-head {{margin-bottom:18px}} .card-head h2 {{font-size:26px}}
        .step-dot {{width:46px;height:46px}}
        div[data-testid="stTextInputRootElement"], div[data-baseweb="base-input"],
        .stTextInput input, .stNumberInput input {{height:54px!important;min-height:54px!important}}
        div[data-baseweb="select"]>div {{min-height:54px}}
        div[role="radiogroup"] label {{min-height:54px;padding:13px 18px}}
        div[data-testid="stElementContainer"]:has(.footer-pin) + div[data-testid="stHorizontalBlock"] {{padding-top:16px;padding-bottom:16px}}
      }}
      @media(max-width:800px) {{.block-container{{padding:0 20px 130px!important}}.mova-header{{height:88px;margin-bottom:35px;padding:0 20px}}
        .mova-status span{{display:none}}.track{{width:88px}}.intro h1{{font-size:38px}}}}
    </style>
    <div class="mova-header"><div class="mova-brand"><span class="mova-logo">⌖</span>MOVA</div>
      <div class="mova-status"><b>STEP {step:02d}</b><span>|</span><span>{progress}% prepared</span><div class="track"><i></i></div></div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([.86, 1.55], gap="large")
    with left:
        st.markdown(f'<section class="intro"><span class="badge">{badge}</span><h1>{title}</h1><p>{description}</p></section>', unsafe_allow_html=True)

    save_current_member = None
    if step == 1:
        pending_index = None
        with left:
            st.markdown('<div class="member-label">MEMBERS</div>', unsafe_allow_html=True)
            st.markdown('<div class="member-tabs-pin"></div>', unsafe_allow_html=True)
            tabs = st.columns([1] * len(members) + [.34])
            for i, member in enumerate(members):
                if tabs[i].button(("• " if i == member_index else "") + member.name,
                                  key=f"member_tab_{i}", use_container_width=True,
                                  type="primary" if i == member_index else "secondary"):
                    pending_index = i
            if tabs[-1].button("＋", key="add_member", use_container_width=True):
                if len(members) < MAX_MEMBERS:
                    members.append(make_default_members(len(members) + 1)[-1])
                    pending_index = len(members) - 1
            st.markdown('<div class="privacy">🛡️ &nbsp; 입력한 위치 정보는 추천을 위해서만 안전하게 사용됩니다.</div>', unsafe_allow_html=True)
            def _on_gu_change() -> None:
                """구를 고르면 곧바로 공공데이터로 전환한다(별도 버튼 없이).

                위젯 값을 gu_choice 로 옮겨 둬야 결과 화면으로 넘어가 위젯이 사라져도
                선택이 유지된다.
                """
                st.session_state.gu_choice = list(st.session_state.selected_gu_widget)
                st.session_state.listing_source = "public" if st.session_state.gu_choice else "bundled"
                st.session_state.listing_source_error = ""

            gu_names = st.multiselect(
                "후보를 찾을 자치구 (서울 25개구)", list(SEOUL_GU_NAMES.values()),
                default=st.session_state.gu_choice, key="selected_gu_widget",
                on_change=_on_gu_change, disabled=not public_data_ready,
                help="고른 구의 최신 실거래 매물을 전부 가져옵니다. 선택하면 바로 반영됩니다.")
            if not public_data_ready:
                st.caption("DATA_GO_KR_SERVICE_KEY와 KAKAO_REST_API_KEY가 필요합니다.")
            elif not gu_names:
                st.caption(f"자치구를 고르면 해당 구의 최신 실거래 매물을 불러옵니다. (현재 내장 후보 {len(listings)}건)")
            else:
                st.caption(f"{'·'.join(gu_names)} 최신 공공데이터 {len(listings)}건 사용 중"
                           + (" · 구가 많으면 첫 수집에 1~2분 걸릴 수 있습니다." if len(gu_names) >= 5 else ""))
            if not public_data_ready:
                st.caption("DATA_GO_KR_SERVICE_KEY와 KAKAO_REST_API_KEY가 필요합니다.")
            elif st.session_state.listing_source_error:
                st.warning(f"공공데이터 갱신 실패: {st.session_state.listing_source_error}")

        current = members[member_index]
        destination = current.destinations[0]
        with right.container(border=True):
            st.markdown(f'<div class="card-head"><div class="step-dot">{member_index + 1}</div><div><h2>{current.name} 정보</h2><p>기본 이동 조건을 입력해 주세요</p></div><span class="required">필수 항목</span></div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            name = a.text_input("구성원 이름", value=current.name, key=f"wiz_name_{current.key}", placeholder="이름을 입력하세요")
            destination_label = b.text_input("목적지", value=destination.label, key=f"wiz_dest_{current.key}", placeholder="장소 또는 주소 검색")
            st.markdown("<br>", unsafe_allow_html=True)
            mode_keys = list(TRAVEL_MODES)
            mode = st.radio("이동 수단", mode_keys, index=mode_keys.index(destination.mode), key=f"wiz_mode_{current.key}", horizontal=True,
                            format_func=lambda x: {"transit":"🚇  대중교통", "drive":"🚗  자동차", "walk":"🚶  도보"}[x])
            selected_place = member_places.get(current.key)
            if selected_place:
                st.success(f"**목적지: {selected_place['place_name']}**  \n"
                           f"주소: {selected_place['address']}  \n선택 완료 ✓")
                if st.button("다시 검색", key=f"wiz_replace_{current.key}"):
                    member_places.pop(current.key, None)
                    st.session_state.pop(f"wiz_results_{current.key}", None)
                    st.rerun()
            else:
                sc1, sc2 = st.columns([3.4, 1])
                search_query = sc1.text_input(
                    "목적지 위치 검색", key=f"wiz_query_{current.key}",
                    placeholder="건물/장소 이름을 입력하세요 (예: 고려대학교)")
                if sc2.button("🔍 검색", key=f"wiz_search_{current.key}", use_container_width=True):
                    results, err = run_place_search(search_query)
                    st.session_state[f"wiz_results_{current.key}"] = {"items": results, "error": err}
                found = st.session_state.get(f"wiz_results_{current.key}")
                if found is None:
                    st.caption("아직 목적지가 선택되지 않았습니다. 장소 이름으로 검색해 주세요.")
                elif found["error"]:
                    st.warning(found["error"])
                else:
                    for j, p in enumerate(found["items"]):
                        rc1, rc2 = st.columns([4.2, 1])
                        addr_lines = [ln for ln in
                                      (p["road_address_name"],
                                       f"(지번) {p['address_name']}" if p["address_name"] else "")
                                      if ln]
                        rc1.markdown(
                            f"**{p['place_name']}**<br>"
                            f"<span style='color:#728197;font-size:14px'>{'<br>'.join(addr_lines)}</span>",
                            unsafe_allow_html=True)
                        if rc2.button("선택", key=f"wiz_pick_{current.key}_{j}",
                                      use_container_width=True):
                            member_places[current.key] = {
                                "place_name": p["place_name"],
                                "address": p["road_address_name"] or p["address_name"],
                                "latitude": p["latitude"],
                                "longitude": p["longitude"],
                            }
                            st.session_state.pop(f"wiz_results_{current.key}", None)
                            st.rerun()
            card_count, card_prev, card_next = st.columns([4, .85, .85])
            card_count.markdown(f'<div class="counter">{member_index + 1} / {len(members)}</div>', unsafe_allow_html=True)
            if card_prev.button("이전", key="previous_member", disabled=member_index == 0, use_container_width=True):
                pending_index = member_index - 1
            if card_next.button("다음", key="next_member", disabled=member_index == len(members) - 1,
                                type="primary", use_container_width=True):
                pending_index = member_index + 1

        def _save_member():
            old = members[member_index]
            sel = member_places.get(old.key)
            if sel:
                lat, lon = float(sel["latitude"]), float(sel["longitude"])
            else:
                lat, lon = destination.lat, destination.lon
            updated_destination = Destination(destination.key, destination_label.strip() or destination.label,
                                              lat, lon, mode, destination.weight, True)
            members[member_index] = Member(old.key, name.strip() or old.name, old.color,
                                           [updated_destination] + old.destinations[1:], old.priority,
                                           old.min_satisfaction, old.active)
            draft["members"] = members
        save_current_member = _save_member
        if pending_index is not None and pending_index != member_index:
            _save_member()
            st.session_state.member_index = pending_index
            st.rerun()

    elif step == 2:
        with right.container(border=True):
            st.markdown('<div class="card-head"><div class="step-dot">2</div><div><h2>필수 주거 조건</h2><p>타협할 수 없는 기준을 선택해 주세요</p></div><span class="required">필수 항목</span></div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            c_rooms = a.slider("최소 방 개수", 1, 7, int(val("c_rooms", 3)))
            c_area = b.slider("최소 전용면적 (m²)", 29, 200, int(val("c_area", 60)))
            lock_rooms = a.checkbox("방 개수는 절대 사수", value=bool(val("lock_rooms", True)))
            lock_area = b.checkbox("면적은 절대 사수", value=bool(val("lock_area", False)))
            a, b = st.columns(2)
            c_park = a.slider("최소 주차 가능 대수", 0.0, 3.0, float(val("c_park", .5)), .05)
            c_walk = b.slider("역까지 도보 상한 (분)", 1, 45, int(val("c_walk", 10)))
            a, b = st.columns(2)
            c_age = a.slider("최대 준공 경과년수", 0, 46, int(val("c_age", 30)))
            c_noise = b.slider("최대 주간 소음도 (dB)", 33, 74, int(val("c_noise", 74)))
            allowed_gu = st.multiselect("지역 한정 (미선택 시 전체)", sorted(listings["gu"].unique()), default=val("allowed_gu", []))
            allowed_or = st.multiselect("허용할 향", list(ORIENTATION_SCORE), default=val("allowed_or", list(ORIENTATION_SCORE)))

    elif step == 3:
        with right.container(border=True):
            st.markdown('<div class="card-head"><div class="step-dot">3</div><div><h2>선호 중요도</h2><p>추천에 반영할 비중을 조정해 주세요</p></div></div>', unsafe_allow_html=True)
            alpha = st.slider("🏠 집 자체  ← 중요도 →  이동 시간 🚶", 0.0, 1.0, float(val("alpha", .5)), .05)
            st.caption(f"주택 조건 {alpha:.0%}  ·  이동 시간 {1-alpha:.0%}")
            weights = {}
            weight_cols = st.columns(2)
            saved_weights = draft.get("common_weights", saved.get("common_weights", {}))
            for i, feature in enumerate(COMMON_FEATURES):
                weights[feature.key] = weight_cols[i % 2].slider(feature.axis, 0.0, 2.0,
                    float(saved_weights.get(feature.key, feature.default_weight)), .1, key=f"wizard_weight_{feature.key}")
            priorities, floors = {}, {}
            st.markdown("#### 구성원별 중요도")
            member_cols = st.columns(2)
            for i, member in enumerate(members):
                priorities[member.key] = member_cols[i % 2].slider(f"{member.name} 발언권", 0.0, 3.0, float(member.priority), .1, key=f"priority_{member.key}")
                floors[member.key] = member_cols[i % 2].slider(f"{member.name} 최소 만족도", 0, 90, int(member.min_satisfaction), 5, key=f"floor_{member.key}")

    else:
        with right.container(border=True):
            st.markdown('<div class="card-head"><div class="step-dot">4</div><div><h2>추천 설정</h2><p>결과 방식을 확인하면 준비가 완료됩니다</p></div></div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            k = a.slider("추천 매물 수", 3, 30, int(val("k", 10)))
            min_candidates = b.slider("최소 확보 후보", 1, 30, int(val("min_candidates", 5)))
            sort_mode = st.radio("정렬 기준", SORT_MODES, index=SORT_MODES.index(val("sort_mode", SORT_MODES[0])), horizontal=True)
            # 경로 API는 사용자가 끄지 않아도 되도록 항상 활성화한다.
            use_api = True
            st.markdown(
                '<div class="api-grid">'
                f'<span class="api-chip{"" if kakao_ready else " off"}">Kakao 자동차·대중교통 {"연결" if kakao_ready else "키 필요"}</span>'
                f'<span class="api-chip{"" if tmap_ready else " off"}">TMAP 도보 {"연결" if tmap_ready else "키 필요"}</span>'
                '</div>', unsafe_allow_html=True)
            scaler_kind = st.selectbox("계산 스케일러", ["robust", "minmax", "standard"], index=["robust", "minmax", "standard"].index(val("scaler", "robust")), format_func=lambda x: {"robust":"RobustScaler (권장)","minmax":"MinMax","standard":"Standard"}[x])
            clip_q = st.slider("이상치 클리핑 (%)", 0.0, 5.0, float(val("clip_q", .01))*100, .5) / 100
            st.caption(f"현재 {'최신 공공' if st.session_state.listing_source == 'public' else '내장'} 후보 {len(listings)}건을 사용합니다.")

    st.markdown('<div class="footer-pin"></div>', unsafe_allow_html=True)
    counter, spacer, prev_col, next_col = st.columns([1, 5, 1, 1.25])
    counter.markdown(f"**{step}** &nbsp; / &nbsp; 4")
    back = prev_col.button("←  이전", disabled=step == 1, use_container_width=True)
    onward = next_col.button("맞춤 추천 보기  →" if step == 4 else "다음  →", type="primary", use_container_width=True)

    if step == 1 and onward:
        save_current_member()
        draft["n_members"] = len(members)
        missing = [m.name for m in members if m.key not in member_places]
        if missing:
            st.error("목적지를 아직 선택하지 않은 구성원이 있습니다: "
                     + ", ".join(missing) + " — 장소를 검색해 선택해 주세요.")
            onward = False
    elif step == 2 and (onward or back):
        draft.update(c_rooms=c_rooms, c_area=c_area, lock_rooms=lock_rooms, lock_area=lock_area,
                     c_park=c_park, c_walk=c_walk, c_age=c_age, c_noise=c_noise,
                     allowed_gu=allowed_gu, allowed_or=allowed_or)
    elif step == 3 and (onward or back):
        draft.update(alpha=alpha, common_weights=weights)
        for member in members:
            member.priority = priorities[member.key]
            member.min_satisfaction = float(floors[member.key])
    elif step == 4 and (onward or back):
        draft.update(k=k, min_candidates=min_candidates, sort_mode=sort_mode, use_api=use_api,
                     scaler=scaler_kind, clip_q=clip_q)

    if back:
        st.session_state.setup_step = step - 1
        st.rerun()
    if onward and step < 4:
        st.session_state.setup_step = step + 1
        st.rerun()
    if onward and step == 4:
        constraints = [
            Constraint("rooms", "min", val("c_rooms", 3), priority=0, locked=val("lock_rooms", True), label="방 개수"),
            Constraint("area_m2", "min", val("c_area", 60), priority=1, locked=val("lock_area", False), label="전용면적"),
            Constraint("parking_slots", "min", val("c_park", .5), priority=2, label="주차 가능 대수"),
            Constraint("transit_walk_min", "max", val("c_walk", 10), priority=2, label="역까지 도보"),
            Constraint("building_age", "max", val("c_age", 30), priority=3, label="준공 경과년수"),
        ]
        if val("c_noise", 74) < 74:
            constraints.append(Constraint("noise_db", "max", val("c_noise", 74), priority=3, label="소음도"))
        categories = []
        orientations = val("allowed_or", list(ORIENTATION_SCORE))
        if len(orientations) < len(ORIENTATION_SCORE):
            categories.append(CategoryConstraint("orientation", orientations, priority=2, label="향"))
        if val("allowed_gu", []):
            categories.append(CategoryConstraint("gu", val("allowed_gu", []), priority=1, label="지역"))
        common_weights = draft.get("common_weights", {f.key: f.default_weight for f in COMMON_FEATURES})
        satisficing = {d.column(m.key): 25.0 for m in members for d in m.enabled_destinations}
        st.session_state.config = {
            "members": members, "constraints": constraints, "categories": categories,
            "min_candidates": int(val("min_candidates", 5)), "alpha": float(val("alpha", .5)),
            "common_weights": common_weights, "satisficing": satisficing,
            "scaler": val("scaler", "robust"), "clip_q": float(val("clip_q", .01)),
            "k": int(val("k", 10)), "sort_mode": val("sort_mode", SORT_MODES[0]),
            "use_api": True, "raw": dict(draft),
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

    # 전체 화면 지도 위에 왼쪽 추천 점수 오버레이를 표시한다.
    # 상세 진단과 표/차트는 지도 안의 [상세 정보 보기] 버튼으로 연다.
    st.markdown("""
        <style>
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        section[data-testid="stMain"] { overflow: hidden; }
        iframe[title$="kakao_map"] { position: fixed; inset: 0; z-index: 10;
                                     width: 100vw !important;
                                     height: 100vh !important; border: 0; }
        [data-stale="true"], .stale-element { opacity: 0.9 !important; }
        </style>""", unsafe_allow_html=True)

    map_sel = st.session_state.get("map_sel")
    if map_sel not in top.index:
        map_sel = top.index[0]
    r_sel = top.loc[map_sel]

    routes: dict[tuple[str, str], dict] = {}
    settings = load_settings()
    tmap_key = read_api_key("TMAP_APP_KEY")
    if settings.has_kakao_key or tmap_key:
        with st.spinner("실제 동선 조회 중..."):
            for m in active:
                for dst in m.enabled_destinations:
                    routes[(m.key, dst.key)] = fetch_real_route(
                        float(r_sel["lat"]), float(r_sel["lon"]),
                        float(dst.lat), float(dst.lon), dst.mode,
                        settings.kakao_rest_api_key, tmap_key)

    # 카카오 JS 키가 있으면 카카오맵, 없으면 Leaflet(OSM)을 사용한다.
    # 어느 쪽이든 왼쪽 추천·점수 오버레이 UI는 동일하다.
    js_key = settings.kakao_javascript_key if settings.has_kakao_js_key else ""
    act = viz_kakao.render_map_view(top, active, map_sel, routes, js_key,
                                    height=640, key="map_view")
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


@st.dialog("📊 상세 분석", width="large")
def detail_dialog(rec, top, active, d, conf, map_sel: str, js_key: str) -> None:
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
