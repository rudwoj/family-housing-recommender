"""
시각화 모듈 (Plotly + Folium).

  - 구성원 만족도 레이더 / 피처 프로파일 레이더
  - 이해충돌 트레이드오프 산점도 + 파레토 프론티어
  - 거리 기여도 분해 막대 (왜 이 매물인가)
  - 가중치 구성 막대
  - 구성원별 이동 동선 지도 (folium 우선, 미설치 시 plotly 대체)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .model import label_of
from .schema import COMMON_KEYS, TRAVEL_MODES, Member

try:  # folium 은 선택 의존성
    import folium
    HAS_FOLIUM = True
except Exception:  # pragma: no cover
    HAS_FOLIUM = False

# plotly 5.24+ 의 MapLibre 기반 트레이스. 구형 Scattermapbox 는 plotly 6.x 에서
# Mapbox 토큰을 요구하므로(타일에 "API KEY REQUIRED" 표시) 가능하면 이쪽을 쓴다.
# 베이스맵은 API 키가 필요 없는 Carto 계열을 쓴다.
#   ("open-street-map" 은 plotly 6 에서 MapTiler 경유라 키를 요구한다)
HAS_SCATTERMAP = hasattr(go, "Scattermap")
MapTrace = go.Scattermap if HAS_SCATTERMAP else go.Scattermapbox
MAP_LAYOUT_KEY = "map" if HAS_SCATTERMAP else "mapbox"
BASEMAP_STYLE = "carto-positron"

GRID = dict(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
LAYOUT = dict(margin=dict(l=10, r=10, t=44, b=10), template="plotly_white",
              font=dict(family="Apple SD Gothic Neo, Malgun Gothic, sans-serif", size=12))


# --------------------------------------------------------------------------- #
# 1. 구성원 만족도 레이더 (이해충돌 균형점 시각화)
# --------------------------------------------------------------------------- #
def member_radar(rec, listing_ids: list[str], members: list[Member]) -> go.Figure:
    axes, keys = [], []
    if "sat__공통" in rec.scored.columns:
        axes.append("공통(주거)"); keys.append("sat__공통")
    for m in members:
        col = f"sat__{m.key}"
        if col in rec.scored.columns:
            axes.append(m.name); keys.append(col)

    fig = go.Figure()
    for lid in listing_ids:
        row = rec.scored.loc[lid]
        vals = [float(row[k]) for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=axes + axes[:1], fill="toself",
            name=f"{row['name']}", opacity=0.55,
            hovertemplate="%{theta}: %{r:.1f}점<extra>" + str(row["name"]) + "</extra>"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="구성원별 만족도 균형 (100 = 개인 이상점)",
        legend=dict(orientation="h", y=-0.12), **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 2. 피처 프로파일 레이더 (스케일 공간, 이상점 대비)
# --------------------------------------------------------------------------- #
def feature_radar(rec, listing_ids: list[str], columns: list[str], title: str) -> go.Figure:
    """만족도(효용) 공간 프로파일. 이상점은 모든 축이 1 이다."""
    axes = [label_of(c) for c in columns]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(columns) + 1), theta=axes + axes[:1], name="가족 이상점",
        line=dict(dash="dot", color="#111827"), fill="toself", opacity=0.10))
    for lid in listing_ids:
        v = [float(rec.utility.loc[lid, c]) for c in columns]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself",
                                      opacity=0.45, name=str(rec.scored.loc[lid, "name"]),
                                      hovertemplate="%{theta}: 만족도 %{r:.2f}<extra></extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title=title, legend=dict(orientation="h", y=-0.12), **LAYOUT)
    return fig


def utility_curve_fig() -> go.Figure:
    """비선형 효용 곡선 설명용 차트."""
    import numpy as np

    from .preprocess import utility
    from .schema import FEATURE_BY_KEY

    t = np.linspace(0, 1, 101)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=1 - t, name="선형 (기존 방식)",
                             line=dict(color="#94a3b8", dash="dash")))
    f_time = FEATURE_BY_KEY["transit_walk_min"]
    fig.add_trace(go.Scatter(x=t, y=utility(t, f_time), name="이동시간 (볼록 · 불만 가속)",
                             line=dict(color="#dc2626", width=3)))
    f_area = FEATURE_BY_KEY["area_m2"]
    fig.add_trace(go.Scatter(x=t, y=utility(t, f_area), name="면적 (오목 · 체감 효용 감소)",
                             line=dict(color="#2563eb", width=3)))
    fig.update_layout(title="비선형 효용 변환 — 같은 1단위 변화라도 구간마다 체감이 다르다",
                      xaxis=dict(title="정규화된 피처 위치 (0=최소, 1=최대)", **GRID),
                      yaxis=dict(title="만족도", **GRID), height=380,
                      legend=dict(orientation="h", y=-0.25), **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 3. 트레이드오프 산점도 + 파레토 프론티어
# --------------------------------------------------------------------------- #
def tradeoff_scatter(rec, x_key: str, y_key: str, x_name: str, y_name: str,
                     highlight: list[str] | None = None) -> go.Figure:
    d = rec.scored
    xs, ys = f"sat__{x_key}", f"sat__{y_key}"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=d[xs], y=d[ys], mode="markers", name="후보 매물",
        marker=dict(size=7, color="rgba(148,163,184,0.55)"),
        text=d["name"], customdata=np.stack([d.index, d["fit_score"]], axis=-1),
        hovertemplate="%{text}<br>" + x_name + " %{x:.1f} / " + y_name +
                      " %{y:.1f}<br>종합 %{customdata[1]:.1f}점<extra></extra>"))

    # 선택한 두 구성원 평면에서의 2차원 파레토 프론티어(계단형)
    from .features import pareto_mask
    pair = d[pareto_mask(d[[xs, ys]].to_numpy(dtype=float))].sort_values(xs)
    fig.add_trace(go.Scatter(
        x=pair[xs], y=pair[ys], mode="lines+markers", name="2인 파레토 프론티어",
        line=dict(color="#f59e0b", width=2.5, shape="hv"),
        marker=dict(size=10, color="#f59e0b", symbol="diamond"), text=pair["name"],
        hovertemplate="⭐ %{text}<br>" + x_name + " %{x:.1f} / " + y_name +
                      " %{y:.1f}<extra>2인 파레토</extra>"))

    # 전 구성원(공통 포함) 기준 파레토 최적 — 다차원이라 2D 평면에서는 흩어져 보인다
    par = d[d["pareto"] & ~d.index.isin(pair.index)]
    fig.add_trace(go.Scatter(
        x=par[xs], y=par[ys], mode="markers", name="전체 구성원 기준 파레토",
        marker=dict(size=8, color="rgba(245,158,11,0.45)", symbol="diamond-open",
                    line=dict(width=1.5, color="#f59e0b")), text=par["name"],
        hovertemplate="◇ %{text}<extra>다차원 파레토</extra>"))

    if highlight:
        h = d.loc[[i for i in highlight if i in d.index]]
        fig.add_trace(go.Scatter(
            x=h[xs], y=h[ys], mode="markers", name="추천 Top-K",
            marker=dict(size=13, color="#2563eb", symbol="star",
                        line=dict(width=1, color="white")),
            text=h["name"], hovertemplate="⬤ %{text}<extra>추천</extra>"))

    lo = float(min(d[xs].min(), d[ys].min())) - 2
    fig.add_shape(type="line", x0=lo, y0=lo, x1=100, y1=100,
                  line=dict(color="rgba(15,23,42,0.25)", dash="dot"))
    fig.add_annotation(x=99, y=lo + 3, text="↘ 아래쪽 = " + x_name + " 우세",
                       showarrow=False, font=dict(size=10, color="#64748b"), xanchor="right")
    fig.update_layout(title=f"이해충돌 트레이드오프: {x_name} ↔ {y_name}",
                      xaxis=dict(title=f"{x_name} 만족도", **GRID),
                      yaxis=dict(title=f"{y_name} 만족도", **GRID),
                      legend=dict(orientation="h", y=-0.18), height=520, **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 4. 거리 기여도 분해
# --------------------------------------------------------------------------- #
def contribution_bar(rec, listing_id: str, top_n: int = 10) -> go.Figure:
    s = rec.contributions.loc[listing_id].sort_values(ascending=False).head(top_n)[::-1]
    owners = ["공통" if c in COMMON_KEYS else c.split("__", 1)[0] for c in s.index]
    palette = {o: px.colors.qualitative.Set2[i % 8] for i, o in enumerate(dict.fromkeys(owners))}
    fig = go.Figure(go.Bar(
        x=(s.values * 100), y=[label_of(c) for c in s.index], orientation="h",
        marker_color=[palette[o] for o in owners],
        hovertemplate="%{y}: 이상점과의 거리 중 %{x:.1f}% 기여<extra></extra>"))
    fig.update_layout(title="이상점과 벌어진 원인 분해 (기여도 %)",
                      xaxis=dict(title="거리 기여율 (%)", **GRID), height=380, **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 5. 가중치 구성
# --------------------------------------------------------------------------- #
def weight_bar(rec) -> go.Figure:
    w = rec.weights.sort_values(ascending=True)
    owners = ["공통" if c in COMMON_KEYS else c.split("__", 1)[0] for c in w.index]
    palette = {o: px.colors.qualitative.Set2[i % 8] for i, o in enumerate(dict.fromkeys(owners))}
    fig = go.Figure(go.Bar(x=w.values * 100, y=[label_of(c) for c in w.index], orientation="h",
                           marker_color=[palette[o] for o in owners],
                           hovertemplate="%{y}: ω = %{x:.2f}%<extra></extra>"))
    fig.update_layout(title="최종 피처 가중치 ω (합계 100%)",
                      xaxis=dict(title="가중치 (%)", **GRID),
                      height=max(320, 22 * len(w)), **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 6. 군집 프로파일
# --------------------------------------------------------------------------- #
def cluster_profile_fig(profile: pd.DataFrame) -> go.Figure:
    if profile.empty:
        return go.Figure()
    cols = [c for c in ["area_m2", "green_ratio", "parking_per_unit", "building_age",
                        "travel_mean_min", "travel_std_min", "stability_index"] if c in profile.columns]
    norm = profile[cols].copy()
    for c in cols:
        lo, hi = norm[c].min(), norm[c].max()
        norm[c] = 0.5 if hi == lo else (norm[c] - lo) / (hi - lo)
    fig = go.Figure()
    for cid in norm.index:
        v = norm.loc[cid].tolist()
        fig.add_trace(go.Scatterpolar(
            r=v + v[:1], theta=[label_of(c) for c in cols] + [label_of(cols[0])],
            fill="toself", opacity=0.45,
            name=f"군집 {cid} ({int(profile.loc[cid, '매물수'])}건, {profile.loc[cid, '대표지역']})"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title="추천 매물 군집별 입지·환경 성격 (군집 내 상대 비교)",
                      legend=dict(orientation="h", y=-0.15), **LAYOUT)
    return fig


# --------------------------------------------------------------------------- #
# 7. 지도 (구성원 동선 마커 + 연결선)
# --------------------------------------------------------------------------- #
def build_map(top: pd.DataFrame, members: list[Member], selected_id: str | None = None,
              routes: dict[tuple[str, str], dict] | None = None):
    """추천 매물 위치 + 선택 매물에서 각 구성원 목적지까지의 동선.

    routes: {(member.key, destination.key): {"path": [(lat, lon), ...], "mode": "drive"|"transit"}}
    카카오 실제 길찾기로 얻은 경로. 없거나 좌표가 2점 미만이면 출발-도착 직선으로 폴백한다
    (API 미설정/실패를 이렇게 시각적으로 구분해서 보여준다).
    """
    center = [float(top["lat"].mean()), float(top["lon"].mean())]
    sel = selected_id if selected_id in top.index else top.index[0]
    r = top.loc[sel]
    routes = routes or {}

    def route_for(mem, d):
        info = routes.get((mem.key, d.key)) or {}
        path = info.get("path") or []
        if len(path) >= 2:
            return path, True
        return [(r["lat"], r["lon"]), (d.lat, d.lon)], False

    if HAS_FOLIUM:
        m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
        for mem in members:
            for d in mem.enabled_destinations:
                folium.Marker([d.lat, d.lon], tooltip=f"{mem.name} · {d.label}",
                              icon=folium.Icon(color="gray", icon="flag", prefix="fa")).add_to(m)
                path, is_real = route_for(mem, d)
                folium.PolyLine([list(p) for p in path],
                                color=mem.color, weight=4 if is_real else 3,
                                opacity=0.8 if is_real else 0.5,
                                dashArray=None if is_real else "6,6",
                                tooltip=f"{mem.name} → {d.label}"
                                        + ("" if is_real else " (실제 경로 조회 실패 · 직선 추정)")
                                ).add_to(m)
        for lid, row in top.iterrows():
            is_sel = lid == sel
            folium.CircleMarker([row["lat"], row["lon"]], radius=9 if is_sel else 6,
                                color="#2563eb" if is_sel else "#94a3b8", fill=True,
                                fill_opacity=0.9 if is_sel else 0.6,
                                tooltip=f"{int(row['rank'])}위 · {row['name']}").add_to(m)
        return "folium", m

    fig = go.Figure()
    fig.add_trace(MapTrace(lat=top["lat"], lon=top["lon"], mode="markers",
                           marker=dict(size=[26 if i == sel else 20 for i in top.index],
                                       color="white"),
                           hoverinfo="skip", showlegend=False))
    fig.add_trace(MapTrace(
        lat=top["lat"], lon=top["lon"], mode="markers+text",
        marker=dict(size=[22 if i == sel else 16 for i in top.index],
                    color=["#2563eb" if i == sel else "#475569" for i in top.index]),
        text=[str(int(v)) for v in top["rank"]], textposition="middle center",
        textfont=dict(color="white", size=10),
        customdata=np.stack([top["name"], top["fit_score"]], axis=-1),
        hovertemplate="%{customdata[0]}<br>종합 %{customdata[1]:.1f}점<extra></extra>",
        name="추천 매물"))

    for mem in members:
        for d in mem.enabled_destinations:
            minutes = r.get(d.column(mem.key), float("nan"))
            path, is_real = route_for(mem, d)
            lats = [p[0] for p in path]
            lons = [p[1] for p in path]
            tag = "" if is_real else " · 직선 추정"
            fig.add_trace(MapTrace(
                lat=lats, lon=lons, mode="lines+markers",
                line=dict(width=4 if is_real else 3, color=mem.color),
                marker=dict(size=[0] * (len(path) - 1) + [14], color=mem.color),
                name=f"{mem.name} → {d.label} ({TRAVEL_MODES[d.mode]} {minutes:.0f}분{tag})",
                hovertemplate=f"{mem.name} → {d.label}{tag}<extra></extra>"))

    fig.update_layout(**{MAP_LAYOUT_KEY: dict(style=BASEMAP_STYLE,
                                              center=dict(lat=center[0], lon=center[1]), zoom=10)},
                      margin=dict(l=0, r=0, t=30, b=0), height=560,
                      legend=dict(orientation="h", y=-0.05),
                      title=f"선택 매물 기준 구성원 동선 — {r['name']}")
    return "plotly", fig
