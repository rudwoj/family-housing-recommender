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
from .schema import COMMON_KEYS, Member

try:  # folium 은 선택 의존성
    import folium
    HAS_FOLIUM = True
except Exception:  # pragma: no cover
    HAS_FOLIUM = False

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
    axes = [label_of(c) for c in columns]
    fig = go.Figure()
    ideal = [float(rec.ideal_scaled[c]) for c in columns]
    fig.add_trace(go.Scatterpolar(r=ideal + ideal[:1], theta=axes + axes[:1],
                                  name="가족 이상점", line=dict(dash="dot", color="#111827"),
                                  fill="toself", opacity=0.12))
    for lid in listing_ids:
        v = [float(rec.scaled.loc[lid, c]) for c in columns]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], fill="toself",
                                      opacity=0.45, name=str(rec.scored.loc[lid, "name"])))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title=title, legend=dict(orientation="h", y=-0.12), **LAYOUT)
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
def build_map(top: pd.DataFrame, members: list[Member], selected_id: str | None = None):
    """folium 이 있으면 folium.Map, 없으면 plotly Figure 를 반환한다."""
    center = [float(top["lat"].mean()), float(top["lon"].mean())]
    sel = selected_id if selected_id in top.index else top.index[0]

    if HAS_FOLIUM:
        m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
        for m_ in members:
            if m_.anchor_lat is None:
                continue
            folium.Marker([m_.anchor_lat, m_.anchor_lon],
                          tooltip=f"{m_.name} · {m_.anchor_label}",
                          icon=folium.Icon(color="gray", icon="briefcase", prefix="fa")).add_to(m)
        for lid, r in top.iterrows():
            is_sel = lid == sel
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=9 if is_sel else 6,
                color="#2563eb" if is_sel else "#94a3b8", fill=True,
                fill_opacity=0.9 if is_sel else 0.6,
                tooltip=f"{int(r['rank'])}위 · {r['name']} · {r['fit_score']}점").add_to(m)
        r = top.loc[sel]
        for m_ in members:
            if m_.anchor_lat is None:
                continue
            folium.PolyLine([[r["lat"], r["lon"]], [m_.anchor_lat, m_.anchor_lon]],
                            color=m_.color, weight=3, opacity=0.75,
                            tooltip=f"{m_.name} 동선").add_to(m)
        return "folium", m

    fig = go.Figure()
    # 지도 배경 위에서도 마커가 보이도록 흰색 헤일로를 먼저 깐다
    fig.add_trace(go.Scattermapbox(
        lat=top["lat"], lon=top["lon"], mode="markers",
        marker=dict(size=[26 if i == sel else 20 for i in top.index], color="white"),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermapbox(
        lat=top["lat"], lon=top["lon"], mode="markers+text",
        marker=dict(size=[22 if i == sel else 16 for i in top.index],
                    color=["#2563eb" if i == sel else "#475569" for i in top.index]),
        text=[str(int(r)) for r in top["rank"]], textposition="middle center",
        textfont=dict(color="white", size=10),
        customdata=np.stack([top["name"], top["fit_score"]], axis=-1),
        hovertemplate="%{customdata[0]}<br>종합 %{customdata[1]:.1f}점<extra></extra>",
        name="추천 매물"))
    r = top.loc[sel]
    for m_ in members:
        if m_.anchor_lat is None:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=[r["lat"], m_.anchor_lat], lon=[r["lon"], m_.anchor_lon],
            mode="lines+markers", line=dict(width=3, color=m_.color),
            marker=dict(size=[0, 14], color=m_.color),
            name=f"{m_.name} → {m_.anchor_label}",
            hovertemplate=f"{m_.name} 동선<extra></extra>"))
    fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=center[0], lon=center[1]), zoom=10),
                      margin=dict(l=0, r=0, t=30, b=0), height=560,
                      legend=dict(orientation="h", y=-0.05),
                      title=f"선택 매물 기준 구성원 동선 — {r['name']}")
    return "plotly", fig
