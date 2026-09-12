"""
카카오맵 JavaScript SDK 기반 지도 렌더러.

Streamlit 은 파이썬이 서버에서 돌기 때문에, 카카오맵을 쓰려면 브라우저에서
SDK 를 직접 로드하는 HTML 을 components.html 로 심어야 한다.

필요한 것 (REST API 키와 별개)
  1) 카카오 디벨로퍼스 > 내 애플리케이션 > 앱 키 > **JavaScript 키**
  2) 플랫폼 > Web > 사이트 도메인에 앱이 서비스되는 도메인 등록
     - 로컬:   http://localhost:8520
     - 배포본: https://family-housing-recommender.streamlit.app
  도메인이 등록되지 않으면 SDK 가 로드되지 않으므로, 지도 자리에 안내 문구를 띄운다.
"""

from __future__ import annotations

import json

from .schema import TRAVEL_MODES, Member

SDK_URL = "//dapi.kakao.com/v2/maps/sdk.js"


def _payload(top, members: list[Member], selected_id: str, routes: dict) -> dict:
    sel = selected_id if selected_id in top.index else top.index[0]
    r = top.loc[sel]

    listings = [{
        "id": str(i), "lat": float(row["lat"]), "lon": float(row["lon"]),
        "rank": int(row["rank"]), "name": str(row["name"]),
        "score": float(row["fit_score"]), "selected": i == sel,
    } for i, row in top.iterrows()]

    lines = []
    for m in members:
        for d in m.enabled_destinations:
            info = routes.get((m.key, d.key)) or {}
            path = [list(p) for p in info.get("path", [])]
            real = len(path) >= 2
            if not real:                       # 실경로가 없으면 출발-도착 직선
                path = [[float(r["lat"]), float(r["lon"])], [float(d.lat), float(d.lon)]]
            minutes = r.get(d.column(m.key))
            lines.append({
                "member": m.name, "dest": d.label, "color": m.color,
                "mode": TRAVEL_MODES.get(d.mode, d.mode), "real": real, "path": path,
                "destLat": float(d.lat), "destLon": float(d.lon),
                "minutes": None if minutes is None or minutes != minutes else float(minutes),
            })

    return {"center": [float(r["lat"]), float(r["lon"])], "listings": listings, "lines": lines,
            "selectedName": str(r["name"])}


def kakao_map_html(top, members: list[Member], selected_id: str,
                   routes: dict | None, js_key: str, height: int = 560) -> str:
    data = _payload(top, members, selected_id, routes or {})
    return f"""
<div id="wrap" style="position:relative;width:100%;height:{height}px;">
  <div id="map" style="width:100%;height:100%;border-radius:8px;"></div>
  <div id="fail" style="display:none;position:absolute;inset:0;padding:24px;
       background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
       font:14px/1.7 -apple-system,'Apple SD Gothic Neo',sans-serif;color:#334155;">
    <b>카카오맵 SDK를 불러오지 못했습니다.</b><br>
    다음을 확인하세요.
    <ol style="margin:8px 0 0 18px;padding:0;">
      <li><b>JavaScript 키</b>를 넣었는지 (REST API 키와 다릅니다)</li>
      <li>카카오 디벨로퍼스 &gt; 플랫폼 &gt; Web &gt; <b>사이트 도메인</b>에 현재 주소를 등록했는지</li>
    </ol>
  </div>
  <div id="legend" style="position:absolute;left:10px;bottom:10px;z-index:5;
       background:rgba(255,255,255,.94);padding:8px 10px;border-radius:8px;
       font:12px/1.6 -apple-system,'Apple SD Gothic Neo',sans-serif;
       box-shadow:0 1px 4px rgba(0,0,0,.18);"></div>
</div>
<script>
  window.__HOUSE_DATA__ = {json.dumps(data, ensure_ascii=False)};
</script>
<script src="{SDK_URL}?appkey={js_key}&autoload=false"
        onerror="document.getElementById('fail').style.display='block';"></script>
<script>
(function () {{
  var D = window.__HOUSE_DATA__;
  function fail() {{ document.getElementById('fail').style.display = 'block'; }}
  if (typeof kakao === 'undefined' || !kakao.maps) {{ fail(); return; }}

  kakao.maps.load(function () {{
    var map = new kakao.maps.Map(document.getElementById('map'), {{
      center: new kakao.maps.LatLng(D.center[0], D.center[1]), level: 7
    }});
    map.addControl(new kakao.maps.ZoomControl(), kakao.maps.ControlPosition.RIGHT);
    var bounds = new kakao.maps.LatLngBounds();

    D.lines.forEach(function (ln) {{
      var pts = ln.path.map(function (p) {{ return new kakao.maps.LatLng(p[0], p[1]); }});
      pts.forEach(function (p) {{ bounds.extend(p); }});
      new kakao.maps.Polyline({{
        map: map, path: pts, strokeWeight: ln.real ? 5 : 3, strokeColor: ln.color,
        strokeOpacity: ln.real ? 0.9 : 0.55,
        strokeStyle: ln.real ? 'solid' : 'shortdash'
      }});
      var t = ln.minutes === null ? '' : ' · ' + Math.round(ln.minutes) + '분';
      new kakao.maps.CustomOverlay({{
        map: map, position: new kakao.maps.LatLng(ln.destLat, ln.destLon), yAnchor: 1.35,
        content: '<div style="background:' + ln.color + ';color:#fff;padding:3px 8px;'
               + 'border-radius:12px;font:11px/1.4 sans-serif;white-space:nowrap;'
               + 'box-shadow:0 1px 3px rgba(0,0,0,.3)">' + ln.member + ' · ' + ln.dest + t + '</div>'
      }});
    }});

    D.listings.forEach(function (h) {{
      var pos = new kakao.maps.LatLng(h.lat, h.lon);
      bounds.extend(pos);
      var bg = h.selected ? '#2563eb' : '#475569';
      var size = h.selected ? 30 : 24;
      var ov = new kakao.maps.CustomOverlay({{
        map: map, position: pos, zIndex: h.selected ? 10 : 1,
        content: '<div title="' + h.name + ' (' + h.score.toFixed(1) + '점)" '
               + 'style="width:' + size + 'px;height:' + size + 'px;line-height:' + size + 'px;'
               + 'border-radius:50%;background:' + bg + ';color:#fff;text-align:center;'
               + 'font:bold 12px sans-serif;border:2px solid #fff;'
               + 'box-shadow:0 1px 4px rgba(0,0,0,.35)">' + h.rank + '</div>'
      }});
      void ov;
    }});

    // 컴포넌트 iframe 은 처음에 크기가 0 이라, 그 상태로 setBounds 를 부르면
    // 카카오가 레벨을 최대로 빼버린다(한반도 전체가 보인다). 크기가 잡힌 뒤
    // relayout + setBounds 를 다시 부른다.
    var fitted = false;
    function fit() {{
      var el = document.getElementById('map');
      if (!el.clientWidth || !el.clientHeight) return;
      map.relayout();
      if (!bounds.isEmpty()) map.setBounds(bounds);
      fitted = true;
    }}
    fit();
    if (!fitted) {{
      var tries = 0;
      var timer = setInterval(function () {{
        fit();
        if (fitted || ++tries > 40) clearInterval(timer);
      }}, 150);
    }}

    var seen = {{}}, html = '';
    D.lines.forEach(function (ln) {{
      if (seen[ln.member]) return;
      seen[ln.member] = 1;
      html += '<div><span style="display:inline-block;width:18px;height:3px;'
            + 'background:' + ln.color + ';vertical-align:middle;margin-right:6px"></span>'
            + ln.member + '</div>';
    }});
    html += '<div style="margin-top:4px;color:#64748b">실선 = 실제 경로 · 점선 = 직선 추정</div>';
    document.getElementById('legend').innerHTML = html;
  }});
}})();
</script>
"""

# ---------------------------------------------------------------------------
# Streamlit 커스텀 컴포넌트로 렌더링
# ---------------------------------------------------------------------------
# components.html 은 about:srcdoc 문서를 쓰기 때문에 카카오 SDK 가 프로토콜을
# http 로 오판해 mixed-content 에 걸린다(자세한 이유는 kakao_frontend/index.html).
# 커스텀 컴포넌트는 /component/<name>/index.html 실제 URL 로 서빙되므로
# location.protocol 이 https 가 되고 Referer 도 등록 도메인으로 나간다.

import os

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "kakao_frontend")
_component = None


def _get_component():
    global _component
    if _component is None:
        import streamlit.components.v1 as components
        _component = components.declare_component("kakao_map",
                                                  path=_FRONTEND_DIR)
    return _component


def render_kakao_map(top, members: list[Member], selected_id: str,
                     routes: dict | None, js_key: str,
                     height: int = 560, key: str | None = None) -> None:
    html = kakao_map_html(top, members, selected_id, routes, js_key, height)
    _get_component()(html=html, height=height + 20, key=key, default=None)
