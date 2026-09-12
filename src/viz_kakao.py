"""
지도 결과 뷰 — 전체 화면 지도 위에 추천 목록을 오버레이로 얹는다 (카카오맵 스타일).

  - 카카오 JavaScript 키가 있으면 카카오맵, 없으면 Leaflet(OpenStreetMap)으로
    같은 화면을 그린다. 두 경우 모두 왼쪽 오버레이 패널·동선·마커 UI 는 동일하다.
  - 목록(또는 지도 마커)에서 매물을 클릭하면 컴포넌트가
    streamlit:setComponentValue 로 매물 id 를 파이썬에 돌려주고,
    앱이 그 매물의 실제 동선을 조회해 다시 그린다.

카카오맵을 쓰려면 (REST API 키와 별개)
  1) 카카오 디벨로퍼스 > 내 애플리케이션 > 앱 키 > **JavaScript 키**
  2) 플랫폼 > Web > 사이트 도메인에 앱이 서비스되는 도메인 등록
     - 로컬:   http://localhost:8520
     - 배포본: https://family-housing-recommender.streamlit.app
  도메인이 등록되지 않으면 SDK 가 로드되지 않으므로, 지도 자리에 안내 문구를 띄운다.
"""

from __future__ import annotations

import json
import os

from .schema import TRAVEL_MODES, Member

SDK_URL = "//dapi.kakao.com/v2/maps/sdk.js"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"


def _far(a, b, tol: float = 1e-5) -> bool:
    """두 좌표가 사실상 같은 지점인지 (약 1m 이내면 같다고 본다)."""
    return abs(a[0] - b[0]) > tol or abs(a[1] - b[1]) > tol


def _num(v):
    """NaN/None 을 JSON null 로 보내기 위한 안전 변환."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _payload(top, members: list[Member], selected_id: str, routes: dict) -> dict:
    sel = selected_id if selected_id in top.index else top.index[0]
    r = top.loc[sel]

    listings = []
    for i, row in top.iterrows():
        chips = []
        for m in members:
            for d in m.enabled_destinations:
                chips.append({"member": m.name, "dest": d.label, "color": m.color,
                              "mode": TRAVEL_MODES.get(d.mode, d.mode),
                              "minutes": _num(row.get(d.column(m.key)))})
        listings.append({
            "id": str(i), "lat": float(row["lat"]), "lon": float(row["lon"]),
            "rank": int(row["rank"]), "name": str(row["name"]),
            "gu": str(row.get("gu", "")), "score": float(row["fit_score"]),
            "area": _num(row.get("area_m2")), "rooms": _num(row.get("rooms")),
            "satMin": _num(row.get("sat_min")),
            "selected": i == sel, "chips": chips,
        })

    lines = []
    for m in members:
        for d in m.enabled_destinations:
            info = routes.get((m.key, d.key)) or {}
            path = [list(p) for p in info.get("path", [])]
            real = len(path) >= 2
            origin = [float(r["lat"]), float(r["lon"])]
            target = [float(d.lat), float(d.lon)]
            if not real:                       # 실경로가 없으면 출발-도착 직선
                path = [origin, target]
            else:
                # 대중교통 경로는 가장 가까운 정류장에서 시작해 매물 마커와 수백 m
                # 떨어져 끊겨 보인다. 양 끝을 매물/목적지에 붙여 선이 닿게 한다.
                if _far(path[0], origin):
                    path.insert(0, origin)
                if _far(path[-1], target):
                    path.append(target)
            # 실제 경로를 받아왔으면 그때의 소요시간이 더 정확하다.
            minutes = info.get("minutes")
            if minutes is None:
                minutes = r.get(d.column(m.key))
            lines.append({
                "member": m.name, "dest": d.label, "color": m.color,
                "mode": TRAVEL_MODES.get(d.mode, d.mode), "real": real, "path": path,
                "destLat": float(d.lat), "destLon": float(d.lon),
                "minutes": _num(minutes),
            })

    return {"center": [float(r["lat"]), float(r["lon"])], "listings": listings,
            "lines": lines, "selectedId": str(sel), "selectedName": str(r["name"])}


# --------------------------------------------------------------------------- #
# HTML 템플릿
# --------------------------------------------------------------------------- #
# f-string 이 아니라 .replace() 치환을 쓴다 — JS 중괄호를 일일이 이스케이프하면
# 유지보수가 불가능해진다. __DATA__ / __ENGINE_TAGS__ 두 자리만 치환한다.
_TEMPLATE = r"""
<style>
  #wrap { position: relative; width: 100%; height: 100vh; min-height: 520px;
          font-family: -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif; }
  #map  { width: 100%; height: 100%; border-radius: 10px; background: #e5e7eb; }
  #fail { display: none; position: absolute; inset: 0; padding: 24px; z-index: 30;
          background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
          font: 14px/1.7 inherit; color: #334155; }

  /* ---------------- 왼쪽 오버레이 컬럼 (버튼 + 목록 + 버튼) ---------------- */
  #leftcol { position: absolute; left: 12px; top: 12px; bottom: 12px; width: 330px;
             z-index: 20; display: flex; flex-direction: column; gap: 10px; }
  .obtn { flex: 0 0 auto; background: rgba(255,255,255,.96); border: 0;
          border-radius: 10px; padding: 11px 14px; cursor: pointer;
          font: 600 13px/1.2 -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic',
                sans-serif;
          color: #0f172a; box-shadow: 0 2px 10px rgba(0,0,0,.2); text-align: center; }
  .obtn:hover { background: #f1f5f9; }
  #panel { flex: 1; min-height: 0; display: flex; flex-direction: column;
           background: rgba(255,255,255,.96); border-radius: 12px;
           box-shadow: 0 4px 18px rgba(0,0,0,.22); overflow: hidden; }
  #panel header { padding: 12px 14px 8px; border-bottom: 1px solid #e2e8f0; }
  #panel header b { font-size: 15px; color: #0f172a; }
  #panel header span { float: right; font-size: 12px; color: #64748b; margin-top: 2px; }
  #cards { flex: 1; overflow-y: auto; padding: 6px; }
  .card { display: flex; gap: 10px; padding: 10px; border-radius: 10px; cursor: pointer;
          border: 1.5px solid transparent; }
  .card:hover { background: #f1f5f9; }
  .card.active { background: #eff6ff; border-color: #2563eb; }
  .rank { flex: 0 0 26px; width: 26px; height: 26px; line-height: 26px; margin-top: 2px;
          border-radius: 50%; background: #475569; color: #fff; text-align: center;
          font: bold 12px/26px sans-serif; }
  .card.active .rank { background: #2563eb; }
  .body { flex: 1; min-width: 0; }
  .row1 { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }
  .row1 b { font-size: 13px; color: #0f172a; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }
  .row1 .score { font-size: 12px; font-weight: 700; color: #2563eb; white-space: nowrap; }
  .row2 { font-size: 11px; color: #64748b; margin-top: 2px; }
  .chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  .chip { font-size: 10px; color: #334155; background: #f1f5f9; border-radius: 8px;
          padding: 1px 7px 1px 5px; border-left: 3px solid #999; white-space: nowrap; }

  /* ---------------- 중앙 로딩 오버레이 (경로 조회 중) ---------------- */
  #loading { display: none; position: absolute; inset: 0; z-index: 40;
             align-items: center; justify-content: center;
             background: rgba(255,255,255,.15); }
  #loading.on { display: flex; }
  #loading .box { background: rgba(255,255,255,.97); border-radius: 12px;
                  padding: 15px 22px; box-shadow: 0 4px 18px rgba(0,0,0,.25);
                  display: flex; align-items: center; gap: 10px;
                  font: 600 14px/1.2 inherit; color: #0f172a; }
  #loading .spin { width: 20px; height: 20px; border-radius: 50%; flex: 0 0 20px;
                   border: 3px solid #e2e8f0; border-top-color: #2563eb;
                   animation: mapspin 0.8s linear infinite; }
  @keyframes mapspin { to { transform: rotate(360deg); } }

  /* ---------------- 범례 (오른쪽 아래) ---------------- */
  #legend { position: absolute; right: 12px; bottom: 12px; z-index: 20;
            background: rgba(255,255,255,.94); padding: 8px 10px; border-radius: 8px;
            font: 12px/1.6 inherit; box-shadow: 0 1px 4px rgba(0,0,0,.18); color: #0f172a; }

  .pill { background: #333; color: #fff; padding: 3px 8px; border-radius: 12px;
          font: 11px/1.4 sans-serif; white-space: nowrap;
          box-shadow: 0 1px 3px rgba(0,0,0,.3); }
  .dot  { border-radius: 50%; color: #fff; text-align: center; border: 2px solid #fff;
          box-shadow: 0 1px 4px rgba(0,0,0,.35); font-weight: bold; font-size: 12px;
          cursor: pointer; box-sizing: border-box; }

  @media (max-width: 640px) {
    #leftcol { width: auto; right: 12px; bottom: auto; max-height: 55%; }
  }
</style>

<div id="wrap">
  <div id="map"></div>
  <div id="fail">
    <b>카카오맵 SDK를 불러오지 못했습니다.</b><br>
    다음을 확인하세요.
    <ol style="margin:8px 0 0 18px;padding:0;">
      <li><b>JavaScript 키</b>를 넣었는지 (REST API 키와 다릅니다)</li>
      <li>카카오 디벨로퍼스 &gt; 플랫폼 &gt; Web &gt; <b>사이트 도메인</b>에 현재 주소를 등록했는지</li>
    </ol>
  </div>
  <div id="leftcol">
    <button class="obtn" id="btn-back">⬅ 설정 수정</button>
    <div id="panel">
      <header><b>🏡 추천 결과</b><span id="count"></span></header>
      <div id="cards"></div>
    </div>
    <button class="obtn" id="btn-detail">📊 상세 정보 보기</button>
  </div>
  <div id="legend"></div>
  <div id="loading"><div class="box"><div class="spin"></div>실제 경로 조회 중...</div></div>
</div>

<script>
  window.__HOUSE_DATA__ = __DATA__;
</script>
__ENGINE_TAGS__
<script>
(function () {
  var D = window.__HOUSE_DATA__;

  // ---------------- 공통: 액션 → 파이썬으로 통보 ---------------- //
  // 컴포넌트 값은 세션 동안 유지되므로, 같은 버튼을 두 번 눌러도 새 이벤트로
  // 인식되도록 nonce(n)를 붙인다. 파이썬은 마지막으로 처리한 n 을 기억한다.
  function emit(action, extra) {
    if (!window.__STREAMLIT_SET_VALUE__) return;
    var v = Object.assign({action: action, n: Date.now()}, extra || {});
    window.__STREAMLIT_SET_VALUE__(v);
  }
  function select(id) {
    if (id === D.selectedId) return;
    var cards = document.querySelectorAll('.card');
    cards.forEach(function (c) {
      c.classList.toggle('active', c.dataset.id === id);
    });
    // 새 경로가 도착하면 컴포넌트가 새 HTML 로 갈아끼워지며 자연히 사라진다.
    document.getElementById('loading').classList.add('on');
    emit('select', {id: id});
  }
  window.__SELECT_LISTING__ = select;
  document.getElementById('btn-back').addEventListener('click', function () {
    emit('back');
  });
  document.getElementById('btn-detail').addEventListener('click', function () {
    emit('detail');
  });

  // ---------------- 오버레이 패널 ---------------- //
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');
  }
  function card(h) {
    var meta = [];
    if (h.gu) meta.push(esc(h.gu));
    if (h.area !== null) meta.push(Math.round(h.area) + 'm²');
    if (h.rooms !== null) meta.push('방 ' + Math.round(h.rooms));
    if (h.satMin !== null) meta.push('최약자 ' + h.satMin.toFixed(0) + '점');
    var chips = h.chips.map(function (c) {
      var t = c.minutes === null ? '-' : Math.round(c.minutes) + '분';
      return '<span class="chip" style="border-left-color:' + c.color + '">'
           + esc(c.member) + '·' + esc(c.dest) + ' ' + t + '</span>';
    }).join('');
    return '<div class="card' + (h.selected ? ' active' : '') + '" data-id="' + esc(h.id) + '">'
         + '<div class="rank">' + h.rank + '</div>'
         + '<div class="body">'
         +   '<div class="row1"><b>' + esc(h.name) + '</b>'
         +   '<span class="score">' + h.score.toFixed(1) + '점</span></div>'
         +   '<div class="row2">' + meta.join(' · ') + '</div>'
         +   '<div class="chips">' + chips + '</div>'
         + '</div></div>';
  }
  document.getElementById('count').textContent = D.listings.length + '건';
  var cardsEl = document.getElementById('cards');
  cardsEl.innerHTML = D.listings.map(card).join('');
  cardsEl.addEventListener('click', function (e) {
    var el = e.target.closest('.card');
    if (el) select(el.dataset.id);
  });
  var activeCard = cardsEl.querySelector('.card.active');
  if (activeCard) activeCard.scrollIntoView({block: 'nearest'});

  // ---------------- 범례 ---------------- //
  (function () {
    var seen = {}, html = '';
    D.lines.forEach(function (ln) {
      if (seen[ln.member]) return;
      seen[ln.member] = 1;
      html += '<div><span style="display:inline-block;width:18px;height:3px;'
            + 'background:' + ln.color + ';vertical-align:middle;margin-right:6px"></span>'
            + esc(ln.member) + '</div>';
    });
    html += '<div style="margin-top:4px;color:#64748b">실선 = 실제 경로 · 점선 = 직선 추정</div>';
    document.getElementById('legend').innerHTML = html;
  })();

  function dotHtml(h) {
    var size = h.selected ? 30 : 24;
    return '<div class="dot" title="' + esc(h.name) + ' (' + h.score.toFixed(1) + '점)" '
         + 'style="width:' + size + 'px;height:' + size + 'px;line-height:' + (size - 4) + 'px;'
         + 'background:' + (h.selected ? '#2563eb' : '#475569') + '" '
         + 'onclick="__SELECT_LISTING__(\'' + esc(h.id) + '\')">' + h.rank + '</div>';
  }
  function pillHtml(ln) {
    var t = ln.minutes === null ? '' : ' · ' + Math.round(ln.minutes) + '분';
    return '<div class="pill" style="background:' + ln.color + '">'
         + esc(ln.member) + ' · ' + esc(ln.dest) + t + '</div>';
  }
  // 패널이 왼쪽 ~350px 를 덮으므로 화면 맞춤 시 왼쪽 여백을 크게 준다.
  var PAD = {top: 60, right: 60, bottom: 60, left: 380};

  // ---------------- 카카오맵 ---------------- //
  function initKakao() {
    var map = new kakao.maps.Map(document.getElementById('map'), {
      center: new kakao.maps.LatLng(D.center[0], D.center[1]), level: 7
    });
    map.addControl(new kakao.maps.ZoomControl(), kakao.maps.ControlPosition.RIGHT);
    var bounds = new kakao.maps.LatLngBounds();

    D.lines.forEach(function (ln) {
      var pts = ln.path.map(function (p) { return new kakao.maps.LatLng(p[0], p[1]); });
      pts.forEach(function (p) { bounds.extend(p); });
      new kakao.maps.Polyline({
        map: map, path: pts, strokeWeight: ln.real ? 5 : 3, strokeColor: ln.color,
        strokeOpacity: ln.real ? 0.9 : 0.55,
        strokeStyle: ln.real ? 'solid' : 'shortdash'
      });
      new kakao.maps.CustomOverlay({
        map: map, position: new kakao.maps.LatLng(ln.destLat, ln.destLon),
        yAnchor: 1.35, content: pillHtml(ln)
      });
    });

    D.listings.forEach(function (h) {
      var pos = new kakao.maps.LatLng(h.lat, h.lon);
      bounds.extend(pos);
      new kakao.maps.CustomOverlay({
        map: map, position: pos, zIndex: h.selected ? 10 : 1, content: dotHtml(h)
      });
    });

    // 컴포넌트 iframe 은 처음에 크기가 0 이고, 비활성 상태면 계속 0 이다.
    // 그 상태의 setBounds 는 레벨을 최대로 빼버리므로, 크기가 "바뀔 때마다"
    // relayout + setBounds 를 다시 부른다.
    var lastW = 0, lastH = 0;
    function fit() {
      var el = document.getElementById('map');
      var w = el.clientWidth, h = el.clientHeight;
      if (!w || !h || (w === lastW && h === lastH)) return;
      lastW = w; lastH = h;
      map.relayout();
      if (!bounds.isEmpty()) {
        map.setBounds(bounds, PAD.top, PAD.right, PAD.bottom, PAD.left);
      }
    }
    fit();
    if (window.ResizeObserver) {
      new ResizeObserver(fit).observe(document.getElementById('map'));
    } else {
      setInterval(fit, 300);
    }
    window.addEventListener('resize', fit);
  }

  // ---------------- Leaflet (카카오 키가 없을 때) ---------------- //
  function initLeaflet() {
    var map = L.map('map', {zoomControl: true}).setView(D.center, 11);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
    var pts = [];

    D.lines.forEach(function (ln) {
      L.polyline(ln.path, {
        color: ln.color, weight: ln.real ? 5 : 3, opacity: ln.real ? 0.9 : 0.55,
        dashArray: ln.real ? null : '6,6'
      }).addTo(map);
      ln.path.forEach(function (p) { pts.push(p); });
      L.marker([ln.destLat, ln.destLon], {
        icon: L.divIcon({className: '', html: pillHtml(ln), iconAnchor: [0, 30]})
      }).addTo(map);
    });

    D.listings.forEach(function (h) {
      pts.push([h.lat, h.lon]);
      var size = h.selected ? 30 : 24;
      L.marker([h.lat, h.lon], {
        zIndexOffset: h.selected ? 1000 : 0,
        icon: L.divIcon({className: '', html: dotHtml(h),
                         iconSize: [size, size], iconAnchor: [size / 2, size / 2]})
      }).addTo(map);
    });

    function fit() {
      map.invalidateSize();
      if (pts.length) {
        map.fitBounds(pts, {paddingTopLeft: [PAD.left, PAD.top],
                            paddingBottomRight: [PAD.right, PAD.bottom]});
      }
    }
    fit();
    if (window.ResizeObserver) {
      new ResizeObserver(function () { map.invalidateSize(); })
        .observe(document.getElementById('map'));
    }
    setTimeout(fit, 250);   // iframe 크기가 잡히기 전에 그려졌을 때 재맞춤
  }

  function fail() { document.getElementById('fail').style.display = 'block'; }
  if (typeof kakao !== 'undefined' && kakao.maps) {
    kakao.maps.load(initKakao);
  } else if (typeof L !== 'undefined') {
    initLeaflet();
  } else {
    fail();
  }
})();
</script>
"""


def map_view_html(top, members: list[Member], selected_id: str,
                  routes: dict | None, js_key: str = "") -> str:
    data = _payload(top, members, selected_id, routes or {})
    if js_key:
        engine = ('<script src="{sdk}?appkey={key}&autoload=false" '
                  'onerror="document.getElementById(\'fail\').style.display=\'block\';">'
                  '</script>').format(sdk=SDK_URL, key=js_key)
    else:
        engine = ('<link rel="stylesheet" href="{css}">\n'
                  '<script src="{js}"></script>').format(css=LEAFLET_CSS, js=LEAFLET_JS)
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__ENGINE_TAGS__", engine))


# ---------------------------------------------------------------------------
# Streamlit 커스텀 컴포넌트로 렌더링
# ---------------------------------------------------------------------------
# components.html 은 about:srcdoc 문서를 쓰기 때문에 카카오 SDK 가 프로토콜을
# http 로 오판해 mixed-content 에 걸린다(자세한 이유는 kakao_frontend/index.html).
# 커스텀 컴포넌트는 /component/<name>/index.html 실제 URL 로 서빙되므로
# location.protocol 이 https 가 되고 Referer 도 등록 도메인으로 나간다.

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


def render_map_view(top, members: list[Member], selected_id: str,
                    routes: dict | None, js_key: str = "",
                    height: int = 640, key: str | None = None):
    """전체 화면 지도 뷰를 그리고, 사용자가 클릭한 매물 id(또는 None)를 반환한다."""
    html = map_view_html(top, members, selected_id, routes, js_key)
    return _get_component()(html=html, height=height, key=key, default=None)
