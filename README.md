# 🏡 가족 맞춤 주거지 추천 (Multi-Person Weighted K-NN)

가격 정보를 **완전히 배제**하고, 주택 자체의 공간·환경 조건(**공통 데이터**)과
가족 구성원 각자의 생활 동선(**개별 데이터**)만으로 최적 주거지를 추천하는 웹 애플리케이션.

```bash
pip install -r requirements.txt
python data/generate_mock_data.py --n 400 --seed 42   # 가상 데이터 생성
streamlit run app.py                                  # 앱 실행
python tests/test_model.py                            # 엔진 스모크 테스트
```

## 디렉토리 구조

```
home_recommender/
├── app.py                        # Streamlit 앱 (가중치 슬라이더 / 트레이드오프 UI / 지도)
├── requirements.txt
├── README.md
├── src/
│   ├── schema.py                 # 공통·개별 피처 정의, 구성원 모델, 가격 배제 가드
│   ├── model.py                  # 하드필터 → 스케일링 → KNNImputer → 가중 K-NN → 파레토·군집
│   ├── features.py               # 파생 변수(이동 부담·불평등도·안정성 지수·파레토 마스크)
│   └── viz.py                    # Plotly 레이더/트레이드오프/기여도, Folium 동선 지도
├── data/
│   ├── generate_mock_data.py     # 가상 주택 + 구성원 동선 데이터 생성
│   ├── listings.csv              # 공통 데이터 (주택 물리/환경)
│   ├── individual_features.csv   # 개별 데이터 (구성원 × 동선)
│   └── members.json              # 구성원 정의 + 직장/학원 좌표
└── tests/test_model.py
```

## 데이터 분리 원칙

| 구분 | 내용 | 컬럼 예시 |
|---|---|---|
| **공통** (가족 공유) | 전용면적, 방/욕실 수, 향, 준공 경과년수, 층 쾌적도, 녹지 비율, 세대당 주차, 단지 규모, 일조, 소음 | `area_m2`, `green_ratio`, `noise_db` |
| **개별** (구성원별) | 통근 시간, 환승 횟수, 역·마트·병원 거리 / 학교 도보, 학원가·공원·도서관 거리 | `dad__commute_min`, `child__school_walk_m` |
| **가격** | **사용하지 않음** — `assert_no_price_columns()` 이 매매/전세/월세/관리비 계열 컬럼 유입을 차단 | – |

## 알고리즘

1. **하드 제약 필터링** — K-NN 이전에 최소 방 수·면적·최대 연식·구성원별 통근 상한 등으로 후보 축소.
2. **스케일링** — MinMax(기본) 또는 Standard 로 `m²`·`분`·`m` 단위 왜곡 제거.
3. **KNNImputer** — 스케일 공간에서 누락된 공간/환경/동선 정보를 거리 가중 평균으로 보정.
4. **다차원 가중치 결합** — 공통 가중치 $W_c$ 와 구성원 가중치 $P_i \cdot W_i$ 를 균형계수 $\alpha$ 로 결합:

$$\omega_c = \alpha\frac{W_c}{\sum W_c},\qquad
  \omega_{i,f} = (1-\alpha)\frac{P_i}{\sum_j P_j}\cdot\frac{W_{i,f}}{\sum_g W_{i,g}},\qquad \sum\omega = 1$$

5. **가족 이상점 벡터** — 피처 방향성(higher/lower)에 따른 최적값. *만족 임계값(satisficing)* 을 켜면
   "통근 25분이면 충분" 같은 기준을 넘는 초과 성능은 점수 차이로 계산하지 않음.
6. **가중 유클리드 K-NN** — 좌표를 $\sqrt{\omega}$ 로 변환해 sklearn `NearestNeighbors` 로 Top-K 탐색.

$$d(x,x^{*}) = \sqrt{\sum_f \omega_f\,(x_f - x^{*}_f)^2}$$

7. **해석 계층** — 구성원별 만족도, 파레토 최적 집합, 거리 기여도 분해, KMeans 군집 프로파일.

## 이해충돌 해결 (Conflict Resolution)

- **트레이드오프 산점도** — 두 구성원의 만족도 평면에 후보를 배치하고 **파레토 프론티어**를 강조.
- **사회적 선택 규칙 3종 비교** — 공리주의(가중 합) / 롤스적 최약자 우선(Maximin) / 형평성 우선(격차 최소).
- **레이더 차트** — 구성원별 만족도 균형을 한눈에 비교(최대 4개 매물 중첩).
- **인터랙티브 슬라이더** — 발언권 $P_i$, 피처별 가중치, $\alpha$ 균형을 실시간 조정.

## 파생 변수 (Feature Engineering)

| 지표 | 정의 |
|---|---|
| `burden_min__*` | 구성원별 체감 이동 부담(분). 시간·도보거리·환승을 분 단위로 환산 합산 |
| `travel_mean_min` / `travel_std_min` | 가족 평균 이동시간 / 구성원 간 표준편차 |
| `inequality_index` | 이동 부담 변동계수(CV) — 구성원 간 불평등도 |
| `equity_index` | `1 - inequality_index` — 높을수록 부담이 균등 |
| `stability_index` | 녹지·주차·일조·단지규모·연식·소음 가중 합성 주거 안정성 (0~1) |
| `sat_min` / `sat_gap` | 최약자 만족도 / 구성원 간 만족도 격차(이해충돌 지수) |

## 실제 데이터로 교체하기

`data/listings.csv`, `data/individual_features.csv` 를 동일한 컬럼 규약으로 바꾸면 된다.
개별 피처 컬럼은 `{구성원key}__{피처key}` 형식이며, 구성원 구성·역할은 `src/schema.py` 의
`ROLE_FEATURES` / `DEFAULT_MEMBERS` 를 수정해 확장한다. 통근 시간은 실제 경로 API
(대중교통 길찾기) 결과를 넣으면 그대로 동작한다.
