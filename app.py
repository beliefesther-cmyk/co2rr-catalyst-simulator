import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="CO₂RR 촉매 선정 시뮬레이터",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main > div {padding-top: 1.4rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        .block-container {max-width: 1450px;}
        .small-note {color:#64748b; font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. 촉매 데이터
# =========================================================
CATALYST_DATA = pd.DataFrame(
    {
        "활성": [492.5, 58.0, 2.91, 2.0, 500.0],
        "선택성": [98.5, 76.7, 18.2, 98.0, 87.1],
        "효용성": [9.188, 9.328, 0.00905, 1.866, 9.328],
        "안정성": [48.0, 25.0, 100.0, 20.0, 53.0],
    },
    index=[
        "CuCo-DSAC",
        "Bi-Sb/CP 나노리프",
        "Cu-Ni 코어-쉘",
        "Cu97Sn3 SAA",
        "CeO2-SnO2",
    ],
)

CRITERIA = ["활성", "선택성", "효용성", "안정성"]


# =========================================================
# 3. 계산 함수
# =========================================================
def min_max_normalize(data: pd.DataFrame) -> pd.DataFrame:
    minimum = data.min(axis=0)
    maximum = data.max(axis=0)
    difference = (maximum - minimum).replace(0, 1)
    return (data - minimum) / difference


def calculate_topsis(data: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    matrix = data.to_numpy(dtype=float)

    column_norms = np.sqrt(np.sum(matrix**2, axis=0))
    column_norms[column_norms == 0] = 1
    normalized_matrix = matrix / column_norms

    weighted_matrix = normalized_matrix * weights
    positive_ideal = np.max(weighted_matrix, axis=0)
    negative_ideal = np.min(weighted_matrix, axis=0)

    distance_to_positive = np.sqrt(
        np.sum((weighted_matrix - positive_ideal) ** 2, axis=1)
    )
    distance_to_negative = np.sqrt(
        np.sum((weighted_matrix - negative_ideal) ** 2, axis=1)
    )

    denominator = distance_to_positive + distance_to_negative
    denominator[denominator == 0] = 1
    scores = distance_to_negative / denominator

    result = pd.DataFrame({"촉매": data.index, "TOPSIS 점수": scores})
    result = result.sort_values("TOPSIS 점수", ascending=False).reset_index(drop=True)
    result.insert(0, "순위", np.arange(1, len(result) + 1))
    return result


@st.cache_data
def find_optimal_weights():
    """10~50%, 5% 간격 조합 중 TOPSIS 점수 표준편차가 최대인 조합 탐색."""
    best_weights = None
    best_result = None
    best_spread = -1.0
    tested_count = 0

    possible_values = range(10, 51, 5)

    for activity in possible_values:
        for selectivity in possible_values:
            for productivity in possible_values:
                stability = 100 - activity - selectivity - productivity

                if stability < 10 or stability > 50 or stability % 5 != 0:
                    continue

                weights = np.array(
                    [activity, selectivity, productivity, stability], dtype=float
                ) / 100

                result = calculate_topsis(CATALYST_DATA, weights)
                spread = result["TOPSIS 점수"].std(ddof=0)
                tested_count += 1

                if spread > best_spread:
                    best_spread = spread
                    best_weights = weights
                    best_result = result

    return best_weights, best_result, best_spread, tested_count


NORMALIZED_DATA = min_max_normalize(CATALYST_DATA)


# =========================================================
# 4. 세션 상태
# =========================================================
DEFAULT_WEIGHTS = {"활성": 25, "선택성": 25, "효용성": 25, "안정성": 25}

for criterion, value in DEFAULT_WEIGHTS.items():
    st.session_state.setdefault(f"weight_{criterion}", value)

st.session_state.setdefault("auto_info", None)


def set_equal_weights():
    for criterion in CRITERIA:
        st.session_state[f"weight_{criterion}"] = 25
    st.session_state.auto_info = None


def set_automatic_weights():
    weights, _, spread, tested_count = find_optimal_weights()
    for criterion, value in zip(CRITERIA, weights * 100):
        st.session_state[f"weight_{criterion}"] = int(round(value))
    st.session_state.auto_info = {
        "tested_count": tested_count,
        "spread": spread,
    }


# =========================================================
# 5. 사이드바 입력
# =========================================================
with st.sidebar:
    st.header("⚙️ 평가 가중치")
    st.caption("각 기준의 중요도를 0~100 사이에서 조절하세요. 합계는 자동으로 100%로 환산됩니다.")

    for criterion in CRITERIA:
        st.slider(
            criterion,
            min_value=0,
            max_value=100,
            step=1,
            key=f"weight_{criterion}",
        )

    button_col1, button_col2 = st.columns(2)
    with button_col1:
        st.button("동일 가중치", use_container_width=True, on_click=set_equal_weights)
    with button_col2:
        st.button("자동 탐색", use_container_width=True, on_click=set_automatic_weights)

    st.divider()
    st.markdown("**자동 탐색 기준**")
    st.caption(
        "각 기준을 10~50% 범위에서 5% 간격으로 조합하고, 다섯 촉매의 TOPSIS 점수 표준편차가 가장 큰 조합을 선택합니다."
    )


raw_weights = np.array(
    [st.session_state[f"weight_{criterion}"] for criterion in CRITERIA],
    dtype=float,
)

if raw_weights.sum() == 0:
    st.error("가중치 중 하나 이상을 0보다 크게 설정하세요.")
    st.stop()

weights = raw_weights / raw_weights.sum()
result = calculate_topsis(CATALYST_DATA, weights)
best_catalyst = result.loc[0, "촉매"]
best_score = float(result.loc[0, "TOPSIS 점수"])


# =========================================================
# 6. 제목 및 핵심 결과
# =========================================================
st.title("🧪 CO₂RR 촉매 선정 시뮬레이터")
st.write(
    "활성·선택성·효용성·안정성을 가중합하여 TOPSIS 방식으로 CO₂ 전기환원 촉매를 비교합니다."
)

metric_col1, metric_col2, metric_col3 = st.columns([1.35, 1, 1])
metric_col1.metric("🏆 현재 최적 촉매", best_catalyst)
metric_col2.metric("최고 TOPSIS 점수", f"{best_score:.4f}")
metric_col3.metric("분석 대상", f"{len(CATALYST_DATA)}개 촉매")

weight_text = " · ".join(
    f"{criterion} {weight * 100:.1f}%" for criterion, weight in zip(CRITERIA, weights)
)
st.info(f"현재 적용 가중치: {weight_text}")

if st.session_state.auto_info:
    info = st.session_state.auto_info
    st.success(
        f"자동 탐색 완료: {info['tested_count']}개 조합 분석 · "
        f"TOPSIS 점수 표준편차 {info['spread']:.4f}"
    )


# =========================================================
# 7. 순위표와 원자료
# =========================================================
st.subheader("최종 순위")

ranking_display = result.copy()
ranking_display["TOPSIS 점수"] = ranking_display["TOPSIS 점수"].map(lambda x: f"{x:.4f}")
ranking_display["순위"] = ranking_display["순위"].map(lambda x: f"{x}위")

st.dataframe(
    ranking_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "순위": st.column_config.TextColumn(width="small"),
        "촉매": st.column_config.TextColumn(width="large"),
        "TOPSIS 점수": st.column_config.TextColumn(width="medium"),
    },
)


# =========================================================
# 8. 시각화
# =========================================================
left_chart, right_chart = st.columns([1.2, 1])

with left_chart:
    st.subheader("가중치 적용 후 기준별 점수")
    weighted_scores = NORMALIZED_DATA.mul(weights, axis=1) * 100

    heatmap = go.Figure(
        data=go.Heatmap(
            z=weighted_scores.values,
            x=CRITERIA,
            y=weighted_scores.index,
            colorscale="YlGnBu",
            text=np.round(weighted_scores.values, 1),
            texttemplate="%{text:.1f}",
            hovertemplate="촉매: %{y}<br>기준: %{x}<br>기여 점수: %{z:.2f}<extra></extra>",
            colorbar=dict(title="기여 점수"),
        )
    )
    heatmap.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="평가 기준",
        yaxis_title="촉매",
    )
    st.plotly_chart(heatmap, use_container_width=True)

with right_chart:
    st.subheader("촉매별 정규화 성능 비교")

    radar = go.Figure()
    radar_categories = CRITERIA + [CRITERIA[0]]

    for catalyst in NORMALIZED_DATA.index:
        values = (NORMALIZED_DATA.loc[catalyst].tolist() + [NORMALIZED_DATA.loc[catalyst].iloc[0]])
        is_best = catalyst == best_catalyst

        radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=radar_categories,
                mode="lines",
                name=(f"1위: {catalyst}" if is_best else catalyst),
                line=dict(width=4 if is_best else 1.5),
                fill="toself" if is_best else "none",
                opacity=1.0 if is_best else 0.55,
            )
        )

    radar.update_layout(
        height=430,
        margin=dict(l=30, r=30, t=20, b=30),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    st.plotly_chart(radar, use_container_width=True)

st.subheader("TOPSIS 종합점수 비교")
bar_data = result.sort_values("TOPSIS 점수", ascending=True)
bar_chart = px.bar(
    bar_data,
    x="TOPSIS 점수",
    y="촉매",
    orientation="h",
    text=bar_data["TOPSIS 점수"].map(lambda x: f"{x:.4f}"),
)
bar_chart.update_traces(textposition="outside", cliponaxis=False)
bar_chart.update_layout(
    height=440,
    xaxis_range=[0, 1.05],
    margin=dict(l=10, r=50, t=20, b=10),
    showlegend=False,
)
st.plotly_chart(bar_chart, use_container_width=True)


# =========================================================
# 9. 데이터 확인 및 설명
# =========================================================
with st.expander("사용한 촉매 원자료 보기"):
    st.dataframe(CATALYST_DATA, use_container_width=True)
    st.caption(
        "활성: 전류밀도(mA) · 선택성: 패러데이 효율(%) · "
        "효용성: 생성속도(mmol·h⁻¹·cm⁻²) · 안정성: 평가 점수"
    )

with st.expander("TOPSIS 계산 원리"):
    st.markdown(
        """
        1. 각 평가 기준의 값을 벡터 정규화합니다.  
        2. 사용자가 정한 가중치를 적용합니다.  
        3. 모든 기준에서 가장 좋은 값으로 이루어진 **양의 이상해**와 가장 나쁜 값으로 이루어진 **음의 이상해**를 설정합니다.  
        4. 각 촉매가 두 이상해에서 얼마나 떨어져 있는지 계산합니다.  
        5. 좋은 값에 가깝고 나쁜 값에서 멀수록 TOPSIS 점수가 높아집니다.
        """
    )

st.caption("CO₂RR Catalyst Simulator · TOPSIS-based multi-criteria evaluation")
