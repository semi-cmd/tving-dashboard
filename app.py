import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="TVING Prototype Dashboard", layout="wide")

DATA_FILES = {
    "시청(watch)": {"path": "watch_data.xlsx", "sheet": None},
    "검색(search)": {"path": "search_data.xlsx", "sheet": None},
    "추천(recommend)": {"path": "recommend_data.xlsx", "sheet": None},
    "이탈(churn)": {"path": "churn_final_data.xlsx", "sheet": None},
}

@st.cache_data(show_spinner=False)
def load_excel(path: str, sheet_name=None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    # Common cleaning
    for col in df.columns:
        if "timestamp" in col.lower() or col.lower().endswith("_date") or col.lower().endswith(" date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def infer_types(df: pd.DataFrame):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns if c not in numeric + datetime]
    return numeric, datetime, cat

def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def date_filter(df: pd.DataFrame, date_col: str):
    if date_col not in df.columns:
        return df
    dmin = df[date_col].min()
    dmax = df[date_col].max()
    if pd.isna(dmin) or pd.isna(dmax):
        return df
    start, end = st.sidebar.date_input(
        "기간 필터",
        value=(dmin.date(), dmax.date()),
        min_value=dmin.date(),
        max_value=dmax.date()
    )
    start = pd.to_datetime(start)
    end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df[date_col] >= start) & (df[date_col] <= end)]

def quick_chart_builder(df: pd.DataFrame, key_prefix: str):
    st.subheader("빠른 차트 빌더 (변수 미확정일 때 유용)")
    numeric, datetime_cols, cat = infer_types(df)

    chart_type = st.selectbox(
        "차트 타입",
        ["라인(시간추이)", "바(집계)", "히스토그램", "박스플롯", "산점도"],
        key=f"{key_prefix}_chart_type"
    )

    with st.expander("설정", expanded=True):
        x = st.selectbox("X 축", datetime_cols + cat + numeric, key=f"{key_prefix}_x")
        if chart_type in ["히스토그램"]:
            y = None
        else:
            y_candidates = numeric if numeric else df.columns.tolist()
            y = st.selectbox("Y 축", y_candidates, key=f"{key_prefix}_y") if chart_type != "박스플롯" else st.selectbox("값(숫자)", numeric, key=f"{key_prefix}_y_box") if numeric else None

        color = st.selectbox("색상(그룹)", ["(없음)"] + cat, key=f"{key_prefix}_color")
        agg = st.selectbox("집계", ["count", "sum", "mean", "median"], key=f"{key_prefix}_agg")
        topn = st.slider("카테고리 Top N(바 차트)", 5, 50, 15, key=f"{key_prefix}_topn")

    d = df.copy()

    # Build chart
    try:
        if chart_type == "라인(시간추이)":
            if x not in datetime_cols:
                st.info("라인 차트는 X축이 날짜/시간 컬럼일 때 가장 좋아요. (지금은 그래도 그려봅니다)")
            if agg == "count":
                g = d.groupby(pd.Grouper(key=x, freq="D")).size().reset_index(name="value")
                fig = px.line(g, x=x, y="value")
            else:
                g = d.groupby(pd.Grouper(key=x, freq="D"))[y].agg(agg).reset_index(name="value")
                fig = px.line(g, x=x, y="value")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "바(집계)":
            if agg == "count":
                g = d.groupby(x).size().reset_index(name="value").sort_values("value", ascending=False).head(topn)
            else:
                g = d.groupby(x)[y].agg(agg).reset_index(name="value").sort_values("value", ascending=False).head(topn)
            fig = px.bar(g, x=x, y="value", color=None if color == "(없음)" else color)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "히스토그램":
            if x in numeric:
                fig = px.histogram(d, x=x, color=None if color == "(없음)" else color)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("히스토그램은 숫자 컬럼을 X로 선택하는 걸 추천해요.")
                fig = px.histogram(d, x=x)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "박스플롯":
            if y is None:
                st.warning("숫자 컬럼이 없어 박스플롯을 만들기 어렵습니다.")
                return
            fig = px.box(d, x=x, y=y, color=None if color == "(없음)" else color)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "산점도":
            if not numeric:
                st.warning("숫자 컬럼이 없어 산점도를 만들기 어렵습니다.")
                return
            x_num = x if x in numeric else st.selectbox("X(숫자)", numeric, key=f"{key_prefix}_x_scatter_num")
            y_num = y if (y in numeric) else st.selectbox("Y(숫자)", numeric, key=f"{key_prefix}_y_scatter_num")
            fig = px.scatter(d, x=x_num, y=y_num, color=None if color == "(없음)" else color)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"차트 생성 중 에러: {e}")

def watch_tab(df: pd.DataFrame):
    st.header("시청 데이터")
    date_col = "timestamp" if "timestamp" in df.columns else None
    if date_col:
        df2 = date_filter(df, date_col)
    else:
        df2 = df

    # Sidebar filters
    with st.sidebar.expander("시청 필터", expanded=False):
        if "device_type" in df2.columns:
            devices = st.multiselect("기기", sorted(df2["device_type"].dropna().unique().tolist()), default=None)
            if devices:
                df2 = df2[df2["device_type"].isin(devices)]
        if "genre_primary" in df2.columns:
            genres = st.multiselect("장르", sorted(df2["genre_primary"].dropna().unique().tolist()), default=None)
            if genres:
                df2 = df2[df2["genre_primary"].isin(genres)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("이벤트 수", f"{len(df2):,}")
    with c2:
        kpi_card("유니크 유저", f"{df2['user_id'].nunique():,}" if "user_id" in df2.columns else "-")
    with c3:
        kpi_card("세션 수", f"{df2['session_id'].nunique():,}" if "session_id" in df2.columns else "-")
    with c4:
        if "watch_duration_minutes" in df2.columns:
            kpi_card("총 시청(분)", f"{df2['watch_duration_minutes'].sum():,.0f}")
        else:
            kpi_card("총 시청(분)", "-")

    left, right = st.columns([1.2, 1])
    with left:
        if date_col:
            # DAU (events-based)
            g = df2.groupby(pd.Grouper(key=date_col, freq="D"))["user_id"].nunique().reset_index(name="DAU")
            fig = px.line(g, x=date_col, y="DAU")
            st.subheader("DAU 추이(일)")
            st.plotly_chart(fig, use_container_width=True)

        if "watch_duration_minutes" in df2.columns:
            st.subheader("시청시간 분포")
            st.plotly_chart(px.histogram(df2, x="watch_duration_minutes"), use_container_width=True)

    with right:
        if "device_type" in df2.columns:
            st.subheader("기기별 이벤트 비중")
            g = df2["device_type"].value_counts().reset_index()
            g.columns = ["device_type", "count"]
            st.plotly_chart(px.pie(g, names="device_type", values="count"), use_container_width=True)

        if "title" in df2.columns:
            st.subheader("Top 콘텐츠(이벤트 기준)")
            top = df2["title"].value_counts().head(15).reset_index()
            top.columns = ["title", "count"]
            st.plotly_chart(px.bar(top, x="count", y="title", orientation="h"), use_container_width=True)

    st.divider()
    quick_chart_builder(df2, "watch_qcb")
    st.divider()
    st.subheader("데이터 미리보기")
    st.dataframe(df2.head(200), use_container_width=True)

def search_tab(df: pd.DataFrame):
    st.header("검색 데이터")
    date_col = "timestamp" if "timestamp" in df.columns else None
    df2 = date_filter(df, date_col) if date_col else df

    with st.sidebar.expander("검색 필터", expanded=False):
        if "device_type" in df2.columns:
            devices = st.multiselect("기기", sorted(df2["device_type"].dropna().unique().tolist()), default=None, key="search_device")
            if devices:
                df2 = df2[df2["device_type"].isin(devices)]
        if "used_filters" in df2.columns:
            uf = st.multiselect("필터 사용 여부", sorted(df2["used_filters"].dropna().unique().tolist()), default=None, key="search_used_filters")
            if uf:
                df2 = df2[df2["used_filters"].isin(uf)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("검색 이벤트", f"{len(df2):,}")
    with c2:
        kpi_card("유니크 유저", f"{df2['user_id'].nunique():,}" if "user_id" in df2.columns else "-")
    with c3:
        if "results_returned" in df2.columns:
            kpi_card("평균 결과 수", f"{df2['results_returned'].mean():.1f}")
        else:
            kpi_card("평균 결과 수", "-")
    with c4:
        if "clicked_result_position" in df2.columns:
            click_rate = df2["clicked_result_position"].notna().mean()
            kpi_card("클릭 발생률", f"{click_rate*100:.1f}%")
        else:
            kpi_card("클릭 발생률", "-")

    left, right = st.columns([1.2, 1])
    with left:
        if date_col:
            g = df2.groupby(pd.Grouper(key=date_col, freq="D")).size().reset_index(name="검색 수")
            st.subheader("검색량 추이(일)")
            st.plotly_chart(px.line(g, x=date_col, y="검색 수"), use_container_width=True)

        if "search_duration_seconds" in df2.columns:
            st.subheader("검색 소요시간 분포(초)")
            st.plotly_chart(px.histogram(df2, x="search_duration_seconds"), use_container_width=True)

    with right:
        if "search_query" in df2.columns:
            st.subheader("Top 검색어")
            top = df2["search_query"].astype(str).value_counts().head(20).reset_index()
            top.columns = ["search_query", "count"]
            st.plotly_chart(px.bar(top, x="count", y="search_query", orientation="h"), use_container_width=True)

    st.divider()
    quick_chart_builder(df2, "search_qcb")
    st.divider()
    st.subheader("데이터 미리보기")
    st.dataframe(df2.head(200), use_container_width=True)

def recommend_tab(df: pd.DataFrame):
    st.header("추천 데이터")
    date_col = "timestamp" if "timestamp" in df.columns else None
    df2 = date_filter(df, date_col) if date_col else df

    with st.sidebar.expander("추천 필터", expanded=False):
        if "recommendation_type" in df2.columns:
            rtypes = st.multiselect("추천 타입", sorted(df2["recommendation_type"].dropna().unique().tolist()), default=None)
            if rtypes:
                df2 = df2[df2["recommendation_type"].isin(rtypes)]
        if "device_type" in df2.columns:
            devices = st.multiselect("기기", sorted(df2["device_type"].dropna().unique().tolist()), default=None, key="rec_device")
            if devices:
                df2 = df2[df2["device_type"].isin(devices)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("노출(행) 수", f"{len(df2):,}")
    with c2:
        kpi_card("유니크 유저", f"{df2['user_id'].nunique():,}" if "user_id" in df2.columns else "-")
    with c3:
        if "was_clicked" in df2.columns:
            ctr = df2["was_clicked"].mean()
            kpi_card("CTR", f"{ctr*100:.2f}%")
        else:
            kpi_card("CTR", "-")
    with c4:
        if "recommendation_score" in df2.columns:
            kpi_card("평균 점수", f"{df2['recommendation_score'].mean():.3f}")
        else:
            kpi_card("평균 점수", "-")

    left, right = st.columns([1.2, 1])
    with left:
        if date_col and "was_clicked" in df2.columns:
            g = df2.groupby(pd.Grouper(key=date_col, freq="D"))["was_clicked"].mean().reset_index(name="CTR")
            st.subheader("CTR 추이(일)")
            st.plotly_chart(px.line(g, x=date_col, y="CTR"), use_container_width=True)
        elif date_col:
            g = df2.groupby(pd.Grouper(key=date_col, freq="D")).size().reset_index(name="노출 수")
            st.subheader("노출 추이(일)")
            st.plotly_chart(px.line(g, x=date_col, y="노출 수"), use_container_width=True)

        if "position_in_list" in df2.columns and "was_clicked" in df2.columns:
            st.subheader("포지션별 CTR")
            g = df2.groupby("position_in_list")["was_clicked"].mean().reset_index(name="CTR").sort_values("position_in_list")
            st.plotly_chart(px.bar(g, x="position_in_list", y="CTR"), use_container_width=True)

    with right:
        if "title" in df2.columns and "was_clicked" in df2.columns:
            st.subheader("클릭 Top 콘텐츠")
            g = df2[df2["was_clicked"] == 1]["title"].value_counts().head(15).reset_index()
            g.columns = ["title", "clicks"]
            st.plotly_chart(px.bar(g, x="clicks", y="title", orientation="h"), use_container_width=True)

        if "algorithm_version" in df2.columns and "was_clicked" in df2.columns:
            st.subheader("버전별 CTR")
            g = df2.groupby("algorithm_version")["was_clicked"].mean().reset_index(name="CTR").sort_values("CTR", ascending=False)
            st.plotly_chart(px.bar(g, x="algorithm_version", y="CTR"), use_container_width=True)

    st.divider()
    quick_chart_builder(df2, "rec_qcb")
    st.divider()
    st.subheader("데이터 미리보기")
    st.dataframe(df2.head(200), use_container_width=True)

def churn_tab(df: pd.DataFrame):
    st.header("이탈(churn) 데이터")
    # churn data often has no timestamp; use signup_date if present
    date_col = "signup_date" if "signup_date" in df.columns else None
    df2 = date_filter(df, date_col) if date_col else df

    with st.sidebar.expander("이탈 필터", expanded=False):
        if "subscription_plan" in df2.columns:
            plans = st.multiselect("요금제", sorted(df2["subscription_plan"].dropna().unique().tolist()), default=None)
            if plans:
                df2 = df2[df2["subscription_plan"].isin(plans)]
        if "device" in df2.columns:
            dev = st.multiselect("주사용 기기(device)", sorted(df2["device"].dropna().unique().tolist()), default=None)
            if dev:
                df2 = df2[df2["device"].isin(dev)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("유저 수", f"{len(df2):,}")
    with c2:
        if "churn_status" in df2.columns:
            churn_rate = df2["churn_status"].mean()
            kpi_card("Churn rate", f"{churn_rate*100:.1f}%")
        else:
            kpi_card("Churn rate", "-")
    with c3:
        if "watch_hours" in df2.columns:
            kpi_card("평균 시청(시간)", f"{df2['watch_hours'].mean():.1f}")
        else:
            kpi_card("평균 시청(시간)", "-")
    with c4:
        if "days_since_last_login" in df2.columns:
            kpi_card("평균 미접속일", f"{df2['days_since_last_login'].mean():.1f}")
        else:
            kpi_card("평균 미접속일", "-")

    # Churn breakdown
    if "churn_status" in df2.columns:
        st.subheader("이탈 여부 분포")
        g = df2["churn_status"].value_counts().reset_index()
        g.columns = ["churn_status", "count"]
        st.plotly_chart(px.pie(g, names="churn_status", values="count"), use_container_width=True)

    # Simple drivers (boxplots)
    st.subheader("이탈 vs 지표 비교(간단)")
    numeric, _, _ = infer_types(df2)
    candidates = [c for c in ["watch_hours", "days_since_last_login", "avg_weekly_usage_hours", "tenure_months", "completion_rate"] if c in df2.columns]
    if "churn_status" in df2.columns and candidates:
        cols = st.columns(min(3, len(candidates)))
        for i, col in enumerate(candidates[:3]):
            with cols[i]:
                fig = px.box(df2, x="churn_status", y=col)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("churn_status 또는 비교할 숫자 지표가 부족해서 박스플롯을 생략했어요.")

    st.divider()
    quick_chart_builder(df2, "churn_qcb")
    st.divider()
    st.subheader("데이터 미리보기")
    st.dataframe(df2.head(200), use_container_width=True)

def main():
    st.title("TVING 변수 미확정 단계용 Streamlit 대시보드(프로토타입)")
    st.caption("목적: 변수 확정 전에도 빠르게 KPI/추이/분포를 훑고, 필요한 변수를 자연스럽게 '발견'할 수 있게 만들기")

    dataset = st.sidebar.selectbox("데이터 선택", list(DATA_FILES.keys()))
    df = load_excel(DATA_FILES[dataset]["path"], sheet_name=DATA_FILES[dataset]["sheet"])

    with st.sidebar.expander("공통 옵션", expanded=False):
        st.write(f"행: {len(df):,} / 열: {df.shape[1]:,}")
        show_nulls = st.checkbox("결측치 요약 보기", value=False)
        if show_nulls:
            nulls = df.isna().mean().sort_values(ascending=False).head(20)
            st.dataframe((nulls*100).round(1).rename("% null").to_frame(), use_container_width=True)

    tab = st.tabs(["요약", "대시보드"])[1]

    # Render by dataset
    if dataset.startswith("시청"):
        watch_tab(df)
    elif dataset.startswith("검색"):
        search_tab(df)
    elif dataset.startswith("추천"):
        recommend_tab(df)
    else:
        churn_tab(df)

if __name__ == "__main__":
    main()
