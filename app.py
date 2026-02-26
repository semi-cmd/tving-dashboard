import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="TVING Analytics Dashboard", page_icon="📺", layout="wide")

# =========================
# 0) Dummy data (나중에 실제 df로 교체)
# =========================
@st.cache_data
def load_data(n_users=3000, n_events=80000, seed=42):
    rng = np.random.default_rng(seed)

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "status": rng.choice(["신규", "장기", "이탈위험", "이탈"], size=n_users, p=[0.25, 0.45, 0.2, 0.1]),
        "plan": rng.choice(["베이직", "스탠다드", "프리미엄"], size=n_users, p=[0.4, 0.45, 0.15]),
        "device": rng.choice(["모바일", "TV", "PC"], size=n_users, p=[0.6, 0.25, 0.15]),
        "acq_channel": rng.choice(["검색", "SNS", "제휴", "직접"], size=n_users, p=[0.35, 0.25, 0.15, 0.25]),
    })

    start = pd.Timestamp("2025-12-01")
    ts = start + pd.to_timedelta(rng.integers(0, 60 * 24 * 60, size=n_events), unit="m")  # 60일 분
    watch_min = np.clip(rng.gamma(2.0, 20.0, size=n_events), 1, 240)
    genre = rng.choice(["드라마", "예능", "스포츠", "영화", "애니", "다큐"], size=n_events,
                       p=[0.30, 0.28, 0.10, 0.18, 0.08, 0.06])

    events = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, size=n_events),
        "ts": ts,
        "watch_min": watch_min,
        "genre": genre,
    })
    events["date"] = pd.to_datetime(events["ts"]).dt.date
    events["dow"] = pd.to_datetime(events["ts"]).dt.day_name()
    events["hour"] = pd.to_datetime(events["ts"]).dt.hour

    risk = users[["user_id"]].copy()
    risk["risk_score"] = np.clip(rng.normal(50, 18, size=n_users), 0, 100)

    return users, events, risk


users_df, events_df, risk_df = load_data()

STATUS_ORDER = ["신규", "장기", "이탈위험", "이탈"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def safe_div(a, b):
    return (a / b) if b else np.nan


# =========================
# 1) Sidebar filters
# =========================
st.sidebar.title("⚙️ 필터")

min_date = pd.to_datetime(events_df["date"]).min()
max_date = pd.to_datetime(events_df["date"]).max()
date_range = st.sidebar.date_input("기간", value=(min_date, max_date), min_value=min_date, max_value=max_date)

status_sel = st.sidebar.multiselect("상태군", STATUS_ORDER, default=STATUS_ORDER)
plan_sel = st.sidebar.multiselect("요금제", sorted(users_df["plan"].unique()), default=sorted(users_df["plan"].unique()))
device_sel = st.sidebar.multiselect("디바이스", sorted(users_df["device"].unique()), default=sorted(users_df["device"].unique()))
channel_sel = st.sidebar.multiselect("유입채널", sorted(users_df["acq_channel"].unique()), default=sorted(users_df["acq_channel"].unique()))

st.sidebar.divider()
topn = st.sidebar.slider("Top N", 5, 20, 10, 1)
show_table = st.sidebar.checkbox("표(데이터)도 보기", value=False)

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

events_f = events_df[
    (pd.to_datetime(events_df["date"]) >= start_date) &
    (pd.to_datetime(events_df["date"]) <= end_date)
].copy()

users_f = users_df[
    (users_df["status"].isin(status_sel)) &
    (users_df["plan"].isin(plan_sel)) &
    (users_df["device"].isin(device_sel)) &
    (users_df["acq_channel"].isin(channel_sel))
].copy()

events_f = events_f.merge(users_f[["user_id", "status"]], on="user_id", how="inner")
risk_f = risk_df.merge(users_f[["user_id"]], on="user_id", how="inner")


# =========================
# 2) KPI
# =========================
def compute_kpis(users, events, risk):
    if len(events) == 0:
        return dict(dau=np.nan, wau=np.nan, r7=np.nan, risk_share=np.nan, avg_watch=np.nan, avg_risk=np.nan)

    dau = events.groupby("date")["user_id"].nunique().mean()
    wau = (
        events.set_index(pd.to_datetime(events["date"]))
        .groupby(pd.Grouper(freq="W"))["user_id"]
        .nunique()
        .mean()
    )

    end = pd.to_datetime(events["date"]).max()
    last7 = events[pd.to_datetime(events["date"]) >= (end - pd.Timedelta(days=6))]["user_id"].nunique()
    last14 = events[pd.to_datetime(events["date"]) >= (end - pd.Timedelta(days=13))]["user_id"].nunique()
    r7 = safe_div(last7, last14)

    risk_share = users["status"].eq("이탈위험").mean() if len(users) else np.nan
    avg_watch = events["watch_min"].mean() if len(events) else np.nan
    avg_risk = risk["risk_score"].mean() if len(risk) else np.nan

    return dict(dau=dau, wau=wau, r7=r7, risk_share=risk_share, avg_watch=avg_watch, avg_risk=avg_risk)


k = compute_kpis(users_f, events_f, risk_f)

# =========================
# 3) Header + Flow box
# =========================
st.title("📺 TVING 행동 기반 개입 타이밍 대시보드 (Prototype)")

st.markdown(
    """
<div style="padding:12px 14px; border:1px solid #e6e6e6; border-radius:14px; background:#fafafa;">
  <div style="font-weight:700; font-size:16px; margin-bottom:6px;">의사결정 흐름</div>
  <div style="font-size:14px; line-height:1.5;">
    ① <b>Overview</b>: 지금 상태(KPI/추이/구성)를 빠르게 파악 →
    ② <b>비교</b>: 상태군별 패턴 차이(시간대/장르)를 확인 →
    ③ <b>액션</b>: 위험 점수 기반으로 <b>언제/누구에게 개입할지</b> 힌트 + 추천 결과
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.caption("변수 미확정 단계에서도 ‘필터 → 핵심 확인 → 차이 확인 → 액션’ 순서로 논의가 바로 시작되게 구성했어요.")

# KPI row
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("평균 DAU", "-" if pd.isna(k["dau"]) else f"{k['dau']:,.0f}")
c2.metric("평균 WAU", "-" if pd.isna(k["wau"]) else f"{k['wau']:,.0f}")
c3.metric("7일 리텐션(Proxy)", "-" if pd.isna(k["r7"]) else f"{k['r7']*100:.1f}%")
c4.metric("이탈위험군 비중", "-" if pd.isna(k["risk_share"]) else f"{k['risk_share']*100:.1f}%")
c5.metric("평균 시청시간(분)", "-" if pd.isna(k["avg_watch"]) else f"{k['avg_watch']:.1f}")
c6.metric("평균 위험점수", "-" if pd.isna(k["avg_risk"]) else f"{k['avg_risk']:.1f}")

# =========================
# 4) Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 패턴/상태군 비교", "🎯 개입 타이밍/추천"])


# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    st.markdown("### 1) 지금 무슨 일이 일어나고 있지?")
    st.caption("전체 규모(KPI)와 추이(DAU), 구성(상태군/장르)을 먼저 확인해서 ‘이상 신호’를 찾는 단계예요.")

    left, right = st.columns([1.45, 1])

    with left:
        st.subheader("DAU 추이")
        if len(events_f) == 0:
            st.info("선택한 필터에서 이벤트가 없습니다.")
        else:
            dau_series = events_f.groupby("date")["user_id"].nunique().reset_index(name="DAU")
            fig = px.line(dau_series, x="date", y="DAU")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("상태군 분포")
        status_cnt = users_f["status"].value_counts().reindex(STATUS_ORDER).fillna(0).reset_index()
        status_cnt.columns = ["status", "users"]
        fig = px.bar(status_cnt, x="status", y="users")
        # 막대 얇게
        fig.update_traces(marker_line_width=0, width=0.55)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 장르 & 유저 시청 분포")
    a, b = st.columns(2)

    with a:
        if len(events_f) == 0:
            st.info("선택한 필터에서 이벤트가 없습니다.")
        else:
            g = (
                events_f.groupby("genre")["watch_min"]
                .sum()
                .sort_values(ascending=False)
                .head(topn)
                .reset_index(name="watch_min")
            )
            fig = px.bar(g.sort_values("watch_min"), x="watch_min", y="genre", orientation="h")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

    with b:
        if len(events_f) == 0:
            st.info("선택한 필터에서 이벤트가 없습니다.")
        else:
            u = events_f.groupby("user_id")["watch_min"].sum().reset_index(name="watch_sum")
            fig = px.histogram(u, x="watch_sum")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.info("다음: 상태군별로 패턴이 다른지 확인하려면 ‘📊 패턴/상태군 비교’ 탭으로 이동하세요.")

    if show_table:
        st.divider()
        st.write("events_f (head)")
        st.dataframe(events_f.head(200), use_container_width=True)


# -------------------------
# Tab 2: Pattern compare
# -------------------------
with tab2:
    st.markdown("### 2) 어떤 집단에서, 어떤 패턴이 다르지?")
    st.caption("상태군별로 시간대/요일(히트맵)과 장르 구성비를 비교해서 ‘차이’를 찾는 단계예요.")

    if len(events_f) == 0:
        st.info("선택한 필터에서 이벤트가 없습니다.")
    else:
        st.subheader("상태군별 시청시간 요약(유저 기준)")
        user_watch = events_f.groupby(["user_id", "status"])["watch_min"].sum().reset_index(name="watch_sum")
        grp = (
            user_watch.groupby("status")["watch_sum"]
            .agg(users="count", mean="mean", median="median")
            .reset_index()
        )
        grp["status"] = pd.Categorical(grp["status"], STATUS_ORDER)
        grp = grp.sort_values("status")
        st.dataframe(grp, use_container_width=True)

        col1, col2 = st.columns([1.25, 1])

        with col1:
            st.subheader("요일 × 시간 히트맵")
            status_one = st.selectbox("상태군 선택", STATUS_ORDER, index=2, key="heat_status")
            e = events_f[events_f["status"] == status_one].copy()

            if len(e) == 0:
                st.info("선택한 상태군 데이터가 없습니다.")
            else:
                pivot = e.pivot_table(index="dow", columns="hour", values="watch_min", aggfunc="sum", fill_value=0)
                pivot = pivot.reindex([d for d in DOW_ORDER if d in pivot.index])
                pivot = pivot.sort_index(axis=1)

                fig = px.imshow(pivot, aspect="auto")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("상태군별 장르 구성비")
            genre_tbl = (
                events_f.groupby(["status", "genre"])["watch_min"]
                .sum()
                .reset_index(name="watch_sum")
            )
            genre_tbl["share"] = genre_tbl["watch_sum"] / genre_tbl.groupby("status")["watch_sum"].transform("sum")

            genre_top = (
                genre_tbl.sort_values(["status", "share"], ascending=[True, False])
                .groupby("status", as_index=False)
                .head(topn)
            )

            g1, g2 = st.columns(2)
            for i, s in enumerate(STATUS_ORDER):
                sub = genre_top[genre_top["status"] == s].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values("share")
                fig = px.bar(sub, x="share", y="genre", orientation="h", title=s)
                fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=250)
                if i % 2 == 0:
                    g1.plotly_chart(fig, use_container_width=True)
                else:
                    g2.plotly_chart(fig, use_container_width=True)

    st.info("다음: 개입 타이밍 힌트를 보려면 ‘🎯 개입 타이밍/추천’ 탭으로 이동하세요.")

    if show_table:
        st.divider()
        st.write("users_f (head)")
        st.dataframe(users_f.head(200), use_container_width=True)


# -------------------------
# Tab 3: Action timing
# -------------------------
with tab3:
    st.markdown("### 3) 그래서 언제/누구에게 개입할까?")
    st.caption("위험 점수 구간별 행동(시청시간 등)을 보고 ‘개입 타이밍’을 정하는 단계예요. 마지막에 추천 결과를 확인해요.")

    st.subheader("위험 점수 분포")
    if len(risk_f) == 0:
        st.info("선택한 필터에서 유저가 없습니다.")
    else:
        fig = px.histogram(risk_f, x="risk_score")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decile별 ‘행동 지표’ 비교")
    st.caption("Decile에서 유저 수는 qcut 특성상 비슷하게 나뉘어 의미가 약해서, 행동 지표(예: 시청시간)로 비교해요.")

    if len(risk_f) >= 10 and len(events_f) > 0:
        tmp = risk_f.copy()
        tmp["decile"] = pd.qcut(tmp["risk_score"], 10, labels=[f"D{i}" for i in range(1, 11)])

        user_watch = events_f.groupby("user_id")["watch_min"].sum().reset_index(name="watch_sum")
        tmp = tmp.merge(user_watch, on="user_id", how="left").fillna({"watch_sum": 0})

        dec = tmp.groupby("decile")["watch_sum"].mean().reset_index()
        fig = px.bar(dec, x="decile", y="watch_sum", title="Decile별 평균 시청시간(기간 합계)")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Decile 분석을 하려면 유저가 충분히 있고(events도 필요), 현재 필터에선 조건이 부족합니다.")

    st.subheader("추천 결과(프로토타입)")
    st.caption("실제 추천 로직이 붙으면 아래 테이블만 결과로 교체하면 돼요.")

    sample_users = users_f["user_id"].sample(min(10, len(users_f)), random_state=1).tolist() if len(users_f) else []
    target_user = st.selectbox("사용자 선택", sample_users) if sample_users else None

    if target_user:
        rec = pd.DataFrame({
            "rank": [1, 2, 3, 4, 5],
            "content_id": [f"C{n:04d}" for n in range(101, 106)],
            "title": ["콘텐츠A", "콘텐츠B", "콘텐츠C", "콘텐츠D", "콘텐츠E"],
            "expected_score": np.round(np.sort(np.random.rand(5))[::-1], 3)
        })
        st.dataframe(rec, use_container_width=True)
    else:
        st.info("필터 결과에 사용자가 없습니다.")

    if show_table:
        st.divider()
        st.write("risk_f (head)")
        st.dataframe(risk_f.head(200), use_container_width=True)
