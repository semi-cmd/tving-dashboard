import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TVING 대시보드(업로드 버전)", layout="wide")

st.title("TVING 변수 미확정 대시보드 (업로드 버전)")
st.caption("Streamlit Cloud에서 가장 안정적으로 동작하도록: 엑셀을 화면에서 업로드 받는 방식")

uploaded = st.file_uploader("엑셀 파일(.xlsx) 업로드", type=["xlsx"])

if uploaded is None:
    st.info("위에서 엑셀 파일을 업로드하면 자동으로 미리보기/차트를 보여줄게요.")
    st.stop()

df = pd.read_excel(uploaded)

st.subheader("1) 데이터 미리보기")
st.write(f"행 {len(df):,} / 열 {df.shape[1]:,}")
st.dataframe(df.head(200), use_container_width=True)

st.subheader("2) 결측치 Top 20")
nulls = df.isna().mean().sort_values(ascending=False).head(20)
st.dataframe((nulls * 100).round(1).rename("% null").to_frame(), use_container_width=True)

# 숫자 컬럼 자동 찾기
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols]

st.subheader("3) 빠른 차트 (변수 미확정용)")
if len(num_cols) == 0:
    st.warning("숫자 컬럼이 없어서 히스토그램을 만들기 어려워요. (숫자 컬럼이 있으면 자동으로 차트가 나와요)")
else:
    col = st.selectbox("숫자 컬럼 선택", num_cols)
    st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

if len(cat_cols) > 0:
    cat = st.selectbox("카테고리 컬럼 선택(Top 20)", cat_cols)
    top = df[cat].astype(str).value_counts().head(20).reset_index()
    top.columns = [cat, "count"]
    st.plotly_chart(px.bar(top, x="count", y=cat, orientation="h"), use_container_width=True)
