import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import ollama

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="🤖 AI Data Analysis Agent", layout="wide")

st.title("🤖 AI Data Analysis Agent (EDA + Cleaning + LLM)")
st.write("Upload CSV → Auto EDA → Cleaning → Ask AI Questions")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def auto_clean_dataset(df):
    df_clean = df.copy()

    # Drop duplicates
    df_clean = df_clean.drop_duplicates()

    # Fill numeric missing with median
    for col in df_clean.select_dtypes(include=["int64", "float64"]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill categorical missing with mode
    for col in df_clean.select_dtypes(include=["object"]).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean


def relationship_finder(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    # Top correlations
    corr_pairs = corr.abs().unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1]

    return corr_pairs.head(5)


def ask_llm_about_data(df, question):
    """
    Sends dataset summary + user question to Llama3 via Ollama
    """

    summary = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    Missing Values: {df.isnull().sum().sum()}
    Numeric Columns: {list(df.select_dtypes(include=['int64','float64']).columns)}
    """

    prompt = f"""
    You are a professional Data Analyst AI Agent.

    Dataset Summary:
    {summary}

    User Question:
    {question}

    Answer clearly with insights and recommendations.
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# ---------------------------------------------------------
# Upload Dataset
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Loaded Successfully!")

    # Preview
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head(10))

    # Dataset Summary
    st.subheader("📊 Dataset Info")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------
    st.subheader("✨ Auto Cleaning Agent")

    if st.button("Clean Dataset Automatically"):
        df = auto_clean_dataset(df)
        st.success("✅ Dataset Cleaned Successfully!")
        st.dataframe(df.head(10))

    # ---------------------------------------------------------
    # Relationship Finder
    # ---------------------------------------------------------
    st.subheader("📌 Top Feature Relationships")

    top_corr = relationship_finder(df)

    if top_corr is not None:
        st.write("🔥 Strongest Correlations:")
        st.write(top_corr)
    else:
        st.warning("Not enough numeric data for correlation.")

    # ---------------------------------------------------------
    # Auto EDA Report
    # ---------------------------------------------------------
    st.subheader("📑 Generate Full EDA Report")

    if st.button("Generate EDA Report"):

        with st.spinner("⏳ Creating report..."):
            profile = ProfileReport(df, title="EDA Report", explorative=True)
            profile.to_file("eda_report.html")

        st.success("✅ Report Generated!")

        with open("eda_report.html", "r", encoding="utf-8") as f:
            html_data = f.read()

        components.html(html_data, height=900, scrolling=True)

    # ---------------------------------------------------------
    # LLM Data Analyst Chat Agent
    # ---------------------------------------------------------
    st.subheader("🧠 Ask AI Analyst Questions")

    question = st.text_input("Ask something about your dataset:")

    if st.button("Ask AI") and question:

        with st.spinner("🤖 Thinking..."):
            answer = ask_llm_about_data(df, question)

        st.success("✅ AI Response:")
        st.write(answer)

# Footer
st.markdown("---")
st.markdown("🚀 Phase 4: Full AI Analyst Agent Ready")
