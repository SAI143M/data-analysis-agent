import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import google.generativeai as genai
import os

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="🤖 AI Data Analyst Agent", layout="wide")

st.title("🤖 AI Data Analyst Agent (Cloud Version)")
st.write("Upload CSV → Auto EDA → Cleaning → Gemini AI Insights")

# ---------------------------------------------------------
# GEMINI CONFIGURATION
# ---------------------------------------------------------
# Streamlit Cloud Secrets: GEMINI_API_KEY
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")


# ---------------------------------------------------------
# SAFE CLEANING FUNCTION
# ---------------------------------------------------------
def auto_clean_dataset(df):
    df_clean = df.copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Fill numeric missing values
    for col in df_clean.select_dtypes(include=["int64", "float64"]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill categorical missing values
    for col in df_clean.select_dtypes(include=["object"]).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean


# ---------------------------------------------------------
# GEMINI AI INSIGHT AGENT
# ---------------------------------------------------------
def generate_ai_report(df):

    summary = f"""
    Dataset Shape: {df.shape}

    Columns: {list(df.columns)}

    Missing Values:
    {df.isnull().sum().to_dict()}

    Numeric Columns:
    {list(df.select_dtypes(include=['int64','float64']).columns)}

    Categorical Columns:
    {list(df.select_dtypes(include=['object']).columns)}
    """

    prompt = f"""
    You are a professional senior Data Analyst AI.

    Dataset Summary:
    {summary}

    TASK:
    1. Suggest cleaning improvements
    2. Give top 5 key insights
    3. Explain strong relationships between columns
    4. Suggest best charts for dashboard
    5. Recommend next analysis or ML steps

    Answer in a structured report format.
    """

    response = model.generate_content(prompt)
    return response.text


def ask_ai_question(df, question):

    summary = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    """

    prompt = f"""
    You are an AI Data Analyst.

    Dataset Summary:
    {summary}

    User Question:
    {question}

    Give a clear helpful answer.
    """

    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------
# FILE UPLOAD
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
    # CLEANING BUTTON
    # ---------------------------------------------------------
    st.subheader("✨ Auto Cleaning Agent")

    if st.button("Clean Dataset Automatically"):

        df = auto_clean_dataset(df)

        st.success("✅ Dataset Cleaned Successfully!")
        st.dataframe(df.head(10))

        # Download cleaned dataset
        csv_data = df.to_csv(index=False)

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    # ---------------------------------------------------------
    # AI INSIGHT REPORT
    # ---------------------------------------------------------
    st.subheader("🧠 Gemini AI Full Insight Report")

    if st.button("Generate AI Analyst Report"):

        with st.spinner("🤖 Gemini is analyzing your dataset..."):

            ai_report = generate_ai_report(df)

        st.success("✅ AI Report Generated!")
        st.markdown("### 📌 AI Analyst Report")
        st.write(ai_report)

    # ---------------------------------------------------------
    # AI CHAT QUESTION
    # ---------------------------------------------------------
    st.subheader("💬 Ask Gemini Questions")

    user_q = st.text_input("Ask something about your dataset:")

    if st.button("Ask AI") and user_q:

        with st.spinner("🤖 Thinking..."):

            answer = ask_ai_question(df, user_q)

        st.write("### ✅ AI Answer")
        st.write(answer)

    # ---------------------------------------------------------
    # FULL EDA REPORT
    # ---------------------------------------------------------
    st.subheader("📑 Auto EDA Profiling Report")

    if st.button("Generate Full EDA Report"):

        profile = ProfileReport(df, title="EDA Report", explorative=True)
        profile.to_file("eda_report.html")

        st.success("✅ EDA Report Created!")

        with open("eda_report.html", "r", encoding="utf-8") as f:
            html_data = f.read()

        components.html(html_data, height=900, scrolling=True)


st.markdown("---")
st.markdown("🚀 Deployed AI Data Analyst Agent (Gemini Cloud Version)")
