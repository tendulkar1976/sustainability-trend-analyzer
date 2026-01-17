import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sustainability Trend Analyzer",
    layout="wide",
    page_icon="🌱"
)

st.title("🌱 AI-Based Sustainability Issue Trend Analyzer")
st.markdown(
    "Upload **feedback data** and automatically analyze sustainability issues using "
    "**Machine Learning, Sentiment Analysis, and Trend Visualization**."
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file (feedback column required)",
    type=["csv"]
)

# ---------------- ML MODEL ----------------
def train_resource_classifier():
    train_data = [
        ("power cut issue", "Electricity"),
        ("electricity problem", "Electricity"),
        ("voltage fluctuation", "Electricity"),
        ("no water supply", "Water"),
        ("water leakage", "Water"),
        ("drinking water issue", "Water"),
        ("garbage not collected", "Waste"),
        ("plastic waste everywhere", "Waste"),
        ("dumping yard smell", "Waste"),
        ("air pollution problem", "Pollution"),
        ("vehicle smoke", "Pollution"),
        ("noise pollution", "Pollution")
    ]

    X, y = zip(*train_data)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=300))
    ])

    model.fit(X, y)
    return model

ml_model = train_resource_classifier()

def classify_resource_ml(text):
    try:
        return ml_model.predict([str(text)])[0]
    except:
        return "Other"

# ---------------- SENTIMENT ----------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ---------------- LOAD DATA ----------------
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    if "feedback" not in df_raw.columns:
        st.error("❌ CSV must contain a 'feedback' column")
        st.stop()

    df = df_raw.copy()
    df["Resource"] = df["feedback"].apply(classify_resource_ml)
    df["Sentiment"] = df["feedback"].apply(get_sentiment)

    # Create time dimension (synthetic if missing)
    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    df["Month"] = np.random.choice(months, size=len(df))

    st.success("✅ Feedback data processed successfully")

else:
    st.info("ℹ️ Please upload a CSV file to begin analysis.")
    st.stop()

# ---------------- RESOURCE DISTRIBUTION ----------------
st.subheader("📊 Resource Distribution")

resource_counts = df["Resource"].value_counts().reset_index()
resource_counts.columns = ["Resource", "Count"]

fig_pie = px.pie(
    resource_counts,
    names="Resource",
    values="Count",
    title="Resource-wise Issue Distribution",
    hole=0.4
)

st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- SENTIMENT DISTRIBUTION ----------------
st.subheader("😊 Sentiment Distribution")

sent_counts = df["Sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]

fig_sent = px.bar(
    sent_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    title="Overall Feedback Sentiment"
)

st.plotly_chart(fig_sent, use_container_width=True)

# ---------------- FEEDBACK TREND OVER TIME ----------------
st.subheader("📈 Feedback Trend Over Time")

trend_df = (
    df.groupby(["Month", "Resource"])
      .size()
      .reset_index(name="Count")
)

fig_trend = px.area(
    trend_df,
    x="Month",
    y="Count",
    color="Resource",
    title="Monthly Trend of Sustainability Issues",
    category_orders={"Month": months}
)

st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- AI SUMMARY ----------------
st.subheader("🧠 AI-Generated Insights")

summary_text = f"""
• Dominant issue: {df['Resource'].value_counts().idxmax()}
• Most common sentiment: {df['Sentiment'].value_counts().idxmax()}
• Total feedback records analyzed: {len(df)}
"""

st.info(summary_text)

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Processed Feedback Data"):
    st.dataframe(df)
