import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

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

# ---------------- CSS FOR ANIMATED DOTS ----------------
st.markdown("""
<style>
.dot-container { text-align:center; margin-top:10px; }
.dot { display:inline-block; font-size:22px; margin:0 6px; color:#bbb;
       transition: all 0.3s ease-in-out; }
.dot.active { color:#2ecc71; transform:scale(1.6); animation:pulse 0.8s; }
@keyframes pulse {
  0% { transform:scale(1.2); }
  50% { transform:scale(1.8); }
  100% { transform:scale(1.6); }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "graph_index" not in st.session_state:
    st.session_state.graph_index = 0
if "fullscreen" not in st.session_state:
    st.session_state.fullscreen = False

# ---------------- TITLE ----------------
st.title("🌱 AI-Based Sustainability Issue Trend Analyzer")
st.markdown(
    "Transform **unstructured citizen feedback** into **explainable, actionable sustainability intelligence**."
)

# ---------------- MONTHS ----------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ---------------- ML MODEL ----------------
def train_resource_classifier():
    train_data = [
        ("power cut issue", "Electricity"),
        ("voltage fluctuation", "Electricity"),
        ("no water supply", "Water"),
        ("water leakage", "Water"),
        ("garbage not collected", "Waste"),
        ("plastic waste problem", "Waste"),
        ("air pollution", "Pollution"),
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

def classify_with_confidence(text):
    probs = ml_model.predict_proba([str(text)])[0]
    classes = ml_model.classes_
    idx = np.argmax(probs)
    return classes[idx], round(probs[idx] * 100, 2)

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

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload CSV (must contain 'feedback' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("CSV must contain a 'feedback' column")
        st.stop()
else:
    df = pd.DataFrame(columns=["feedback"])

# ---------------- PROCESS DATA ----------------
if not df.empty:
    df["Resource"], df["Confidence"] = zip(*df["feedback"].apply(classify_with_confidence))
    df["Sentiment"] = df["feedback"].apply(get_sentiment)
    df["Month"] = np.random.choice(months, size=len(df))

# ---------------- MANUAL INPUT ----------------
st.markdown("---")
st.subheader("✍️ Analyze Feedback Manually")

user_feedback = st.text_area(
    "Enter sustainability-related feedback",
    placeholder="Example: Frequent power cuts in my area at night..."
)

if st.button("🔍 Analyze Feedback") and user_feedback.strip():
    predicted_resource, confidence = classify_with_confidence(user_feedback)
    sentiment = get_sentiment(user_feedback)

    st.markdown("### 🤖 AI Analysis Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resource", predicted_resource)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Sentiment", sentiment)

    if st.button("➕ Add to Dataset"):
        df.loc[len(df)] = {
            "feedback": user_feedback,
            "Resource": predicted_resource,
            "Confidence": confidence,
            "Sentiment": sentiment,
            "Month": months[datetime.now().month - 1]
        }
        st.success("Added to dataset. Graphs updated.")
        st.rerun()

# ---------------- STOP IF NO DATA ----------------
if df.empty:
    st.info("Upload a CSV or add manual feedback to see analytics.")
    st.stop()

# ---------------- TIME CONTEXT ----------------
current_month = months[datetime.now().month - 1]
current_index = months.index(current_month)
last_month = months[current_index - 1] if current_index > 0 else None

# ---------------- GRAPHS ----------------
# Resource Distribution
res_counts = df["Resource"].value_counts().reset_index()
res_counts.columns = ["Resource", "Count"]
fig_pie = px.pie(res_counts, names="Resource", values="Count", hole=0.4)

# Sentiment Distribution
sent_counts = df["Sentiment"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
fig_sent = px.bar(sent_counts, x="Sentiment", y="Count", color="Sentiment")

# Trend Over Time
trend_df = df.groupby(["Month","Resource"]).size().reset_index(name="Count")
fig_trend = px.area(
    trend_df,
    x="Month",
    y="Count",
    color="Resource",
    category_orders={"Month": months}
)

# Severity Index
severity_df = df.copy()
severity_df["NegWeight"] = severity_df["Sentiment"].apply(
    lambda x: 1 if x == "Negative" else 0.3 if x == "Neutral" else 0
)
severity_scores = (
    severity_df.groupby("Resource")
    .apply(lambda x: int(
        x["NegWeight"].mean()*40 +
        x["Confidence"].mean()*0.4 +
        (len(x)/len(df))*20
    ))
    .reset_index(name="Severity_Index")
)

# ---------------- GRAPH SWITCHER ----------------
def render_animated_dots(total, current):
    html = "<div class='dot-container'>"
    for i in range(total):
        html += "<span class='dot active'>●</span>" if i == current else "<span class='dot'>○</span>"
    html += "</div>"
    return html

graphs = [
    ("📊 Resource Distribution", fig_pie),
    ("😊 Sentiment Distribution", fig_sent),
    ("📈 Feedback Trend Over Time", fig_trend),
    ("🚨 Severity Index", None)
]

title, current_graph = graphs[st.session_state.graph_index]
st.subheader("🔄 Interactive Graph Explorer")
st.markdown(f"### {title}")

if not st.session_state.fullscreen:
    if current_graph:
        st.plotly_chart(current_graph, use_container_width=True)
    else:
        st.dataframe(severity_scores.sort_values("Severity_Index", ascending=False))

st.markdown(render_animated_dots(len(graphs), st.session_state.graph_index), unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,2,1])
with c1:
    if st.button("⬅️ Previous"):
        st.session_state.graph_index = (st.session_state.graph_index - 1) % len(graphs)
        st.rerun()
with c3:
    if st.button("➡️ Next"):
        st.session_state.graph_index = (st.session_state.graph_index + 1) % len(graphs)
        st.rerun()
with c2:
    if st.button("🔍 Full Screen"):
        st.session_state.fullscreen = True
        st.rerun()

# ---------------- FULL SCREEN ----------------
if st.session_state.fullscreen:
    with st.modal(f"{title} — Full Screen"):
        if current_graph:
            st.plotly_chart(current_graph, use_container_width=True)
        else:
            st.dataframe(severity_scores, use_container_width=True)
        if st.button("❌ Exit Full Screen"):
            st.session_state.fullscreen = False
            st.rerun()

# ---------------- MINI BAR: LAST MONTH VS NOW ----------------
if last_month:
    lm = len(df[(df["Month"]==last_month) & (df["Resource"]==predicted_resource)])
    cm = len(df[(df["Month"]==current_month) & (df["Resource"]==predicted_resource)])
    cmp_df = pd.DataFrame({"Period":[last_month,current_month],"Count":[lm,cm]})
    fig_cmp = px.bar(cmp_df, x="Period", y="Count", text="Count",
                     title=f"{predicted_resource}: Last Month vs Now")
    st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Dataset"):
    st.dataframe(df)
