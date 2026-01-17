import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import os
import pickle

from wordcloud import WordCloud
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

nltk.download("stopwords")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sustainability Trend Analyzer", layout="wide")

# ---------------- UI THEME ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);}
h1,h2,h3 {color:#e8f5e9;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="text-align:center">
<h1>🌱 AI-Based Sustainability Issue Trend Analyzer</h1>
<p>Word Cloud • Sentiment Analysis • ML Classification • Dashboards</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file (must contain a `feedback` column)",
    type=["csv"]
)

# ---------------- ISSUE LABEL FUNCTION ----------------
def classify_issue(text):
    text = text.lower()
    if any(w in text for w in ['water','river','drinking','scarcity','sanitation']):
        return "Water"
    elif any(w in text for w in ['waste','garbage','plastic','landfill']):
        return "Waste"
    elif any(w in text for w in ['energy','electricity','power','renewable']):
        return "Energy"
    elif any(w in text for w in ['pollution','air','smoke','noise']):
        return "Pollution"
    else:
        return "Other"

# ---------------- PDF REPORT ----------------
def generate_pdf_report(log_acc, nb_acc, keywords):
    file_name = "model_comparison_report.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    w, h = A4
    y = h - 50

    c.setFont("Helvetica-Bold",16)
    c.drawString(50,y,"AI Sustainability Analyzer – Model Report")

    y -= 40
    c.setFont("Helvetica",12)
    c.drawString(50,y,f"Logistic Regression Accuracy: {log_acc:.2f}")
    y -= 20
    c.drawString(50,y,f"Naive Bayes Accuracy: {nb_acc:.2f}")

    y -= 30
    c.setFont("Helvetica-Bold",14)
    c.drawString(50,y,"Top Keywords per Issue Category")

    c.setFont("Helvetica",11)
    for cat, words in keywords.items():
        y -= 20
        c.drawString(50,y,f"{cat}: {', '.join(words)}")
        if y < 80:
            c.showPage()
            y = h - 50

    c.save()
    return file_name

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "feedback" not in data.columns:
        st.error("CSV must contain a `feedback` column")
        st.stop()

    # ---------- WORD CLOUD ----------
    text = " ".join(data["feedback"].astype(str))
    stop_words = set(stopwords.words("english"))
    filtered_text = " ".join([w for w in text.split() if w.lower() not in stop_words])

    wc = WordCloud(width=900,height=400,background_color="white").generate(filtered_text)
    st.subheader("🔍 Dominant Sustainability Issues")
    fig, ax = plt.subplots()
    ax.imshow(wc); ax.axis("off")
    st.pyplot(fig)

    # ---------- SENTIMENT ----------
    analyzer = SentimentIntensityAnalyzer()
    data["Sentiment Score"] = data["feedback"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    data["Sentiment"] = data["Sentiment Score"].apply(
        lambda x: "Positive" if x>0.05 else "Negative" if x<-0.05 else "Neutral"
    )

    # ---------- ML LABELS ----------
    data["ML_Label"] = data["feedback"].apply(classify_issue)

    X_train, X_test, y_train, y_test = train_test_split(
        data["feedback"], data["ML_Label"], test_size=0.2, random_state=42
    )

    # ---------- MODELS ----------
    logistic_pipeline = Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",LogisticRegression(max_iter=1000))
    ])

    nb_pipeline = Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",MultinomialNB())
    ])

    logistic_pipeline.fit(X_train,y_train)
    nb_pipeline.fit(X_train,y_train)

    log_acc = logistic_pipeline.score(X_test,y_test)
    nb_acc = nb_pipeline.score(X_test,y_test)

    # ---------- MODEL COMPARISON ----------
    st.subheader("⚖️ Model Comparison")
    comp_df = pd.DataFrame({
        "Model":["Logistic Regression","Naive Bayes"],
        "Accuracy":[log_acc,nb_acc]
    })
    st.dataframe(comp_df)

    # ---------- CONFUSION MATRICES ----------
    y_pred_log = logistic_pipeline.predict(X_test)
    y_pred_nb = nb_pipeline.predict(X_test)

    cm_log = confusion_matrix(y_test,y_pred_log)
    cm_nb = confusion_matrix(y_test,y_pred_nb)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Regression")
        fig1,ax1 = plt.subplots()
        sns.heatmap(cm_log,annot=True,fmt="d",cmap="Greens",
                    xticklabels=logistic_pipeline.classes_,
                    yticklabels=logistic_pipeline.classes_,ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("### Naive Bayes")
        fig2,ax2 = plt.subplots()
        sns.heatmap(cm_nb,annot=True,fmt="d",cmap="Blues",
                    xticklabels=nb_pipeline.classes_,
                    yticklabels=nb_pipeline.classes_,ax=ax2)
        st.pyplot(fig2)

    # ---------- EXPLAINABILITY ----------
    vectorizer = logistic_pipeline.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()

    def top_keywords(model,features,n=8):
        result={}
        for i,cls in enumerate(model.classes_):
            coef=model.named_steps["clf"].coef_[i]
            top_idx=coef.argsort()[-n:]
            result[cls]=[features[j] for j in top_idx]
        return result

    keywords = top_keywords(logistic_pipeline,feature_names)

    st.subheader("🔍 Top Keywords per Issue Category")
    for k,v in keywords.items():
        st.markdown(f"**{k}**: {', '.join(v)}")

    # ---------- FILTERS ----------
    st.sidebar.header("🔎 Filters")
    cat_filter = st.sidebar.multiselect(
        "Issue Category", data["ML_Label"].unique(), data["ML_Label"].unique()
    )
    sent_filter = st.sidebar.multiselect(
        "Sentiment", data["Sentiment"].unique(), data["Sentiment"].unique()
    )

    filtered_data = data[
        (data["ML_Label"].isin(cat_filter)) &
        (data["Sentiment"].isin(sent_filter))
    ]

    st.subheader("🧠 Automated Insights")
    if len(filtered_data)>0:
        st.info(
            f"{len(filtered_data)} feedbacks analyzed. "
            f"Dominant issue: {filtered_data['ML_Label'].value_counts().idxmax()} | "
            f"Dominant sentiment: {filtered_data['Sentiment'].value_counts().idxmax()}"
        )

    # ---------- DOWNLOAD CSV ----------
    csv = filtered_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Results (CSV)",
        csv,
        "filtered_sustainability_feedback.csv",
        "text/csv"
    )

    # ---------- PDF EXPORT ----------
    if st.button("📄 Generate Model Comparison PDF"):
        pdf = generate_pdf_report(log_acc,nb_acc,keywords)
        with open(pdf,"rb") as f:
            st.download_button("⬇️ Download PDF",f,file_name=pdf,mime="application/pdf")

    st.markdown("---")
    st.markdown("<center>Academic AI–ML Sustainability Project</center>",unsafe_allow_html=True)
