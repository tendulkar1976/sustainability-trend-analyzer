import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')

st.set_page_config(page_title="Sustainability Trend Analyzer", layout="centered")

st.title("🌱 AI-Based Sustainability Issue Trend Analyzer")
st.subheader("Word Cloud • Sentiment • Issue Classification • Dashboards")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain 'feedback' column)", type=["csv"]
)

def classify_issue(text):
    text = text.lower()
    if any(w in text for w in ['water', 'river', 'drinking', 'scarcity', 'sanitation']):
        return "Water"
    elif any(w in text for w in ['waste', 'garbage', 'plastic', 'landfill']):
        return "Waste"
    elif any(w in text for w in ['energy', 'electricity', 'power', 'renewable']):
        return "Energy"
    elif any(w in text for w in ['pollution', 'air', 'smoke', 'noise']):
        return "Pollution"
    else:
        return "Other"

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'feedback' not in data.columns:
        st.error("CSV must contain a column named 'feedback'")
    else:
        text = " ".join(data['feedback'].astype(str))
        stop_words = set(stopwords.words('english'))
        filtered_text = " ".join(
            [w for w in text.split() if w.lower() not in stop_words]
        )

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(filtered_text)

        st.subheader("🔍 Sustainability Issues – Word Cloud")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")
        st.pyplot(fig)

        analyzer = SentimentIntensityAnalyzer()
        data['Sentiment Score'] = data['feedback'].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound']
        )

        data['Sentiment'] = data['Sentiment Score'].apply(
            lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral'
        )

        data['Issue Category'] = data['feedback'].apply(classify_issue)

        st.subheader("📊 Category-wise Sentiment Dashboard")
        table = pd.crosstab(data['Issue Category'], data['Sentiment'])
        st.write(table)

        fig2, ax2 = plt.subplots()
        table.plot(kind='bar', stacked=True, ax=ax2)
        st.pyplot(fig2)
