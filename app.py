import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Sustainability Trend Analyzer",
    layout="wide",
    page_icon="🌱"
)

st.title("🌱 AI-Based Sustainability Resource Usage Trend Analyzer")
st.markdown(
    "Upload sustainability data (CSV / TXT / PDF) or use AI-generated data to analyze "
    "**resource usage trends, sentiment trends, and issue dominance** across Indian states."
)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV / TXT / PDF file",
    type=["csv", "txt", "pdf"]
)

# -----------------------------
# DEFAULT VALUES
# -----------------------------
states = [
    "Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh", "Telangana",
    "Delhi", "Maharashtra", "Gujarat", "Rajasthan", "West Bengal",
    "Assam", "Puducherry"
]

months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

resources = ["Electricity", "Water", "Waste"]

# -----------------------------
# LOAD UPLOADED DATA (CSV/TXT/PDF)
# -----------------------------
def load_uploaded_data(file):
    if file.type == "text/csv":
        return pd.read_csv(file)

    if file.type == "text/plain":
        return pd.read_csv(file, sep=",", engine="python")

    if file.type == "application/pdf":
        rows = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            if row and len(row) >= 4:
                                rows.append(row[:4])
                else:
                    text = page.extract_text()
                    if text:
                        for line in text.split("\n"):
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) == 4:
                                rows.append(parts)

        df = pd.DataFrame(
            rows,
            columns=["State", "Month", "Resource", "Usage_Level"]
        )
        df["Usage_Level"] = pd.to_numeric(df["Usage_Level"], errors="coerce")
        return df

    return None

# -----------------------------
# AI-GENERATED FALLBACK DATA
# -----------------------------
def generate_ai_data():
    np.random.seed(42)
    data = []

    for state in states:
        for month in months:
            for resource in resources:
                base = np.random.randint(40, 70)

                if resource == "Electricity" and month in ["March", "April", "May"]:
                    base += np.random.randint(15, 25)
                if resource == "Water" and month in ["March", "April", "May"]:
                    base += np.random.randint(10, 20)
                if resource == "Waste" and month in ["October", "November"]:
                    base += np.random.randint(10, 15)

                data.append([state, month, resource, base])

    return pd.DataFrame(
        data,
        columns=["State", "Month", "Resource", "Usage_Level"]
    )

# -----------------------------
# SELECT DATA SOURCE
# -----------------------------
if uploaded_file:
    try:
        df = load_uploaded_data(uploaded_file)
        st.success("✅ Uploaded data loaded successfully")
    except Exception:
        st.error("❌ Failed to read file. Using AI-generated data.")
        df = generate_ai_data()
else:
    st.info("ℹ️ No file uploaded. Using AI-generated sustainability data.")
    df = generate_ai_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    df["State"].unique(),
    default=df["State"].unique()[:5]
)

selected_resources = st.sidebar.multiselect(
    "Select Resource(s)",
    df["Resource"].unique(),
    default=df["Resource"].unique()
)

filtered_df = df[
    (df["State"].isin(selected_states)) &
    (df["Resource"].isin(selected_resources))
]

# -----------------------------
# MAIN TREND CHART
# -----------------------------
fig_trend = px.line(
    filtered_df,
    x="Month",
    y="Usage_Level",
    color="Resource",
    line_group="State",
    markers=True,
    title="📈 Resource Usage Trends Across States"
)

st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------
# STACKED ISSUE TREND
# -----------------------------
fig_stack_issue = px.area(
    df,
    x="Month",
    y="Usage_Level",
    color="Resource",
    title="📊 Stacked Resource Usage Trend"
)
st.plotly_chart(fig_stack_issue, use_container_width=True)

# -----------------------------
# AI SUMMARY
# -----------------------------
st.subheader("🧠 AI-Generated Insights")

summary_text = (
    f"• Dominant resource: {df['Resource'].value_counts().idxmax()}\n"
    f"• Highest usage level observed: {int(df['Usage_Level'].max())}\n"
    f"• Total records analyzed: {len(df)}"
)

st.info(summary_text)

# -----------------------------
# EXPORT CHARTS AS IMAGES
# -----------------------------
def save_plotly_chart(fig, filename):
    fig.write_image(filename, format="png", scale=2)
    return filename

trend_img = save_plotly_chart(fig_trend, "trend_chart.png")
stack_img = save_plotly_chart(fig_stack_issue, "stacked_issue_trend.png")

# -----------------------------
# PDF REPORT WITH LIST OF FIGURES
# -----------------------------
def generate_pdf_report(figures, summary):
    file_name = "sustainability_analysis_report.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    w, h = A4
    y = h - 50

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI-Based Sustainability Trend Analysis Report")

    # Summary
    y -= 40
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "AI-Generated Insights")

    c.setFont("Helvetica", 11)
    for line in summary.split("\n"):
        y -= 18
        c.drawString(50, y, line)

    # List of Figures
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "List of Figures")

    c.setFont("Helvetica", 11)
    for i, (title, _) in enumerate(figures, start=1):
        y -= 18
        c.drawString(60, y, f"Figure {i}: {title}")

    c.showPage()
    y = h - 50

    # Figures
    for i, (title, img) in enumerate(figures, start=1):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Figure {i}: {title}")
        y -= 10
        c.drawImage(ImageReader(img), 50, y - 220, width=500, height=220)
        y -= 260
        if y < 100:
            c.showPage()
            y = h - 50

    c.save()
    return file_name

# -----------------------------
# PDF DOWNLOAD BUTTON
# -----------------------------
if st.button("📄 Generate Full PDF Report"):
    figures = [
        ("Resource Usage Trend Across States", trend_img),
        ("Stacked Resource Usage Trend", stack_img)
    ]
    pdf_file = generate_pdf_report(figures, summary_text)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "⬇️ Download PDF Report",
            f,
            file_name=pdf_file,
            mime="application/pdf"
        )

# -----------------------------
# DATA PREVIEW
# -----------------------------
with st.expander("📊 View Processed Dataset"):
    st.dataframe(filtered_df)
