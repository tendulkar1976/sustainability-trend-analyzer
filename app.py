import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 Universal Forms Analyzer",
    layout="wide"
)

st.title("🎛️ Universal Forms Analyzer")
st.markdown(
    "**Employee feedback • Customer surveys • Student forms • Sales data • ANY CSV!**"
)

# ---------------- ANALYSIS FUNCTION ----------------
def universal_analyze(df, prompt):
    prompt = prompt.lower()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    score_col = None
    group_col = None
    feedback_col = None

    for col in numeric_cols:
        if any(k in col.lower() for k in ["score", "rate", "rating", "mark", "grade", "satisf"]):
            score_col = col
            break
    if score_col is None and numeric_cols:
        score_col = numeric_cols[0]

    group_keywords = ["group", "dept", "team", "class", "division", "department", "branch", "region", "location"]
    for col in text_cols:
        if any(k in col.lower() for k in group_keywords):
            group_col = col
            break

    for col in text_cols:
        if any(k in col.lower() for k in ["feedback", "comment", "remark", "review", "note"]):
            feedback_col = col
            break

    # ---------------- RESPONSES ----------------
    if "report" in prompt or "summary" in prompt or "analyze" in prompt:
        result = f"""
### 📊 COMPLETE REPORT

• **Records:** {len(df)}
• **Columns:** {len(df.columns)}

**Numeric Columns:** {', '.join(numeric_cols) if numeric_cols else 'None'}
**Text Columns:** {', '.join(text_cols) if text_cols else 'None'}
"""
        if score_col:
            result += f"""
**Key Metric ({score_col})**
• Avg: {df[score_col].mean():.2f}
• Min: {df[score_col].min():.2f}
• Max: {df[score_col].max():.2f}
"""
        if group_col:
            result += f"\n**Groups Found:** {df[group_col].nunique()}"

        return result

    elif "lowest" in prompt or "worst" in prompt:
        if score_col and group_col:
            g = df.groupby(group_col)[score_col].mean()
            return f"### 📉 Lowest Performing\n**{g.idxmin()}** → {g.min():.2f}"
        return "No group or score column detected."

    elif "highest" in prompt or "best" in prompt:
        if score_col and group_col:
            g = df.groupby(group_col)[score_col].mean()
            return f"### 📈 Best Performing\n**{g.idxmax()}** → {g.max():.2f}"
        return "No group or score column detected."

    elif "trend" in prompt or "correlation" in prompt:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            return "### 📈 Correlation Matrix\n" + corr.round(2).to_string()
        return "Not enough numeric columns for trends."

    else:
        return """
### 🔍 INSTANT INSIGHTS

Try asking:
• **full report**
• **lowest performing**
• **best performing**
• **show trends**
"""

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Upload CSV File")
uploaded_file = st.sidebar.file_uploader(
    "Upload Employee / Student / Customer CSV",
    type=["csv"]
)

# ---------------- MAIN APP ----------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV loaded successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("📈 Auto Stats")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Records", len(df))
            st.metric("Columns", len(df.columns))
            if len(numeric_cols) > 0:
                st.metric("Avg Score", f"{df[numeric_cols[0]].mean():.2f}")
            else:
                st.metric("Avg Score", "N/A")

        st.markdown("---")
        st.subheader("💬 Ask About Your Data")

        prompt = st.text_input(
            "Type here (example: full report, lowest performing, trends)"
        )

        if prompt:
            with st.spinner("🔍 Analyzing..."):
                response = universal_analyze(df, prompt)
                st.markdown(response)

    except Exception as e:
        st.error("❌ Error reading CSV")
        st.code(str(e))

else:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis")

st.markdown("---")
st.caption("🚀 Works 100% online • GitHub + Streamlit • No APIs")
