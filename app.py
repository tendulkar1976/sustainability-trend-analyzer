import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 Universal Forms AI Analyzer",
    layout="wide"
)

st.title("🎛️ Universal Forms AI Analyzer")
st.markdown(
    "**Employee feedback • Customer surveys • Student forms • Sales data • ANY CSV!**"
)

# ---------------- AI ANALYSIS FUNCTION ----------------
def universal_analyze(df, prompt):
    """
    This function represents the core 'Rule-Based AI' of the system.
    It interprets natural language prompts and dynamically analyzes any CSV.
    """
    prompt_lower = prompt.lower()

    # Auto-detect columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    score_col = None
    group_col = None

    for col in numeric_cols:
        if any(x in col.lower() for x in ["score", "rate", "rating", "mark", "grade", "satisf"]):
            score_col = col
            break
    if score_col is None and numeric_cols:
        score_col = numeric_cols[0]

    for col in text_cols:
        if any(x in col.lower() for x in ["group", "dept", "team", "class", "division", "department"]):
            group_col = col
            break

    # ---------------- RESPONSE LOGIC ----------------
    if "group" in prompt_lower or "department" in prompt_lower:
        if score_col and group_col:
            stats = df.groupby(group_col)[score_col].agg(["mean", "count"]).round(2)
            return f"### 🏢 Group Analysis\n{stats.to_string()}"
        return "Group or score column not detected."

    elif "lowest" in prompt_lower or "worst" in prompt_lower:
        if score_col and group_col:
            g = df.groupby(group_col)[score_col].mean()
            return f"### 📉 Lowest Performing\n**{g.idxmin()}** → {g.min():.2f}"
        return "Insufficient data for lowest analysis."

    elif "highest" in prompt_lower or "best" in prompt_lower:
        if score_col and group_col:
            g = df.groupby(group_col)[score_col].mean()
            return f"### 📈 Best Performing\n**{g.idxmax()}** → {g.max():.2f}"
        return "Insufficient data for highest analysis."

    elif "trend" in prompt_lower or "correlation" in prompt_lower:
        if len(numeric_cols) >= 2:
            return "### 📈 Correlation Matrix\n" + df[numeric_cols].corr().round(2).to_string()
        return "Not enough numeric columns for trend analysis."

    elif "report" in prompt_lower or "summary" in prompt_lower:
        return f"""
### 📊 Complete Report
• Records: {len(df)}
• Columns: {len(df.columns)}
• Numeric Columns: {numeric_cols if numeric_cols else "None"}
• Text Columns: {text_cols if text_cols else "None"}
"""

    else:
        return """
### 🤖 Smart AI Suggestions
Try asking:
- full report
- lowest performing group
- best performing department
- show trends
"""

# ---------------- SIDEBAR ----------------
st.sidebar.title("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader(
    "Employee / Student / Customer CSV",
    type=["csv"]
)

# ---------------- MAIN APP ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("📈 Auto Stats")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Records", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Avg Score", f"{df[numeric_cols[0]].mean():.2f}" if len(numeric_cols) > 0 else "N/A")

    st.markdown("---")
    st.subheader("💬 Ask the AI about your data")

    prompt = st.text_input("Example: full report, lowest performing, trends")

    if prompt:
        with st.spinner("🤖 AI is analyzing..."):
            response = universal_analyze(df, prompt)
            st.markdown(response)

else:
    st.info("👈 Upload a CSV file to begin")

# ---------------- AI EXPLANATION SECTION ----------------
st.markdown("---")
st.subheader("🤖 Where AI Is Used in This Project")

st.markdown("""
### ❌ No External AI APIs or LLMs Used

This project uses **Rule-Based Artificial Intelligence**, implemented using Python and Pandas.

### ✅ AI Techniques Implemented
- **Natural Language Keyword Matching** – Interprets user queries like *“lowest”, “report”, “departments”*
- **Auto Column Detection** – Automatically finds score and group columns
- **Dynamic Group Analysis** – Performs real-time aggregation and comparisons
- **Smart Fallback Logic** – Works with ANY CSV structure

### 🎯 Why This Still Qualifies as AI
- Mimics human decision-making logic
- Adapts to unknown data schemas
- Produces intelligent, context-aware insights
- Requires no manual configuration
""")

# ---------------- SDG SECTION ----------------
st.markdown("---")
st.subheader("🌍 UN Sustainable Development Goal Alignment")

st.markdown("""
### 🎯 SDG 9: Industry, Innovation & Infrastructure

**Why this project supports SDG 9:**
- ✅ Democratizes data analysis (no coding required)
- ✅ Enables digital decision-making tools
- ✅ Promotes innovation using lightweight AI
- ✅ Works in low-resource environments
- ✅ Supports evidence-based organizational improvements

**Specific SDG Targets Addressed:**
- **9.5** – Enhance research and technological capability
- **9.C** – Increase access to digital technologies
""")

st.caption("🚀 100% Free • No APIs • Pure Python • Rule-Based AI")
