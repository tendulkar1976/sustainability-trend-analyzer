import streamlit as st
import pandas as pd
import openai
import os
from io import StringIO

# Page config
st.set_page_config(page_title="📊 Student Forms AI Analyzer", layout="wide")

# Sidebar for file upload and API key
st.sidebar.title("⚙️ Setup")
uploaded_file = st.sidebar.file_uploader("Upload CSV forms", type="csv")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                               help="Get free key from platform.openai.com")

if api_key:
    openai.api_key = api_key

# Main title
st.title("🤖 Student Forms AI Analyzer")
st.markdown("**Ask any question about your student forms data!**")

# Load data
if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file)
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Quick Stats")
        st.metric("Total Forms", len(df))
        st.metric("Avg Satisfaction", f"{df.get('satisfaction', pd.Series([0])).mean():.1f}/10")
        st.metric("Departments", df.get('department', pd.Series(['N/A'])).nunique())
    
    # Chat interface
    st.subheader("💬 Ask Questions About Your Data")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about trends, satisfaction, departments, feedback..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare data for AI
        sample_data = df.sample(min(20, len(df))).to_csv(index=False)
        stats_summary = {
            'rows': len(df),
            'avg_satisfaction': df.get('satisfaction', pd.Series([0])).mean(),
            'departments': df.get('department', pd.Series(['N/A'])).value_counts().to_dict()
        }
        
        # AI Prompt
        full_prompt = f"""
        You are analyzing student/office worker forms data. 
        
        SAMPLE DATA (first 20 rows):
        {sample_data}
        
        STATS SUMMARY: {stats_summary}
        
        Question: "{prompt}"
        
        Answer concisely with specific insights from the data. Use bullet points.
        """
        
        with st.chat_message("assistant"):
            with st.spinner("AI is analyzing your data..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": full_prompt}],
                        max_tokens=400,
                        temperature=0.2
                    )
                    ai_response = response.choices[0].message.content
                    
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
else:
    st.info("👈 Upload your CSV file and enter OpenAI API key in the sidebar to start!")
    st.markdown("""
    ## Example Questions to Ask:
    - "What are the main feedback themes?"
    - "Which department has lowest satisfaction?"
    - "Show trends by attendance"
    - "Recommendations to improve scores?"
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit + OpenAI GPT | Lightweight prototype*")
