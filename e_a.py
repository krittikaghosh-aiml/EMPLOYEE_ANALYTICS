import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import openai
import os
import re
import io
import plotly.io as pio

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# --- Setup your OpenAI key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- User Credentials ---
USERS = {
    "admin": "admin123",
    "kg": "kg123",
    "sg": "sg123",
    "analyst": "insights"
}

# --- Streamlit UI ---
st.set_page_config(page_title="🐝 HIVE RADAR 📡", layout="centered", page_icon="📊")

# --- Styling ---
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    div.stButton > button {
        background-color: #6a0dad; color: white; border-radius: 8px;
        font-size: 18px; font-weight: bold; transition: 0.3s; animation: pulse 2s infinite;
    }
    div.stButton > button:hover {
        background-color: #5c0099; transform: scale(1.05);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(106, 13, 173, 0); }
        100% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <style>
    .footer-left-animated {
        position: fixed; bottom: 0; left: 0; padding: 10px 20px; font-size: 16px;
        color: white; background-color: #6a0dad; border-top-right-radius: 12px;
        animation: glow 3s ease-in-out infinite; z-index: 9999;
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px #6a0dad; }
        50% { box-shadow: 0 0 20px #6a0dad; }
        100% { box-shadow: 0 0 5px #6a0dad; }
    }
    </style>
    <div class="footer-left-animated">👩‍💻 Created by <b>Krittika Ghosh</b></div>
""", unsafe_allow_html=True)

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center;color:#6a0dad;'>🔐 Login to 🐝 HIVE RADAR 📡</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("🔐 Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.stop()

# --- Logout Button ---
if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# --- Load Employee Data ---
csv_path = "employee_data.csv"
df = pd.read_csv(csv_path)

# 🔁 MAP categorical performance to numeric
performance_map = {
    "Poor": 1, "Below Average": 2, "Average": 3, "Good": 4, "Excellent": 5
}
if "PERFORMANCE" in df.columns and df["PERFORMANCE"].dtype == "object":
    df["Performance_Score"] = df["PERFORMANCE"].map(performance_map)

# Show data if user wants
if st.checkbox("👁️ Show Employee Dataset"):
    st.dataframe(df, use_container_width=True)

# --- Heading ---
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🐝 HIVE RADAR 📡</h1>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:#333;'>Welcome <b>{st.session_state.username}</b>! Generate employee insights below.</h5>", unsafe_allow_html=True)

# --- Sample Questions ---
sample_questions = [
    "Top 5 performers", "Average salary by department", "Pie chart of employees by gender",
    "Employees with attendance below 75%", "Department-wise count of employees",
    "Top 5 highest paid employees", "Bar chart of experience vs salary"
]
selected_sample = st.selectbox("💡 Choose a sample question (or type your own):", [""] + sample_questions)
user_query = st.text_input("💬 Or type your question here:", value=selected_sample)

# --- Ask Button ---
if st.button("🔍 Ask"):
    if not user_query.strip():
        st.markdown("### 🙋‍♀️ Please ask a question.")
    else:
        cleaned_query = user_query.strip().lower()
        casual_inputs = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hi there! 👋 What would you like to know about your employees?",
            "thank you": "You're welcome! 😊",
            "thanks": "Glad to help! 🙌",
            "hey": "Hey there! Ask me anything about your team.",
            "good morning": "Good morning! Ready to analyze your data?",
            "how are you": "I'm just a chatbot, but I'm ready to help!"
        }

        if cleaned_query in casual_inputs:
            st.markdown(f"### 🤖 {casual_inputs[cleaned_query]}")
        else:
            with st.spinner("🔎 Thinking..."):
                if "top 5" in user_query.lower():
                    user_query += ". Please list exactly 5 employees based on the dataset."

                loader = CSVLoader(file_path=csv_path)
                data = loader.load()
                vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
                qa = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model="gpt-3.5-turbo"),
                    retriever=vectorstore.as_retriever()
                )
                answer = qa.run(user_query)

                st.markdown("### 📊📡 Radar Scan Result: (if applicable):")

                # 🔁 IMPROVED PROMPT
                prompt = f"""
                You are a Python data analyst. The user asked: '{user_query}'.

                You are working with a pandas DataFrame named 'df' with these columns: {list(df.columns)}.

                If the user asks for something based on non-numeric values (like 'PERFORMANCE'), use the mapped numeric version (e.g., 'Performance_Score') if available.

                Generate raw Python code that either:
                1. Assigns a Plotly chart to a variable named 'fig'
                2. Assigns a filtered pandas table to 'result'

                Do NOT include markdown, explanation, or code fencing. Just output raw code.
                """

                chat_completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a Python data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                raw_code = chat_completion.choices[0].message.content.strip()
                code_blocks = re.findall(r"```(?:python)?\\n(.*?)```", raw_code, re.DOTALL)
                code = code_blocks[0].strip() if code_blocks else raw_code

                try:
                    local_vars = {"df": df}
                    exec(code, {}, local_vars)

                    if "fig" in local_vars:
                        fig = local_vars["fig"]
                        st.plotly_chart(fig, use_container_width=True)

                        pdf_buf = io.BytesIO()
                        pio.write_image(fig, pdf_buf, format="pdf")
                        pdf_buf.seek(0)
                        st.download_button("📥 Download Chart as PDF", data=pdf_buf, file_name="chart.pdf", mime="application/pdf")

                    elif "result" in local_vars:
                        st.dataframe(local_vars["result"])

                    else:
                        st.markdown("ℹ️ No valid chart or table generated. Please ask a different question.")
                except Exception as e:
                    st.error(f"⚠️ Error running visualization code: {e}")




