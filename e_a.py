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
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or use st.secrets["openai_api_key"]

# --- User Credentials ---
USERS = {
    "admin": "admin123",
    "kg": "kg123",
    "sg": "sg123",
    "analyst": "insights"
}

# --- Streamlit UI ---
st.set_page_config(page_title="🐝Hive Radar📡", layout="centered", page_icon="📊")

# --- UI Styling ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    body {
        background-color: #e6ccff;
        color: #2c3e50;
    }
    div.stButton > button {
        background-color: #6a0dad;
        color: white;
        padding: 10px 30px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        animation: pulse 2s infinite;
        white-space: nowrap;
    }
    div.stButton > button:hover {
        background-color: #5c0099;
        transform: scale(1.05);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(106, 13, 173, 0); }
        100% { box-shadow: 0 0 0 0 rgba(106, 13, 173, 0); }
    }
    
    </style>
""", unsafe_allow_html=True)

# ========= FOOTER (ALWAYS VISIBLE) ==========
st.markdown("""
    <style>
    .footer-left-animated {
        position: fixed;
        bottom: 0;
        left: 0;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #6a0dad;
        border-top-right-radius: 12px;
        animation: glow 3s ease-in-out infinite;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    @keyframes glow {
        0% { box-shadow: 0 0 5px #6a0dad; }
        50% { box-shadow: 0 0 20px #6a0dad; }
        100% { box-shadow: 0 0 5px #6a0dad; }
    }

    .emoji {
        animation: bounce 1.5s infinite;
        font-size: 18px;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    </style>

    <div class="footer-left-animated">
        <span class="emoji">👩‍💻</span>
        Created by <b>Krittika Ghosh</b>
    </div>
""", unsafe_allow_html=True)

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center; color:#6a0dad;'>🔐 Login to 🐝  Hive Radar  📡</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #333; font-size: 16px;'>Please enter your credentials below.</p>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("🔐 Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful! Reloading...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

    st.stop()

# --- Logout Button ---
logout_center = st.columns([4, 1, 4])
with logout_center[1]:
    if st.button("🚪 Logout", key="logout_top"):
        st.session_state.logged_in = False
        if "username" in st.session_state:
            del st.session_state["username"]
        st.rerun()

st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🐝  Hive Radar  📡</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #333;'>Smart Radar for Smarter Teams 📊</h4>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#6a0dad;'>📈 Employee Analytics Dashboard</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:#333;'>Welcome <b>{st.session_state.username}</b>! Generate employee insights below.</h5>", unsafe_allow_html=True)

# --- Load your uploaded employee CSV ---
csv_path = "employee_data.csv"  # You uploaded this earlier
df = pd.read_csv(csv_path)
if st.checkbox("👁️ Show Employee Dataset"):
    st.dataframe(df, use_container_width=True)

# --- Info for the user ---
# st.info("💡 Try asking: 'Top 5 performers', 'Pie chart of department', 'Average salary by gender', 'Employees with low attendance'")

# --- Text input box ---
# Sample questions
sample_questions = [
    "Top 5 performers",
    "Average salary by department",
    "Pie chart of employees by gender",
    "Employees with attendance below 75%",
    "Department-wise count of employees",
    "Top 5 highest paid employees",
    "Bar chart of experience vs salary"
]

selected_sample = st.selectbox("💡 Choose a sample question (or type your own below):", [""] + sample_questions)
user_query = st.text_input("💬 Or type your own question here:", value=selected_sample)

# --- On click of Ask button ---
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

        # If user input is casual small talk
        if cleaned_query in casual_inputs:
            st.markdown(f"### 🤖 {casual_inputs[cleaned_query]}")
        
        # Otherwise, run the full visualization pipeline
        else:
            with st.spinner("🔎 Thinking..."):
                custom_query = user_query
                if "top 5" in user_query.lower():
                    custom_query += ". Please list exactly 5 employees based on the dataset."

                # --- LangChain Q&A ---
                loader = CSVLoader(file_path=csv_path)
                data = loader.load()
                vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
                qa = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model="gpt-3.5-turbo"),
                    retriever=vectorstore.as_retriever()
                )
                answer = qa.run(custom_query)

                # Optional: Hide the answer if not needed
                # st.markdown("### 🧠 Answer:")
                # st.write(answer)

                # --- Ask GPT to generate visual code ---
                st.markdown("### 📊 Auto Visualization (if applicable):")
                prompt = f"""
                You are a Python data analyst. The user asked: '{user_query}'.

                Based on this DataFrame with columns: {list(df.columns)},
                generate only raw Python code using pandas and plotly to create a chart or a filtered table.

                Rules:
                - Use the dataframe named 'df'
                - If chart, assign it to variable 'fig'
                - If table, assign it to variable 'result'
                - Do NOT use triple backticks, markdown, or explanations. Only return valid code.
                """

                chat_completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a Python data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                raw_response = chat_completion.choices[0].message.content.strip()
                code_blocks = re.findall(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
                code = code_blocks[0].strip() if code_blocks else raw_response

                try:
                    local_vars = {"df": df}
                    exec(code, {}, local_vars)

                    if "fig" in local_vars:
                        fig = local_vars["fig"]
                        st.plotly_chart(fig, use_container_width=True)

    # Convert to PDF in memory
                        pdf_buffer = io.BytesIO()
                        pio.write_image(fig, pdf_buffer, format='pdf')
                        pdf_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Chart as PDF",
                            data=pdf_buffer,
                            file_name="employee_chart.pdf",
                            mime="application/pdf"
                        )
                    elif "result" in local_vars:
                        st.dataframe(local_vars["result"])
                    else:
                        st.warning("❗No chart or table generated from this query.")
                except Exception as e:
                    st.error(f"⚠️ Error running visualization code: {e}")



