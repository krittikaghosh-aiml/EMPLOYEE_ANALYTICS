import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import openai
import os

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# --- Setup your OpenAI key securely (store as secret in deployment) ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # Use .env or Streamlit secrets

# --- Streamlit UI ---
st.set_page_config(page_title="AskEmployeeAI - Smart Chat & Auto Visuals", layout="wide")
st.title("🧠 AskEmployeeAI")
st.markdown("Ask any question — I’ll generate visual insights from the employee dataset!")

# --- Load the default CSV ---
df = pd.read_csv("employee.csv")
st.success("✅ Loaded default dataset: `employee.csv`")
st.dataframe(df.head())

# --- Save CSV temporarily ---
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    df.to_csv(tmp.name, index=False)
    temp_file_path = tmp.name

# --- LangChain setup ---
loader = CSVLoader(file_path=temp_file_path)
data = loader.load()
vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=vectorstore.as_retriever())

# --- User input ---
user_query = st.text_input("💬 Ask anything about employees:")
if st.button("Ask") and user_query:
    with st.spinner("🔎 Thinking..."):
        answer = qa.run(user_query)
        st.markdown("### 🧠 Answer:")
        st.write(answer)

    # GPT chart/table generation
    st.markdown("### 📊 Auto Visualization (if applicable):")
    prompt = f"""
    You are a Python data analyst. The user asked: '{user_query}'.

    Based on the CSV dataframe columns: {list(df.columns)} — write Python code using pandas and plotly to generate a chart or table as appropriate.

    1. Use dataframe named 'df'
    2. The output should be stored in a variable named 'fig' (for charts) or 'result' (for tables)
    3. Do NOT include import statements
    4. ONLY generate the code block needed, no explanation
    """
    chat_completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Python data analyst"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    code = chat_completion.choices[0].message.content.strip()

    try:
        local_vars = {'df': df}
        exec(code, {}, local_vars)
        if 'fig' in local_vars:
            st.plotly_chart(local_vars['fig'], use_container_width=True)
        elif 'result' in local_vars:
            st.dataframe(local_vars['result'])
        else:
            st.warning("❗No visualization generated from this query.")
    except Exception as e:
        st.error(f"⚠️ Error in generating visualization: {e}")

