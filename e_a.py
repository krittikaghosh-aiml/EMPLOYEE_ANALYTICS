import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import openai
import os
import re

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# --- Setup your OpenAI key ---
openai.api_key = os.getenv("OPENAI_API_KEY")  # or st.secrets["openai_api_key"]

# --- Streamlit UI ---
st.set_page_config(page_title="AskEmployeeAI - Smart Chat & Auto Visuals", layout="wide")
st.title("🧠 AskEmployeeAI")
st.markdown("Ask any question — I’ll generate smart reports using the employee dataset!")

# --- Load built-in dataset ---
df = pd.read_csv("employee.csv")
st.success("✅ Loaded default dataset: `employee.csv`")
st.dataframe(df.head())

# --- Save temporarily for LangChain ---
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    df.to_csv(tmp.name, index=False)
    temp_file_path = tmp.name

# --- LangChain setup ---
loader = CSVLoader(file_path=temp_file_path)
data = loader.load()
vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever()
)

# --- User query ---
user_query = st.text_input("💬 Ask anything about employees:")
if st.button("Ask") and user_query:
    with st.spinner("🔎 Thinking..."):
        # 1. Answer the question
        answer = qa.run(user_query)
        st.markdown("### 🧠 Answer:")
        st.write(answer)

        # 2. Prompt for code generation
        st.markdown("### 📊 Auto Visualization (if applicable):")

        prompt = f"""
        You are a Python data analyst. The user asked: '{user_query}'.

        Based on the DataFrame with columns: {list(df.columns)},
        write ONLY raw Python code using pandas and plotly to generate a chart or table.

        Rules:
        - Use the DataFrame named 'df'
        - If chart, assign to variable 'fig'
        - If table, assign to variable 'result'
        - Do NOT return any text, markdown, comments, or triple backticks.
        """

        chat_completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # 3. Extract clean code
        raw_response = chat_completion.choices[0].message.content.strip()
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else raw_response

        st.code(code, language="python")

        # 4. Run the generated code
        try:
            local_vars = {"df": df}
            exec(code, {}, local_vars)

            if "fig" in local_vars:
                st.plotly_chart(local_vars["fig"], use_container_width=True)
            elif "result" in local_vars:
                st.dataframe(local_vars["result"])
            else:
                st.warning("❗No chart or table generated from this query.")
        except Exception as e:
            st.error(f"⚠️ Error running visualization code: {e}")

