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
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or st.secrets["openai_api_key"]

# --- Streamlit UI ---
st.set_page_config(page_title="AskEmployeeAI - Smart Chat & Auto Visuals", layout="wide")
st.title("🧠 AskEmployeeAI")
st.markdown("Ask any question about your employee dataset and auto-generate reports!")

# --- Load your uploaded employee CSV ---
csv_path = "/mnt/data/employee.csv"  # Path to uploaded file
df = pd.read_csv(csv_path)
st.success("✅ Loaded your uploaded dataset successfully!")
st.dataframe(df.head())

# --- Save temporarily for LangChain processing ---
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    df.to_csv(tmp.name, index=False)
    temp_file_path = tmp.name

# --- LangChain Setup ---
loader = CSVLoader(file_path=temp_file_path)
data = loader.load()
vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever()
)

# --- User question input ---
user_query = st.text_input("💬 Ask your employee-related question:")
if st.button("Ask") and user_query:
    with st.spinner("🔎 Thinking..."):
        # Force 5 results if asked for top 5
        custom_query = user_query
        if "top 5" in user_query.lower():
            custom_query += ". Please list exactly 5 employees based on the dataset."

        # Ask LangChain
        answer = qa.run(custom_query)
        st.markdown("### 🧠 Answer:")
        st.write(answer)

        # GPT prompt for visual generation
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

        # Extract clean code block
        raw_response = chat_completion.choices[0].message.content.strip()
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
        code = code_blocks[0].strip() if code_blocks else raw_response

        # Run generated code
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


