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
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or use st.secrets["openai_api_key"]

# --- Streamlit UI ---
st.set_page_config(page_title="AskEmployeeAI - Smart Chat & Auto Visuals", layout="wide")
st.title("🧠 AskEmployeeAI")
st.markdown("Ask any question about your employee dataset and auto-generate reports!")

# --- Load your uploaded employee CSV ---
csv_path = "employee.csv"  # You uploaded this earlier
df = pd.read_csv(csv_path)
st.success("✅ Loaded your uploaded dataset successfully!")
st.dataframe(df, use_container_width=True)

# --- Info for the user ---
# st.info("💡 Try asking: 'Top 5 performers', 'Pie chart of department', 'Average salary by gender', 'Employees with low attendance'")

# --- Text input box ---
user_query = st.text_input("💬 Ask your employee-related question:")

# --- On click of Ask button ---
if st.button("Ask") and user_query.strip():
    # --- Basic check for meaningful input ---
    if len(user_query.strip().split()) <= 2 and not any(word in user_query.lower() for word in ["show", "list", "top", "who", "what", "how", "which", "thank you","average", "employees", "salary", "performance"]):
        st.warning("HOW CAN I HELP YOU TODAY!")
    else:
        with st.spinner("🔎 Thinking..."):
            # --- Adjust query for top 5 fix ---
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

            #st.markdown("### 🧠 Answer:")
            #st.write(answer)

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
                    st.plotly_chart(local_vars["fig"], use_container_width=True)
                elif "result" in local_vars:
                    st.dataframe(local_vars["result"])
                else:
                    st.warning("❗No chart or table generated from this query.")
            except Exception as e:
                st.error(f"⚠️ Error running visualization code: {e}")


