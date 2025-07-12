import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# --- Setup your OpenAI key securely (store as secret in deployment) ---
openai.api_key = os.getenv("OPENAI_API_KEY")  

# --- Streamlit UI ---
st.set_page_config(page_title="AskEmployeeAI - Smart Chat & Auto Visuals", layout="wide")
st.title("🧠 AskEmployeeAI")
st.markdown("Upload your employee data and ask any question — it will auto-generate tables or charts!")

uploaded_file = st.file_uploader("📤 Upload your Employee CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV Loaded!")
    st.dataframe(df.head())

    # Save CSV temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_file_path = tmp.name

    # LangChain Embed CSV
    loader = CSVLoader(file_path=temp_file_path)
    data = loader.load()
    vectorstore = FAISS.from_documents(data, OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=vectorstore.as_retriever())

    # User Q&A
    user_query = st.text_input("💬 Ask anything about employees:")
    if st.button("Ask"):
        with st.spinner("🔎 Thinking..."):
            answer = qa.run(user_query)
            st.markdown("### 🧠 Answer:")
            st.write(answer)

        # Call GPT again to generate code for visual
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
            messages=[{"role": "system", "content": "You are a Python data analyst"},
                      {"role": "user", "content": prompt}],
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
else:
    st.info("📁 Upload a CSV file to begin.")
