import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np
import openai
import tempfile
import os  

st.set_page_config(
    page_title="PDF CHATBOT", layout="centered"
    page_icon="📄",
)

openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("📄 Chat with your PDF")


pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

texts, index, embed_model = [], None, None

if pdf_file and openai.api_key:
    st.info("Processing PDF...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    reader = PdfReader(tmp_path)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content + "\n"

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(raw_text)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("✅ PDF processed. Ask a question below!")

query = st.text_input("Ask a question about the PDF")

if query and texts and index is not None:
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([texts[i] for i in indices[0]])

    prompt = f"""
You are an assistant that answers questions based only on the context below.

Context:
{context}

Question: {query}
Answer:
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        st.markdown(f"**Answer:** {answer}")
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
elif pdf_file and not openai.api_key:
    st.warning("⚠️ No OpenAI API key found. Please add it in Streamlit secrets.")
