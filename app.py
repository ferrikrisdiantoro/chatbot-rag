# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# --- Konfigurasi Model & PDF ---
PDF_FILE_PATH = "putri_kumalasari.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"  # Ringan dan cocok untuk CPU

# --- Sidebar ---
st.sidebar.title("📘 Info")
st.sidebar.markdown("**Chatbot Cerita Rakyat: Putri Kumalasari**")
st.sidebar.markdown("Bertanya berdasarkan isi dokumen PDF.")
st.sidebar.markdown("---")

# --- Load dan Cache Resource ---
@st.cache_resource(show_spinner="🔍 Memuat dan memproses dokumen...")
def load_rag_components():
    # Load PDF
    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM
    llm = HuggingFaceHub(
        repo_id=LLM_MODEL,
        model_kwargs={"temperature": 0.3, "max_length": 256},
    )

    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    return qa_chain

# --- UI ---
st.title("📚 Chatbot Cerita Rakyat: Putri Kumalasari")
st.markdown("Tanyakan isi dari cerita rakyat berdasarkan dokumen PDF yang dimuat.")

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"File PDF `{PDF_FILE_PATH}` tidak ditemukan.")
else:
    rag_chain = load_rag_components()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Apa yang ingin kamu ketahui?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Menjawab..."):
                try:
                    result = rag_chain(user_input)
                    answer = result["result"]
                    sources = result["source_documents"]
                except Exception as e:
                    answer = f"⚠️ Maaf, terjadi error: {e}"
                    sources = []

                st.markdown(answer)
                if sources:
                    with st.expander("🔎 Sumber Konteks"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Doc {i+1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
