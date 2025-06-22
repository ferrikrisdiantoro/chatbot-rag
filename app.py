import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from langchain_community.document_loaders import PyPDFLoader # Untuk membaca PDF

# --- Konfigurasi Model LLM ---
# Pilihan model LLM (Llama 2 7B atau Gemma 2B IT)
# Pilih salah satu saja dengan menghapus tanda komentar (#)
# 1. Google Gemma 2B Instruction Tuned (Direkomendasikan untuk kemudahan akses dan performa di CPU/GPU lokal)
LLM_MODEL_ID = "google/gemma-2b-it"
# 2. Mistral 7B Instruct v0.2 (Performa sangat bagus, tapi butuh lebih banyak VRAM/RAM jika tanpa quantization)
# LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# 3. Llama 2 7B Chat HF (Membutuhkan akses dan Hugging Face Token)
# LLM_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

# Jika menggunakan Llama, masukkan token Hugging Face Anda
# HATI-HATI! JANGAN PUBLIKASIKAN TOKEN ANDA
HF_TOKEN = os.getenv("HF_TOKEN")

# Pastikan ada GPU jika ingin performa baik dengan model 7B
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {DEVICE}")

# --- Fungsi Inisialisasi Sumber Daya (untuk cache Streamlit) ---
# st.cache_resource akan menyimpan model dan vector store di memori agar tidak di-load ulang setiap interaksi
@st.cache_resource
def load_resources(pdf_path):
    # --- 1. Memuat Data dari PDF ---
    st.info(f"Memuat dokumen dari: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.success(f"Berhasil memuat {len(documents)} halaman.")

    # --- 2. Chunking (Membagi Dokumen) ---
    st.info("Melakukan chunking dokumen...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Ukuran chunk, bisa disesuaikan
        chunk_overlap=100, # Overlap antar chunk
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Dokumen dipecah menjadi {len(chunks)} chunks.")

    # --- 3. Inisialisasi Model Embedding ---
    st.info("Memuat model embedding...")
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    st.success(f"Model embedding: {embedding_model_name} berhasil dimuat.")

    # --- 4. Membuat dan Mengisi Vector Database (FAISS) ---
    st.info("Membuat dan mengisi vector database (FAISS)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    st.success("Vector database (FAISS) berhasil dibuat dan diisi.")

    # --- 5. Inisialisasi LLM ---
    st.info(f"Memuat LLM: {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32, # Bfloat16 untuk GPU, float32 untuk CPU
        device_map="auto" if DEVICE == "cuda" else None, # Auto device map hanya untuk GPU
        token=HF_TOKEN,
        load_in_8bit=True if DEVICE == "cuda" else False, # Opsional: load in 8bit untuk hemat VRAM GPU
    )
    model.eval()
    st.success(f"LLM {LLM_MODEL_ID} berhasil dimuat ke {DEVICE}.")

    return vector_store, model, tokenizer

# --- Fungsi Generasi Respons (diadaptasi untuk format prompt yang sesuai) ---
def generate_response(query, context_docs, llm_model, llm_tokenizer):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Sesuaikan prompt template dengan model yang dipilih
    if "gemma" in LLM_MODEL_ID.lower():
        prompt = f"""<s>[INST] <<SYS>>
        Anda adalah asisten yang cerdas dan informatif.
        Berikan jawaban yang akurat dan ringkas berdasarkan informasi kontekstual berikut dari buku cerita.
        Jika jawaban tidak dapat ditemukan dalam konteks, nyatakan bahwa Anda tidak memiliki informasi yang cukup dari konteks yang diberikan.
        <<SYS>>

        Konteks Cerita:
        {context_text}

        Pertanyaan: {query} [/INST]
        """
    elif "llama" in LLM_MODEL_ID.lower():
         prompt = f"""<s>[INST] <<SYS>>
        Anda adalah asisten yang cerdas dan informatif.
        Berikan jawaban yang akurat dan ringkas berdasarkan informasi kontekstual berikut dari buku cerita.
        Jika jawaban tidak dapat ditemukan dalam konteks, nyatakan bahwa Anda tidak memiliki informasi yang cukup dari konteks yang diberikan.
        <<SYS>>

        Konteks Cerita:
        {context_text}

        Pertanyaan: {query} [/INST]
        """
    elif "mistral" in LLM_MODEL_ID.lower():
        prompt = f"""[INST]
        Berikan jawaban yang akurat dan ringkas berdasarkan informasi kontekstual berikut dari buku cerita.
        Jika jawaban tidak dapat ditemukan dalam konteks, nyatakan bahwa Anda tidak memiliki informasi yang cukup dari konteks yang diberikan.

        Konteks Cerita:
        {context_text}

        Pertanyaan: {query} [/INST]
        """
    else: # Default/fallback
        prompt = f"""
        Berikan jawaban yang akurat dan ringkas berdasarkan informasi kontekstual berikut dari buku cerita.
        Jika jawaban tidak dapat ditemukan dalam konteks, nyatakan bahwa Anda tidak memiliki informasi yang cukup.

        Konteks Cerita:
        {context_text}

        Pertanyaan: {query}

        Jawaban:
        """

    input_ids = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(
        **input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=llm_tokenizer.eos_token_id if hasattr(llm_tokenizer, 'eos_token_id') else None
    )

    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Logika untuk memotong prompt dari respons
    try:
        if "[/INST]" in response: # Untuk Gemma/Llama/Mistral instruct
            response = response.split("[/INST]", 1)[1].strip()
        # Hapus token <s> yang mungkin muncul di awal
        if response.startswith("<s>"):
            response = response[len("<s>"):].strip()
        # Hapus token </s> yang mungkin muncul di akhir
        if response.endswith("</s>"):
            response = response[:-len("</s>")].strip()
    except Exception as e:
        st.warning(f"Error extracting response (keeping full response): {e}")
        pass # Biarkan response asli jika gagal ekstrak

    return response

# --- Fungsi utama Chatbot RAG ---
def chatbot_rag(query, vector_db, llm_model, llm_tokenizer, top_k=3):
    retrieved_docs = vector_db.similarity_search(query, k=top_k)

    # Streamlit akan menampilkan ini di terminal lokalmu, bukan di UI
    st.sidebar.markdown(f"--- Dokumen yang Ditemukan (Relevansi Top {top_k}) ---")
    for i, doc in enumerate(retrieved_docs):
        st.sidebar.text(f"Dokumen {i+1} (Source: {doc.metadata.get('source', 'N/A')} Page: {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:200]}...") # Tampilkan sebagian kecil
    st.sidebar.markdown("---")

    answer = generate_response(query, retrieved_docs, llm_model, llm_tokenizer)
    return answer

# --- Antarmuka Pengguna Streamlit ---
st.title("📚 Chatbot Cerita Rakyat: Putri Kumalasari")
st.write("Tanyakan apa saja tentang cerita 'Putri Kumalasari' yang ada di PDF ini.")

# Pastikan file PDF ada di direktori yang sama atau berikan path lengkap
PDF_FILE_PATH = "putri_kumalasari.pdf"

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"File PDF '{PDF_FILE_PATH}' tidak ditemukan. Mohon pastikan file ada di direktori yang sama dengan `app.py`.")
else:
    vector_store, llm_model, llm_tokenizer = load_resources(PDF_FILE_PATH)

    # Inisialisasi riwayat chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan pesan dari riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input chat dari user
    user_query = st.chat_input("Tanyakan sesuatu tentang Putri Kumalasari...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari jawaban..."):
                try:
                    response = chatbot_rag(user_query, vector_store, llm_model, llm_tokenizer)
                except Exception as e:
                    response = f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {e}"
                    st.error(f"Error detail: {e}")
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})