import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from langchain_community.document_loaders import PyPDFLoader

# --- Konfigurasi Model LLM ---
# Menggunakan TinyLlama 1.1B Chat untuk kompatibilitas Streamlit Cloud gratisan
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Untuk TinyLlama, tidak perlu Hugging Face Token karena ini model publik
HF_TOKEN = None # Tidak perlu token

# Pastikan ada GPU jika ingin performa baik dengan model 7B, tapi ini untuk CPU di Streamlit Cloud
DEVICE = "cpu" # Kita paksa ke CPU karena Streamlit Cloud gratisan tidak ada GPU yang dijamin
print(f"Menggunakan device: {DEVICE}")

# --- Fungsi Inisialisasi Sumber Daya (untuk cache Streamlit) ---
@st.cache_resource
def load_resources(pdf_path):
    st.info(f"Memuat dokumen dari: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.success(f"Berhasil memuat {len(documents)} halaman.")

    st.info("Melakukan chunking dokumen...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    st.success(f"Dokumen dipecah menjadi {len(chunks)} chunks.")

    st.info("Memuat model embedding...")
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    st.success(f"Model embedding: {embedding_model_name} berhasil dimuat.")

    st.info("Membuat dan mengisi vector database (FAISS)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    st.success("Vector database (FAISS) berhasil dibuat dan diisi.")

    st.info(f"Memuat LLM: {LLM_MODEL_ID} (Ini mungkin memakan waktu)...")
    # TinyLlama memiliki tokenizer di repo yang sama
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float32, # Pakai float32 untuk CPU, atau float16 jika GPU tersedia
        device_map=None,           # Paksa None untuk CPU (tidak ada auto mapping ke GPU)
    )
    model.eval()
    st.success(f"LLM {LLM_MODEL_ID} berhasil dimuat ke {DEVICE}.")

    return vector_store, model, tokenizer

# --- Fungsi Generasi Respons (diadaptasi untuk TinyLlama Chat Format) ---
def generate_response(query, context_docs, llm_model, llm_tokenizer):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # Format prompt untuk TinyLlama Chat (bisa mirip dengan Llama/Gemma Instruct)
    # Referensi: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0#usage
    prompt = f"<|im_start|>system\nAnda adalah asisten yang cerdas dan informatif. Berikan jawaban yang akurat dan ringkas berdasarkan informasi kontekstual berikut dari buku cerita. Jika jawaban tidak dapat ditemukan dalam konteks, nyatakan bahwa Anda tidak memiliki informasi yang cukup dari konteks yang diberikan.<|im_end|>\n<|im_start|>user\nKonteks Cerita:\n{context_text}\n\nPertanyaan: {query}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(
        **input_ids,
        max_new_tokens=256, # Batasi agar tidak terlalu panjang
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=llm_tokenizer.eos_token_id,
        eos_token_id=llm_tokenizer.convert_tokens_to_ids("<|im_end|>") # TinyLlama menggunakan ini sebagai EOS
    )

    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Logika untuk memotong prompt dari respons TinyLlama
    try:
        # Cari marker asisten
        response_start_marker = "<|im_start|>assistant\n"
        if response_start_marker in response:
            response = response.split(response_start_marker, 1)[1].strip()
        
        # Hapus token khusus yang mungkin masih ada
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>user", "").strip() # Jika ada pengulangan prompt
        response = response.replace("<|im_start|>system", "").strip() # Jika ada pengulangan prompt
        
    except Exception as e:
        st.warning(f"Error extracting response (keeping full response): {e}")
        pass

    return response

# --- Fungsi utama Chatbot RAG ---
def chatbot_rag(query, vector_db, llm_model, llm_tokenizer, top_k=3):
    retrieved_docs = vector_db.similarity_search(query, k=top_k)

    st.sidebar.markdown(f"--- Dokumen yang Ditemukan (Relevansi Top {top_k}) ---")
    for i, doc in enumerate(retrieved_docs):
        st.sidebar.text(f"Dokumen {i+1} (Source: {doc.metadata.get('source', 'N/A')} Page: {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:200]}...")
    st.sidebar.markdown("---")

    answer = generate_response(query, retrieved_docs, llm_model, llm_tokenizer)
    return answer

# --- Antarmuka Pengguna Streamlit ---
st.title("📚 Chatbot Cerita Rakyat: Putri Kumalasari")
st.write("Tanyakan apa saja tentang cerita 'Putri Kumalasari' yang ada di PDF ini.")

# Pastikan file PDF ada di direktori yang sama atau berikan path lengkap
PDF_FILE_PATH = "putri_kumalasari.pdf" # Ganti dengan nama file PDF yang benar jika berbeda

if not os.path.exists(PDF_FILE_PATH):
    st.error(f"File PDF '{PDF_FILE_PATH}' tidak ditemukan. Mohon pastikan file ada di direktori yang sama dengan `app.py`.")
else:
    vector_store, llm_model, llm_tokenizer = load_resources(PDF_FILE_PATH)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
