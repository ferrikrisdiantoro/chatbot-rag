import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec

# Load .env
load_dotenv()

# ENV variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_DB_NAME")

# Initialize Pinecone with new SDK
pc = Pinecone(api_key=PINECONE_API_KEY)

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_vectorstore(uploaded_files):
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
    
    print("📁 File paths:", file_paths)

    # Load PDF and split
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    
    print("📄 Total documents loaded:", len(docs))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)
    
    print("✂️ Total chunks after split:", len(texts))

    # Embedding
    embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Creating/connecting to Pinecone index...")

    # Check if index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Created new index: {INDEX_NAME}")
    else:
        print(f"✅ Using existing index: {INDEX_NAME}")

    # Create vectorstore
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return vectorstore