import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

LLM_API=os.environ.get("LLM_API_KEY")

def get_llm_chain(vectorstore):
    llm=ChatGroq(
        groq_api_key=LLM_API,
        model_name="llama3-70b-8192"
    )
    retriever=vectorstore.as_retriever(seacrh_kwargs={"k":3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    