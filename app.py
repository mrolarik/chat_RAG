import os
import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# === CONFIG ===
GROQ_API_KEY = "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"  # 🔑 เปลี่ยนเป็น API Key ของคุณ
MODEL_NAME = "llama3-70b-8192"  # หรือ "llama3-8b-8192" ที่รองรับ Groq


#MODEL_NAME = "mixtral-8x7b-32768"   # ✅ ใช้ได้
#MODEL_NAME = "llama3-8b-8192"       # ✅ ใช้ได้
#MODEL_NAME = "llama3-70b-8192"      # ✅ ใช้ได้
#MODEL_NAME = "gemma-7b-it"          # ✅ ใช้ได้


# === Load documents ===
def load_documents():
    all_docs = []
    folder_path = "docs"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            continue

        all_docs.extend(loader.load())

    return all_docs

# === Split documents ===
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# === Build vectorstore with FAISS ===
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    return vectordb

# === Create RAG chain ===
def create_qa_chain(vectordb):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return chain

# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("💬 RAG Chatbot สำหรับข้อมูลองค์กร")

    # 👇 เก็บประวัติคำถาม-คำตอบ
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ปุ่มล้างประวัติ
    if st.button("🔁 ล้างประวัติการสนทนา"):
        st.session_state.chat_history = []

    with st.spinner("📚 กำลังโหลดเอกสารและเตรียมฐานข้อมูล..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = build_vectorstore(chunks)
        qa_chain = create_qa_chain(vectordb)

    # 👇 รับคำถามจากผู้ใช้
    query = st.text_input("📥 พิมพ์คำถามของคุณ:", placeholder="เช่น โครงสร้าง_

