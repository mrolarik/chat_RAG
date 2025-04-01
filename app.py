import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatGroq
from langchain.chains import RetrievalQA

# === SETUP ===
GROQ_API_KEY = "your-groq-api-key"  # 🔑 เปลี่ยนเป็น API Key ของคุณ
MODEL_NAME = "mixtral-8x7b"  # หรือ "llama3-8b-8192" ที่รองรับ Groq

# === Load documents from multiple file types ===
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

# === Build Vectorstore ===
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    return vectordb

# === RAG Chain using Groq ===
def create_qa_chain(vectordb):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return chain

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("💬 องค์กร Chatbot ด้วย RAG + Groq")

    with st.spinner("📚 กำลังโหลดข้อมูล..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = build_vectorstore(chunks)
        qa_chain = create_qa_chain(vectordb)

    query = st.text_input("ถามเกี่ยวกับองค์กรของคุณที่นี่:", placeholder="เช่น โครงสร้างองค์กรเป็นอย่างไร?")
    if query:
        with st.spinner("🧠 กำลังประมวลผล..."):
            answer = qa_chain.run(query)
            st.markdown(f"**คำตอบ:** {answer}")

if __name__ == "__main__":
    main()
