import os
import streamlit as st
import panel as pn
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

pn.extension()

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

# === Load and Prepare Vectorstore ===
docs = load_documents()
chunks = split_documents(docs)
vectordb = build_vectorstore(chunks)
qa_chain = create_qa_chain(vectordb)

# === Panel UI ===
chat_history = pn.Column()
input_box = pn.widgets.TextInput(name="📥 คำถาม", placeholder="พิมพ์คำถาม เช่น หน่วยงานมีกี่ฝ่าย?")
send_button = pn.widgets.Button(name="ส่งคำถาม", button_type="primary")
clear_button = pn.widgets.Button(name="🔁 ล้างประวัติ", button_type="warning")

def send_callback(event):
    question = input_box.value.strip()
    if question:
        answer = qa_chain.run(question)
        chat_history.append(pn.pane.Markdown(f"**🧑‍💼 คุณ:** {question}"))
        chat_history.append(pn.pane.Markdown(f"**🤖 บอท:** {answer}"))
        input_box.value = ""

def clear_chat(event):
    chat_history.clear()

send_button.on_click(send_callback)
clear_button.on_click(clear_chat)

app_layout = pn.Column(
    "# 💬 RAG Chatbot ด้วย Groq + Panel",
    pn.Row(input_box, send_button, clear_button),
    pn.Spacer(height=10),
    "### 🗂️ ประวัติการสนทนา",
    chat_history,
    sizing_mode="stretch_width"
)

app_layout.servable()

