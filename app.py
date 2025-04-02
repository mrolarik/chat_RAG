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
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

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

# === Create QA Chain with Thai Prompt ===
def create_qa_chain(vectordb):
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-70b-8192"
    )

    prompt_template = """
    คุณเป็นผู้เชี่ยวชาญด้านกฎหมายของประเทศไทย กรุณาตอบคำถามต่อไปนี้เป็นภาษาไทยที่ชัดเจน เข้าใจง่าย และเหมาะสมกับประชาชนทั่วไป:

    เอกสารที่เกี่ยวข้อง:
    {context}

    คำถาม: {question}
    คำตอบ:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ✅ สร้าง LLMChain แยกก่อน
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # ✅ ใส่ LLMChain ลง StuffDocumentsChain
    combine_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # ✅ สร้าง RetrievalQA พร้อม combine chain
    qa_chain = RetrievalQA(
        retriever=vectordb.as_retriever(),
        combine_documents_chain=combine_chain
    )

    return qa_chain


# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("💬 Chatbot พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562")

    # === Sidebar ===
    st.sidebar.title("📘 Chatbot พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562")
    st.sidebar.markdown(
        """
        Chatbot นี้มีวัตถุประสงค์เพื่อทดสอบการทำงานของ chatbot ร่วมกับ LLM และ RAG 
        เพื่อถามตอบคำถามเกี่ยวกับ **พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562** 
        และเนื้อหาสาระต่าง ๆ ที่เกี่ยวข้อง โดยเบื้องต้นใช้เนื้อหาที่รวบรวมมาจากอินเทอร์เน็ต
        """,
        unsafe_allow_html=False
    )

    st.title("💬 Chatbot พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562")

    # เก็บประวัติคำถาม-คำตอบ
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ปุ่มล้างประวัติ
    if st.button("🔁 ล้างประวัติการสนทนา"):
        st.session_state.chat_history = []

    with st.spinner("📚 กำลังโหลดเอกสารและสร้างฐานข้อมูล..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = build_vectorstore(chunks)
        qa_chain = create_qa_chain(vectordb)

    # รับคำถามจากผู้ใช้
    query = st.text_input("📥 พิมพ์คำถามของคุณ:", placeholder="เช่น ข้อมูลส่วนบุคคลคืออะไร?")
    if query:
        with st.spinner("🧠 คิดคำตอบ..."):
            answer = qa_chain.run(query)
            st.session_state.chat_history.append((query, answer))

    # แสดงประวัติการสนทนา (จำกัดความกว้าง)
    if st.session_state.chat_history:
        st.markdown("### 🗂️ ประวัติการสนทนา")
        with st.container():
            st.markdown(
                "<div style='max-width: 800px; margin-left: auto; margin-right: auto;'>",
                unsafe_allow_html=True
            )
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"**{i}. คำถาม:** {q}")
                st.markdown(f"👉 **คำตอบ:** {a}")
                st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
