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

#MODEL_NAME = "mixtral-8x7b-32768"   # ✅ ใช้ได้
#MODEL_NAME = "llama3-8b-8192"       # ✅ ใช้ได้
#MODEL_NAME = "llama3-70b-8192"      # ✅ ใช้ได้
#MODEL_NAME = "gemma-7b-it"          # ✅ ใช้ได้


# === CONFIG ===
GROQ_API_KEY = "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"  # 🔑 เปลี่ยนเป็น API Key ของคุณ

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

# === Build vectorstore ===
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)

# === Create QA Chain ===
def create_qa_chain(vectordb, model_name):
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name=model_name
    )
    prompt_template = """
คุณเป็นผู้เชี่ยวชาญด้านกฎหมายของประเทศไทย กรุณาตอบคำถามต่อไปนี้เป็นภาษาไทยที่ชัดเจน เข้าใจง่าย และเหมาะสมกับประชาชนทั่วไป:

{context}

โปรดสรุปและตอบคำถามด้านบนโดยอ้างอิงจากเนื้อหาที่ให้ไว้เท่านั้น
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    combine_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    return RetrievalQA(retriever=vectordb.as_retriever(), combine_documents_chain=combine_chain)

# === Main App ===
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("💬 Chatbot พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562")

    # Sidebar
    st.sidebar.title("💬 Chatbot พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562")
    st.sidebar.markdown(
        """
        Chatbot นี้มีวัตถุประสงค์เพื่อทดสอบการทำงานของ chatbot ร่วมกับ LLM และ RAG  
        เพื่อถามตอบคำถามเกี่ยวกับ  
        **พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562**  
        โดยใช้เนื้อหาที่รวบรวมมาจากอินเทอร์เน็ต
        """
    )

    selected_model = st.sidebar.selectbox(
        "🧠 เลือกโมเดล LLM",
        options=["llama3-70b-8192", "gemma-7b-it", "mixtral-8x7b-32768"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 ตัวอย่างคำถาม-คำตอบ")
    st.sidebar.markdown("""
    <div style='background-color: #f1f3f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
    <b>คำถาม:</b><br>ข้อมูลส่วนบุคคลหมายถึงอะไร?<br><br>
    👉 <b>คำตอบ:</b><br>ข้อมูลที่ทำให้สามารถระบุตัวบุคคลได้ เช่น ชื่อ ที่อยู่ เลขบัตรประชาชน
    </div>
    <div style='background-color: #f1f3f6; padding: 10px; border-radius: 10px;'>
    <b>คำถาม:</b><br>เจ้าของข้อมูลมีสิทธิอะไรบ้าง?<br><br>
    👉 <b>คำตอบ:</b><br>สิทธิในการเข้าถึง แก้ไข ลบข้อมูล ขอให้ระงับการใช้ หรือเพิกถอนความยินยอม
    </div>
    """, unsafe_allow_html=True)

    # Session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "query" not in st.session_state:
        st.session_state.query = ""

    # Load documents and build chain
    with st.spinner("📚 กำลังโหลดเอกสารและสร้างฐานข้อมูล..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = build_vectorstore(chunks)
        qa_chain = create_qa_chain(vectordb, selected_model)

    # Submit callback
    def submit_question():
        query = st.session_state.query.strip()
        if query:
            with st.spinner("🧠 คิดคำตอบ..."):
                answer = qa_chain.run(query)
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "model": selected_model
                })
                st.session_state.query = ""

    # Input box with callback
    st.text_input(
        "📥 พิมพ์คำถามของคุณ:",
        placeholder="เช่น ข้อมูลส่วนบุคคลคืออะไร?",
        key="query",
        on_change=submit_question
    )

    # ล้างประวัติ
    if st.button("🔁 ล้างประวัติการสนทนา"):
        st.session_state.chat_history = []

    # แสดงประวัติ
    if st.session_state.chat_history:
        st.markdown("### 🗂️ ประวัติการสนทนา")
        with st.container():
            st.markdown("<div style='max-width: 800px; margin-left: auto; margin-right: auto;'>", unsafe_allow_html=True)
            for i, item in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"**{i}. คำถาม:** {item['question']}")
                st.markdown(f"👉 **คำตอบ:** {item['answer']}")
                st.markdown(f"<span style='color: gray; font-size: 0.9em;'>🧠 โมเดลที่ใช้: {item['model']}</span>", unsafe_allow_html=True)
                st.markdown("---")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

