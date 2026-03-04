import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Colors
st.markdown("""
    <style>
    .main-title {
        color: #007BFF;
    }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None

import torch

# Cấu hình model Singleton để tránh load lại
@st.cache_resource
def load_embedding_model():
    # Detect device (cuda or cpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm_model():
    return Ollama(
        model="qwen2.5:7b",
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1
    )

embedder = load_embedding_model()
llm = load_llm_model()

# ============================
# Cấu trúc giao diện
# ============================

# Sidebar
with st.sidebar:
    st.title("⚙️ Cấu hình hệ thống")
    st.info("Hệ thống RAG sử dụng Ollama: Qwen2.5:7b và Embedding MPNet đa ngôn ngữ.")
    st.markdown("---")
    st.markdown("### 📝 Hướng dẫn")
    st.markdown("- Tải lên 1 file PDF (khuyến nghị < 50MB).")
    st.markdown("- Chờ ứng dụng phân tích.")
    st.markdown("- Gõ câu hỏi vào thanh Chat.")

# Main content
st.markdown("<h1 class='main-title'>SmartDoc AI - Q&A System</h1>", unsafe_allow_html=True)
st.write("Giải pháp tìm kiếm và sinh câu trả lời theo tài liệu dựa trên Large Language Models.")

# 1. Khu vực Upload
uploaded_file = st.file_uploader("📥 Tải lên tài liệu PDF của bạn", type=['pdf'])

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file:
    with st.spinner("Đang xử lý tài liệu (Loading & Chunking)..."):
        # Lưu file tạm để PDFPlumber đọc
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        try:
            # Document Loader
            loader = PDFPlumberLoader(temp_path)
            docs = loader.load()

            # Text Splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            documents = text_splitter.split_documents(docs)
            logger.info(f"Processing {len(documents)} chunks")

            # Khởi tạo Vector Store
            st.session_state.vectorstore = FAISS.from_documents(documents, embedder)
            st.session_state.processed_file = uploaded_file.name
            st.success(f"Đã xử lý xong file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Có lỗi khi xử lý tài liệu: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

elif uploaded_file is None:
    # Reset state khi người dùng bấm dấu X tắt file
    st.session_state.vectorstore = None
    st.session_state.processed_file = None

st.markdown("---")

# 2. Khu vực đặt câu hỏi nếu đã xử lý file
if st.session_state.vectorstore is not None:
    st.subheader("💡 Đặt câu hỏi về tài liệu")
    
    # Bọc trong form để tránh rerun liên tục và dễ dàng clear
    with st.form(key='qa_form'):
        user_question = st.text_input("Nhập câu hỏi tại đây:")
        submit_button = st.form_submit_button(label='Gửi')

    if submit_button and user_question:
        with st.spinner("Processing your query..."):
            try:
                logger.info(f"Query: {user_question}")
                
                # Retriever setup
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3, "fetch_k": 20}
                )

                # Prompt logic (Language detection)
                vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
                is_vietnamese = any(char in user_question.lower() for char in vietnamese_chars)

                if is_vietnamese:
                    prompt_template = """Su dung ngu canh sau day de tra loi cau hoi.
Neu ban khong biet, chi can noi la ban khong biet.
Tra loi ngan gon (3-4 cau) BAT BUOC bang tieng Viet.

Ngu canh: {context}

Cau hoi: {question}

Tra loi:"""
                else:
                    prompt_template = """Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep answer concise (3-4 sentences).

Context: {context}

Question: {question}

Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )

                # QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": PROMPT}
                )

                response = qa_chain.invoke({"query": user_question})
                
                if "source_documents" in response:
                    logger.info(f"Retrieved {len(response['source_documents'])} documents")
                
                # Hiển thị câu trả lời
                st.markdown("### 🤖 Response:")
                st.info(response['result'])
            except Exception as e:
                st.error(f"Có lỗi khi tạo câu trả lời: {str(e)}")
else:
    st.warning("Vui lòng upload một file PDF để bắt đầu hệ thống hỏi đáp.")
