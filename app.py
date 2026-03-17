import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
import sqlite3
import datetime

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
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'sessions' not in st.session_state:
    st.session_state.sessions = []

# Đảm bảo thư mục lưu trữ tồn tại
SESSIONS_DIR = "data/sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Cấu hình database SQLite
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_new_session(title="Cuộc trò chuyện mới"):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO sessions (title) VALUES (?)', (title,))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Tạo thư mục cho session
    session_dir = os.path.join(SESSIONS_DIR, str(session_id))
    os.makedirs(session_dir, exist_ok=True)
    return session_id

def get_all_sessions():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT id, title, created_at FROM sessions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows

def save_chat(session_id, question, answer):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO history (session_id, question, answer) VALUES (?, ?, ?)', 
              (session_id, question, answer))
    
    # Tự động cập nhật tiêu đề session bằng câu hỏi đầu tiên nếu chưa có
    c.execute('SELECT count(*) FROM history WHERE session_id = ?', (session_id,))
    if c.fetchone()[0] == 1:
        new_title = question[:30] + "..." if len(question) > 30 else question
        c.execute('UPDATE sessions SET title = ? WHERE id = ?', (new_title, session_id))
        
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT question, answer, timestamp FROM history WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def switch_session(session_id):
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = get_chat_history(session_id)
    # Load lại FAISS index nếu có
    session_dir = os.path.join(SESSIONS_DIR, str(session_id))
    index_path = os.path.join(session_dir, "faiss_index")
    
    if os.path.exists(index_path):
        try:
            st.session_state.vectorstore = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
            
            # Tìm tên file tài liệu trong thư mục session (PDF hoặc DOCX)
            doc_files = [f for f in os.listdir(session_dir) if f.endswith(('.pdf', '.docx', '.doc'))]
            if doc_files:
                st.session_state.processed_file = doc_files[0]
            else:
                st.session_state.processed_file = "Tài liệu từ phiên cũ"
                
        except Exception as e:
            st.warning("⚠️ Không thể tải lại dữ liệu tài liệu của phiên này (có thể file lưu trữ bị lỗi hoặc đã bị xóa). Vui lòng tải lên lại tài liệu mới.")
            st.session_state.vectorstore = None
            st.session_state.processed_file = None
    else:
        st.session_state.vectorstore = None
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

init_db()

# Cập nhật danh sách session
st.session_state.sessions = get_all_sessions()

# Chọn session mặc định nếu chưa có
if st.session_state.current_session_id is None:
    if len(st.session_state.sessions) > 0:
        switch_session(st.session_state.sessions[0][0])
    else:
        new_id = create_new_session()
        st.session_state.sessions = get_all_sessions()
        switch_session(new_id)



# ============================
# Cấu trúc giao diện
# ============================

# Sidebar
with st.sidebar:
    st.title("⚙️ SmartDoc Chat")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        new_id = create_new_session()
        st.session_state.sessions = get_all_sessions()
        switch_session(new_id)
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 🕒 Lịch sử Chat")
    
    for s_id, s_title, s_time in st.session_state.sessions:
        # Highlight session đang chọn
        btn_label = f"💬 {s_title}"
        if s_id == st.session_state.current_session_id:
            btn_label = f"🟢 {s_title}"
            
        if st.button(btn_label, key=f"session_{s_id}", use_container_width=True):
            switch_session(s_id)
            st.rerun()

# Main content
st.markdown("<h1 class='main-title'>SmartDoc AI - Q&A System</h1>", unsafe_allow_html=True)
st.write("Giải pháp tìm kiếm và sinh câu trả lời theo tài liệu dựa trên Large Language Models.")

# Hiển thị lịch sử chat trong main screen (giống Gemini)
for q, a, t in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

st.markdown("---")

# 1. Khu vực Upload hoặc Hiển thị File đã Upload
if st.session_state.vectorstore is not None and st.session_state.processed_file:
    # Nếu đã có VectorDB, giấu khung upload đi và hiện thông báo
    st.success(f"✅ Tài liệu đang sử dụng cho phiên này: **{st.session_state.processed_file}**")
    
    # Cho phép xóa để upload file khác
    if st.button("🔄 Thay thế bằng tài liệu khác", type="secondary"):
        st.session_state.vectorstore = None
        st.session_state.processed_file = None
        st.rerun()
else:
    # Hiện khung upload nếu chưa có DB
    uploaded_file = st.file_uploader("📥 Tải lên tài liệu (PDF, DOCX, DOC) cho phiên này", type=['pdf', 'docx', 'doc'])

    if uploaded_file is not None and uploaded_file.name != st.session_state.processed_file:
        with st.spinner("Đang phân tích và gán AI cho tài liệu..."):
            # Lưu file tài liệu và FAISS vào thư mục của session
            session_dir = os.path.join(SESSIONS_DIR, str(st.session_state.current_session_id))
            file_path = os.path.join(session_dir, uploaded_file.name)
            index_path = os.path.join(session_dir, "faiss_index")
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            try:
                # Document Loader
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext == '.pdf':
                    loader = PDFPlumberLoader(file_path)
                elif file_ext in ['.docx', '.doc']:
                    loader = Docx2txtLoader(file_path)
                else:
                    st.error("❌ Định dạng file không được hỗ trợ.")
                    st.stop()
                docs = loader.load()

                # Text Splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                documents = text_splitter.split_documents(docs)
                
                if not documents:
                    st.error("❌ Không tìm thấy văn bản nào trong tài liệu này. File có thể là bản scan hoặc bị mã hóa chữ.")
                else:
                    logger.info(f"Processing {len(documents)} chunks")

                    # Khởi tạo và LƯU Vector Store
                    st.session_state.vectorstore = FAISS.from_documents(documents, embedder)
                    st.session_state.vectorstore.save_local(index_path)
                    
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"Đã xử lý và lưu xong tài liệu: {uploaded_file.name}")
                    st.rerun() # Refresh lại để ẩn khung upload nhanh
            except Exception as e:
                st.error(f"⚠️ Có lỗi khi xử lý tài liệu: {str(e)}")

st.markdown("---")

# 2. Khu vực đặt câu hỏi nếu đã xử lý file
if st.session_state.vectorstore is not None:    
    # Sử dụng chat_input thay vì form tĩnh giống Gemini
    user_question = st.chat_input("Hỏi AI về tài liệu của bạn...")

    if user_question:
        # Hiển thị câu hỏi ngay lập tức trên UI
        with st.chat_message("user"):
            st.write(user_question)
            
        with st.spinner("AI đang trả lời..."):
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
                    prompt_template = """Bạn là một trợ lý AI phân tích tài liệu chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA VÀO NGỮ CẢNH ĐƯỢC CUNG CẤP.
Tuyệt đối KHÔNG ĐƯỢC bịa đặt thông tin. Nếu trong ngữ cảnh không có thông tin để trả lời, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".

[NGỮ CẢNH BẮT ĐẦU]
{context}
[NGỮ CẢNH KẾT THÚC]

Câu hỏi của người dùng: {question}
Hãy trả lời câu hỏi trên bằng Tiếng Việt một cách ngắn gọn, súc tích (khoảng 3-4 câu) và sử dụng định dạng Markdown (ví dụ: in đậm, list) nếu cần thiết.
Trả lời:"""
                else:
                    prompt_template = """You are a professional document analysis AI assistant. Your task is to answer the user's question BASED ON THE PROVIDED CONTEXT.
Absolutely DO NOT fabricate information. If the context does not contain the information to answer, say "I cannot find this information in the document".

[CONTEXT START]
{context}
[CONTEXT END]

User's question: {question}
Please answer the question concisely (about 3-4 sentences) and use Markdown format (e.g., bold, list) if necessary.
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
                
                # Lưu lịch sử hội thoại vào CSDL và cập nhật session state
                save_chat(st.session_state.current_session_id, user_question, response['result'])
                st.session_state.chat_history = get_chat_history(st.session_state.current_session_id)
                st.session_state.sessions = get_all_sessions() # Cập nhật title nếu cần
                
                # Hiển thị câu trả lời ngay lập tức
                with st.chat_message("assistant"):
                    st.write(response['result'])
            except Exception as e:
                error_msg = str(e)
                if "Connection refused" in error_msg or "Failed to connect" in error_msg:
                    st.error("🚨 Không thể kết nối đến Ollama. Vui lòng kiểm tra chắc chắn bạn đã bật phần mềm Ollama dưới Local.")
                else:
                    st.error(f"⚠️ Có lỗi từ mô hình AI: {error_msg}")
else:
    st.info("💡 Vui lòng tải lên tài liệu (PDF, DOCX, DOC) để bắt đầu đặt câu hỏi cho phiên này.")
